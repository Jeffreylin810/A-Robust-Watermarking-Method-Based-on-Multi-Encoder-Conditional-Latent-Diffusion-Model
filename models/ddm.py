import json
import math
import os
import resource
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import utils
from Noise import Noise
from models.unet import DiffusionUNet
from tensorboardX import SummaryWriter

try:
    from diffusers import AutoencoderKL
except ImportError:
    AutoencoderKL = None

try:
    import kornia.losses as kornia_losses
except ImportError:
    kornia_losses = None


def data_transform(x):
    return 2 * x - 1.0


def inverse_data_transform(x):
    return torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)


class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1.0 - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def _unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def _profile_sync(diffusion):
    if getattr(diffusion, "_detailed_profile", False) and torch.cuda.is_available():
        torch.cuda.synchronize()


def _profile_add(diffusion, name, value):
    if getattr(diffusion, "_detailed_profile", False):
        diffusion._profile_sums[name] = diffusion._profile_sums.get(name, 0.0) + value


def noise_estimation_loss(
    diffusion,
    model,
    x0,
    t,
    e,
    betas,
    message,
    epoch,
    combined_attack,
    target_channels,
    target_rgb_gt=None,
):
    _profile_sync(diffusion)
    section_start = time.time()
    a = (1 - betas).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x_cond = x0[:, :target_channels, :, :]
    x_target = x0[:, target_channels : target_channels * 2, :, :]
    x_noisy = x_target * a.sqrt() + e * (1.0 - a).sqrt()
    core_model = _unwrap_model(model)
    _profile_sync(diffusion)
    _profile_add(diffusion, "forward_setup_time", time.time() - section_start)

    _profile_sync(diffusion)
    section_start = time.time()
    encoded_latent = core_model.encode_watermark(torch.cat([x_cond, x_noisy], dim=1), t.float(), message)
    _profile_sync(diffusion)
    _profile_add(diffusion, "encode_watermark_time", time.time() - section_start)

    rgb_loss_weight = getattr(diffusion.config.watermarking, "rgb_loss_weight", 0)
    rgb_psnr_hinge_weight = getattr(diffusion.config.watermarking, "rgb_psnr_hinge_weight", 0)
    needs_rgb_fidelity = rgb_loss_weight > 0 or rgb_psnr_hinge_weight > 0

    # NOTE(pixel-bridge): the default path attacks the decoded RGB result and
    # then projects it back to latent before message recovery. The
    # latent_direct option is reserved for the bridge-schedule ablation: it
    # keeps the same RGB fidelity objective but lets message recovery bypass
    # the VAE/RGB round-trip during the first training stage.
    _profile_sync(diffusion)
    section_start = time.time()
    if diffusion.message_recovery_path == "latent_direct":
        attacked_latent = encoded_latent
        encoded_rgb = (
            diffusion.decode_latents(encoded_latent, track_grad=True)
            if needs_rgb_fidelity
            else None
        )
    else:
        attacked_output = diffusion.apply_attack_bridge(
            encoded_latent,
            x_cond,
            return_encoded_rgb=needs_rgb_fidelity,
        )
        if needs_rgb_fidelity:
            attacked_latent, encoded_rgb = attacked_output
        else:
            attacked_latent = attacked_output
            encoded_rgb = None
    _profile_sync(diffusion)
    _profile_add(diffusion, "attack_bridge_time", time.time() - section_start)

    _profile_sync(diffusion)
    section_start = time.time()
    output_message = core_model.decode_message(attacked_latent)
    _profile_sync(diffusion)
    _profile_add(diffusion, "decode_message_time", time.time() - section_start)

    _profile_sync(diffusion)
    section_start = time.time()
    criterion = nn.MSELoss()
    x0_loss = criterion(encoded_latent, x_target)
    message_loss_per_bit = nn.functional.mse_loss(output_message, message, reduction="none")
    bit_weights = getattr(diffusion, "bit_weights", None)
    if bit_weights is not None:
        message_loss = (message_loss_per_bit * bit_weights).mean()
    else:
        message_loss = message_loss_per_bit.mean()
    rgb_loss = x0_loss.new_zeros(())
    if needs_rgb_fidelity:
        if diffusion.latent_enabled:
            if encoded_rgb is None:
                encoded_rgb = diffusion.decode_latents(encoded_latent, track_grad=True)
            rgb_loss_target = getattr(diffusion.config.watermarking, "rgb_loss_target", "vae_cover")
            if rgb_loss_target == "original_gt":
                if target_rgb_gt is None:
                    raise ValueError("rgb_loss_target='original_gt' requires target_rgb_gt")
                target_rgb = target_rgb_gt
            elif rgb_loss_target == "vae_cover":
                target_rgb = diffusion.decode_latents(x_target, track_grad=False)
            else:
                raise ValueError(f"Unsupported rgb_loss_target: {rgb_loss_target}")
            rgb_loss = criterion(encoded_rgb, target_rgb)
        else:
            rgb_loss = criterion(encoded_latent, x_target)

    rgb_psnr_hinge_loss = x0_loss.new_zeros(())
    if rgb_psnr_hinge_weight > 0:
        rgb_psnr_target = getattr(diffusion.config.watermarking, "rgb_psnr_target", 30.0)
        rgb_psnr_max_value = getattr(diffusion.config.watermarking, "rgb_psnr_max_value", 2.0)
        target_mse = (rgb_psnr_max_value**2) / (10.0 ** (rgb_psnr_target / 10.0))
        target_mse_tensor = rgb_loss.new_tensor(target_mse)
        rgb_psnr_hinge_loss = torch.relu(rgb_loss - target_mse_tensor)

    decoded_rounded = output_message.detach().cpu().numpy().round().clip(0, 1)
    bitwise_avg_err = np.sum(np.abs(decoded_rounded - message.detach().cpu().numpy())) / (
        x0.shape[0] * message.shape[1]
    )

    if needs_rgb_fidelity:
        rgb_psnr_max_value = getattr(diffusion.config.watermarking, "rgb_psnr_max_value", 2.0)
        max_value_tensor = rgb_loss.new_tensor(rgb_psnr_max_value)
        psnr = 20.0 * torch.log10(max_value_tensor / torch.sqrt(torch.clamp(rgb_loss.detach(), min=1.0e-12)))
    elif diffusion.latent_enabled or kornia_losses is None:
        psnr = torch.zeros(1, device=x0.device).squeeze(0)
    else:
        psnr = kornia_losses.psnr_loss(encoded_latent.detach(), x_target, 2)

    x0_loss_weight = getattr(diffusion.config.watermarking, "x0_loss_weight", 1)
    message_loss_weight = getattr(diffusion.config.watermarking, "message_loss_weight", 10)
    rgb_fidelity_loss = rgb_loss_weight * rgb_loss + rgb_psnr_hinge_weight * rgb_psnr_hinge_loss

    if combined_attack:
        total_loss = x0_loss_weight * x0_loss + message_loss_weight * message_loss + rgb_fidelity_loss
    else:
        if epoch > 5:
            total_loss = 10 * x0_loss + message_loss + rgb_fidelity_loss
        else:
            total_loss = x0_loss_weight * x0_loss + message_loss_weight * message_loss + rgb_fidelity_loss

    _profile_sync(diffusion)
    _profile_add(diffusion, "loss_compute_time", time.time() - section_start)

    return x0_loss, message_loss, rgb_loss, bitwise_avg_err, total_loss, psnr


class DenoisingDiffusion(object):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device
        self.output_dir = getattr(config.data, "output_dir", config.data.data_dir)
        self.latent_enabled = getattr(config, "latent", None) is not None and config.latent.enabled
        self.vae = None
        self.latent_scaling_factor = 1.0
        self.attack_profile = getattr(config.watermarking, "attack_profile", "pixel_bridge_combined")
        self.message_recovery_path = getattr(
            config.watermarking,
            "message_recovery_path",
            "pixel_bridge",
        )
        if self.message_recovery_path not in {"pixel_bridge", "latent_direct"}:
            raise ValueError(
                "Unsupported watermarking.message_recovery_path: "
                f"{self.message_recovery_path}"
            )
        self.bit_weights = self._load_bit_weights()
        # NOTE(pixel-bridge): attacks are executed in RGB after decoding latent outputs.
        self.pixel_noise = Noise(attack_profile=self.attack_profile).to(self.device)

        self._configure_model_space()

        self.model = DiffusionUNet(config)
        self.model.to(self.device)
        self.model = torch.nn.DataParallel(self.model)

        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)

        self.optimizer = utils.optimize.get_optimizer(self.config, self.model.parameters())
        self.start_epoch, self.step = 0, 0
        self.resume_elapsed_sec = 0.0
        self.resume_cpu_user_sec = 0.0
        self.resume_cpu_sys_sec = 0.0

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )

        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = self.betas.shape[0]
        print(
            "Message recovery path: {}; training attack profile: {} -> {}".format(
                self.message_recovery_path,
                self.attack_profile,
                ", ".join(self.pixel_noise.layer_descriptions),
            )
        )

    def _load_bit_weights(self):
        bit_weight_json = getattr(self.config.watermarking, "bit_weight_json", None)
        if not bit_weight_json:
            return None
        with open(bit_weight_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
        values = payload.get("weights", payload)
        expected_length = int(self.config.watermarking.message_length)
        if len(values) != expected_length:
            raise ValueError(
                f"bit_weight_json length mismatch: got {len(values)}, expected {expected_length}"
            )
        weights = torch.tensor(values, dtype=torch.float32, device=self.device).view(1, -1)
        print(f"Loaded bit weights from {bit_weight_json}: {values}")
        return weights

    @staticmethod
    def _format_duration(seconds):
        seconds = int(max(seconds, 0))
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _elapsed_seconds(self, train_start_time):
        return self.resume_elapsed_sec + (time.time() - train_start_time)

    @staticmethod
    def _cpu_times():
        self_usage = resource.getrusage(resource.RUSAGE_SELF)
        child_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        return (
            self_usage.ru_utime + child_usage.ru_utime,
            self_usage.ru_stime + child_usage.ru_stime,
        )

    def _elapsed_cpu_times(self, train_start_cpu):
        current_user_sec, current_sys_sec = self._cpu_times()
        start_user_sec, start_sys_sec = train_start_cpu
        return (
            self.resume_cpu_user_sec + current_user_sec - start_user_sec,
            self.resume_cpu_sys_sec + current_sys_sec - start_sys_sec,
        )

    def _configure_model_space(self):
        if not self.latent_enabled:
            self.config.model.resolution = self.config.data.image_size
            self.target_channels = self.config.model.out_ch
            self.sample_shape = (
                self.config.model.out_ch,
                self.config.model.resolution,
                self.config.model.resolution,
            )
            return

        if AutoencoderKL is None:
            raise ImportError(
                "diffusers is required for latent mode. Install it in the current environment first."
            )

        vae_path = self.config.latent.model_path
        if os.path.isdir(os.path.join(vae_path, "vae")):
            self.vae = AutoencoderKL.from_pretrained(vae_path, subfolder="vae")
        else:
            self.vae = AutoencoderKL.from_pretrained(vae_path)

        self.vae.to(self.device)
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False

        scaling = getattr(self.config.latent, "scaling_factor", None)
        if scaling is None:
            scaling = getattr(self.vae.config, "scaling_factor", 1.0)
        self.latent_scaling_factor = scaling

        with torch.no_grad():
            dummy = torch.zeros(
                1,
                self.config.data.channels,
                self.config.data.image_size,
                self.config.data.image_size,
                device=self.device,
            )
            latent = self.encode_images(data_transform(dummy))

        latent_channels = latent.shape[1]
        latent_resolution = latent.shape[2]
        self.config.model.in_channels = latent_channels
        self.config.model.out_ch = latent_channels
        self.config.model.resolution = latent_resolution
        self.config.latent.channels = latent_channels
        self.config.latent.resolution = latent_resolution
        self.target_channels = latent_channels
        self.sample_shape = (latent_channels, latent_resolution, latent_resolution)

    def encode_images(self, x, track_grad=False):
        if not self.latent_enabled:
            return x
        grad_context = torch.enable_grad() if track_grad else torch.no_grad()
        with grad_context:
            posterior = self.vae.encode(x).latent_dist
            if getattr(self.config.latent, "deterministic_encode", True):
                latents = posterior.mode()
            else:
                latents = posterior.sample()
        return latents * self.latent_scaling_factor

    def decode_latents(self, latents, track_grad=False):
        if not self.latent_enabled:
            return latents
        grad_context = torch.enable_grad() if track_grad else torch.no_grad()
        with grad_context:
            decoded = self.vae.decode(latents / self.latent_scaling_factor).sample
        return decoded

    def apply_attack_bridge(self, encoded_latent, cover_latent, return_encoded_rgb=False):
        if not self.latent_enabled:
            _profile_sync(self)
            section_start = time.time()
            attacked_latent = self.pixel_noise([encoded_latent, cover_latent])
            if return_encoded_rgb:
                return attacked_latent, encoded_latent
            return attacked_latent

        # NOTE(pixel-bridge): decode latent prediction to RGB, run pixel-space
        # attacks, then project the attacked image back into latent space.
        attack_uses_cover = getattr(self.config.watermarking, "attack_uses_cover", True)

        _profile_sync(self)
        section_start = time.time()
        encoded_rgb = self.decode_latents(encoded_latent, track_grad=True)
        _profile_sync(self)
        _profile_add(self, "bridge_decode_encoded_time", time.time() - section_start)

        if attack_uses_cover:
            _profile_sync(self)
            section_start = time.time()
            cover_rgb = self.decode_latents(cover_latent, track_grad=True)
            _profile_sync(self)
            _profile_add(self, "bridge_decode_cover_time", time.time() - section_start)
        else:
            cover_rgb = encoded_rgb
            _profile_add(self, "bridge_decode_cover_time", 0.0)

        _profile_sync(self)
        section_start = time.time()
        attacked_rgb = self.pixel_noise([encoded_rgb, cover_rgb])
        _profile_sync(self)
        _profile_add(self, "bridge_pixel_noise_time", time.time() - section_start)

        # NOTE(pixel-bridge): re-encoding keeps the decoder operating in the
        # same latent domain as the rest of the watermark pipeline.
        _profile_sync(self)
        section_start = time.time()
        attacked_latent = self.encode_images(attacked_rgb, track_grad=True)
        _profile_sync(self)
        _profile_add(self, "bridge_encode_attacked_time", time.time() - section_start)
        if return_encoded_rgb:
            return attacked_latent, encoded_rgb
        return attacked_latent

    def prepare_batch(self, x, return_rgb_target=False):
        x = x.to(self.device)
        x = data_transform(x)
        cond = x[:, : self.config.data.channels, :, :]
        target = x[:, self.config.data.channels : self.config.data.channels * 2, :, :]
        target_rgb = target
        if self.latent_enabled:
            cond = self.encode_images(cond)
            target = self.encode_images(target)
        if return_rgb_target:
            return torch.cat([cond, target], dim=1), cond, target, target_rgb
        return torch.cat([cond, target], dim=1), cond, target

    def random_message(self, batch_size):
        return torch.tensor(
            np.random.choice([0, 1], (batch_size, self.config.watermarking.message_length)),
            device=self.device,
            dtype=torch.float32,
        )

    def random_sample(self, batch_size):
        return torch.randn(batch_size, *self.sample_shape, device=self.device)

    def load_ddm_ckpt(self, load_path, ema=False):
        checkpoint = utils.logging.load_checkpoint(load_path, self.device)
        self.start_epoch = checkpoint["epoch"]
        self.step = checkpoint["step"]
        self.resume_elapsed_sec = checkpoint.get("elapsed_sec", 0.0)
        self.resume_cpu_user_sec = checkpoint.get("cpu_user_sec", 0.0)
        self.resume_cpu_sys_sec = checkpoint.get("cpu_sys_sec", 0.0)
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if getattr(self.config.optim, "reset_lr_on_resume", False):
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.config.optim.lr
            print("=> reset resumed optimizer learning rate to {}".format(self.config.optim.lr))
        self.ema_helper.load_state_dict(checkpoint["ema_helper"])
        if ema:
            self.ema_helper.ema(self.model)
        print("=> loaded checkpoint '{}' (epoch {}, step {})".format(load_path, checkpoint["epoch"], self.step))

    def _checkpoint_state(self, epoch, train_start_time, train_start_cpu):
        save_time = time.time()
        elapsed_sec = self.resume_elapsed_sec + (save_time - train_start_time)
        cpu_user_sec, cpu_sys_sec = self._elapsed_cpu_times(train_start_cpu)
        return {
            "epoch": epoch + 1,
            "step": self.step,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "ema_helper": self.ema_helper.state_dict(),
            "params": self.args,
            "config": self.config,
            "train_start_time": train_start_time,
            "save_time": save_time,
            "elapsed_sec": elapsed_sec,
            "cpu_user_sec": cpu_user_sec,
            "cpu_sys_sec": cpu_sys_sec,
        }

    def train(self, dataset):
        cudnn.benchmark = True
        train_loader, val_loader = dataset.get_loaders()

        if os.path.isfile(self.args.resume):
            self.load_ddm_ckpt(self.args.resume)

        writer = SummaryWriter("./logs/log")
        train_start_time = time.time()
        train_start_cpu = self._cpu_times()
        benchmark_mode = getattr(self.config.training, "benchmark_mode", False)
        benchmark_warmup_steps = int(getattr(self.config.training, "benchmark_warmup_steps", 0) or 0)
        benchmark_measure_steps = int(getattr(self.config.training, "benchmark_measure_steps", 0) or 0)
        benchmark_target_step = benchmark_warmup_steps + benchmark_measure_steps
        benchmark_result_path = getattr(
            self.config.training,
            "benchmark_result_path",
            os.path.join(self.output_dir, "logs", "benchmark_result.json"),
        )
        benchmark_started = False
        benchmark_finished = False
        benchmark_start_wall = None
        benchmark_start_cpu = None
        timing_log_path = os.path.join(self.output_dir, "logs", "training_time.txt")
        os.makedirs(os.path.dirname(timing_log_path), exist_ok=True)
        detailed_profile = getattr(self.config.training, "detailed_profile", False)
        profile_interval = max(1, getattr(self.config.training, "profile_interval", 10))
        profile_log_path = os.path.join(self.output_dir, "logs", "detailed_profile.csv")
        profile_sums = {}
        profile_count = 0
        self._detailed_profile = detailed_profile
        self._profile_sums = profile_sums

        def sync_if_profiled():
            if detailed_profile and torch.cuda.is_available():
                torch.cuda.synchronize()

        def add_profile_time(name, value):
            if detailed_profile:
                profile_sums[name] = profile_sums.get(name, 0.0) + value

        if detailed_profile:
            with open(profile_log_path, "w", encoding="utf-8") as profile_log:
                profile_log.write(
                    "step,epoch,count,data_time,prepare_batch_time,sample_setup_time,"
                    "forward_loss_time,forward_setup_time,encode_watermark_time,"
                    "attack_bridge_time,decode_message_time,loss_compute_time,"
                    "bridge_decode_encoded_time,bridge_decode_cover_time,"
                    "bridge_pixel_noise_time,bridge_encode_attacked_time,"
                    "writer_time,zero_grad_time,backward_time,"
                    "optimizer_time,ema_time,step_time\n"
                )

        with open(timing_log_path, "a", encoding="utf-8") as timing_log:
            timing_log.write(
                "run_start step={} epoch={} image_size={} batch_size={} message_length={} "
                "x0_loss_weight={} message_loss_weight={} message_recovery_path={} "
                "attack_profile={} "
                "resumed_elapsed_sec={:.2f} resumed_cpu_user_sec={:.2f} resumed_cpu_sys_sec={:.2f}\n".format(
                    self.step,
                    self.start_epoch,
                    self.config.data.image_size,
                    self.config.training.batch_size,
                    self.config.watermarking.message_length,
                    getattr(self.config.watermarking, "x0_loss_weight", 1),
                    getattr(self.config.watermarking, "message_loss_weight", 10),
                    self.message_recovery_path,
                    self.attack_profile,
                    self.resume_elapsed_sec,
                    self.resume_cpu_user_sec,
                    self.resume_cpu_sys_sec,
                )
            )
        last_epoch = self.start_epoch - 1
        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            last_epoch = epoch
            print("epoch: ", epoch)
            data_start = time.time()
            data_time = 0
            for i, (x, _) in enumerate(train_loader):
                sync_if_profiled()
                step_start = time.time()
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                n = x.size(0)
                current_data_time = time.time() - data_start
                data_time += current_data_time
                add_profile_time("data_time", current_data_time)
                self.model.train()
                self.step += 1

                sync_if_profiled()
                section_start = time.time()
                x_pair, _, target, target_rgb_gt = self.prepare_batch(x, return_rgb_target=True)
                sync_if_profiled()
                add_profile_time("prepare_batch_time", time.time() - section_start)

                sync_if_profiled()
                section_start = time.time()
                e = torch.randn_like(target)
                message = self.random_message(target.shape[0])

                t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,), device=self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                sync_if_profiled()
                add_profile_time("sample_setup_time", time.time() - section_start)

                sync_if_profiled()
                section_start = time.time()
                x0_loss, message_loss, rgb_loss, bitwise_avg_err, total_loss, psnr = noise_estimation_loss(
                    self,
                    self.model,
                    x_pair,
                    t,
                    e,
                    self.betas,
                    message,
                    epoch,
                    self.config.watermarking.combined_attack,
                    self.target_channels,
                    target_rgb_gt=target_rgb_gt,
                )
                sync_if_profiled()
                add_profile_time("forward_loss_time", time.time() - section_start)

                section_start = time.time()
                writer.add_scalar("message_loss", message_loss, epoch)
                writer.add_scalar("x0_loss", x0_loss, epoch)
                writer.add_scalar("rgb_loss", rgb_loss, epoch)
                writer.add_scalar("all_loss", total_loss, epoch)
                add_profile_time("writer_time", time.time() - section_start)

                if self.step % 10 == 0:
                    pending_log_values = (
                        x0_loss.item(),
                        message_loss.item(),
                        rgb_loss.item(),
                        total_loss.item(),
                        psnr.item(),
                        bitwise_avg_err,
                    )

                section_start = time.time()
                self.optimizer.zero_grad()
                add_profile_time("zero_grad_time", time.time() - section_start)

                sync_if_profiled()
                section_start = time.time()
                total_loss.backward()
                sync_if_profiled()
                add_profile_time("backward_time", time.time() - section_start)

                sync_if_profiled()
                section_start = time.time()
                self.optimizer.step()
                sync_if_profiled()
                add_profile_time("optimizer_time", time.time() - section_start)

                sync_if_profiled()
                section_start = time.time()
                self.ema_helper.update(self.model)
                sync_if_profiled()
                add_profile_time("ema_time", time.time() - section_start)

                sync_if_profiled()
                step_time = time.time() - step_start
                add_profile_time("step_time", step_time)
                if detailed_profile:
                    profile_count += 1
                elapsed_sec = self._elapsed_seconds(train_start_time)
                writer.add_scalar("timing/elapsed_sec", elapsed_sec, self.step)
                writer.add_scalar("timing/step_time_sec", step_time, self.step)
                if self.step % 10 == 0:
                    x0_loss_value, message_loss_value, rgb_loss_value, total_loss_value, psnr_value, bitwise_err_value = (
                        pending_log_values
                    )
                    print(
                        f"step: {self.step}, x0_loss: {x0_loss_value}, message_loss: {message_loss_value}, "
                        f"rgb_loss: {rgb_loss_value}, bitwise_avg_err: {bitwise_err_value}, "
                        f"loss: {total_loss_value}, psnr: {psnr_value}, data time: {data_time / (i + 1)}, "
                        f"step time: {step_time:.4f}, "
                        f"elapsed: {self._format_duration(elapsed_sec)}"
                    )
                    if detailed_profile and profile_count > 0 and self.step % profile_interval == 0:
                        profile_names = (
                            "data_time",
                            "prepare_batch_time",
                            "sample_setup_time",
                            "forward_loss_time",
                            "forward_setup_time",
                            "encode_watermark_time",
                            "attack_bridge_time",
                            "decode_message_time",
                            "loss_compute_time",
                            "bridge_decode_encoded_time",
                            "bridge_decode_cover_time",
                            "bridge_pixel_noise_time",
                            "bridge_encode_attacked_time",
                            "writer_time",
                            "zero_grad_time",
                            "backward_time",
                            "optimizer_time",
                            "ema_time",
                            "step_time",
                        )
                        averages = {name: profile_sums.get(name, 0.0) / profile_count for name in profile_names}
                        print(
                            "profile avg over {} steps: data={:.4f}, prepare={:.4f}, sample={:.4f}, "
                            "forward_loss={:.4f}, attack_bridge={:.4f}, encode_wm={:.4f}, "
                            "decode_msg={:.4f}, bridge_dec_enc={:.4f}, bridge_dec_cover={:.4f}, "
                            "bridge_noise={:.4f}, bridge_enc_attacked={:.4f}, "
                            "loss_compute={:.4f}, backward={:.4f}, optimizer={:.4f}, ema={:.4f}, "
                            "step={:.4f}".format(
                                profile_count,
                                averages["data_time"],
                                averages["prepare_batch_time"],
                                averages["sample_setup_time"],
                                averages["forward_loss_time"],
                                averages["attack_bridge_time"],
                                averages["encode_watermark_time"],
                                averages["decode_message_time"],
                                averages["bridge_decode_encoded_time"],
                                averages["bridge_decode_cover_time"],
                                averages["bridge_pixel_noise_time"],
                                averages["bridge_encode_attacked_time"],
                                averages["loss_compute_time"],
                                averages["backward_time"],
                                averages["optimizer_time"],
                                averages["ema_time"],
                                averages["step_time"],
                            )
                        )
                        with open(profile_log_path, "a", encoding="utf-8") as profile_log:
                            profile_log.write(
                                "{},{},{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f}\n".format(
                                    self.step,
                                    epoch,
                                    profile_count,
                                    averages["data_time"],
                                    averages["prepare_batch_time"],
                                    averages["sample_setup_time"],
                                    averages["forward_loss_time"],
                                    averages["forward_setup_time"],
                                    averages["encode_watermark_time"],
                                    averages["attack_bridge_time"],
                                    averages["decode_message_time"],
                                    averages["loss_compute_time"],
                                    averages["bridge_decode_encoded_time"],
                                    averages["bridge_decode_cover_time"],
                                    averages["bridge_pixel_noise_time"],
                                    averages["bridge_encode_attacked_time"],
                                    averages["writer_time"],
                                    averages["zero_grad_time"],
                                    averages["backward_time"],
                                    averages["optimizer_time"],
                                    averages["ema_time"],
                                    averages["step_time"],
                                )
                            )
                        profile_sums = {}
                        self._profile_sums = profile_sums
                        profile_count = 0
                data_start = time.time()

                if self.step % self.config.training.validation_freq == 0:
                    self.model.eval()
                    self.sample_validation_patches(val_loader, self.step)

                save_checkpoints = getattr(self.config.training, "save_checkpoints", True)
                if save_checkpoints and (self.step % self.config.training.snapshot_freq == 0 or self.step == 1):
                    utils.logging.save_checkpoint(
                        self._checkpoint_state(epoch, train_start_time, train_start_cpu),
                        filename=os.path.join(self.output_dir, "ckpts", self.config.data.dataset + "_ddpm"),
                    )

                if benchmark_mode and (not benchmark_started) and self.step == benchmark_warmup_steps:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    benchmark_start_wall = time.time()
                    benchmark_start_cpu = self._cpu_times()
                    benchmark_started = True
                    print(
                        f"[benchmark] warmup complete at step {self.step}; "
                        f"starting timed region for next {benchmark_measure_steps} steps"
                    )

                if benchmark_mode and benchmark_started and self.step >= benchmark_target_step:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    measured_wall = time.time() - benchmark_start_wall
                    measured_cpu_user, measured_cpu_sys = self._elapsed_cpu_times(benchmark_start_cpu)
                    os.makedirs(os.path.dirname(benchmark_result_path), exist_ok=True)
                    with open(benchmark_result_path, "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "benchmark_mode": True,
                                "warmup_steps": benchmark_warmup_steps,
                                "measured_steps": benchmark_measure_steps,
                                "target_total_steps": benchmark_target_step,
                                "completed_step": self.step,
                                "measured_seconds": measured_wall,
                                "seconds_per_step": measured_wall / max(1, benchmark_measure_steps),
                                "cpu_user_seconds": measured_cpu_user,
                                "cpu_sys_seconds": measured_cpu_sys,
                                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                                "batch_size": self.config.training.batch_size,
                                "image_size": self.config.data.image_size,
                                "dataset_name": self.config.data.dataset,
                                "dataset_root": getattr(self.config.data, "dataset_root", ""),
                                "num_train_images": len(getattr(train_loader, "dataset", [])),
                                "seed": getattr(self.args, "seed", None),
                                "attack_profile": self.attack_profile,
                                "message_length": self.config.watermarking.message_length,
                                "output_dir": self.output_dir,
                            },
                            f,
                            ensure_ascii=False,
                            indent=2,
                        )
                    print(
                        f"[benchmark] completed {benchmark_measure_steps} timed steps "
                        f"in {measured_wall:.2f}s ({measured_wall / max(1, benchmark_measure_steps):.6f}s/step)"
                    )
                    benchmark_finished = True
                    break

            if benchmark_finished:
                break

        save_final_checkpoint = getattr(self.config.training, "save_final_checkpoint", True)
        if save_final_checkpoint and last_epoch >= self.start_epoch:
            utils.logging.save_checkpoint(
                self._checkpoint_state(last_epoch, train_start_time, train_start_cpu),
                filename=os.path.join(self.output_dir, "ckpts", self.config.data.dataset + "_ddpm_final"),
            )
        total_elapsed_sec = self._elapsed_seconds(train_start_time)
        total_cpu_user_sec, total_cpu_sys_sec = self._elapsed_cpu_times(train_start_cpu)
        with open(timing_log_path, "a", encoding="utf-8") as timing_log:
            timing_log.write(
                "run_end step={} epoch={} total_elapsed_sec={:.2f} total_elapsed={} "
                "cpu_user_sec={:.2f} cpu_sys_sec={:.2f}\n".format(
                    self.step,
                    last_epoch,
                    total_elapsed_sec,
                    self._format_duration(total_elapsed_sec),
                    total_cpu_user_sec,
                    total_cpu_sys_sec,
                )
            )
        print("Total training time: {}".format(self._format_duration(total_elapsed_sec)))
        writer.close()

    def sample_image(self, x_cond, x, message, last=True, patch_locs=None, patch_size=None):
        start_value = self.config.diffusion.num_diffusion_timesteps - 1
        num_samples = self.args.sampling_timesteps
        samples = []
        start_tolerance = 1
        tolerance = (start_value / (num_samples - start_tolerance) - start_tolerance) / (
            num_samples - start_tolerance
        )
        for i in range(num_samples):
            samples.append(start_value - i * start_tolerance)
            start_tolerance += tolerance
        samples = [math.floor(value) for value in samples]
        samples.reverse()
        seq = samples
        xs = utils.sampling.generalized_steps(x, x_cond, seq, message, self.model, self.betas, eta=0.0)
        return xs

    def sample_validation_patches(self, val_loader, step):
        image_folder = os.path.join(self.args.image_folder, self.config.data.dataset + str(self.config.data.image_size))
        with torch.no_grad():
            print(f"Processing a single batch of validation images at step: {step}")
            for x, _ in val_loader:
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                break

            n = x.size(0)
            x_cond_rgb = x[:, : self.config.data.channels, :, :]
            x_pair, x_cond, _ = self.prepare_batch(x)
            x_sample = self.random_sample(n)
            message = self.random_message(x_cond.shape[0])

            sampled = self.sample_image(x_cond, x_sample, message)
            x_pred = sampled[0][-1].to(self.device)
            if self.latent_enabled:
                x_pred = self.decode_latents(x_pred)
            x_pred = inverse_data_transform(x_pred)
            message_pred = sampled[2][-1]

            decoded_rounded = message_pred.detach().cpu().numpy().round().clip(0, 1)
            message_detached = message.detach().cpu().numpy()
            print("original: {}".format(message_detached))
            print("decoded : {}".format(decoded_rounded))
            print("error : {:.3f}".format(np.mean(np.abs(decoded_rounded - message_detached))))

            for i in range(n):
                utils.logging.save_image(x_cond_rgb[i], os.path.join(image_folder, str(step), f"{i}_cond.png"))
                utils.logging.save_image(x_pred[i], os.path.join(image_folder, str(step), f"{i}.png"))
