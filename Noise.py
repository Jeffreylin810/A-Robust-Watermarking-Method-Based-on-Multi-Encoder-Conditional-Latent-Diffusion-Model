from noise_layers import *
import numpy as np
import torch.nn as nn

class Noise(nn.Module):

	def __init__(self, attack_profile="pixel_bridge_combined"):
		super(Noise, self).__init__()
		self.attack_profile = attack_profile
		# NOTE(pixel-bridge): the bridge stays the same across profiles; only the
		# pixel-space attack set changes for diagnosis.
		layer_specs = self._build_layers(attack_profile)
		layers = list(layer_specs)
		# layers = ["GF(2)"]
		# layers = ["GF(3)"]
		# layers = ["GF(4)"]
		# layers = ["JpegTest(50)"]
		# layers = ["JpegTest(40)"]
		# layers = ["JpegTest(30)"]
		# layers = ["Identity()"]
		# layers = ["Dropout(0.7)"]
		# layers = ["Dropout(0.2)"]
		# layers = ["Dropout(0.1)"]
		# p=0.035
		# layers = ["Crop(0.1871, 0.1871)"]
		# p = 0.045
		# layers = ["Crop(0.21213, 0.21213)"]
		# p = 0.055
		# layers = ["Crop(0.23452, 0.23452)"]
		# layers = ["Cropout(0.8367, 0.8367)"]
		# layers = ["Cropout(0.5477, 0.5477)"]
		# layers = ["Cropout(0.4472, 0.4472)"]
		# layers = ["SP(0.3)"]
		# layers = ["SP(0.2)"]
		# layers = ["SP(0.1)"]
		# layers = ["MF(3)"]
		# layers = ["MF(5)"]
		# layers = ["MF(7)"]
		# layers = ["MF(9)"]
		# layers = ["GN(0.1)"]
		# layers = ["GN(0.2)"]
		# layers = ["GN(0.3)"]
		# layers = ["Adjust_Brightness(1.1)"]
		# layers = ["Adjust_Brightness(1.3)"]
		# layers = ["Adjust_Brightness(1.5)"]
		# layers = ["Adjust_Contrast(1.1)"]
		# layers = ["Adjust_Contrast(1.3)"]
		# layers = ["Adjust_Contrast(1.5)"]
		# layers = ["Adjust_hue(0.1)"]
		# layers = ["Adjust_hue(0.2)"]
		# layers = ["Adjust_hue(0.3)"]
		# layers = ["WebP(50)"]
		# layers = ["WebP(30)"]
		# layers = ["WebP(10)"]

		for i in range(len(layers)):
			layers[i] = eval(layers[i])
		# NOTE(pixel-bridge): keep string specs for startup logging and debugging.
		self.layer_descriptions = layer_specs
		self.noise = nn.Sequential(*layers)

	def _build_layers(self, attack_profile):
		if attack_profile == "identity_only":
			# NOTE(pixel-bridge): this isolates the frozen VAE round-trip bridge
			# from extra pixel attack damage during joint training.
			return ["Identity()"]
		if attack_profile == "identity_gn":
			# NOTE(pixel-bridge): first robustness add-back for the 256/32 branch.
			return ["Combined([Identity(),GN(0.1)])"]
		if attack_profile == "identity_gf":
			# NOTE(pixel-bridge): filter-only add-back branch reused from the
			# earlier robustness progression.
			return ["Combined([Identity(),GF(2)])"]
		if attack_profile == "identity_gn_gf":
			# NOTE(pixel-bridge): combined light-corruption branch reused from the
			# earlier robustness progression.
			return ["Combined([Identity(),GN(0.1),GF(2)])"]
		if attack_profile == "identity_crop":
			# NOTE(pixel-bridge): legacy crop alias kept for compatibility.
			return ["Combined([Identity(),Crop(0.15, 0.15)])"]
		if attack_profile == "identity_crop_light":
			# NOTE(pixel-bridge): lighter crop stage for crop-severity sweeps.
			return ["Combined([Identity(),Crop(0.10, 0.10)])"]
		if attack_profile == "identity_crop_medium":
			# NOTE(pixel-bridge): medium crop stage for crop-severity sweeps.
			return ["Combined([Identity(),Crop(0.15, 0.15)])"]
		if attack_profile == "identity_crop_heavy":
			# NOTE(pixel-bridge): heavier crop stage for crop-severity sweeps.
			return ["Combined([Identity(),Crop(0.20, 0.20)])"]
		if attack_profile == "identity_jpeg":
			# NOTE(pixel-bridge): compression-specific add-back branch.
			return ["Combined([Identity(),Jpeg(50)])"]
		if attack_profile == "identity_webp":
			# NOTE(pixel-bridge): alternative compression-specific add-back branch.
			return ["Combined([Identity(),WebP(50)])"]
		if attack_profile == "identity_gn_sp_webp":
			# NOTE(pixel-bridge): milder non-geometry curriculum without
			# brightness/contrast pressure, used to recover fidelity between C4
			# and the full nongeom C5 branch.
			return ["Combined([Identity(),GN(0.1),SP(0.1),WebP(50)])"]
		if attack_profile == "identity_gn_sp_webp_bc":
			# NOTE(pixel-bridge): non-geometry robustness add-back for the
			# bitplane branch. Keep Crop/Rotate separate because they are
			# alignment/geometry failures rather than bit representation failures.
			return ["Combined([Identity(),GN(0.1),SP(0.1),WebP(50),Adjust_Brightness(1.1),Adjust_Contrast(1.1)])"]
		if attack_profile == "identity_nongeom_formal":
			# Formal non-geometric base-training suite. Geometry is intentionally
			# excluded from the thesis scope. Jpeg is differentiable; WebP is kept
			# out of training to avoid CPU/file-I/O pressure.
			return ["Combined([Identity(),GN(0.1),SP(0.1),Jpeg(50),GF(2),Dropout(0.1),MF(3)])"]
		if attack_profile == "identity_nongeom_curriculum_light":
			# First non-geometric curriculum stage after an identity-trained base.
			# Combined samples uniformly: 3 Identity entries out of 7 choices
			# produce an actual Identity probability of 3/7 (about 42.9%).
			# Corruptions are intentionally mild so message learning is preserved.
			return ["Combined([Identity(),Identity(),Identity(),GN(0.01),SP(0.02),Jpeg(70),GF(1)])"]
		if attack_profile == "identity_nongeom_targeted_weak":
			# Targeted non-geometric continuation that deliberately emphasizes the
			# weakest measured attack families from the formal BA table:
			# dropout, stronger Gaussian noise, and salt-pepper. Two Identity
			# entries keep a clean-anchor probability of 2/8 (25%), while Jpeg/GF
			# remain as lighter regularizers so the model does not overfit only the
			# weak families.
			return [
				"Combined([Identity(),Identity(),GN(0.3),GN(0.09),Dropout(0.3),SP(0.2),Jpeg(50),GF(2)])"
			]
		if attack_profile == "identity_nongeom_dropout_focus":
			# Scheduling probe after the weak-attack continuation. Dropout remains
			# the min-BA attack, so this profile increases dropout sampling pressure
			# while keeping GN/SP/Jpeg/GF as guards against narrow overfitting.
			return [
				"Combined([Identity(),Dropout(0.3),Dropout(0.3),Dropout(0.2),GN(0.3),SP(0.2),Jpeg(50),GF(2)])"
			]
		if attack_profile == "identity_nongeom_dropout_margin":
			# Base-side margin probe after the exp34 validation base. This keeps
			# the same non-geometry guard families as dropout_focus, but replaces
			# the weaker Dropout(0.2) branch with another native Dropout(0.3).
			# Use only for short continuation; it intentionally tests whether the
			# last Dropout r03 gap belongs to the base watermark margin.
			return [
				"Combined([Identity(),Dropout(0.3),Dropout(0.3),Dropout(0.3),GN(0.3),SP(0.2),Jpeg(50),GF(2)])"
			]
		if attack_profile == "identity_gn_crop":
			# NOTE(pixel-bridge): mixed noise + geometry branch reused from the
			# earlier robustness progression.
			return ["Combined([Identity(),GN(0.1),Crop(0.15, 0.15)])"]
		if attack_profile == "identity_gn_crop_gf":
			# NOTE(pixel-bridge): three-family mixed branch reused from the
			# earlier robustness progression.
			return ["Combined([Identity(),GN(0.1),GF(2),Crop(0.15, 0.15)])"]
		if attack_profile == "pixel_bridge_combined":
			# NOTE(pixel-bridge): this is the previous default joint-training setup.
			return ["Combined([JpegMask(50),Jpeg(10),Identity(),GF(2),Crop(0.15, 0.15)])"]
		raise ValueError(f"Unsupported attack_profile: {attack_profile}")

	def forward(self, image_and_cover):
		noised_image = self.noise(image_and_cover)
		return noised_image
