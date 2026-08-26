# Parameter Origin Notes

Chinese version: `docs/PARAMETER_ORIGIN_NOTES_zh.md`

## Purpose

This note records why several important settings in the later O256 line were not
arbitrary.

It is not a formal proof document. It is a historical bridge between early
exploration and the later curated StageE protocol.

## Why 256x256?

The earlier pixel-bridge line explicitly tested whether larger input resolution
and therefore a larger latent grid could rescue learnability for longer
messages.

The key historical clue appears in:

- `Pixel_bridge_256_msglen32_msgloss20_identity/md/CAPACITY_SCALING_256_MSG32_IDENTITY.md`

That branch was created because earlier lower-capacity settings suggested the
watermark signal was too weak, and 256x256 offered a way to test whether the
problem was partly capacity-driven rather than only bridge-driven.

Later paper-side branches standardized around 256x256 and treated it as the
cleaner research scale.

## Why 32-Bit Messages?

The earlier text archive contains explicit payload diagnostics for:

- 64-bit baseline
- 32-bit diagnostic
- 16-bit diagnostic
- 8-bit diagnostic

These branches were used to test whether the original payload was too heavy for
the latent watermark signal.

The logic that emerged was:

- 64 bits were too hard for the early setup
- 8 and 16 bits were useful for diagnosis, but too small to represent the
  intended practical target
- 32 bits became the workable middle ground once capacity and representation
  were improved

The later paper-side and O256 lines then treated 32 bits as the practical
target payload rather than only a toy diagnostic.

## Why `message_map_mode: bitplane`?

The early bitplane ablation was not added randomly. It came from observing that
some bits behaved like systematically weak bits under the old linear message
map.

The key historical note is:

- `Pixel_bridge_256_msglen32_r30_msgmap_bitplane_identity/md/MESSAGE_MAP_BITPLANE_ABLATION.md`

The rationale was:

- the linear map mixed all 32 bits too early
- some bit positions appeared easier to dilute or destabilize
- bitplane preserved bit identity longer before learned mixing

This is why bitplane should be treated as a historically motivated structural
choice rather than cosmetic parameter tuning.

## Why `message_loss_weight = 30`?

The paper-side cleaned branch explicitly recorded the best current design as:

- image size 256
- message length 32
- message loss ratio `r=30`
- bitplane message map
- non-geometry attack profile

This appears in:

- `paper/Pixel_bridge_256_msglen32_paper_bitplane_nongeom/README_PAPER_PROJECT.md`

Historically, the later `r=30` setting should be read as a stronger watermark
pressure choice that remained compatible with the 256 / 32 / bitplane line
after the earlier capacity bottlenecks were partially relieved.

## Why `identity_gn_sp_webp_bc`?

The non-geometric attack profile was chosen for a specific reason, not because
it listed all attacks.

The historical note is:

- `Pixel_bridge_256_msglen32_r30_bitplane_gn_sp_webp_bc_identity/md/BITPLANE_NONGEOM_PROFILE.md`

The reasoning was:

- bitplane identity-only already showed clean decoding could become strong
- some non-geometric attacks were still weaker
- crop/rotate remained closer to geometry/alignment problems
- therefore non-geometry robustness should be isolated before mixing in
  geometry problems

This logic later echoes the more careful attack scheduling choices in the O256
line.

## Why Early Paper-Side Work Still Matters

The later StageE protocol is not a direct copy of the earlier paper-side
branches.

However, those earlier branches still explain:

- why 256x256 became the stable research scale
- why 32 bits became the practical target
- why bitplane was preferred over the older linear map in some branches
- why non-geometric robustness was often isolated before broader schedules

This is why those branches belong in front-history notes even when their
checkpoints are not preserved in the final NAS package.
