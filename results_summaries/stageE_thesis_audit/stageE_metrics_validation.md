# StageE Metrics Validation

## Checks

### exp52
- per-attack count: `11`.
- Avg BA computed from 11 attacks only: `98.761818`.
- Clean BA is stored separately and is not included in Nongeom Avg BA.
- Min BA from per-attack table: `90.33` at `Paper_Dropout_r03`.
- Summary Min BA / attack: `90.33` / `Paper_Dropout_r03`.
- Status: can be used as final main result under the locked StageE criteria.

### exp53
- per-attack count: `11`.
- Avg BA computed from 11 attacks only: `98.578182`.
- Clean BA is stored separately and is not included in Nongeom Avg BA.
- Min BA from per-attack table: `91.36` at `Paper_Dropout_r03`.
- Summary Min BA / attack: `91.36` / `Paper_Dropout_r03`.
- Status: can be used as final main result under the locked StageE criteria.

### exp54
- per-attack count: `11`.
- Avg BA computed from 11 attacks only: `98.348182`.
- Clean BA is stored separately and is not included in Nongeom Avg BA.
- Min BA from per-attack table: `89.98` at `Paper_Dropout_r03`.
- Summary Min BA / attack: `89.98` / `Paper_Dropout_r03`.
- Status: completed full-from-scratch seed63 validation, but near-miss: PSNR 33.4948 and Dropout/Min BA 89.98 miss the strict gates.
