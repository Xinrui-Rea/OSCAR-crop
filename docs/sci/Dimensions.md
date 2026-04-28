# Dimensions

## Hard-wired dimensions

| Dimension | Description | Notes |
| --- | --- | --- |
| `year` | time axis | |
| `config` | Monte Carlo elements | |
| `reg_code` | model regions | finest: 311 regions |
| `spc_crop` | crop spieces | `mai`, `ri1`, `ri2`, `soy`, `swh`, `wwh` |
| `irr` | irrigation regime | `firr`, `noirr` |
|||

## Soft-wired dimensions

| Dimension | Notes | Models Used |
| --- | --- | --- |
| `mod_clim_gcm` | GCMs used for climate-related parameter calibration | `GFDL-ESM4`, `IPSL-CM6A-LR`, `MPI-ESM1-2-HR`, `MRI-ESM2-0`, `UKESM1-0-LL`|
| `mod_YD_crop` | GGCMs used for crop-related calibration | `CYGMA1p74`, `EPIC-IIASA`, `ISAM`, `LDNDC`, `LPJmL`, `PEPIC`, `PROMET`, `SIMPLACE-LINTUL5` |
||||

# Acronyms

## Crop
| Acronym | Full Name |
|--- | --- |
| `mai` | maize|
| `ri1` | rice (frist growing season) |
| `ri2` | rice (second growing season) |
| `soy` | soybean |
| `swh` | spring wheat |
| `wwh` | winter wheat |
|||

## Irrigation
| Acronym | Full Name |
|--- | --- |
| `firr` | full irrigation |
| `noirr` | no irrigation |
|||