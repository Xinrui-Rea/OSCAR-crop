# Step-by-step

## Prepare Input Data

**Prepare drivers**
- Drivers for OSCAR
- Drivers for crop emulator: nitrogen fertilizer application rate

## Load Parameters

- Load parameters from OSCAR
- Load parameters from `core_fct.Par_CROP`


## Run the Emulator

**Load CROP emulator by running**

```python
from core.OSCAR_crop import CROP
CROP = CROP()
```

## Analyze Output
