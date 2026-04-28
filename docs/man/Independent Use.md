# Step-by-step

## Prepare Input Data

**Prepare drivers**
- CO₂ concentration
- Regional temperature
- Regional precipitation
- Nitrogen fertilizer application rate


**Offset drivers by their preindustrial levels**
- CO₂ concentration
- Regional temperature
- Regional precipitation

## Load Parameters

- Load parameters from `core_fct.Par_CROP`

## Run the Emulator

**Load CROP emulator by running**

```python
from core.OSCAR_crop import CROP
CROP = CROP()
```

## Analyze Output
