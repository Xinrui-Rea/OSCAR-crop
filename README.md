# 🌾 OSCAR-crop

A crop emulator calibrated on GGCMI Phase 3 models.

## 🏗️ Model Architecture

![Crop Emulator Architecture](./docs/sci/crop_fd/Model%20Architecture.png)

## ✨ Features

- ⚡ **High efficiency**
- 🗺️ **Flexible regional aggregation**

## 🚀 Installation

Since this repository is not packaged for `pip`, users will need to clone the repository and run it directly within your Python environment.

### Prerequisites & Dependencies

This project was developed and tested on **Python 3.12** (requires **Python 3.11** or higher).

All required Python packages and their version constraints are listed in [`requirements.txt`](requirements.txt). 

Key dependencies include:

* **Data & Science:** `numpy`, `pandas`, `scipy`, `xarray`
* **Visualization:** `matplotlib`, `cartopy`

### Step-by-step
* **Create and Activate a Virtual Environment:**
It is strongly recommended to use an isolated environment to avoid package conflicts.

1. **Using `venv` (Standard Python):**
   ```bash
   # Create the environment
   python3 -m venv env

   # Activate the environment (Linux/macOS):
   source env/bin/activate
   ```

2. **Clone the repository:**
   ```bash
   git clone https://github.com/Xinrui-Rea/OSCAR-crop.git
   cd OSCAR-crop
   ```

3. **Install Required Packages:**
   ```Bash
   pip install -r requirements.txt
   ```

## 📄 Documentation

- [**Independent Use**](./docs/man/Independent%20Use.md)
- [**Coupling with OSCAR**](./docs/man/Coupling%20with%20OSCAR.md)

## 📚 Scientific Guide

### Food crop emulator

- [**Acronyms**](./docs/sci/crop_fd/Acronyms.md)
- [**Dimensions**](./docs/sci/crop_fd/Dimensions.md)
- [**Parameters**](./docs/sci/crop_fd/Parameters.md)
- [**Variables**](./docs/sci/crop_fd/Variables.md)


## 📝 Notes

Current model structure is consistent with OSCARv3. Regional aggregation scheme is updated to OSCARv4. 

## 📖 Citation

If you use this code or data in your research, please cite:

[Liu, X. et al. A food crop yield emulator for integration in the compact Earth system model OSCAR (OSCAR-crop v1.0). Geosci Model Dev 19, 5857–5880 (2026).](https://doi.org/10.5194/gmd-19-5857-2026)

## 📞 Contact

- **Developer:** Xinrui Liu
- **Email:** liuxinrui@iiasa.ac.at