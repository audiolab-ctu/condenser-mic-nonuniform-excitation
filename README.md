# README

This repository contains Python scripts and reference data used to compute the analytical results presented in the article:

**Šimonová et al., “Analytical modeling of condenser microphones with nonuniform acoustic excitation and backplate perforation patterns.”, J. Acoust. Soc. Am., 159(5), 4512–4523 (2026)**

If you use this code or the accompanying data in your own work, please cite the article above.

## Contents

The Python files in this repository are intended for the analytical modeling of condenser microphones under nonuniform acoustic excitation, including the evaluation of membrane displacement, acoustic pressure fields, and directivity-related quantities for segmented backplate configurations.

### Python scripts

- **`functions_CondMic_NonUnifExc.py`**  
  Core module containing the functions required for the analytical calculations of the condenser microphone response under nonuniform acoustic excitation. It also includes utility functions for post-processing and figure generation.

- **`MeanDisp_CondMic_NonUnifExc.py`**  
  Computes and plots the frequency-dependent mean membrane displacement over the entire membrane area. It also evaluates the mean displacement over individual membrane segments and the displacement differences between opposing segments for a given acoustic incidence angle.

- **`Directivity_CondMic_NonUnifExc.py`**  
  Calculates and plots polar directivity diagrams based on the mean displacement differences between opposing membrane segments at a specified frequency.

- **`FieldDisplPress_CondMic_NonUnifExc.py`**  
  Visualizes spatial fields of membrane displacement and acoustic pressure in the air gap and backing cavity at a selected frequency and incidence angle.

### Reference numerical data

- Files whose names start with **`Ximean`** contain reference numerical results for the frequency-dependent mean membrane displacements shown in the article.
- Files whose names start with **`Angle`** contain reference numerical results for the angle-dependent directivity polar plots shown in the article.

## Installation

A recent Python 3 installation is recommended. The required external packages are listed in `requirements.txt`.

A typical installation in a virtual environment is:

```bash
python -m venv .venv
```

Activate the virtual environment:

On Windows:

```bash
.venv\\Scripts\\activate
```

On Linux/macOS:

```bash
source .venv/bin/activate
```

Then install the required packages:

```bash
pip install -r requirements.txt
```

## Requirements

The main required Python packages are:

- `numpy`
- `scipy`
- `matplotlib`
- `mpmath`
- `alive-progress`
- `joblib`

The remaining imported modules used by the scripts, such as `math`, `time`, `datetime`, `warnings`, `re`, `sys`, `typing`, and `concurrent.futures`, are part of the Python standard library.

## Contact

Corresponding author: Petr Honzík  
Email: <honzikp@fel.cvut.cz>
