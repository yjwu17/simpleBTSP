
# Simple BTSP Project

This repository contains the code for the Simple BTSP project.

## Version Requirements
- **Python**: `3.10`
- **Torch**: `2.1.0.dev20230317+cu117`
- **Numba**: `0.56.4`
- **Scipy**: `1.10.1`

## Execution Instructions
To validate different memory models, including BTSP, RP, and HFN, please execute the file named `main_**`.

## Specific Model Usage
- The **BTSP Feedforward Model** was utilized for Figures 2C-E, 3, 4, and 6.
- The **BTSP Feedback Model** was employed for Figure 5.
- **Random Projections** and the **HFN** with both continuous and binary weights were employed for Figures 3 and 5.
- Functions for theoretical predictions can be found in `theory_prediction_function.py`.


We plan to release the complete versions of the codes for reproducing all main results following the publication of our study.
