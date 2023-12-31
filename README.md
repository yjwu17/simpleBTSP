
# Codes for a learning theory of binary simple BTSP 

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

## Examples
To test HFN performance on non-orthogonal memory item tasks, please directly run the code in 'main_HFN_continuous_weights.py'; adjust the threshold (cdf_vth) to observe the impact of the threshold on performance. To test the performance on orthogonal memory items, simply replace the code lne 96 and 97 as used for BTSP models;
To test the binarized HFN performance, please directly run the code in 'main_HFN_binary_weights.py'


We will release the complete versions of the codes after the publication of our study. Please cite the following reference for our work:
## Reference
Wu Y, Maass W. Memory structure created through behavioral time scale synaptic plasticity[J]. biorxiv, 2023: 2023.04. 04.535572.
