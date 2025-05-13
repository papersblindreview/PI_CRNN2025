# PI_CRNN2025

Supplemental codes for "A Physics-Informed Convolutional Long Short Term Memory Statistical Model for Fluid Thermodynamics Simulations".

There are four scripts in the code folder:

1) `functions.py` contains helper functions to load and preprocess the data
2) `cae_model.py` contains code to run the CAE
3) `pi_crnn_model.py` contains code to run the physics-informed spatiotemporal model, conditional on the trained CAE
4) `uq.py` contains code to reproduce predictions intervals using conformal method

The DNS data for Rayleigh-Benard Convection can be reproduced, using the physical constants from the manuscript, from [this](https://git.uwaterloo.ca/SPINS/SPINS_main) public repository. The data file should be saved as "RB_Data.mat".

In the same folder as the data, a user should download the three files in the code folder. To reproduce the results from the manuscript, the user should first run `cae_model.py` to train the CAE portion of the model. Then, run `pi_crnn_model.py` to train and produce forecasts for the proposed PI-CRNN approach. The `uq.py` can be run after training the model to obtain predictions intervals and verify proper coverage.
