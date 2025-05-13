# PI_CRNN2025

Supplemental codes for "A Physics-Informed Convolutional Long Short Term Memory Statistical Model for Fluid Thermodynamics Simulations".

There are three scripts in the code folder:

1) functions.py contains helper functions to load and preprocess the data
2) cae_model.py contains code to run the CAE
3) pi_crnn_model.py contains code to run the spatiotemporal model, conditional on the trained CAE

A user should first download the three files in the code folder. To reproduce the results from the manuscript, the user should first run cae_model.py to train the CAE portion of the model. Then, run pi_crnn_model.py to train and produce forecasts for the proposed PI-CRNN approach.
