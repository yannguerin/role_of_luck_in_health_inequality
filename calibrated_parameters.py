# --------------------------------------------------- #
# Module to store all the calibrated parameters from  #
# various calibration runs and calibrated simulations #
# --------------------------------------------------- #

from collections import namedtuple

Parameters = namedtuple("Parameters", ['loc', 'scale', 'health_ability_link_cobb_douglas_alpha', 'effort_type', 'shape', 'shock_probability_scaling', 'shock_probability_scaling_post_age'], defaults=[None, None])

# Update the Parameters named tuple after running the calibration

# Parameters from calibration on updated datasets
# Date: January 23rd 2025
# ran with accidental deaths and neonatal deaths
base_model_parameters = Parameters(loc=0.6, scale=0.1, health_ability_link_cobb_douglas_alpha=0.58, effort_type=0.0, shape=1.0, shock_probability_scaling=1.0)