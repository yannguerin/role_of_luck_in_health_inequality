# --------------------------------------------------- #
# Module to store all the calibrated parameters from  #
# various calibration runs and calibrated simulations #
# --------------------------------------------------- #

from collections import namedtuple

Parameters = namedtuple("Parameters", ['loc', 'scale', 'health_ability_link_cobb_douglas_alpha', 'effort_type', 'shape', 'shock_probability_scaling', 'shock_probability_scaling_post_age'], defaults=[None, None])

# Update the Parameters named tuple after running the calibration

# Parameters from calibration on decay (gompertz and annual) for paper revision with bugs fixed
# Date: July 24th 2026
# ran with accidental and neonatal deaths
base_model_parameters = Parameters(loc=0.54, scale=0.5, health_ability_link_cobb_douglas_alpha=0.23, effort_type=0.0, shape=1.0, shock_probability_scaling=1.0)
