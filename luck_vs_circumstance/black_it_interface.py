# --------------------------------------------------------------- #
# Interface between the Luck vs Circumstance model with the 
# black-it calibration package.
# 
# Author: Yann Guerin
# --------------------------------------------------------------- #

# theta is the parameter vector to be optimized: (loc, scale, link)
# N is the length of the time series: number of years
# rndSeed is the random seed (Optional) 
from collections import namedtuple
from typing import Any, Callable, Sequence
import numpy as np
import neworder
from luck_vs_circumstance import dist, params, model

Parameters = namedtuple("Parameters", ['loc', 'scale', 'health_ability_link_cobb_douglas_alpha', 'effort_type', 'shape', 'shock_probability_scaling', 'shock_probability_scaling_post_age'], defaults=[None, None])

def build_interface(**kwargs) -> tuple[Callable[[Sequence[float], int, None], np.ndarray], Callable[[Sequence[float], int], dict[str, Any]]]:
    """
    Builds the interface function for black-it, passing the optional kwargs to the model
    The kwargs are optional extra parameters that change the model for testing purposes
    """
    def get_parameters(theta: Sequence[float], N: int) -> dict[str, Any]:
        """
        Gets the parameters for the LvC Model
        Loads the DataFrames and sets all the parameter values in a dictionary
        to be passed as key word arguments to the LvC model in the LvC interface
        """
        model_params = Parameters(*theta)
        population_size = 10000
        # number_of_years = 80
        # annual_health_score_decay = 1/N
        
        # Indexed by theta parameter so that all parameters are integers as I do not know how to pass strings in black-it
        effort_type_options = ['truncnorm', 'truncweibull_min', 'truncexpon']

        canada_education, health_shock_parameters, mortality = params.parse_empirical_data("./luck_vs_circumstance/datasets")
        accidental_deaths = params.parse_accidental_death_data("./luck_vs_circumstance/datasets")

        circumstance_dist = dist.Circumstance_Distribution(canada_education['Circumstance_Score'],
                                                                            canada_education['Proportion'])
        
        effort_dist = dist.Effort_Distribution({'loc': model_params.loc, 'scale': model_params.scale, 'shape': model_params.shape}, effort_type_options[int(model_params.effort_type)])

        return {
            'population_size': population_size,
            'number_of_years': N - 1,
            'circumstance_dist': circumstance_dist,
            'effort_dist': effort_dist,
            'health_shock_parameters': health_shock_parameters,
            'health_ability_link_cobb_douglas_alpha': model_params.health_ability_link_cobb_douglas_alpha,
            'accidental_deaths': accidental_deaths,
            'shock_probability_scaling': model_params.shock_probability_scaling,
            'shock_probability_scaling_post_age': model_params.shock_probability_scaling_post_age,
            **kwargs
        }
    
    def LvC_interface(theta: Sequence[float], N: int, rndSeed=None) -> np.ndarray:
        """
        Gets the parameters and passes them to the model, runs the model with neworder
        Returns the model mortality DataFrame as a numpy array
        """
        sim_params = get_parameters(theta, N)
        lvc_model = model.LvCHealthInequityModel(**sim_params)
        ok = neworder.run(lvc_model)
        ret = lvc_model.model_mortality.to_numpy()
        return ret
    
    return LvC_interface, get_parameters