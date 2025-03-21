# -------------------------------------------------------- #
# Script to generate cross sectional population and
# population over time datasets for all simulation models.
# For use in health inequity calculations and analyses
#
# Date: July 4th 2024
# Author: Yann Guerin
# -------------------------------------------------------- #


import pathlib

import neworder

from luck_vs_circumstance import dist, params
from luck_vs_circumstance.model import LvCHealthInequityModel
from interactive_plotting import *
from metadata import *
from calibrated_parameters import base_model_parameters


def main():
    models = {}

    # ------------------------------- #
    # Setting up the Model Parameters #
    # ------------------------------- #

    population_size = 10000
    number_of_years = 80

    effort_type_options = ['truncnorm', 'truncweibull_min', 'truncexpon']

    canada_education, health_shock_parameters, mortality = params.parse_empirical_data("./luck_vs_circumstance/datasets")
    circumstance_dist = dist.Circumstance_Distribution(canada_education['Circumstance_Score'],
                                                                        canada_education['Proportion'])

    accidental_deaths = params.parse_accidental_death_data("./luck_vs_circumstance/datasets")

    effort_dist = dist.Effort_Distribution({'loc': base_model_parameters.loc, 'scale': base_model_parameters.scale, 'shape': base_model_parameters.shape}, effort_type_options[int(base_model_parameters.effort_type)])

    # ------------------ #
    # Running the Models #
    # ------------------ # 

    for sim_num in simulation_num_to_description_mapping.keys():

        sim_params =  {
            'population_size': population_size,
            'number_of_years': number_of_years,
            'circumstance_dist': circumstance_dist,
            'effort_dist': effort_dist,
            'health_shock_parameters': health_shock_parameters,
            'health_ability_link_cobb_douglas_alpha': base_model_parameters.health_ability_link_cobb_douglas_alpha,
            'shock_probability_scaling': base_model_parameters.shock_probability_scaling,
            'shock_probability_scaling_post_age': base_model_parameters.shock_probability_scaling_post_age,
            'accidental_deaths': accidental_deaths,
            **simulation_options[sim_num],
            'inspect': True
        }

        lvc_model = LvCHealthInequityModel(**sim_params)
        ok = neworder.run(lvc_model)

        models[sim_num] = lvc_model

    # ------------------------------------- #
    # Creatinh the cross sectional datasets #
    # ------------------------------------- #

    cross_sectional_dfs = build_cross_sectional_datasets(models, alive_only=True)
    cross_sectional_dfs_dead_included = build_cross_sectional_datasets(models, alive_only=False)

    # ------------------ #
    # Saving the Results #
    # ------------------ #

    for sim_num in models.keys():
        dataset_results_path = pathlib.Path(f"./results/simulation_datasets/{sim_num}/")
        if not dataset_results_path.exists():
            dataset_results_path.mkdir()

        dataset = cross_sectional_dfs[sim_num][['id', 'health_score', 'circumstance_score', 'effort_score', 'time_bin']]
        dataset.to_csv(f"./results/simulation_datasets/{sim_num}/cross_sectional_population.csv")

        dataset_dead_included = cross_sectional_dfs_dead_included[sim_num][['id', 'health_score', 'circumstance_score', 'effort_score', 'time_bin']]
        dataset_dead_included.to_csv(f"./results/simulation_datasets/{sim_num}/cross_sectional_population_dead_included.csv")
        
        models[sim_num].population.to_csv(f"./results/simulation_datasets/{sim_num}/population.csv")
        models[sim_num].population_over_time.to_csv(f"./results/simulation_datasets/{sim_num}/population_over_time.csv")

if __name__ == '__main__':
    main()