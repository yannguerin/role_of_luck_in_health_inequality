# --------------------------------------------- #
# For generating interactive simulation results #
# 
# Author: Yann Guerin
# --------------------------------------------- #

# Standard Library Imports
from collections import namedtuple, defaultdict
import html

# Third-party Imports
from jinja2 import Environment, FileSystemLoader, select_autoescape
import neworder

from luck_vs_circumstance import dist, params
from luck_vs_circumstance.model import LvCHealthInequityModel

# Custom module Imports
from interactive_plotting import *
from metadata import *
from calibrated_parameters import base_model_parameters

def main():
    print("Main Results")
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
            'neonatal_deaths': False,
            **simulation_options[sim_num],
            'inspect': True
        }

        lvc_model = LvCHealthInequityModel(**sim_params)
        ok = neworder.run(lvc_model)

        # Adding the cause of death to the population dataframe

        # cause_of_death_data = lvc_model.population_over_time[['id', 'cause_of_death']].copy()
        id_to_cause_of_death = {i: group[group.cause_of_death.notnull() == True].cause_of_death.iloc[0] if group[group.cause_of_death.notnull() == True].cause_of_death.size > 0 else None for i, group in lvc_model.population_over_time.groupby('id')}
        lvc_model.population.reset_index(inplace=True)
        lvc_model.population['cause_of_death'] = lvc_model.population.id.apply(lambda x: id_to_cause_of_death[x])

        models[sim_num] = lvc_model

    print("Models Run")

    # ---------------------------------------------- #
    # Creating the individual simulation model plots #
    # ---------------------------------------------- #

    simulation_plots = defaultdict(dict)
    for sim_num, model in models.items():
        simulation_plots[sim_num]['number_of_health_shocks_vs_age_of_death'] = distribution_of_age_of_death_stacked_by_number_of_health_shocks(model).to_html(full_html=False, include_plotlyjs="cdn")
        simulation_plots[sim_num]['distribution_of_scores'] = plot_distribution_of_scores(model).to_html(full_html=False, include_plotlyjs="cdn")
        
        if "equal_circumstance" not in simulation_options[sim_num].keys():
            simulation_plots[sim_num]['mortality_by_c'] = plot_inequity_mortality_curves(model, c_or_e_or_ha='c').to_html(full_html=False, include_plotlyjs="cdn")
        
        if "equal_effort" not in simulation_options[sim_num].keys():
            simulation_plots[sim_num]['mortality_by_e'] = plot_inequity_mortality_curves(model, c_or_e_or_ha='e').to_html(full_html=False, include_plotlyjs="cdn")

        # simulation_plots[sim_num]['individuals_over_time'] = plot_individuals_in_the_population_from_model(model).to_html(full_html=False, include_plotlyjs="cdn")
        if sim_num != "Pure luck model":
            simulation_plots[sim_num]['individuals_over_time'] = plot_inequity_mortality_curves(model, c_or_e_or_ha='ha').to_html(full_html=False, include_plotlyjs="cdn")

        simulation_plots[sim_num]['age_of_health_shock'] = plot_age_of_health_shocks(model).to_html(full_html=False, include_plotlyjs="cdn")

        simulation_plots[sim_num]['shock_magnitudes'] = plot_shock_magnitudes(model).to_html(full_html=False, include_plotlyjs="cdn")

        simulation_plots[sim_num]['title'] = simulation_num_to_title_mapping[sim_num]
        simulation_plots[sim_num]['explanation'] = simulation_num_to_explanation_mapping[sim_num]
        
    print("Simulation Plots Created")

    # --------------------------------------- #
    # Interactive Plots containing all Models #
    # --------------------------------------- #

    fig = interactive_plot_of_mortality(models, simulation_num_to_description_mapping)
    plot = fig.to_html(full_html=False, include_plotlyjs='cdn')

    fig = interactive_lorenz_curve_plot(models, simulation_num_to_description_mapping)
    lorenz_curve_plot_dead_included = fig.to_html(full_html=False, include_plotlyjs='cdn')

    fig = interactive_lorenz_curve_plot(models, simulation_num_to_description_mapping, alive_only=True)
    lorenz_curve_plot_alive_only = fig.to_html(full_html=False, include_plotlyjs='cdn')

    cross_sectional_dfs = build_cross_sectional_datasets(models, alive_only=True)
    cross_sectional_dfs_dead_included = build_cross_sectional_datasets(models, alive_only=False)

    fig = interactive_cross_sectional_health_scores_plot(cross_sectional_dfs, simulation_num_to_description_mapping)
    cross_sectional_health_scores = fig.to_html(full_html=False, include_plotlyjs='cdn')

    health_scores_by_age = {}
    for age in [20, 40, 60, 80]:
        fig = interative_health_scores_plot_by_age(models, simulation_num_to_description_mapping, age)
        health_scores_plot = fig.to_html(full_html=False, include_plotlyjs='cdn')
        health_scores_by_age[str(age)] = health_scores_plot

    fig = interactive_cross_sectional_lorenz_curve_plot(cross_sectional_dfs, simulation_num_to_description_mapping, alive_only=True)
    cross_sectional_lorenz_alive_only = fig.to_html(full_html=False, include_plotlyjs='cdn')

    fig = interactive_cross_sectional_lorenz_curve_plot(cross_sectional_dfs_dead_included, simulation_num_to_description_mapping, alive_only=False)
    cross_sectional_lorenz_dead_included = fig.to_html(full_html=False, include_plotlyjs='cdn')

    fig = plot_proportion_of_deaths_by_health_ability_decile(models)
    proportion_of_deaths_by_health_ability = fig.to_html(full_html=False, include_plotlyjs='cdn')

    fig = plot_proportion_of_deaths_by_circumstance_score(models)
    proportion_of_deaths_by_circumstance_score = fig.to_html(full_html=False, include_plotlyjs='cdn')

    env = Environment(loader=FileSystemLoader("results/interactive_results/templates"), autoescape=select_autoescape())
    main_template = env.get_template("main_interactive_plots.html")
    simulation_tabs_template = env.get_template("simulation_tabs_results.html")

    print("Mortality Plot Created")

    # ------------------ #
    # Saving the Results #
    # ------------------ #

    with open("./results/interactive_results/main_simulation_results.html", 'w+') as f:
        f.write(html.unescape(main_template.render(mortality_plot=plot, health_scores_age_80=health_scores_by_age, cross_sectional_lorenz_alive_only=cross_sectional_lorenz_alive_only, cross_sectional_lorenz_dead_included=cross_sectional_lorenz_dead_included, lorenz_curve_plot_dead_included=lorenz_curve_plot_dead_included, lorenz_curve_plot_alive_only=lorenz_curve_plot_alive_only, cross_sectional_health_scores=cross_sectional_health_scores, proportion_of_deaths_by_health_ability=proportion_of_deaths_by_health_ability, proportion_of_deaths_by_circumstance_score=proportion_of_deaths_by_circumstance_score)))

    with open("./results/interactive_results/individual_simulation_results.html", 'w+') as f:
        f.write(html.unescape(simulation_tabs_template.render(sim_nums=list(models.keys()), sim_plots=simulation_plots)))

    print("Results saved")

if __name__ == '__main__':
    main()
