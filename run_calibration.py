# --------------------------------- #
# Runs the Calibration of the Model
#
# Author: Yann Guerin
# --------------------------------- #
from typing import Optional
import random
from collections import namedtuple
import datetime

import matplotlib.pyplot as plt
import matplotlib
import scipy
import numpy as np
import pandas as pd
import seaborn as sns

import luck_vs_circumstance as lvc
from luck_vs_circumstance import dist
from luck_vs_circumstance.model import LvCHealthInequityModel
import neworder

from black_it.loss_functions.minkowski import MinkowskiLoss
from black_it.calibrator import Calibrator
from black_it.samplers.halton import HaltonSampler
from black_it.samplers.random_forest import RandomForestSampler
from black_it.samplers.best_batch import BestBatchSampler

from luck_vs_circumstance.black_it_interface import build_interface, Parameters

def plot_model_results(real_data: np.ndarray, model_data: np.ndarray, params: Parameters, title: Optional[str] = "Real Mortality vs. Model Mortality", options: Optional[dict[str, bool] | None] = None, fname: Optional[str | None] = None) -> None:
    """
    Line plots of the mortality data of the model vs. the real data.
    """
    sns.set_style('whitegrid')
    sns.set_palette('colorblind')

    cmap = matplotlib.cm.get_cmap("tab10").colors
    # Plotting the mortality data
    fig, ax1 = plt.subplots()
    ax1.plot(real_data[:, 1], "-", label="Real Survivors", color=cmap[1])
    ax1.plot(model_data[:, 1], "--", label="Model Survivors", color=cmap[2])

    # Adding some labels and title
    ax1.set_ylabel("Number of Survivors")
    ax1.set_xlabel("Age (years)")
    ax1.title.set_text(title)
    ax1.legend(loc=5)

    fig.tight_layout()

    
    # adding a table of parameters and their values
    params_dict = params._asdict()
    params_to_label = [key for key, val in params_dict.items() if val != None]
    
    # Changing the effort_type parameter from a float to the string that the float represents
    effort_type_options = ['truncnorm', 'truncweibull_min', 'truncexpon']
    effort_type_label = effort_type_options[int(params.effort_type)]
    params_dict['effort_type'] = effort_type_label

    # rounding values like 2.7000000000000004
    formatted_params_dict = {key: round(val, 2) if val != None and not isinstance(val, str) else val for key, val in params_dict.items()}
    formatted_params_dict = {key: val for key, val in params_dict.items() if val != None}
    # print(f"{len(formatted_params_dict.values())=}")
    # print(f"{len(params_to_label)=}")

    ax1.table(cellText=[[str(param)] for param in formatted_params_dict.values()],
              rowLabels=params_to_label,
              bbox=[1.5, 0.75, 0.25, 0.25])

    # Creating a sensible filename if not passed
    if not fname:
        if not options:
            options = {'no_options_selected': True}
        options_string = '_'.join([opt for opt, val in options.items() if val])
        fname = f'real_vs_model_mortality_using_{options_string}'
    plt.savefig(f"./results/calibration_results/{fname}.png", bbox_inches='tight')

    plt.cla()
    plt.clf()

def plot_individuals_in_the_population(sample_population_over_time: pd.DataFrame, options: Optional[dict[str, bool] | None] = None, fname: Optional[str | None] = None):
    """
    Plots the health scores over time of twelve individuals in the population along with shocks encountered or taken color coded by shock cause.
    """
    sns.set_style('whitegrid')
    sns.set_palette('colorblind')

    # Creating a sensible filename if not passed
    if not fname:
        if not options:
            options = {'no_options_selected': True}
        options_string = '_'.join([opt for opt, val in options.items() if val])

    from matplotlib.lines import Line2D
    # For plotting the grid of results
    fig, axs = plt.subplots(3, 4, figsize=(30,20))

    # Creating a custom legend for the vertical lines of health shocks
    shock_cause_colors = sns.color_palette()

    shock_causes = ['Ischemic heart disease', 'Total cancers',
        'Diabetes mellitus', "Alzheimer's disease and other dementias"]

    custom_legend_lines = [
        Line2D([0], [0], color=shock_cause_colors[i], lw=4) for i in range(len(shock_causes))
    ]

    fig.suptitle(f"Health Score over Time for Individuals in the Population", fontsize=20, y=0.93)

    # Making a grid of plots for each individual in the population
    for i, index in enumerate(sample_population_over_time.index.unique()): 
        single_individual = sample_population_over_time[sample_population_over_time.index == index]
        # Creating the health_score over time line plot
        sns.lineplot(ax=axs[i % 3, i % 4], data=single_individual, x='time', y='unshocked_health_score', color='dimgrey', linestyle='-.')
        sns.lineplot(ax=axs[i % 3, i % 4], data=single_individual, x='time', y='health_score', color='black')
        axs[i % 3, i % 4].set_ylim([0, 1])
        axs[i % 3, i % 4].set_title(f"Circumstance: {round(single_individual['circumstance_score'].iloc[0], 4)} | Effort: {round(single_individual['effort_score'].iloc[0], 4)} | Health Ability: {round(single_individual['health_ability'].iloc[0], 4)}" )

        # Make the background lightgrey if the individual is dead
        if not single_individual.alive.all():
            axs[i % 3, i % 4].set_facecolor('lightgrey')
        
        # Creating the vertical lines for encountered shocks
        for ix, row in single_individual[single_individual.encountered_shock].iterrows():
            # Different linestyles depending on if a shock is taken or not
            if row['taken_shock']:
                ls = '-'
            else:
                ls = '--'
            axs[i % 3, i % 4].axvline(x=row['time'], color=shock_cause_colors[shock_causes.index(row['shock_causes'])], linewidth=3, linestyle=ls)

    fig.legend(custom_legend_lines, shock_causes, loc='upper right', title='Shock Causes', handleheight=3, handlelength=5)

    fname = f'health_scores_over_time_for_sample_of_population_using_{options_string}'
    plt.savefig(f"./results/calibration_results/{fname}.png")

    plt.cla()
    plt.clf()


def plot_distribution_values(population_df: pd.DataFrame, options: Optional[dict[str, bool] | None] = None, fname: Optional[str | None] = None):
    """
    Histograms of the effort and circumstance scores in the population
    """

    sns.set_style('whitegrid')
    sns.set_palette('colorblind')

    # Creating a sensible filename if not passed
    if not fname:
        if not options:
            options = {'no_options_selected': True}
        options_string = '_'.join([opt for opt, val in options.items() if val])
    
    # Plotting the distributions

    # Circumstance
    fig = sns.histplot(data=population_df, x='circumstance_score').set(title='Histogram of Circumstance Scores in the population')
    fname = f'circumstance_scores_using_{options_string}'
    plt.savefig(f"./results/calibration_results/{fname}.png")

    plt.cla()
    plt.clf()

    # Effort
    fig = sns.histplot(data=population_df, x='effort_score').set(title='Histogram of Effort Scores in the population')
    fname = f'effort_scores_using_{options_string}'
    plt.savefig(f"./results/calibration_results/{fname}.png")

    plt.cla()
    plt.clf()

def plot_losses_to_params(losses: np.ndarray, params: np.ndarray,  options: Optional[dict[str, bool] | None] = None, fname: Optional[str | None] = None):
    sns.set_style('whitegrid')
    sns.set_palette('colorblind')

    # Creating a sensible filename if not passed
    if not fname:
        if not options:
            options = {'no_options_selected': True}
        options_string = '_'.join([opt for opt, val in options.items() if val])

    # plt.tight_layout()

    sns.barplot(x=losses, y=[str(p) for p in params])
    plt.xlabel("Losses (using Minkowski Distance)")
    plt.ylabel("Parameters")
    plt.title("Losses by Parameter Set\n Parameter values: loc, scale, health_ability_link_cobb_douglas_alpha, shape, effort_type, shock_probability_scaling")

    fname = f"losses_to_params_{options_string}"

    plt.savefig(f"./results/calibration_results/{fname}.png", bbox_inches="tight")

    plt.cla()
    plt.clf()


if __name__ == '__main__':
    # Calibration Parameters
    # number_of_batches = 50
    # calibration_seed = 1
    # batch_size = 10
    # ensemble_size = 5
    number_of_batches = 5
    calibration_seed = 1
    batch_size = 5
    ensemble_size = 5

    population_size = 10_000
    number_of_years = 81

    loss = MinkowskiLoss()
    halton_sampler = HaltonSampler(batch_size=batch_size)
    random_forest_sampler = RandomForestSampler(batch_size=batch_size)
    best_batch_sampler = BestBatchSampler(batch_size=batch_size)

    canada_education, health_shock_parameters, mortality = lvc.params.parse_empirical_data("luck_vs_circumstance/datasets")

    mortality['Actual survivors at age x'] = (mortality['Actual survivors at age x'] / (100_000 / population_size)).astype(int)
    real_data = mortality.iloc[:number_of_years + 1, :].to_numpy()

    # accidental death data
    accidental_deaths = lvc.params.parse_accidental_death_data("luck_vs_circumstance/datasets")

    # Parameters when using different shock probability scaling for pre age 50 and post age 50
    # In order: loc, scale, health_ability_link_cobb_douglas_alpha, effort_type, shape, shock_probability_scaling
    bounds = [[0.4, 0.1, 0.4, 0, 0.1, 1., 1.], [0.6, 1.0, 0.6, 2, 1.0, 2.0, 2.0]]
    # Precision steps to explore
    precisions = [0.01, 0.1, 0.01, 1, 0.1, 0.1, 0.1]

    # bounds = [[0, 0, 0], [1.1, 1.1, 1.1]]
    # # Precision steps to explore
    # precisions = [0.1, 0.1, 0.1]

    bounds = [bounds[0][:-1], bounds[1][:-1]]
    precisions = precisions[:-1]

    options = {'accidental_deaths': accidental_deaths, 'custom_unequal_health_score': 0.9, "neonatal_deaths": True}

    if options.get("transport_only"):
        options['accidental_deaths'] = lvc.params.parse_accidental_death_data("luck_vs_circumstance/datasets", cause="Transport injuries")
    elif options.get("unintentional_only"):
        options['accidental_deaths'] = lvc.params.parse_accidental_death_data("luck_vs_circumstance/datasets", cause="Unintentional injuries")

    LvC_interface, get_parameters = build_interface(**options)

    cal = Calibrator(
        samplers=[halton_sampler, random_forest_sampler, best_batch_sampler], # Multiple samplers can be given
        real_data=real_data, # The real data from which loss will be calculated
        model=LvC_interface,
        loss_function=loss, 
        parameters_bounds=bounds,
        parameters_precision=precisions, 
        ensemble_size=ensemble_size,
        convergence_precision=None,
        verbose=True,
        saving_folder=None,
        random_state=1,
        n_jobs=10,
        # sim_length=80
    )

    params, losses = cal.calibrate(number_of_batches)

    if "accidental_deaths" in options.keys():
        options['accidental_deaths'] = True

    # print(params, losses)

    print(params[0])
    
    idxmin  = np.argmin(cal.losses_samp)
    
    ### PLOTTING RESULTS

    calibrated_params = Parameters(*params[0])
    
    plot_model_results(real_data, cal.series_samp[idxmin, 0], calibrated_params, "Real Mortality vs. Model Mortality", options=options)

    try:
        plot_losses_to_params(losses[:10], params[:10], options)
    except Exception as e:
        print(e)

    # Rerunning the model with the best parameters (if I can find a way to extract the model object from the calibrator then I should use that)
    # In order to plot the circumstance and effort distributions along with a sample of individuals in the population
    # Note: with the change to the parameter handling I believe I have solved the problem

    theta = params[0]

    sim_params = get_parameters(theta, number_of_years)
    
    # Plotting the distributions
    # Only need to initialize the model (no need to run) since I just need the initialized effort and circumstance scores
    lvc_model = LvCHealthInequityModel(**sim_params)
    plot_distribution_values(population_df=lvc_model.population, options=options)

    # Plotting a sample of individuals
    sim_params['population_size'] = 12
    sim_params['inspect'] = True

    lvc_model = LvCHealthInequityModel(**sim_params)
    ok = neworder.run(lvc_model)

    plot_individuals_in_the_population(lvc_model.population_over_time, options)

    print(datetime.datetime.now())

