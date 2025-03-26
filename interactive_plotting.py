import random
from collections import defaultdict

import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from metadata import *
from luck_vs_circumstance import dist, params, model

pio.templates.default = "plotly_white"

def model_mortality_from_population(population):
    model_mortality = population['age_of_death'].value_counts()

    # Added the plus one to the number of years as this fixes a reshaping bug 
    # that occurs when everyone is dead before the end of the timeline
    for age in range(0, 80 + 1):
        if age not in model_mortality.index:
            model_mortality.loc[age] = 0

    model_mortality = model_mortality.reset_index(name='count').rename(columns={'age_of_death': 'age'})
    model_mortality = model_mortality.sort_values('age', ascending=False).set_index('age').cumsum().iloc[::-1]
    model_mortality = model_mortality.rename(columns={'count': 'Model survivors at age x'}).reset_index()
    return model_mortality

def interactive_plot_of_mortality(models, sim_num_to_description, defaults: list | None = None):
    all_sims_mortality = pd.DataFrame()

    for sim_num, model in models.items():
        population = model.population
        model_mortality = model_mortality_from_population(population)
        model_mortality['simulation'] = sim_num_to_description[sim_num]
        all_sims_mortality = pd.concat([all_sims_mortality, model_mortality])

    # canada_education, health_shock_parameters, mortality = params.parse_empirical_data("./luck_vs_circumstance/datasets")
    mortality = params.parse_mortality_data("./luck_vs_circumstance/datasets")
    mortality['Model survivors at age x'] = (mortality['Actual survivors at age x'] / (100_000 / 10_000)).astype(int)
    real_world_mortality = mortality.iloc[:81, :]

    real_world_mortality['simulation'] = "real_world"
    all_sims_mortality = pd.concat([all_sims_mortality, real_world_mortality])

    if not defaults:
        defaults = ["Pure luck model", "Mediumly correlated luck model"]
    default_sims_to_show = [simulation_num_to_description_mapping[sim_num] for sim_num in defaults]
    default_sims_to_show.append("real_world")

    fig = px.line(all_sims_mortality, x='age', y='Model survivors at age x', color='simulation', title="Mortality for All Simulations")

    fig.for_each_trace(lambda trace: trace.update(visible="legendonly") if trace.name not in default_sims_to_show else ())

    return fig

def plot_inequity_mortality_curves(lvc_model, c_or_e_or_ha: str, index=None, number_of_years=80, population_size=10_000):
    population = lvc_model.population.copy()
    # population_over_time = lvc_model.population_over_time.copy()

    if c_or_e_or_ha == "c":
        min_pop = population[population.circumstance_score == population.circumstance_score.min()]
        max_pop = population[population.circumstance_score == population.circumstance_score.max()]
        mid_pop = population[population.circumstance_score == population.circumstance_score.median()]
    elif c_or_e_or_ha == "e":
        population['effort_quantiles'] = pd.qcut(population.effort_score, q=[0, 0.1, 0.45, 0.55, 0.9, 1.0])
        min_pop = population[population.effort_quantiles == population.effort_quantiles.min()]
        max_pop = population[population.effort_quantiles == population.effort_quantiles.max()]
        mid_pop = population[population.effort_quantiles == sorted(population.effort_quantiles.unique())[2]] 
    elif c_or_e_or_ha == "ha":
        population['health_ability_quantiles'] = pd.qcut(population.health_ability, q=[0, 0.1, 0.45, 0.55, 0.9, 1.0])
        min_pop = population[population.health_ability_quantiles == population.health_ability_quantiles.min()]
        max_pop = population[population.health_ability_quantiles == population.health_ability_quantiles.max()]
        mid_pop = population[population.health_ability_quantiles == sorted(population.health_ability_quantiles.unique())[2]]                                  
    else:
        raise ValueError("c_or_e_or_ha must be 'c' or 'e' or 'ha'")
    
    # base_model_mortality = pd.read_csv("base_model_mortality.csv", index_col=0)
    # base_model_mortality['Model survivors at age x'] = base_model_mortality['Model survivors at age x'] / population_size
    # base_model_mortality['segment'] = "base_model"


    mortality = params.parse_mortality_data("./luck_vs_circumstance/datasets")
    mortality = mortality[mortality.age <= number_of_years].copy()
    mortality = mortality.rename(columns={'Actual survivors at age x': 'Model survivors at age x'})
    mortality['Model survivors at age x'] =(mortality['Model survivors at age x'] / (100_000 / population_size)).astype(int) / population_size
    mortality['segment'] = "real_world"
    
    # Dataframe for all subpops
    all_model_mortalities = pd.DataFrame()

    for sub_pop, sub_pop_name in zip((population, min_pop, max_pop, mid_pop), ("full", "min", "max", "median")):   
        model_mortality = sub_pop['age_of_death'].value_counts(normalize=True)

        # Added the plus one to the number of years as this fixes a reshaping bug 
        # that occurs when everyone is dead before the end of the timeline
        for age in range(0, number_of_years + 1):
            if age not in model_mortality.index:
                model_mortality.loc[age] = 0

        model_mortality = model_mortality.reset_index(name='count').rename(columns={'age_of_death': 'age'})
        model_mortality = model_mortality.sort_values('age', ascending=False).set_index('age').cumsum().iloc[::-1]
        model_mortality = model_mortality.rename(columns={'count': 'Model survivors at age x'}).reset_index()
        model_mortality = model_mortality
        model_mortality['segment'] = sub_pop_name

        all_model_mortalities = pd.concat([all_model_mortalities, model_mortality])

    all_model_mortalities = pd.concat([all_model_mortalities, mortality])

    fig = px.line(all_model_mortalities, x='age', y='Model survivors at age x', color='segment', title=f"Mortality Curves for varying levels of {c_or_e_or_ha.upper()}")
    
    fig.update_layout(yaxis=dict(
        tickmode='array',
        tickvals=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    ))

    return fig

def distribution_of_age_of_death_stacked_by_number_of_health_shocks(lvc_model, index=None, include_neonatal_deaths = False):
    if not index is None:
        shocks_taken_data = {k: v for k, v in lvc_model.shocks_taken_data.items() if k in index}
        population = lvc_model.population.loc[index].copy()
    else:
        shocks_taken_data = lvc_model.shocks_taken_data
        population = lvc_model.population.copy()
    agent_to_num_shocks = {k:len(v['shocks']) for k, v in shocks_taken_data.items()}
    population['number_of_health_shocks'] = population.id.map(agent_to_num_shocks)
    # Sorting to ensure that plotly assigns colors in a sequential manner
    population = population.sort_values(by='number_of_health_shocks', ascending=True)


    if include_neonatal_deaths:
        return px.histogram(population[(~population.alive) & (population.age_of_death > 0)], x='age_of_death', color='number_of_health_shocks', color_discrete_sequence=px.colors.sequential.Magenta, nbins=int(population.age_of_death.max()), title="Age of Death by Number of Health Shocks")
        # return sns.histplot(data = population[~population.alive], x='age_of_death', hue='number_of_health_shocks', multiple='stack')
    else:
        return px.histogram(population[(~population.alive) & (population.age_of_death > 0)], x='age_of_death', color='number_of_health_shocks', color_discrete_sequence=px.colors.sequential.Magenta, nbins=int(population.age_of_death.max()), title="Age of Death by Number of Health Shocks")
    
def plot_distribution_of_scores(lvc_model, index=None):
    if not index is None:
        population = lvc_model.population.loc[index].copy()
    else:
        population = lvc_model.population.copy()
    
    # Setting all negative health score values to 0
    population.loc[population.health_score < 0, 'health_score'] = 0

    # Removing neonatal deaths
    population = population[population.age_of_death != 0]

    subplot_titles = ["Circumstance", "Effort", "Health Ability", "Health Score"]

    fig = make_subplots(2, 2, subplot_titles=subplot_titles)

    fig.add_trace(go.Histogram(x=population.circumstance_score, name="Circumstance"), row=1, col=1)
    fig.add_trace(go.Histogram(x=population.effort_score, name="Effort"), row=1, col=2)
    fig.add_trace(go.Histogram(x=population.health_ability, name="Health Ability"), row=2, col=1)
    fig.add_trace(go.Histogram(x=population.health_score, name="Health Score"), row=2, col=2)

    return fig

def plot_individuals_in_the_population_from_model(lvc_model, index=None):
    """
    Plots the health scores over time of twelve individuals in the population along with shocks encountered or taken color coded by shock cause.
    """

    if not index is None:
        if len(index) <= 12:
            population = lvc_model.population_over_time.loc[index].copy()
        else:
            pop_index = random.sample(list(index), 12)
            population = lvc_model.population_over_time.loc[index].sample(12).copy()
    else:
        pop_index = random.sample(list(lvc_model.population_over_time[lvc_model.population_over_time.age_of_death != 0].index.unique()), 12)
        population = lvc_model.population_over_time.loc[pop_index].copy()
        
    individuals = sorted(population.index.unique())

    subplot_titles = [f"Circumstance {population[population.index == i].head(1)['circumstance_score'].iloc[0]:.2f} | Effort {population[population.index == i].head(1)['effort_score'].iloc[0]:.2f} | Health Ability {population[population.index == i].head(1)['health_ability'].iloc[0]:.2f}" for i in individuals]

    fig = make_subplots(3, 4, subplot_titles=subplot_titles)

    # Creating a custom legend for the vertical lines of health shocks
    shock_cause_colors = ['seagreen', 'slateblue', 'tomato', 'violet']

    shock_causes = ['Ischemic heart disease', 'Total cancers',
        'Diabetes mellitus', "Alzheimer's disease and other dementias"]
    
    causes_in_legend = []
 
    # Making a grid of plots for each individual in the population
    for i, index in enumerate(individuals): 
        single_individual = population[population.index == index]
        # Creating the health_score over time line plot
        x_cor = i % 3 + 1
        y_cor = i % 4 + 1
        fig.add_trace(go.Scatter(x=single_individual['time'], y=single_individual['unshocked_health_score'], marker={"color": "grey"}, name=index, mode='lines', opacity=0.5, showlegend=False), row=x_cor, col=y_cor)
        fig.add_trace(go.Scatter(x=single_individual['time'], y=single_individual['health_score'], marker={"color": "black"}, name=index, mode='lines', showlegend=False), row=x_cor, col=y_cor)
        # Make the background lightgrey if the individual is dead
        if not single_individual.alive.all():
            fig.add_shape(dict(
                type="rect",
                xref="x2",
                yref="paper",
                x0=0,
                y0=-0.001,
                x1=80,
                y1=1,
                fillcolor="lightgray",
                opacity=0.5,
                layer="below",
                line_width=0,
            ), row=x_cor, col=y_cor)
        
        # Creating the vertical lines for encountered shocks
        for ix, row in single_individual[single_individual.encountered_shock].iterrows():
            # Different linestyles depending on if a shock is taken or not
            if row['taken_shock'] == True:
                ls = 'solid'
            else:
                ls = 'dash'

            # Sometimes I hate plotly for making me write this garbage
            # line colors will only show up in the legend if an individual took that health shock
            if ls == 'solid' and row['shock_causes'] not in causes_in_legend:
                causes_in_legend.append(row['shock_causes'])
                show_legend = True
            else:
                show_legend = False

            fig.add_trace(go.Scatter(x=[row['time'], row['time']], y=[0,1], mode='lines', showlegend=show_legend, name=row['shock_causes'], line=dict(color=shock_cause_colors[shock_causes.index(row['shock_causes'])], width=1, dash=ls)), row=x_cor, col=y_cor)
            # vline was the old approach that I changed because I couldnt add it to the legend, the scatter workaround is kinda better but still has issues
            # fig.add_vline(x=row['time'], line_color=shock_cause_colors[shock_causes.index(row['shock_causes'])], line_width=0.8, line_dash=ls, row=x_cor, col=y_cor)

    yaxis_limit_args = {f'yaxis{i}': dict(range=[0,1]) for i in range(1, 14)}
    fig.update_layout(width=1200, height=800, title="Health Scores over Time for Sample of Individuals", **yaxis_limit_args) # type: ignore
    fig.update_annotations(font_size=7.5)

    return fig


def plot_shock_magnitudes(lvc_model):
    shock_data = lvc_model.shocks_taken_data
    
    shock_magnitudes = defaultdict(list)
    for shock in shock_data.values():
        for shock_cause, shock_magnitude in zip(shock['shocks'], shock['shock_magnitudes']):
            shock_magnitudes[shock_cause].extend([1 - magn for magn in shock['shock_magnitudes']])
    
    shock_magnitudes_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in shock_magnitudes.items()]))
    shock_magnitudes_df = shock_magnitudes_df[['Total cancers', 'Ischemic heart disease', 'Diabetes mellitus', "Alzheimer's disease and other dementias"]]


    return px.histogram(shock_magnitudes_df, title="Shock Magnitudes by Shock Cause", labels={'count': 'Number of Shocks with Magnitude', 'value': 'Shock Magnitude'})


def plot_age_of_health_shocks(lvc_model):
    pop_over_time = lvc_model.population_over_time.copy()

    age_of_health_shock_df = pd.DataFrame()
    age_of_health_shock_df['encountered_shock'] = pop_over_time.groupby('time').encountered_shock.sum()
    age_of_health_shock_df['taken_shock'] = pop_over_time.groupby('time').taken_shock.sum()
    age_of_health_shock_df['time'] = list(range(80))

    return px.bar(age_of_health_shock_df, y=['encountered_shock', 'taken_shock'], x='time', title="Number of Health Shocks by Age", labels={'value': 'Number of Health Shocks', 'time': 'age'})

# ------------- #
# Lorenz Curves #
# ------------- #

def lorenz_curve(data):
    sorted_data = np.sort(data)
    cumsum = np.cumsum(sorted_data)
    rel_cumsum = cumsum / cumsum[-1]
    lorenz_curve = np.insert(rel_cumsum, 0, 0)
    population_percent = np.linspace(0.0, 1.0, len(lorenz_curve))
    return population_percent, lorenz_curve

def interactive_lorenz_curve_plot(models, sim_num_to_description, alive_only=False):
    # Send cross-sectional dataset with dead included to Yukiko
    # Add scaled sampling by population growth (based on data from Nathan)
    fig = go.Figure()

    for sim_num, model in models.items():
        population = model.population.copy()
        if alive_only:
            population = population[population.alive].copy()
        population.loc[population[(population.health_score < 0) | (population.health_score == 1)].index, 'health_score'] = 0
        pop_percent, lorenz = lorenz_curve(population.health_score)        
        fig.add_trace(go.Scatter(
            x=pop_percent,
            y=lorenz,
            mode='lines',
            name=sim_num_to_description[sim_num],
            line=dict()
        ))

    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='equality',
        line=dict(color='red', dash='dash')
    ))

    if alive_only:
        title = "Lorenz Curves for All Simulations (Alive only)"
    else:
        title = "Lorenz Curves for All Simulations (Dead included)"


    fig.update_layout(
        title=title,
        xaxis_title='Percentiles (p)',
        yaxis_title='L(p)',
        showlegend=True
    )

    fig.update_layout(
        updatemenus=[
        dict(
            active=0,
            buttons=list([
                dict(label="Show All",
                     method="update",
                     args=[{"visible": [True] * 13}]),
                dict(label="Cluster 1",
                     method="update",
                     args=[{"visible": [True, True, True, True, True, True, True, False, False, False, False, False, True]}]),
                dict(label="Cluster 2",
                     method="update",
                     args=[{"visible": [False, False, False, False, False, False, False, True, True, True, True, False, False]}]),
                dict(label="Cluster 3",
                     method="update",
                     args=[{"visible": [False, False, False, False, False, False, False, False, False, False, False, True, False]}]),
            ]),
        )
    ])

    return fig

def interactive_cross_sectional_lorenz_curve_plot(cross_sectional_dfs, sim_num_to_description, alive_only=False):
    fig = go.Figure()

    for sim_num, cross_sectional_population in cross_sectional_dfs.items():
        population = cross_sectional_population.copy()
        population.loc[population[(population.health_score < 0) | (population.health_score == 1)].index, 'health_score'] = 0
        pop_percent, lorenz = lorenz_curve(population.health_score)        
        fig.add_trace(go.Scatter(
            x=pop_percent,
            y=lorenz,
            mode='lines',
            name=sim_num_to_description[sim_num],
            line=dict()
        ))

    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='equality',
        line=dict(color='red', dash='dash')
    ))

    if alive_only:
        title = "Cross-sectional Lorenz Curves for All Simulations (Alive only)"
    else:
        title = "Cross-sectional Lorenz Curves for All Simulations (Dead included)"


    fig.update_layout(
        title=title,
        xaxis_title='Percentiles (p)',
        yaxis_title='L(p)',
        showlegend=True
    )

    fig.update_layout(
        updatemenus=[
        dict(
            active=0,
            buttons=list([
                dict(label="Show All",
                     method="update",
                     args=[{"visible": [True] * 13}]),
                dict(label="Cluster 1",
                     method="update",
                     args=[{"visible": [True, True, True, True, True, True, True, False, False, False, False, False, True]}]),
                dict(label="Cluster 2",
                     method="update",
                     args=[{"visible": [False, False, False, False, False, False, False, True, True, True, True, False, False]}]),
                dict(label="Cluster 3",
                     method="update",
                     args=[{"visible": [False, False, False, False, False, False, False, False, False, False, False, True, False]}]),
            ]),
        )
    ])


    return fig

def cross_sectional_health_score(model):
    proportions = [
        0.0491,
        0.0555,
        0.0571,
        0.0583,
        0.0684,
        0.0760,
        0.0778,
        0.0734,
        0.0699,
        0.0642,
        0.0635,
        0.0666,
        0.0701,
        0.0622,
        0.0500,
        0.0381
    ]

    bins = [
        "0 to 4 years",
        "5 to 9 years",
        "10 to 14 years",
        "15 to 19 years",
        "20 to 24 years",
        "25 to 29 years",
        "30 to 34 years",
        "35 to 39 years",
        "40 to 44 years",
        "45 to 49 years",
        "50 to 54 years",
        "55 to 59 years",
        "60 to 64 years",
        "65 to 69 years",
        "70 to 74 years",
        "75 to 79 years",
    ]
    # zipping a list of bin (ex: 0 to 4 years, with proportion of population (ex: 0.0491))
    bins_to_proportion = dict(zip(bins, proportions))

    pop_over_time = model.population_over_time.copy()
    # setting the time bin for each time in the population over time df
    pop_over_time['time_bin'] = pop_over_time.time.apply(lambda x: bins[int(x//5)])

    cross_sectional_population = pd.DataFrame()
    # Only keeping the alive population
    for time_bin, pop in pop_over_time[pop_over_time.alive == True].groupby("time_bin"):
        time = int(time_bin.replace(" to ", " ").replace(" years", " ").split(" ")[1])
        # sampling the number of agents in the time bin equal to the proportion to ensure there are a total of 10_000 agents
        cross_sectional_population = pd.concat([cross_sectional_population, pop[pop.time == time].sample(n=int(10_000 * bins_to_proportion[time_bin]), replace=True)])
    
    cross_sectional_population.reset_index(inplace=True)

    return px.histogram(cross_sectional_population, x='health_score')

def build_cross_sectional_datasets(models: dict, alive_only: bool = True) -> dict[str, pd.DataFrame]:

    bins = [
        "0 to 4 years",
        "5 to 9 years",
        "10 to 14 years",
        "15 to 19 years",
        "20 to 24 years",
        "25 to 29 years",
        "30 to 34 years",
        "35 to 39 years",
        "40 to 44 years",
        "45 to 49 years",
        "50 to 54 years",
        "55 to 59 years",
        "60 to 64 years",
        "65 to 69 years",
        "70 to 74 years",
        "75 to 79 years",
    ]

    if alive_only:
        proportions = [
            0.0491,
            0.0555,
            0.0571,
            0.0583,
            0.0684,
            0.0760,
            0.0778,
            0.0734,
            0.0699,
            0.0642,
            0.0635,
            0.0666,
            0.0701,
            0.0622,
            0.0500,
            0.0381
        ]
    else:
        proportions = [
            0.10260975606132532,
            0.09531822961433355,
            0.08854484452122442,
            0.08225278126765409,
            0.076407836761663,
            0.07097823840873327,
            0.06593447139881217,
            0.0612491182664499,
            0.05689670985191442,
            0.05285358685312175,
            0.04909777121579009,
            0.045608846662706426,
            0.04236784771267325,
            0.03935715658585248,
            0.0365604074350931,
            0.033962397382652755
        ]

    # zipping a list of bin (ex: 0 to 4 years, with proportion of population (ex: 0.0491))
    bins_to_proportion = dict(zip(bins, proportions))

    cross_sectional_dfs = {}

    for sim_num, model in models.items():
        pop_over_time = model.population_over_time.copy()
        # setting the time bin for each time in the population over time df
        pop_over_time['time_bin'] = pop_over_time.time.apply(lambda x: bins[int(x//5)])

        cross_sectional_population = pd.DataFrame()

        if alive_only:
            groups = pop_over_time[pop_over_time.alive == True].groupby("time_bin")
        else:
            groups = pop_over_time.groupby("time_bin")

        for time_bin, pop in groups:
            time = int(time_bin.replace(" to ", " ").replace(" years", " ").split(" ")[1])
            # sampling the number of agents in the time bin equal to the proportion to ensure there are a total of 10_000 agents
            cross_sectional_population = pd.concat([cross_sectional_population, pop[pop.time == time].sample(n=int(10_000 * bins_to_proportion[time_bin]), replace=True)])
        
        cross_sectional_population.reset_index(inplace=True)
        cross_sectional_dfs[sim_num] = cross_sectional_population

    return cross_sectional_dfs


def interactive_cross_sectional_health_scores_plot(cross_sectional_dfs, simulation_num_to_description_mapping):
    subplot_titles = [simulation_num_to_description_mapping[sim_num] for sim_num in list(cross_sectional_dfs.keys())]

    fig_shape = (2, 3)

    fig = make_subplots(*fig_shape, subplot_titles=subplot_titles)
    for i, (sim_num, cross_sectional_population) in enumerate(cross_sectional_dfs.items()):
        row, col = np.unravel_index(i, fig_shape)
        row += 1
        col += 1

        fig.add_trace(go.Histogram(x=cross_sectional_population.health_score, name=simulation_num_to_description_mapping[sim_num], nbinsx=50), row=row, col=col)

    fig.update_layout(title_text="Cross Sectional Health Scores")

    fig.update_layout(
        updatemenus=[
        dict(
            active=0,
            buttons=list([
                dict(label="Show All",
                     method="update",
                     args=[{"visible": [True] * 13}]),
                dict(label="Cluster 1",
                     method="update",
                     args=[{"visible": [True, True, True, True, True, True, True, False, False, False, False, False, True]}]),
                dict(label="Cluster 2",
                     method="update",
                     args=[{"visible": [False, False, False, False, False, False, False, True, True, True, True, False, False]}]),
                dict(label="Cluster 3",
                     method="update",
                     args=[{"visible": [False, False, False, False, False, False, False, False, False, False, False, True, False]}]),
            ]),
        )
    ])


    return fig


def interative_health_scores_plot(models, simulation_num_to_description_mapping):
    subplot_titles = [simulation_num_to_description_mapping[sim_num] for sim_num in list(models.keys())]

    fig_shape = (2, 3)

    fig = make_subplots(*fig_shape, subplot_titles=subplot_titles)
    for i, (sim_num, model) in enumerate(models.items()):
        row, col = np.unravel_index(i, fig_shape)
        row += 1
        col += 1

        population = model.population.copy()
        population.loc[population[(population.health_score < 0) | (population.health_score == 1) | (~population.alive)].index, 'health_score'] = 0
        fig.add_trace(go.Histogram(x=population.health_score, name=simulation_num_to_description_mapping[sim_num], nbinsx=50), row=row, col=col)

    fig.update_layout(title_text="Health Scores at Age 80")

    fig.update_layout(
        updatemenus=[
        dict(
            active=0,
            buttons=list([
                dict(label="Show All",
                     method="update",
                     args=[{"visible": [True] * 13}]),
                dict(label="Cluster 1",
                     method="update",
                     args=[{"visible": [True, True, True, True, True, True, True, False, False, False, False, False, True]}]),
                dict(label="Cluster 2",
                     method="update",
                     args=[{"visible": [False, False, False, False, False, False, False, True, True, True, True, False, False]}]),
                dict(label="Cluster 3",
                     method="update",
                     args=[{"visible": [False, False, False, False, False, False, False, False, False, False, False, True, False]}]),
            ]),
        )
    ])


    return fig

def interative_health_scores_plot_by_age(models, simulation_num_to_description_mapping, age):
    subplot_titles = [simulation_num_to_description_mapping[sim_num] for sim_num in list(models.keys())]

    fig_shape = (2, 3)

    fig = make_subplots(*fig_shape, subplot_titles=subplot_titles)
    for i, (sim_num, model) in enumerate(models.items()):
        row, col = np.unravel_index(i, fig_shape)
        row += 1
        col += 1

        population = model.population_over_time.copy()
        population = population[population.time == (age - 1)]
        population.loc[population[(population.health_score < 0) | (population.health_score == 1) | (~population.alive)].index, 'health_score'] = 0
        fig.add_trace(go.Histogram(x=population.health_score, name=simulation_num_to_description_mapping[sim_num], nbinsx=50), row=row, col=col)

    fig.update_layout(title_text=f"Health Scores at Age {age}")

    fig.update_layout(
        updatemenus=[
        dict(
            active=0,
            buttons=list([
                dict(label="Show All",
                     method="update",
                     args=[{"visible": [True] * 13}]),
                dict(label="Cluster 1",
                     method="update",
                     args=[{"visible": [True, True, True, True, True, True, True, False, False, False, False, False, True]}]),
                dict(label="Cluster 2",
                     method="update",
                     args=[{"visible": [False, False, False, False, False, False, False, True, True, True, True, False, False]}]),
                dict(label="Cluster 3",
                     method="update",
                     args=[{"visible": [False, False, False, False, False, False, False, False, False, False, False, True, False]}]),
            ]),
        )
    ])


    return fig

def interactive_plot_of_health_score_distributions_over_time(models):
    population_over_time = pd.DataFrame()

    for sim_num, model in models.items():
        pop = model.population_over_time.copy()
        pop['simulation'] = str(sim_num.replace("simulation_", ""))
        population_over_time = pd.concat([population_over_time, pop])

    population_over_time.loc[population_over_time[(population_over_time.health_score < 0) | (population_over_time.health_score == 1) | (~population_over_time.alive)].index, 'health_score'] = 0

    fig = px.histogram(population_over_time, x='health_score', animation_frame='time', range_y=[0, 2500], facet_col='simulation', facet_col_wrap=2, title='Health Scores over Time')

    return fig


def plot_proportion_of_deaths_by_circumstance_score(models):
    prop_frame = pd.DataFrame()

    for sim_num in models.keys():
        if sim_num == "Pure luck model":
            # Adding imaginary noise to simulation 0
            canada_education, _, _ = params.parse_empirical_data("/home/yann/health_inequity_simulation/luck_vs_circumstance/datasets")
            c_dist = dist.Circumstance_Distribution(canada_education['Circumstance_Score'], canada_education['Proportion'])
            population = models[sim_num].population.copy()
            population['circumstance_score'] = c_dist.draw(10000)
        else:
            population = models[sim_num].population.copy()

        population = population[~population.cause_of_death.isin(['accidental', 'neonatal'])]
        temp = population.groupby('circumstance_score').age_of_death.value_counts(True).to_frame().reset_index()
        subtemp = temp[temp.age_of_death == 80].copy()
        for c in population.circumstance_score.unique():
            if c not in subtemp.circumstance_score.unique():
                subtemp = pd.concat([subtemp, pd.DataFrame({'circumstance_score': c, 'age_of_death': 80, 'proportion': 0}, index=[0])])
        subtemp = subtemp.sort_values('circumstance_score')
        subtemp['simulation'] = str(sim_num)
        prop_frame = pd.concat([prop_frame, subtemp])
        # sns.lineplot(subtemp, x='circumstance_score', y='proportion', label=sim_num).set_title("Proportion of Agents surviving to age 80 by Circumstance")

    fig = px.line(prop_frame, x='circumstance_score', y='proportion', color='simulation', title="Proportion of Agents surviving to age 80 by Circumstance")

    return fig

def plot_proportion_of_deaths_by_health_ability_decile(models):
    prop_frame = pd.DataFrame()

    for sim_num in models.keys():
        if sim_num == "Pure luck model":
            continue
        population = models[sim_num].population.copy()
        population = population[~population.cause_of_death.isin(['accidental', 'neonatal'])]
        population['ha_decile_2'] = pd.qcut(population.health_ability, q=10)
        temp = population.groupby('ha_decile_2').age_of_death.value_counts(True).to_frame().reset_index()
        temp['health_ability_decile_index'] = temp.ha_decile_2.apply(lambda x: sorted(list(temp.ha_decile_2.unique())).index(x))
        temp = temp[temp.age_of_death == 80].copy()
        temp['simulation'] = str(sim_num)
        prop_frame = pd.concat([prop_frame, temp])

    fig = px.line(prop_frame, x='health_ability_decile_index', y='proportion', color='simulation', title="Proportion of Agents surviving to age 80 by Health Ability Decile")

    return fig