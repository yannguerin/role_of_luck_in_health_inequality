Health Inequity Simulation: Luck vs Circumstance
================================================

Health inequity simulation project to better characterise role of luck and circumstance in unexplained residuals of health outcome models.

Goal of this SSHRC Insight Development-funded project is to use simulation approaches to explore the unexplained residuals in health outcome regressions.
Specifically, we are trying to decompose these residuals into fixed (circumstance) and random (luck) effects.

To explore this we have adopted an agent-based model approach in which each person (agent) will randomly encounter health shocks (i.e., the luck component).
Each person then has a probability of being impacted by this health shock determined by their assigned health ability (i.e., the fixed component built from circumstance and health effort).

We can summarise this model by the mortality, health scores, and health ability (and its decomposition) of suriviving people over time.

## Installation

Models are implemented in neworder and be run by installing the ananconda distribution, using conda to install neworder/pandas/scipy/numpy, then executing `python run.py`

## MicroSimulation Model

- N people (agents):
    - Assigned `health_score` of 1 (equivalent to "full health").
    - Assigned `circumstance_score` (representative of demographic indicators of health outcome including SES components) drawn from **empirical** CIHI Educational categories (decided to make this simpler and not use income).
    - Assigned `health_effort` score (representative of health behaviours) **fitted** in model calibration.
    - Assigned a `health_ability` that is a function of `circumstance_score` and `health_effort` with the link function **fitted** in model calibration.
- Progression:
    - Number of `health_shock`s = **?** drawn from **empirical** distribution based on DALY/GDB data representing the number of expected health shocks per year of life 
    - Distribution of `health_shock_magnitudes` = **?** (representative of the size of health shock in terms of `health_score` lost) drawn from **empirical** distribution based on DALY/GBD data representing the DALYs associated with health shocks.
    - If no `health_shock` encountered => no change in `health_score`
    - If `health_shock` encountered => `health_score` decreases with a probability inversely proportional to `health_ability` of that individual.
    - If `health_score` reduced to 0 then individual is removed (i.e., death)
- Calibration: 
    - Number of people with `health_score` of 0 per step match real world mortality per year of life. 

### Data Sources

Empirical data sources used and intended model component they are meant to empirically inform.

### Real-world Mortality Calibration

- `1310083701_StatCan_Life_Table_2020_databaseLoadingData.csv`
    - "Real distribution of mortality" from distribution of mortality probability per year in Canada for 2020 
    - [Source](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1310083701&pickMembers%5B0%5D=1.1&pickMembers%5B1%5D=3.1&pickMembers%5B2%5D=4.5&cubeTimeFrame.startYear=2020&cubeTimeFrame.endYear=2020&referencePeriods=20200101%2C20200101) 
        - Survivors at age X for all Canada, for both genders

### `circumstance_score` Distribution

The empirical part of the `health_ability` score (`heath_effort` portion will be fitted using mortality data). Decided to use CIHI educational discrete categories as the primary representation of circumstance (may use income later).

- `Canada_income_2021.csv`
    - Income distribution from 2021 census - represent as a pareto/gamma distribution(?)
    - [Source](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=9810005501)

- CIHI `measuring-health-inequalities-toolkit-education-stratifier-en.pdf`
   - Highest self-reported educational attainment
   - [Source](https://github.com/maguire-lab/health_inequity_simulation/issues/url) 2023-08-04
   - 
- `Canada_Education_2021.csv`  
    - Education distribution from 2021 census.
    - [Source](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=9810038401)
 

### `health_shock` number and `health_shock_magnitudes` distribution

Split these into the separate categories - split incidence 

- `IHME_GBD_2019_DISABILITY_WEIGHTS.XLSX`
    - Disability weights used in 2019 GDB estimates
    - [Source](https://ghdx.healthdata.org/record/ihme-data/gbd-2019-disability-weights)

- `IHME-GBD_2019_DATA.csv`
    - Includes deaths, disability adjusted life years (DALYs), years of life lost (YLLs), incidence, and prevalence for four chronic diseases: ischemic heart disease, alzheimer's disease (and other dementias), diabetes mellitus (types 1 and 2, both individually and combined), and all cancers combined. Age- and sex-specific estimates included.
    - [Source](https://vizhub.healthdata.org/gbd-results/)
    - Use the following to cite data included in this download:
        - Global Burden of Disease Collaborative Network.
        - Global Burden of Disease Study 2019 (GBD 2019) Results.
        - Seattle, United States: Institute for Health Metrics and Evaluation (IHME), 2020.

## Analyses 

### Simulations to Evaluate 
- binary circumstance (0.001 and 0.999 circumstance)
- Draw circumstance from income then education.
- all luck (initial circumstance all the same) - minimum amount of inequality
- all circumstance (fill the board with fixed negative effects)
- Differences in initial healthscores
- Different interaction between circumstance and luck (instead of circumstance impacting probability of health shock)? Alternatively could have circumstance determining random shock.

### Potential Modifications
- Non-uniform `health_shock` distribution
- Dynamic `health_shock` and `health_shock_magnitude` over model steps


