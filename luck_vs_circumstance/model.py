import luck_vs_circumstance as lvc
import pandas as pd
import numpy as np
import scipy
import neworder
from .utils import *

class LvCHealthInequityModel(neworder.Model):
    """
    This model extends the builtin neworder.Model class by providing
    implementations of the following methods:
    - modify (optional)
    - step
    - check (optional)
    - finalise (optional)
    
    The neworder.run() function will execute the model, looping over
    the timeline and calling the methods above
    """

    def __init__(self,
                 population_size: int,
                 number_of_years: int,
                 circumstance_dist: lvc.dist.Circumstance_Distribution,
                 effort_dist: lvc.dist.Effort_Distribution,
                 health_shock_parameters: pd.DataFrame,
                 health_ability_link_cobb_douglas_alpha: float,
                 shock_probability_scaling: float | None,
                 shock_probability_scaling_post_age: float | None, 
                 **kwargs
                ) -> None:
        """
        """
        super().__init__(neworder.LinearTimeline(0, number_of_years, number_of_years),
                         neworder.MonteCarlo.nondeterministic_stream)
        
        # ------------- #
        # MODEL OPTIONS #
        # ------------- #

        # For health score decay using only gompertz deductions
        self.use_gompertz = kwargs.get("use_gompertz", False)
        # For health score decay using only annual linear deductions
        self.annual_health_score_decay = kwargs.get("annual_health_score_decay", 1/500)
        health_ability_link_function = kwargs.get("health_ability_link_function", None)
        self.use_both = kwargs.get("use_both", True)
        self.use_qx = kwargs.get("use_qx", False)
        self.gompertz_scaling = kwargs.get("gompertz_scaling", 1)
        self.shock_probability_conditional_on_effort = kwargs.get("shock_probability_conditional_on_effort", False)
        self.shock_probability_conditional_on_circumstance = kwargs.get("shock_probability_conditional_on_circumstance", False)
        self.shock_probability_conditional_on_health_ability = kwargs.get("shock_probability_conditional_on_health_ability", False)
        self.unequal_health_score = kwargs.get("unequal_health_score", False)
        self.slightly_unequal_health_score = kwargs.get("slightly_unequal_health_score", False)
        self.custom_unequal_health_score = kwargs.get("custom_unequal_health_score", 0.9)
        self.taking_shock_determined_by_health_score = kwargs.get("taking_shock_determined_by_health_score", False)
        self.shock_magnitude_influenced_by_health_ability = kwargs.get("shock_magnitude_influenced_by_health_ability", False)
        self.compounded_shock_probability_exp = kwargs.get("compounded_shock_probability_exp", False)
        self.compounded_shock_probability_prod = kwargs.get("compounded_shock_probability_prod", False)
        self.shock_probability_inversely_proportional_to_health_ability = kwargs.get("shock_probability_inversely_proportional_to_health_ability", False)
        self.triangular_l_take = kwargs.get("triangular_l_take", False)

        # Deterministic Model and its variants
        self.deterministic_shocks = kwargs.get("deterministic_shocks", False)
        self.deterministic_encounters = kwargs.get("deterministic_encounteres", False)
        self.deterministic_taken = kwargs.get("deterministic_taken", False)
        self.deterministic_magnitude = kwargs.get("deterministic_magnitude", False)
 
        # Extras, not currently used for anything
        self.random_health_score = kwargs.get("random_health_score", False)
        self.modified_shock_magnitude = kwargs.get("modified_shock_magnitude", False)
        self.uniform_probability_of_taking_shock = kwargs.get("uniform_probability_of_taking_shock", False)
        self.random_circumstance = kwargs.get("random_circumstance", False)

        # Accidental Deaths
        self.accidental_deaths = kwargs.get("accidental_deaths", pd.DataFrame())

        # Neonatal Deaths
        self.neonatal_deaths = kwargs.get("neonatal_deaths", True)

        # Inspect Option adds extra tracking of the population data for each iteration of the model
        self.inspect = kwargs.get("inspect", False)
        if self.inspect:
            self.population_over_time = pd.DataFrame()
            self.options = kwargs
            self.mc_hazard_vals = []

        self.population_size = population_size
        self.number_of_years = number_of_years
        self.health_shock_parameters = health_shock_parameters.copy(deep=True)
        self.health_shock_parameters['Shock Probability'] *= shock_probability_scaling # type: ignore
        # For age dependant shock probability scaling
        self.shock_probability_scaling_pre_50 = shock_probability_scaling
        self.shock_probability_scaling_post_50 = shock_probability_scaling_post_age

        # --------------------------------- #
        # CREATING THE POPULATION DATAFRAME #
        # --------------------------------- #

        # health_ability is a product of circumstance and effort
        if kwargs.get("equal_circumstance"):
            circumstance_values = np.array([circumstance_dist.draw(population_size).mean()] * population_size)
        elif self.random_circumstance:
            circumstance_values = np.random.uniform(low=0, high=1, size=population_size)
        else:
            circumstance_values = circumstance_dist.draw(population_size)
        if kwargs.get("equal_effort"):
            effort_values = np.array([effort_dist.draw(population_size).mean()] * population_size)
        else:
            effort_values = effort_dist.draw(population_size)

        # cobb_douglas link function
        if not health_ability_link_function:
            health_ability = (effort_values**health_ability_link_cobb_douglas_alpha) * (circumstance_values**(1-health_ability_link_cobb_douglas_alpha))    
        elif health_ability_link_function == 'mean':
            health_ability = np.mean(np.vstack([effort_values, circumstance_values]), axis=0)
        elif health_ability_link_function == 'median':
            health_ability = np.median(np.vstack([effort_values, circumstance_values]), axis=0)
        else:
            raise ValueError("health_ability_link_functions must be one of: [mean, median]")

        if self.unequal_health_score:
            health_score = 0.5 + (circumstance_values / 2)
        elif self.slightly_unequal_health_score:
            health_score = 0.99 + (circumstance_values / 100)
        elif self.custom_unequal_health_score:
            health_score = self.custom_unequal_health_score + (circumstance_values / (1 / (1 - self.custom_unequal_health_score)))
        elif self.random_health_score:
            health_score = np.random.uniform(low=0.6, high=1.0, size=population_size)
        else:
            health_score = 1.0 

        # create a population of number_of_agents size
        self.population = pd.DataFrame(index=neworder.df.unique_index(population_size),
                                       data={"health_score": health_score,
                                             "unshocked_health_score": health_score,
                                             "health_ability": health_ability,
                                             "circumstance_score": circumstance_values,
                                             "effort_score": effort_values,
                                             "alive": True,
                                             "age_of_death": number_of_years})
        self.shocks_taken_data = {id: {'shocks': [], 'shock_magnitudes': []} for id in self.population.index}
        self.population.index.name = "id"

        # ------------------------ #
        # IMPOSING NEONATAL DEATHS #
        # ------------------------ #

        if self.neonatal_deaths:
            # add neonatal mortality fixed % randomly dying
            # from https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1310071301&pickMembers%5B0%5D=1.1&cubeTimeFrame.startYear=2019&cubeTimeFrame.endYear=2019&referencePeriods=20190101%2C20190101
            neonatal_deaths = self.population.sample(int(population_size / 1000 * 4.4))
            self.population.loc[neonatal_deaths.index, "alive"] = False
            self.population.loc[neonatal_deaths.index, "age_of_death"] = 0
            self.neonatal_death_ids = neonatal_deaths.index

    def step(self) -> None:
        """
        Transitions to run at each timestep.
        This method must be implemented.
        Arguments: self
        Returns: NoneType
        """
        # randomly have people take health shocks
        self.impose_health_shock()

        if self.use_both:
            self.population.loc[self.population['alive'], 'health_score'] -= (self._gompertz_health_decay() + self.annual_health_score_decay)
            self.population.loc[self.population['alive'], 'unshocked_health_score'] -= (self._gompertz_health_decay() + self.annual_health_score_decay)
        elif self.use_gompertz:
            # decay health_scores by gompertz distribution
            self.population.loc[self.population['alive'], 'health_score'] -= self._gompertz_health_decay() #  + self.annual_health_score_decay
            self.population.loc[self.population['alive'], 'unshocked_health_score'] -= self._gompertz_health_decay() #  + self.annual_health_score_decay
        elif self.use_qx:
            # decay health_scores by gompertz distribution
            val = self._gompertz_health_decay()
            self.population.loc[self.population['alive'], 'health_score'] -= self._gompertz_health_decay() #  + self.annual_health_score_decay
            self.population.loc[self.population['alive'], 'unshocked_health_score'] -= self._gompertz_health_decay() #  + self.annual_health_score_decay
            # Create random number between [0, 1) for each person in the population
            # There may be a slight issue here where the precision of the random numbers is too small to ever return a value of to kill for gompertz values at early times in the timeline
            # which would lead to too few people dying early on, which can be seen in the plots created using this option
            # though, the error in the plot could also be due to the gompertz being an approximation of the mortality curve
            random_nums = pd.Series(np.random.rand(self.population.index.size), index=self.population.index)
            # Bool depending on if the random number of less than or equal to the gompertz value at this time
            to_kill = random_nums <= val
            # If the random num is less and or equal to, then kill that person
            if to_kill.any():
                self.population.loc[to_kill[to_kill].index, 'health_score'] = 0
                self.population.loc[to_kill[to_kill].index, 'unshocked_health_score'] = 0
        else:
            self.population.loc[self.population['alive'], 'health_score'] -= self.annual_health_score_decay
            self.population.loc[self.population['alive'], 'unshocked_health_score'] -= self.annual_health_score_decay

        # check for deaths and update age of death for new deaths
        new_deaths = self.population[(self.population['alive'] == True) & (self.population['health_score'] <= 0)].index
        self.population.loc[new_deaths, 'alive'] = False
        self.population.loc[new_deaths, 'age_of_death'] = self.timeline.time # type: ignore
        if self.inspect:
            death_from_decay = set(new_deaths) - set(self.death_from_shock_ids)

        # apply accidental deaths if given
        if not self.accidental_deaths.empty:
            self.impose_accidental_deaths()

        # Oh boy do i gotta clean this mess up
        if self.inspect:
            self.population_copy = self.population.copy()
            self.population_copy['time'] = self.timeline.time
            self.population_copy['encountered_shock'] = False
            self.population_copy['taken_shock'] = False
            self.population_copy['shock_causes'] = None
            self.population_copy['cause_of_death'] = None
            self.population_copy.loc[self.encountered_shock_ids, 'encountered_shock'] = True 
            self.population_copy.loc[self.taken_shock_ids, 'taken_shock'] = True 
            self.population_copy.loc[self.encountered_shock_ids, 'shock_causes'] = self.shock_causes 
            self.population_copy.loc[self.death_from_shock_ids, 'cause_of_death'] = [self.shocks_taken_data[index_key]['shocks'][-1] for index_key in self.death_from_shock_ids]
            self.population_copy.loc[list(death_from_decay), 'cause_of_death'] = "decay"
            if not self.accidental_deaths.empty:
                self.population_copy.loc[self.accidental_death_ids, 'cause_of_death'] = "accidental"
            if self.timeline.time == 0 and self.neonatal_deaths:
                self.population_copy.loc[self.neonatal_death_ids, 'cause_of_death'] = "neonatal" 
            self.population_over_time = pd.concat([self.population_over_time, self.population_copy])
            
    def impose_health_shock(self) -> None:
        age_specific_health_shocks = self.health_shock_parameters[(self.health_shock_parameters['age_start'] <= self.timeline.time) & (self.health_shock_parameters['age_end'] >= self.timeline.time)]

        if self.inspect:
            self.encountered_shock_ids = []
            self.shock_causes = []
            self.taken_shock_ids = []
            self.death_from_shock_ids = []

        # Get the population that is still alive, to impose the health shocks on
        alive_pop = self.population[self.population['alive']]
        # If everybody dead -> no health left to shock
        if alive_pop.empty:
            return
        
        # shock probability scaling dependant on age
        age_dependant_shock_probability_scaling_factor = None
        if self.shock_probability_scaling_pre_50 and self.shock_probability_scaling_post_50:            
            age_dependant_shock_probability_scaling_factor = self.shock_probability_scaling_pre_50 if self.timeline.time <= 50 else self.shock_probability_scaling_post_50

        for ix, shock in age_specific_health_shocks.iterrows():
            # only shock individuals that are alive and have not taken single-hit hits before
            if shock['Shock Type'] == 'single-hit':
                not_taken_single_hit_shock = [shock['cause'] not in self.shocks_taken_data[x]['shocks'] for x in alive_pop.index]
                shock_possible_pop = alive_pop[not_taken_single_hit_shock]
            # otherwise the susceptible pop is just everyone alive
            else:
                shock_possible_pop = alive_pop

            # scaling the shock probability if the age dependant scaling factor is defined
            if age_dependant_shock_probability_scaling_factor:
                shock['Shock Probability'] *= age_dependant_shock_probability_scaling_factor


            if self.shock_probability_inversely_proportional_to_health_ability:
                inverse_health_ability = 1 / shock_possible_pop.health_ability
                shock_probability_by_individual = (inverse_health_ability - inverse_health_ability.mean() + 1).to_numpy()
                vals = [shock['Shock Probability'] * shock_prob for shock_prob in shock_probability_by_individual]
                encountered = np.array([self.mc.hazard(val, 1)[0] if val > 0 else 0 for val in vals]).astype(bool)
            # scaling the shock probability for scenarios with unequal shock probability conditional on cirucmstance or effort or health ability
            elif self.shock_probability_conditional_on_health_ability:
                centered_health_ability_values = (shock_possible_pop['health_ability'] - shock_possible_pop['health_ability'].mean())
                # Added replace to remove the np.inf vals that show up when dividing by zero
                unsquashed_shock_probabilities = (1 / centered_health_ability_values).replace(np.inf, 0) * shock["Shock Probability"]
                shock_probability_by_individual = scipy.special.expit(list(unsquashed_shock_probabilities)) * 0.01
                encountered = np.array([self.mc.hazard(shock_prob, 1)[0] for shock_prob in shock_probability_by_individual]).astype(bool)
            elif self.shock_probability_conditional_on_circumstance:
                centered_circumstance_values = (shock_possible_pop['circumstance_score'] - shock_possible_pop['circumstance_score'].mean())
                # Added replace to remove the np.inf vals that show up when dividing by zero
                unsquashed_shock_probabilities = (1 / centered_circumstance_values).replace(np.inf, 0) * shock["Shock Probability"]
                if self.unequal_health_score:
                    shock_probability_by_individual = scipy.special.expit(list(unsquashed_shock_probabilities)) * 0.01
                else:
                    shock_probability_by_individual = scipy.special.expit(list(unsquashed_shock_probabilities)) * 0.01
                encountered = np.array([self.mc.hazard(shock_prob, 1)[0] for shock_prob in shock_probability_by_individual]).astype(bool)
                # encountered = self.mc.hazard(np.array(shock_probability_by_individual), len(shock_possible_pop)).astype(bool)
            elif self.shock_probability_conditional_on_effort:
                centered_effort_values = (shock_possible_pop['effort_score'] - shock_possible_pop['effort_score'].mean())
                # Added replace to remove the np.inf vals that show up when dividing by zero
                unsquashed_shock_probabilities = (1 / centered_effort_values).replace(np.inf, 0) * shock["Shock Probability"]
                shock_probability_by_individual = scipy.special.expit(list(unsquashed_shock_probabilities)) * 0.01                
                encountered = np.array([self.mc.hazard(shock_prob, 1)[0] for shock_prob in shock_probability_by_individual]).astype(bool)
                # encountered = self.mc.hazard(np.array(shock_probability_by_individual), len(shock_possible_pop)).astype(bool)
            elif self.deterministic_shocks or self.deterministic_encounters:
                encountered = sum(self.mc.hazard(shock['Shock Probability'], len(shock_possible_pop)).astype(bool))
            elif self.compounded_shock_probability_exp:
                num_shocks_per_agent = [len(self.shocks_taken_data[x]['shocks']) for x in shock_possible_pop.index]
                shock_probability_by_individual = [shock['Shock Probability'] ** (1 / (num_shocks + 1)) for num_shocks in num_shocks_per_agent]
                encountered = np.array([self.mc.hazard(shock_prob, 1)[0] for shock_prob in shock_probability_by_individual]).astype(bool)
            elif self.compounded_shock_probability_prod:
                num_shocks_per_agent = [len(self.shocks_taken_data[x]['shocks']) for x in shock_possible_pop.index]
                shock_probability_by_individual = [shock['Shock Probability'] * (num_shocks + 1) for num_shocks in num_shocks_per_agent]
                encountered = np.array([self.mc.hazard(shock_prob, 1)[0] for shock_prob in shock_probability_by_individual]).astype(bool)
            else:
                # determine if eligible individuals are exposed to shock with the incidence probability of that shock
                encountered = self.mc.hazard(shock['Shock Probability'], len(shock_possible_pop)).astype(bool)
            
            if self.inspect:
                self.mc_hazard_vals.append((self.timeline.time, encountered.sum(), shock['Shock Probability'], shock['cause'], len(shock_possible_pop)))
            
            if self.deterministic_shocks or self.deterministic_encounters:
                # Getting encountered number of individuals with the lowest health scores for deterministic shocks
                encountered_shock = shock_possible_pop.sort_values(by='health_score', ascending=True).iloc[:encountered, :]
            else:
                encountered_shock = shock_possible_pop[encountered]

            #neworder.log(f"Individuals encountering shock {shock['cause']}: {encountered_shock.index}")
            # determine if they actually take a shock based on their health ability
            if self.uniform_probability_of_taking_shock:
                taken_shock = encountered_shock[0.5 <= np.random.random(len(encountered_shock))]
            elif self.deterministic_shocks or self.deterministic_taken:
                # For the deterministic model I think the best approach to making this deterministic
                # is to take the bottom % of the mean of health abilities in the encountered population
                mean_health_ability = encountered_shock['health_ability'].mean()
                if encountered.any():
                    num_taken = round(encountered_shock.shape[0] * (1 - mean_health_ability))
                else:
                    num_taken = 0
                taken_shock = encountered_shock.iloc[:num_taken, :]
            elif self.taking_shock_determined_by_health_score:
                taken_shock = encountered_shock[encountered_shock['health_score'] <= np.random.random(len(encountered_shock))]
            elif self.triangular_l_take:
                triangular_health_abilities = encountered_shock['health_ability'].copy()
                triangular_health_abilities.loc[triangular_health_abilities[triangular_health_abilities < triangular_health_abilities.mean()].index] = 1 - triangular_health_abilities[triangular_health_abilities < triangular_health_abilities.mean()]
                encountered_shock['health_ability'] = triangular_health_abilities.copy()
                taken_shock = encountered_shock[encountered_shock['health_ability'] <= np.random.random(len(encountered_shock))]
            else:
                taken_shock = encountered_shock[encountered_shock['health_ability'] <= np.random.random(len(encountered_shock))]

            if self.inspect:
                # For adding only encountered shocks that were not taken
                # self.encountered_shock_ids.extend(list(encountered_shock.index.difference(taken_shock.index)))
                # For adding all encountered shocks including the ones that were taken
                self.encountered_shock_ids.extend(list(encountered_shock.index))
                self.taken_shock_ids.extend(list(taken_shock.index))
                self.shock_causes.extend([shock['cause']] * len(encountered_shock.index))

            #neworder.log(f"Individuals taking shock {shock['cause']}: {taken_shock.index}")

            # Skip the rest of the iteration if there are no shocks taken
            if len(taken_shock) <= 0:
                continue

            # sample shock mangitude uniformly between disability weights
            shock_magnitudes = np.random.uniform(low=shock['Disability Weights'][0],
                                                 high=shock['Disability Weights'][1],
                                                 size=len(taken_shock))
            #neworder.log(f"Shock magnitudes: {shock_magnitudes}")

            # Sorting the shock magnitudes and the taken shock dataframe
            # so that the highest shock magnitude gets applied to the individual with the lowest health score
            if self.deterministic_shocks or self.deterministic_magnitude:
                shock_magnitudes = np.linspace(shock['Disability Weights'][0], shock['Disability Weights'][1], len(taken_shock))
                taken_shock = taken_shock.sort_values(by='health_score', ascending=True)   
            elif self.shock_magnitude_influenced_by_health_ability and len(taken_shock) > 1:
                norm_uniform_magnitudes = min_max(shock_magnitudes)
                # norm_health_abilities = lvc.utils.min_max(taken_shock.health_abilities.to_numpy())
                norm_health_abilities = taken_shock.health_ability.to_numpy()
                norm_shock_magnitudes = ((1 - norm_health_abilities) / 2) + (norm_uniform_magnitudes / 2)
                shock_magnitudes = linear_scale(norm_shock_magnitudes, high=shock['Disability Weights'][1], low=shock['Disability Weights'][0])

            for positional_index, index_key in enumerate(taken_shock.index):
                # append shock name to shocks list
                # append shock magnitudes to shocks_magnitudes list
                self.shocks_taken_data[index_key]['shocks'].append(shock['cause'])
                if self.modified_shock_magnitude and self.timeline.time <= 50:
                    # magnitude = (shock_magnitudes[positional_index] / 2) * (1/self.population.loc[index_key, 'health_ability']) / 2 #  * (self.population.loc[index_key, 'health_score'] * (self.number_of_years - self.timeline.time) * 0.8) #  * age_scaling #  * (((self.number_of_years - 1) - self.timeline.time) / (self.number_of_years/2))
                    magnitude = (shock_magnitudes[positional_index] / 2) * (1/self.population.loc[index_key, 'health_ability']) / ((1/3.5*(-np.sqrt((-self.timeline.time + self.number_of_years))) + 4))
                else:
                    magnitude = shock_magnitudes[positional_index]

                if len(shock['cause']) > 1:
                    self.shocks_taken_data[index_key]['shock_magnitudes'].append(1 - magnitude)
                    # apply shock using compounding rule (health_score - (1-weight1) - (1-weight1) * (1-weight2)) etc
                    # need to use unshocked health score (i.e., just decay) to correctly applying compounding
                    self.population.loc[index_key, 'health_score'] = self.population.loc[index_key, 'health_score'] - np.prod(self.shocks_taken_data[index_key]['shock_magnitudes'])
                else:
                    self.shocks_taken_data[index_key]['shock_magnitudes'].append(1 - magnitude)
                    self.population.loc[index_key, 'health_score'] -= magnitude

                if self.population.loc[index_key, 'health_score'] <= 0 and self.inspect:                
                    self.death_from_shock_ids.append(index_key)

    def impose_accidental_deaths(self):
        # Get the population that is still alive, to impose the health shocks on
        alive_pop = self.population[self.population['alive']]
        # If everybody dead -> no health left to shock
        if alive_pop.empty:
            return
        
        age_specific_accidental_deaths = self.accidental_deaths[(self.accidental_deaths['age_start'] <= self.timeline.time) & (self.accidental_deaths['age_end'] >= self.timeline.time)]

        accidental_death = alive_pop[age_specific_accidental_deaths.val.iloc[0] >= np.random.random(len(alive_pop))].index

        self.population.loc[accidental_death, 'health_score'] = 0

        # check for deaths and update age of death for new deaths
        new_deaths = self.population[(self.population['alive'] == True) & (self.population['health_score'] <= 0)].index
        self.population.loc[new_deaths, 'alive'] = False
        self.population.loc[new_deaths, 'age_of_death'] = self.timeline.time # type: ignore
        self.accidental_death_ids = new_deaths

    def _gompertz_health_decay(self, a: int = 1, b: int = 77, c: float = 0.041) -> float:
        # From Michel
        # 1, 77, 0.041
        a = self.gompertz_scaling
        return a*np.exp(-b*np.exp(-c*(self.timeline.time)))
        
    def finalise(self) -> None:
        """
        This method (optional, if defined) is run at the end of the timeline
        Arguments: self
        Returns: NoneType
        """
        model_mortality = self.population['age_of_death'].value_counts()
        
        # Added the plus one to the number of years as this fixes a reshaping bug 
        # that occurs when everyone is dead before the end of the timeline
        for age in range(0, self.number_of_years + 1):
            if age not in model_mortality.index:
                model_mortality.loc[age] = 0

        model_mortality = model_mortality.reset_index(name='count')
        model_mortality = model_mortality.rename(columns={'age_of_death': 'age'})
        model_mortality = model_mortality.sort_values('age', ascending=False).set_index('age').cumsum().iloc[::-1]
        model_mortality = model_mortality.rename(columns={'count': 'Model survivors at age x'}).reset_index()
        self.model_mortality = model_mortality
