import luck_vs_circumstance as lvc
import pandas as pd
import numpy as np
import neworder

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
                 **kwargs
                ) -> None:
        """
        """
        super().__init__(neworder.LinearTimeline(0, number_of_years, number_of_years),
                         neworder.MonteCarlo.nondeterministic_stream)
        
        self.use_both = kwargs.get("use_both", True)

        # The annual health score decay to apply if use_both == True
        self.annual_health_score_decay = kwargs.get("annual_health_score_decay", 0.0046)

        # The parameters for the Gompertz function
        self.a = kwargs.get("a", 1.1)
        self.b = kwargs.get("b", 90)
        self.c = kwargs.get("c", 0.025)
        self.custom_unequal_health_score = kwargs.get("custom_unequal_health_score", 0.9)
        self.shock_probability_inversely_proportional_to_health_ability = kwargs.get("shock_probability_inversely_proportional_to_health_ability", False)

        # Deterministic Model
        self.deterministic_shocks = kwargs.get("deterministic_shocks", False)
 
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

        # health_ability is a product of circumstance and effort
        if kwargs.get("equal_circumstance"):
            circumstance_values = np.array([circumstance_dist.draw(population_size).mean()] * population_size)
        else:
            circumstance_values = circumstance_dist.draw(population_size)
        if kwargs.get("equal_effort"):
            effort_values = np.array([effort_dist.draw(population_size).mean()] * population_size)
        else:
            effort_values = effort_dist.draw(population_size)

        health_ability = (effort_values**health_ability_link_cobb_douglas_alpha) * (circumstance_values**(1-health_ability_link_cobb_douglas_alpha))    

        if self.custom_unequal_health_score:
            health_score = self.custom_unequal_health_score + (circumstance_values / (1 / (1 - self.custom_unequal_health_score)))
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

        if self.neonatal_deaths:
            # add neonatal mortality fixed % randomly dying
            # from https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1310071301&pickMembers%5B0%5D=1.1&cubeTimeFrame.startYear=2019&cubeTimeFrame.endYear=2019&referencePeriods=20190101%2C20190101
            neonatal_deaths = self.population.sample(int(population_size / 1000 * 4.4))
            self.population.loc[neonatal_deaths.index, "alive"] = False
            self.population.loc[neonatal_deaths.index, "age_of_death"] = 0
            self.population.loc[neonatal_deaths.index, "health_score"] = 0
            self.population.loc[neonatal_deaths.index, "unshocked_health_score"] = 0
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
            self.population.loc[self.population['alive'], 'health_score'] -= self._gompertz_health_decay(a=self.a, b=self.b, c=self.c) + self.annual_health_score_decay
            self.population.loc[self.population['alive'], 'unshocked_health_score'] -= self._gompertz_health_decay(a=self.a, b=self.b, c=self.c) + self.annual_health_score_decay
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
        
        for ix, shock in age_specific_health_shocks.iterrows():
            # only shock individuals that are alive and have not taken single-hit hits before
            if shock['Shock Type'] == 'single-hit':
                not_taken_single_hit_shock = [shock['cause'] not in self.shocks_taken_data[x]['shocks'] for x in alive_pop.index]
                shock_possible_pop = alive_pop[not_taken_single_hit_shock]
            # otherwise the susceptible pop is just everyone alive
            else:
                shock_possible_pop = alive_pop

            # ---------- #
            # Encounters #
            # ---------- #
            if self.shock_probability_inversely_proportional_to_health_ability:
                inverse_health_ability = 1 / shock_possible_pop.health_ability
                shock_probability_by_individual = (inverse_health_ability - inverse_health_ability.mean() + 1).to_numpy()
                vals = [shock['Shock Probability'] * shock_prob if (shock['Shock Probability'] * shock_prob) <= 1 else 1 for shock_prob in shock_probability_by_individual]
                encountered = np.array([self.mc.hazard(val, 1)[0] if val > 0 else 0 for val in vals]).astype(bool)
            elif self.deterministic_shocks:
                encountered = sum(self.mc.hazard(shock['Shock Probability'], len(shock_possible_pop)).astype(bool))
            else:
                # determine if eligible individuals are exposed to shock with the incidence probability of that shock
                encountered = self.mc.hazard(shock['Shock Probability'], len(shock_possible_pop)).astype(bool)
            
            if self.inspect:
                self.mc_hazard_vals.append((self.timeline.time, encountered.sum(), shock['Shock Probability'], shock['cause'], len(shock_possible_pop)))
            
            if self.deterministic_shocks:
                # Getting encountered number of individuals with the lowest health scores for deterministic shocks
                encountered_shock = shock_possible_pop.sort_values(by='health_score', ascending=True).iloc[:encountered, :]
            else:
                encountered_shock = shock_possible_pop[encountered]

            # determine if they actually take a shock based on their health ability
            if self.deterministic_shocks:
                # For the deterministic model I think the best approach to making this deterministic
                # is to take the bottom % of the mean of health abilities in the encountered population
                mean_health_ability = encountered_shock['health_ability'].mean()
                if encountered.any():
                    num_taken = round(encountered_shock.shape[0] * (1 - mean_health_ability))
                else:
                    num_taken = 0
                taken_shock = encountered_shock.iloc[:num_taken, :]
            else:
                taken_shock = encountered_shock[encountered_shock['health_ability'] <= np.random.random(len(encountered_shock))]

            if self.inspect:
                # For adding only encountered shocks that were not taken
                # self.encountered_shock_ids.extend(list(encountered_shock.index.difference(taken_shock.index)))
                # For adding all encountered shocks including the ones that were taken
                self.encountered_shock_ids.extend(list(encountered_shock.index))
                self.taken_shock_ids.extend(list(taken_shock.index))
                self.shock_causes.extend([shock['cause']] * len(encountered_shock.index))

            # Skip the rest of the iteration if there are no shocks taken
            if len(taken_shock) <= 0:
                continue

            # sample shock mangitude uniformly between disability weights
            shock_magnitudes = np.random.uniform(low=shock['Disability Weights'][0],
                                                 high=shock['Disability Weights'][1],
                                                 size=len(taken_shock))

            # Sorting the shock magnitudes and the taken shock dataframe
            # so that the highest shock magnitude gets applied to the individual with the lowest health score
            if self.deterministic_shocks:
                shock_magnitudes = np.linspace(shock['Disability Weights'][0], shock['Disability Weights'][1], len(taken_shock))
                taken_shock = taken_shock.sort_values(by='health_score', ascending=False)   

            for positional_index, index_key in enumerate(taken_shock.index):
                # append shock name to shocks list
                # append shock magnitudes to shocks_magnitudes list
                self.shocks_taken_data[index_key]['shocks'].append(shock['cause'])
                magnitude = shock_magnitudes[positional_index]

                if len(self.shocks_taken_data[index_key]['shocks']) > 1:
                    self.shocks_taken_data[index_key]['shock_magnitudes'].append(1 - magnitude)
                    # apply shock using compounding rule (health_score - (1-weight1) - (1-weight1) * (1-weight2)) etc
                    # need to use unshocked health score (i.e., just decay) to correctly applying compounding
                    self.population.loc[index_key, 'health_score'] = self.population.loc[index_key, 'health_score'] - np.prod(self.shocks_taken_data[index_key]['shock_magnitudes'])
                else:
                    # Appending (1 - magnitude) since that is used for compounding
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

        model_mortality = model_mortality.reset_index(name='count').rename(columns={'age_of_death': 'age'})
        model_mortality = model_mortality.sort_values('age', ascending=False).set_index('age').cumsum().iloc[::-1]
        model_mortality = model_mortality.rename(columns={'count': 'Model survivors at age x'}).reset_index()
        self.model_mortality = model_mortality
