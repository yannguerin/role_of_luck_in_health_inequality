import numpy as np
import scipy

class Circumstance_Distribution():
    """
    Supports random draws of specific discrete values (`circumstance_score_bins`)
    proportion to probabilities associated with each value (`circumstance_score_probabilities`).
    """
    def __init__(self, circumstance_score_bins, circumstance_score_probabilities):
        self.circumstance_score_bins = circumstance_score_bins
        self.circumstance_score_probabilities =  circumstance_score_probabilities

    def draw(self, number_of_draws: int):
        return np.random.choice(self.circumstance_score_bins,
                                size=number_of_draws,
                                replace=True,
                                p=self.circumstance_score_probabilities)

class Effort_Distribution():
    """
    Supports random draws from a specified (indicated in string `effort_type`)
    continuous probability distribution parameterised via dict `effort_params`
    """
    def __init__(self, effort_params: dict, effort_type: str):
        self.effort_params = effort_params
        self.effort_type = effort_type

    def draw(self, number_of_draws: int):
        # truncweibull_min, truncpareto, truncexpon
        if self.effort_type == 'truncnorm':
            a_transformed, b_transformed = (0 - self.effort_params['loc']) / self.effort_params['scale'], (1 - self.effort_params['loc']) / self.effort_params['scale'] 
            return scipy.stats.truncnorm.rvs(a=a_transformed, b=b_transformed,
                                             loc=self.effort_params['loc'],
                                             scale=self.effort_params['scale'],
                                             size=number_of_draws)
        elif self.effort_type == 'truncexpon':
            return scipy.stats.truncexpon.rvs(1, 
                                             loc=self.effort_params['loc'],
                                             scale=self.effort_params['scale'],
                                             size=number_of_draws) - self.effort_params['loc']
        elif self.effort_type == 'truncweibull_min':
            return scipy.stats.truncweibull_min.rvs(c=self.effort_params['shape'],
                                                   a=0,
                                                   b=1,
                                                   loc=self.effort_params['loc'],
                                                   scale=self.effort_params['scale'],
                                                   size=number_of_draws) - self.effort_params['loc']
        else:
            raise ValueError(f"Effort Type {self.effort_type} not supported. Please choose from truncnorm, truncexpon, or truncweibull_min.")


