# --------------------------------------------------------------- #
# Metadata needed for creating the interactive results files
# And, for keeping track of each simulation's options for the model
# 
# Author: Yann Guerin
# --------------------------------------------------------------- #

# These are the options passed to the model as kwargs
simulation_options = {
    "Pure luck model": {"custom_unequal_health_score": 0.9, "equal_circumstance": True, "equal_effort": True, "uniform_probability_of_taking_shock": True},
    "Mediumly correlated luck model": {"custom_unequal_health_score": 0.9}, # is the same as the base model
    "Highly correlated luck model": {"custom_unequal_health_score": 0.9, "shock_probability_inversely_proportional_to_health_ability": True},
    "Deterministic model": {"custom_unequal_health_score": 0.9, "deterministic_shocks": True},
}

# The following mappings are mainly used in the generation of the interactive results. Allows for looping over each model and getting the relevant text from here.

simulation_num_to_description_mapping = {
    "Pure luck model": "0_pure_luck",
    "Mediumly correlated luck model": "4_unequal_c_e",
    "Highly correlated luck model": "new_9_unequal_L_enc_plus_4",
    "Deterministic model": "12_determ_all",
}

simulation_num_to_name_mapping = {
    "Pure luck model": "Pure luck model",
    "Mediumly correlated luck model": "Unequal society modified by effort",
    "Highly correlated luck model": "Unequal society mod. by E w/ unequal L-enc.",
    "Deterministic model": "Deterministic model",
}

simulation_num_to_title_mapping = {
    "Pure luck model": "Pure luck model",
    "Mediumly correlated luck model": "Unequal society with equal start modified by effort",
    "Highly correlated luck model": "Unequal society with equal start modified by effort with unequal shock encounter probability",
    "Deterministic model": "Deterministic model with equal health score at entry",
}

simulation_num_to_explanation_mapping = {
    "Pure luck model": "Random health shocks in equal society",
    "Mediumly correlated luck model": "Whether to take health shock is determined by circumstance and effort – the same model as  the calibration “base” model",
    "Highly correlated luck model": "Whether to take health shock is determined by circumstance and effort, whether to encounter a health shock is determined by health ability and age",
    "Deterministic model": "No randomness in life in unequal society – Brave New World or dystopian society, except that health score at entry is equal",
}
