import pandas as pd
from typing import Tuple

def parse_empirical_data(datasets: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	# canada education data to inform circumstance distribution
	canada_education = pd.read_csv(datasets + "/2016_Education_data.csv")
	canada_education = canada_education.loc[:, ['Highest certificate, diploma or degree', 'Percent']]

	advanced_degree = canada_education.loc[[10,11,12,13]]
	advanced_degree = pd.DataFrame(data={'Highest certificate, diploma or degree': ['Advanced Degree/Diploma'],
	                                     'Percent': [advanced_degree['Percent'].sum()]})
	canada_education = canada_education.loc[[0,1,3,6,7,9]]
	canada_education = pd.concat([canada_education, advanced_degree]).reset_index(drop=True)
	circumstance_rank = 1/(canada_education.index.max() + 1) # i.e. 1/number of ranks used
	canada_education['Circumstance_Score'] = (canada_education.index + 1) * circumstance_rank
	canada_education['Proportion'] = canada_education['Percent'] / 100


	# health_shock
	gbd_ages = ['<5 years', '5-9 years', '10-14 years', '15-19 years', '20-24 years', '25-29 years', '30-34 years',
	            '35-39 years', '40-44 years', '45-49 years', '50-54 years', '55-59 years', '60-64 years', '65-69 years',
	            '70-74 years', '75-79 years', '80-84', '85-89', '90-94', '95+ years']
	gbd_causes = ["Total cancers",
	              "Ischemic heart disease",
	              "Alzheimer's disease and other dementias",
	              "Diabetes mellitus"]

	health_shock_data = pd.read_csv(datasets + "/IHME-GBD_2019_DATA.csv")
	health_shock_data = health_shock_data[health_shock_data['age'].isin(gbd_ages)]
	health_shock_data = health_shock_data[health_shock_data['cause'].isin(gbd_causes)]

	## probability of health shock occurring
	health_shock_incidence = health_shock_data[(health_shock_data['measure'] == 'Incidence') & (health_shock_data['sex'] == 'Both') & (health_shock_data['metric'] == 'Rate')]
	health_shock_incidence = health_shock_incidence[['age', 'cause', 'val']].rename(columns={'val': 'New cases per 100,000 population'})
	health_shock_incidence['Shock Probability'] = health_shock_incidence['New cases per 100,000 population'] / 100_000
	health_shock_incidence = health_shock_incidence.set_index(['age', 'cause'])

	## preavalence of health shocks
	health_shock_prevalence = health_shock_data[(health_shock_data['measure'] == 'Prevalence') & (health_shock_data['sex'] == 'Both') & (health_shock_data['metric'] == 'Rate')]
	health_shock_prevalence = health_shock_prevalence[['age', 'cause', 'val']].rename(columns={'val': 'Prevalence per 100,000 population'}).set_index(['age', 'cause'])

	# probability of death from health shock
	health_shock_deaths = health_shock_data[(health_shock_data['measure'] == 'Deaths') & (health_shock_data['sex'] == 'Both') & (health_shock_data['metric'] == 'Rate')]
	health_shock_deaths = health_shock_deaths[['age', 'cause', 'val']].rename(columns={'val': 'Deaths per 100,000 population'}).set_index(['age', 'cause'])

	# directly determine disability weights from Yukiko/Nathan
	disability_weights = {'Diabetes mellitus': (0.049, 0.633),
	                      "Alzheimer's disease and other dementias": (0.069, 0.449),
	                      'Total cancers':  (0.288, 0.569),
	                      'Ischemic heart disease': (0.033, 0.432)}

	shock_type = {'Diabetes mellitus': 'single-hit',
	              "Alzheimer's disease and other dementias": 'single-hit',
	              'Total cancers': 'multi-hit',
	              'Ischemic heart disease': 'multi-hit'}

	health_shock_parameters = pd.concat([health_shock_incidence, health_shock_prevalence, health_shock_deaths], axis=1).reset_index()
	health_shock_parameters['Disability Weights'] = health_shock_parameters['cause'].apply(lambda x: disability_weights[x])
	health_shock_parameters['Shock Type'] = health_shock_parameters['cause'].apply(lambda x: shock_type[x])
	health_shock_parameters['age'] = health_shock_parameters['age'].replace({'<5 years': '0-4 years',
	                                                                         '95+ years': '95-100000'}).str.replace(' years', '')
	health_shock_parameters['age_start'] = health_shock_parameters['age'].str.split('-').str.get(0).astype(int)
	health_shock_parameters['age_end']= health_shock_parameters['age'].str.split('-').str.get(1).astype(int)

	# observed mortality data
	mortality = pd.read_csv(datasets + "/2019_Life_Table_data.csv")
	mortality['age'] = mortality['Age group'].str.replace(' year', '').str.replace('s', '').str.replace(' and over', '').astype(int)
	mortality['Actual survivors at age x'] = mortality['VALUE']
	mortality = mortality[['age', 'Actual survivors at age x']].set_index('age').reset_index()

	return canada_education, health_shock_parameters, mortality

def parse_mortality_data(datasets: str) -> pd.DataFrame:
	# observed mortality data
	mortality = pd.read_csv(datasets + "/2019_Life_Table_data.csv")
	mortality['age'] = mortality['Age group'].str.replace(' year', '').str.replace('s', '').str.replace(' and over', '').astype(int)
	mortality['Actual survivors at age x'] = mortality['VALUE']
	mortality = mortality[['age', 'Actual survivors at age x']].set_index('age').reset_index()
	return mortality

def parse_accidental_death_data(datasets: str) -> pd.DataFrame:
    fname = datasets + "/IHME-GBD_2019_Accidental_Mortality.csv"
    df = pd.read_csv(fname)

    accidental_deaths = df.groupby('age').val.sum().to_frame().reset_index()

    accidental_deaths['age'] = accidental_deaths['age'].replace({'<5 years': '0-4 years',
                                                                            '95+ years': '95-100000'}).str.replace(' years', '')
    accidental_deaths['age_start'] = accidental_deaths['age'].str.split('-').str.get(0).astype(int)
    accidental_deaths['age_end'] = accidental_deaths['age'].str.split('-').str.get(1).astype(int)

    accidental_deaths['val'] = accidental_deaths['val'] / 100_000

    return accidental_deaths


if __name__ == "__main__":
	canada_education, health_shock_parameters, mortality = parse_empirical_data("./luck_vs_circumstance/datasets")