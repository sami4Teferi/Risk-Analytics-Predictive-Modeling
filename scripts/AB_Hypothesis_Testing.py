import pandas as pd
from scipy.stats import chi2_contingency , ttest_ind , f_oneway
import os

# function to load data
def data_load(filepath):
    return pd.read_csv(filepath)

# function to see the null values
def data_preprocess(data):
    return data.isnull().sum()

# detect unique values (potential outliers) in a categorical column
def categorical_outliers_detecting(data , column):
    return data[column].unique()

# replace or remove invalid categorical values
def handle_catagorical_outliers(data,column , valid_values , replecment_values=None):
    invalid_entries = data[~data[column].isin(valid_values)]
    if replecment_values is not None:
        data.loc[~data[column].isin(valid_values),column] = replecment_values
        return 'the outlier is replaced and here is the data', data
    else:
        data = data[data[column].isin(valid_values)]
        return 'remove rows with invalid rows' , data


def calculate_kpis(data):
    data['ClaimFrequency'] = (data['TotalClaims'] > 0).astype(int)
    data['ClaimSeverity'] = data['TotalClaims']  # or any logic you want
    data['Margin'] = data['TotalPremium'] - data['TotalClaims']
    return data


# perform chi-squared test for Province vs Claim Frequency
def ab_test_provinces(data):
    contingency_table = pd.crosstab(data['Province'], data['ClaimFrequency'])
    chi2,p_value, _, _ = chi2_contingency(contingency_table)
    return chi2,p_value,contingency_table

# perform chi-squared test for Zip Code vs Claim Frequency
def ab_test_zipcodes(data):
    contingency_table = pd.crosstab(data['PostalCode'], data['ClaimFrequency'])
    chi2 , p_value, _,_ = chi2_contingency(contingency_table)
    return chi2,p_value,contingency_table

# perform ANOVA test for Margin difference across zip codes
def ab_test_zipcode_margin(data):
    f_stat,p_value = f_oneway(*[data[data['PostalCode'] == code]['Margin'] for code in data['PostalCode'].unique()])
    return f_stat,p_value

# perform T-test for gender-based risk difference (Claim Frequency)
def ab_test_gender(data):
    male_claims = data[data['Gender'] == 'Male']['ClaimFrequency']
    female_claims = data[data['Gender'] == 'Female']['ClaimFrequency']
    t_test,p_value = ttest_ind(male_claims,female_claims)
    return t_test,p_value

# report test result interpretation
def report_results(p_value,alpha=0.05):
    if p_value < alpha:
        return "Reject the null hypothesis (statistically significant)"
    else:
        return "Fail to reject the null hypothesis (not statistically significant)"
