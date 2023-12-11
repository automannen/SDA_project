import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


# Suppress SettingWithCopyWarning
pd.options.mode.chained_assignment = None  # default='warn'

# Load datasets
life_expectancy_df = pd.read_csv('../data/life_expectancy.csv')
pharma_sales_df = pd.read_csv('../data/pharma_sales_ppp.csv')

# Filter data for the year 2014
filtered_life_expectancy_df = life_expectancy_df[
  (life_expectancy_df['Variable'].isin(['Females at age 40', 'Males at age 40'])) &
  (life_expectancy_df['Measure'] == 'Years') &
  (life_expectancy_df['Year'] == 2014)
]
filtered_pharma_sales_df = pharma_sales_df[
  (pharma_sales_df['Measure'] == '/capita, US$ purchasing power parity') &
  (pharma_sales_df['Year'] == 2014)
]

# Drop unnecessary columns
columns_to_drop = ['VAR', 'UNIT', 'Measure', 'COU', 'YEA', 'Year', 'Flag Codes', 'Flags']
filtered_life_expectancy_df.drop(columns=columns_to_drop, inplace=True)
filtered_pharma_sales_df.drop(columns=columns_to_drop, inplace=True)

# Rename columns for clarity
filtered_life_expectancy_df.rename(columns={'Value': 'Life_Expectancy_Value', 'Variable': 'Life_Expectancy_Variable'}, inplace=True)
filtered_pharma_sales_df.rename(columns={'Value': 'Pharma_Sales_Value', 'Variable': 'Pharma_Sales_Variable'}, inplace=True)

# Identify common countries
common_countries = set(filtered_life_expectancy_df['Country']).intersection(set(filtered_pharma_sales_df['Country']))

# Filter the dataframes to include only common countries
filtered_life_expectancy_df = filtered_life_expectancy_df[filtered_life_expectancy_df['Country'].isin(common_countries)]
filtered_pharma_sales_df = filtered_pharma_sales_df[filtered_pharma_sales_df['Country'].isin(common_countries)]

# Merge datasets
merged_df = filtered_life_expectancy_df.merge(filtered_pharma_sales_df, on=['Country'], how='outer')


for life_exp_variable in ['Females at age 40', 'Males at age 40']:
    current_df = merged_df[merged_df['Life_Expectancy_Variable'] == life_exp_variable]
    print("\nLife_Expectancy_Variable: " + str(life_exp_variable) + "\n")

    current_df.drop(columns=['Life_Expectancy_Variable',], inplace=True)

    unique_pharma_sales_variables = current_df['Pharma_Sales_Variable'].unique()
    country_variable_count = {}

    for pharma_variable in unique_pharma_sales_variables:
        current_pharma_variable_df = current_df[current_df['Pharma_Sales_Variable'] == pharma_variable]

        for country in current_pharma_variable_df['Country'].unique():
            if country not in country_variable_count:
                country_variable_count[country] = 0

            country_variable_count[country] += 1

    print(country_variable_count)
    # Add a new column to current_df for country_variable_count
    current_df['Country_Variable_Count'] = current_df['Country'].map(country_variable_count)
    # Now, current_df includes the count of unique pharma sales variables for each country
    # print(current_df.head())

    # Iterate over each pharmaceutical sales variable
    for pharma_variable in current_df['Pharma_Sales_Variable'].unique():
        print(f"\nGLM Results for Pharma Sales Variable: {pharma_variable}")

        # Filter data for the current pharmaceutical sales variable
        current_pharma_variable_df = current_df[current_df['Pharma_Sales_Variable'] == pharma_variable]

        current_pharma_variable_df.to_csv('current_pharma_variable_df.csv', index=False)

        # Fit GLM
        # Note: You might want to handle cases where Pharma_Sales_Value is NaN
        # model = smf.glm(formula='Life_Expectancy_Value ~ Pharma_Sales_Value + Country_Variable_Count',
        #                 data=current_pharma_variable_df, family=sm.families.Gaussian()).fit()
        model = smf.glm(formula='Life_Expectancy_Value ~ np.log(Pharma_Sales_Value + 1) + Country_Variable_Count + np.log(Pharma_Sales_Value + 1) * Country_Variable_Count',
                        data=current_pharma_variable_df, family=sm.families.Gaussian()).fit()

        # with open('glm_result_pharma_sales.pkl', 'wb') as f:
        #     pickle.dump(model, f)


        # # Print the summary of the model
        print(model.summary())



# # Get unique Pharma Sales Variables
# unique_pharma_sales_variables = merged_df['Pharma_Sales_Variable'].unique()

# # Loop over each Pharma Sales Variable and run regression for each life expectancy variable
# for pharma_variable in unique_pharma_sales_variables:
#     for life_exp_variable in ['Females at age 40', 'Males at age 40']:
#         print(f"\nResults for Life Expectancy: {life_exp_variable}, Pharma Sales Variable: {pharma_variable}")

#         # Filter the data for the current variables
#         current_df = merged_df[(merged_df['Life_Expectancy_Variable'] == life_exp_variable) &
#                                (merged_df['Pharma_Sales_Variable'] == pharma_variable)]

#         if current_df.empty:
#             print("No data available for this combination.")
#             continue

#         # Prepare regression model
#         X = current_df[['Pharma_Sales_Value']]
#         missing_variable = pharma_variable + '_Missing'
#         X['Interaction'] = X['Pharma_Sales_Value'] * current_df[missing_variable]

#         y = current_df['Life_Expectancy_Value']

#         # Replace missing values in Pharma_Sales_Value with 0 for regression
#         X['Pharma_Sales_Value'].fillna(0, inplace=True)

#         # Add a constant to the model (intercept)
#         X = sm.add_constant(X)

#         # Fit the model
#         try:
#             model = sm.OLS(y, X, missing='drop')
#             results = model.fit()

#             # Print key results
#             print(f"R-squared: {results.rsquared:.3f}")
#             print(f"F-statistic: {results.fvalue:.3f}, P-value: {results.f_pvalue:.3f}")
#             for param in results.params.index:
#                 print(f"{param}: Coef = {results.params[param]:.3f}, P-value = {results.pvalues[param]:.3f}")
#         except ValueError as e:
#             print(f"Unable to compute regression for this combination: {e}")
#         print("-" * 80)
