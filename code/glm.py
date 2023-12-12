import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

merged_df = pd.read_csv('merged_data.csv')

# Columns:
# Country,Pharma_Sales_Variable,Pharma_Sales_Value,Life_Expectancy_Variable,Life_Expectancy_Value,Missingness_Indicator


# Run generalized linear models for each life expectancy variable
for life_exp_variable in merged_df['Life_Expectancy_Variable'].unique():

  print(f"\nResults for Life Expectancy: {life_exp_variable}")
  current_life_exp_df = merged_df[merged_df['Life_Expectancy_Variable'] == life_exp_variable]

  for pharma_sales_variable in current_life_exp_df['Pharma_Sales_Variable'].unique():
    print(f"Results for Pharma Sales: {pharma_sales_variable}")

    # Filter the DataFrame to include only the current life expectancy variable and pharma sales variable
    current_pharma_df = current_life_exp_df[current_life_exp_df['Pharma_Sales_Variable'] == pharma_sales_variable]


    model = smf.glm(formula='Life_Expectancy_Value ~ Pharma_Sales_Value + Missingness_Indicator',
                  data=current_pharma_df, family=sm.families.Gaussian()).fit()

    print(model.summary())


