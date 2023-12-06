import pandas as pd

# Read the CSV file
df = pd.read_csv('../data/gdp.csv')

# Optionally, filter out rows with certain measures
# df = df[df['MEASURE'] == 'SomeMeasure']

# Group by LOCATION, MEASURE, INDICATOR, and SUBJECT, and check for duplicates
for (location, measure, indicator, subject), group_df in df.groupby(['LOCATION', 'MEASURE', 'INDICATOR', 'SUBJECT']):
    duplicated_years = group_df[group_df.duplicated(subset='TIME', keep=False)]

    if not duplicated_years.empty:
        print(f"Duplicates found for {location}, {measure}, {indicator}, {subject}:")
        print(duplicated_years[['TIME', 'Value']])
