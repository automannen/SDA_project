# SDA project
In this project we researched the link between pharmaceutical sales and the life expectancy at age 40. This is done by looking at correlations and fitting a linear regression model.

# Table of contents
- Getting started
    - Prequisites
    - Installing
- Structure
- Usage
- Dataset

# Prerequisites
The libraries and packages needed for this project can be installed from the requirements.txt file with the following command:

<pre>pip install -r requirements.txt</pre>

# Installing
The installation steps:
1. Clone the repository
2. Cd into the project
<pre>cd SDA_project</pre>

3. Install the requirements
<pre>pip install -r requirements.txt</pre>

# Structure of the project
The structure of the project is as follows:

1. Code file
In the code file we have all our code that is used in the project

The files are:
- combinations_regression.py
In this file our models with the different drug variables are made and scores
for these models are calculated.
- data_exploration.ipynb
This notebook is used for data explorations and training the first prototype
of our model.
- life_expectancy_norm.py
Plotting the histogram of the dependent variable life expectancy to see if it is normally distributed
- old_pipeline_without_outlierdetection.py
Trying different transformations on the independent variables to see which transformation yields the highest
pearson correlation coefficient. This is done without first removing the outliers.
- outlier_detection_with_Kmeans.py
This file tries the outlier detection with Kmeans clustering
- outlier_detection.py
In this file the outlier detection is done with KNN.
- pipeline.py
This pipeline consists of first outlier detection with KNN and after that the
data is transformed in the pipeline
- prepare_data.py
This file is used to make the merged_data.csv file where the life expectancy and
pharmasales are merged. We also sampled a random value from the corresponding column
if there was a missing value in a row for a specific pharmaceutical variable.
- vif.py
The file for doing the vif test on 3 pharmaceutical sales variables to see if there is
multicollinearity.

2. Data
This file consists of the data used for the research and some data for feature research

3. Data visualization
consists of all the different plots.

# Usage
The Python files can be runned with the command if you are in the directory with:
<pre>python3 code_file.py</pre>

# Dataset
Link to dataset: https://stats.oecd.org/index.aspx?DataSetCode=HEALTH_PHMC#
