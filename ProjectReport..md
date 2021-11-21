## Group 8 
Charlie Barry (hbe2mx)
Emma Cooper (eyc4xd)
Aishwarya Pradhan (xaz3kw)
Christopher Lee (cl7zn)

# Introduction

Question: What are some key factors leading to hesitancy for the COVID vaccine?
Hypothesis: Demographics (Proximity to medical care), Political Affiliation, Education, Religion, Age, Gender, Race, Socioeconomic Status, Occupation

# Data
- [Us Vaccindation Data](https://github.com/owid/covid-19-data/tree/master/public/data/vaccinations/#united-states-vaccination-data)
- [Us State Vaccinations](https://ourworldindata.org/us-states-vaccinations)
- [Census Race Data](https://data.census.gov/cedsci/table?q=United%20States&t=Race%20and%20Ethnicity&tid=DECENNIALPL2020.P2)
- [Precinct level Election Results](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/NH5S2I)
- [County Level Election Data](https://github.com/tonmcg/US_County_Level_Election_Results_08-20/blob/master/2016_US_County_Level_Presidential_Results.csv)
- [Interesting query system for medical data based on County](https://hcupnet.ahrq.gov/#setup)
- [COVID-19 County Hesitancy](https://data.cdc.gov/Vaccinations/COVID-19-County-Hesitancy/c4bi-8ytd)

- Starting dataset: COVID-19 County Hesitancy data (CDC)
- Joined dataset #1: Census data
- Joined dataset #2: Election data
- Joined dataset #3: Healthcare data

All data joined on County (will likely have to use either Census tract or Fips Code)

# Experimental Design
Take data and join into one dataset based on location (either zip or census tracts)
Pre-processing and data clean-up
Correlation between variables
Regression analysis (Model for predicting outcome on new data)
Machine Learning techniques (Random Forest, Clustering)
Using ML techniques, identify variables of interest. Then create a model to predict outcomes on new data
Visualizations

# Project Management
**Milestones:**
- Identifying data sources
- Pre-Processing data and combining
- Initial Analyses to identify variables of interest
- Model Creation
- Model Validation
- Write-up

**Team Roles:**
_Charlie_
- Submitter
- Git Leader
_Aishwarya_
-Editor/Compiler
_Emma_
- Model Validation
_Christopher_
-Pre-Processing 
_Whole group:_
Combining data
-Initial Analyses
-Model Creation
-Write-up

# Results
Visualizations
ML Model Outputs
ML Demo Video
Linear Model with use of test data (training/test split)
PowerPoint Deck

# Testing
Data will be split into two groups (66% training, 34% test)
Models and variable selection will be done on training set
Model validation done on test set
Will use unit tests for python code
OUTCOME
Goal is to combine data to effectively analyze populations, groups, and areas that may be predisposed to a higher level of vaccine hesitancy.
This could be used by governments to focus education on the groups to elaborate on understanding of the vaccine, and alleviate any discomfort/hesitancy towards the vaccine
Summarize your plan and explain how your findings could be used by others (if applicable).


# Project Management Plan:

- Task #1: Set up GitHub repository (Emma)
- Task #2: Acquire and load Data (Charlie)
- Task #3: Join Data (Aishwarya & Chris)
- Task #4: Plan Analyses (Group)
