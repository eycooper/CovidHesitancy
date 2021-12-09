## Project Report - Group 8 
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
- Editor/Compiler


_Emma_
- Model Validation


_Christopher_
- Pre-Processing 


_Whole group:_

- Combining data
- Initial Analyses
- Model Creation
- Write-up

# Data Processing

To begin, we loaded our data and created a processing function to run on the data which would handle NA's, improper data types, and consistency across the joined datasets. We used the following code to do so:

```{python}
def convert(data):
    number = preprocessing.LabelEncoder()
    float_cols_vec = [3,4,5,6,8,10,11,12,13,14,15,16,22,23,24,25]
    float_cols = np.array(final_df.columns)[float_cols_vec]
    for i in float_cols:
      data[i] = pd.to_numeric(data[i],errors='coerce')
    data=data.fillna(0)
    return data

final_df_processed=convert(final_df)

#Set matrix with desired columns
col_names=[3, 4, 5, 10, 11, 13, 14, 24, 25, 32]
np.array(final_df.columns)[col_names]

#Aggregate data by state
by_state= final_df.groupby(['State'])[np.array(final_df.columns)[col_names]].mean().reset_index()
Est_hes_corr = by_state.corr().iloc[0]
by_state['political_affiliation'] = np.where(by_state['per_dem'] > by_state['per_gop'], "Democrat", "Republican")
by_state.head()
```

# Intial Analyses

After Processing, we began exploring relationships seen in the data. First, we put together a Correlation Matrix to look at which variables were highly correlated. As most were not highly correlated, but also many were hovering around 0.5, we immediately realized this would have a large impact on the type of analyses we chose.

![image](https://user-images.githubusercontent.com/89169474/145312275-039029cb-da52-47d6-80c0-c1f5336696a6.png)

After exploring correlation, we also looked at a scatterplot of the different counties, with their Social Vulnerability Index (SVI), and the Estimated Hesitant value. As this plot was interactive, it will only be found within the code. Following this, we began looking at relationships of Estimated Hesitant with other variables.

![image](https://user-images.githubusercontent.com/89169474/145312557-7939279e-0fd0-4a1e-90c7-a94bf7ef3738.png)

![image](https://user-images.githubusercontent.com/89169474/145312576-68119655-3a96-4447-aaca-c56624946e1a.png)

![image](https://user-images.githubusercontent.com/89169474/145312588-048a059f-e6e4-43ab-9b01-d74ff9fad1e9.png)

# Regression Analyses

For proof of concept, we then did a single Linear Regression with the Percent GOP (per_gop) column, and Estimated Hesitant.
```
#Train Regression Model On Full Data Set

x=np.array(final_df['per_gop'])
y=np.array(final_df['Estimated hesitant'])
x=np.nan_to_num(x)
y=np.nan_to_num(y)

x= x.reshape(-1, 1)
y= y.reshape(-1, 1)

model = LinearRegression(fit_intercept =True).fit(x, y)
model.score(x, y)

#Plot Residuals
from yellowbrick.regressor import ResidualsPlot

visualizer = ResidualsPlot(model)
visualizer.fit(x, y)
```
![image](https://user-images.githubusercontent.com/89169474/145312715-993f7b6b-fb02-4274-866e-2f97faea43d8.png)

Given our model score was relatively low and residuals did not seem evenly dispersed around 0, we then went about performing a transformation.

```
#Transformation
from scipy import stats

prediction = model.predict(x)
residual = np.array(y - prediction)

tdata = stats.boxcox(y.flatten())[0]
plt.figure(figsize = (8, 8))
sns.distplot(tdata)
plt.show()
```

```
## Lambda value of -1 suggests transformation y_star=1/y
y_star = 1/y
model2 = LinearRegression(fit_intercept =True).fit(x, y_star)
model2.score(x, y_star)
```

# Random Forest
With our Linear Regression done, we then went about using a common Machine Learning algorithm to analyze the data further. Given the high level of correlation between predictors, we wanted to use a method that would incorporate this into the model, rather than having it detract from the model. Random Forest allows for correlation between predictors, so it seemed to fit perfect for the task.

To start, we split our data 70% for training, and 30% for testing.

```
predictor_cols=[6, 10, 11,12,13,14,15,16,24,25]

X = final_df_processed[np.array(final_df_processed.columns)[predictor_cols]]
y = final_df_processed['Estimated hesitant']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test
```

Next, we trained our Random Forest model on the training dataset.

```
# Instantiate model
rf = RandomForestRegressor()

# Train the model on training data
rf.fit(X_train, y_train);

#Observe predictions
y_pred=rf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
```

After, we looked at Feature Importance to identify variables of importance. Of note, Percent GOP, Percent non-Hispanic Black, Percent non-Hispanic American Indian/Alaska Native, and percent Hispanic were at the top.

```
#Access Random Forest Feature Importance

feature_importance = rf.feature_importances_
col_names = np.array(final_df.columns)[predictor_cols]

feat_importance_matrix = pd.DataFrame(
    {'Column Names': col_names,
     'Feature Importance': feature_importance
    })

feat_importance_matrix.sort_values(['Feature Importance'], ascending=False)
```


Finally, we tested the accuracy of our model on the test set by looking at how many predictions came with 0.1, 0.05, and 0.01 of the true value.

```
y_acc = y_test - y_pred
y_acc

a1 = 0
a05 = 0
a01 = 0
for i in y_acc:
  if abs(i) <= 0.1:
    a1 = a1+1
  if abs(i) <= 0.05:
    a05 = a05+1
  if abs(i) <= 0.01:
    a01 = a01+1

acc1 = a1/len(y_acc)
acc05 = a05/len(y_acc)
acc01 = a01/len(y_acc)

print('% of Predictions within 0.1 of true value: ' + str(round(acc1,3)))
print('% of Predictions within 0.05 of true value: ' + str(round(acc05, 3)))
print('% of Predictions within 0.01 of true value: ' + str(round(acc01, 3)))
```

% of Predictions within 0.1 of true value: 0.988
% of Predictions within 0.05 of true value: 0.862
% of Predictions within 0.01 of true value: 0.273


# Conclusions

In the end, we were able to put together a highly performing model that was above 85% accurate at getting within 0.05 of the true value for predictions (Estimated Hesitant has a range of 0.0269 - 0.267).
