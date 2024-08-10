#ESTIMATION OF THE EFFECTS OF CLIMATE CHANGE BY ANNUAL SURFACE TEMPERATURE

# Importing the necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import missingno as msno
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

# Load data
data_url = 'https://opendata.arcgis.com/datasets/4063314923d74187be9596f10d034914_0.csv'
df = pd.read_csv(data_url)

# EXPLORATORY DATA ANALYSIS

print(df.head())

# Basic statistics
print(df.describe())

# Missing value check
print(df.isnull().sum())

# Numeric columns
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Correlation matrix
corr_matrix = numeric_df.corr()

# Heeat map
plt.figure(figsize=(12, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# List column names by year
year_columns = [col for col in df.columns if col.startswith('F')]

# Calculating the average temperature change by year
df_melted = df.melt(id_vars=['Country'], value_vars=year_columns, var_name='Year', value_name='Temperature')
df_melted['Year'] = df_melted['Year'].str.extract('(\d+)').astype(int)


plt.figure(figsize=(12, 6))
df_melted.groupby('Year')['Temperature'].mean().plot()
plt.title('Average Temperature Change by Years')
plt.xlabel('Year')
plt.ylabel('Average Temperature Change (°C)')
plt.show()

# DATA PREPROCESSING AND FEATURE ENGINEERING

# Outliers

# Temperature change columns
temp_columns = df.columns[10:]

# Function that determines outlier boundaries
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

for col in df.columns:
    if col.startswith('F'):
        low, up = outlier_thresholds(df, col)
        print(f"For {col}: low limit: {low:.3f}, up limit: {up:.3f}")

# Checking for outliers
def check_outliers(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in df.columns:
    if col.startswith('F'):
        result = check_outliers(df, col)
        print(f"For {col}: result: {result}")

# Accessing outliers
def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    outliers = dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)]

    if outliers.shape[0] > 10:
        print(f"Column: {col_name}")
        print(outliers.head())
    else:
        print(f"Column: {col_name}")
        print(outliers)

    if index:
        outlier_index = outliers.index
        return outlier_index

for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:
        grab_outliers(df, col)

# Replace outliers with threshold values
def replace_with_threshold(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[dataframe[variable] < low_limit, variable] = low_limit
    dataframe.loc[dataframe[variable] > up_limit, variable] = up_limit

for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:
        replace_with_threshold(df, col)

# Check after changing with threshold value
for col in df.columns:
    if col.startswith('F'):
        result = check_outliers(df, col)
        print(f"For {col}: result: {result}")

# Missing Values

df.isnull().values.any()
df.isnull().sum()
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)

# Filling in missing values;

# -For numeric variables
df_melted = df_melted.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)

# -For categorical variables
df_melted = df_melted.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 225) else x, axis=0)

# Checking for missing values
print(df_melted.isnull().values.any())
print(df_melted.isnull().sum().sort_values(ascending=False))

# Encoding of Categorical Data
label_encoder = LabelEncoder()
df_melted['Country'] = label_encoder.fit_transform(df_melted['Country'])

# Time Series Feature Engineering
df_melted['Year_sin'] = np.sin(2 * np.pi * df_melted['Year'] / max(df_melted['Year']))
df_melted['Year_cos'] = np.cos(2 * np.pi * df_melted['Year'] / max(df_melted['Year']))

# Separating Data into Training and Test Sets
X = df_melted.drop(columns=['Temperature', 'Year'])
y = df_melted['Temperature']

# Make sure you only have numeric data
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
X = X[numeric_features]

# Check and fill in missing values
X = X.apply(lambda x: x.fillna(x.mean()), axis=0)
y = y.fillna(y.mean())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standart Scaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# MODELLING

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Random Forest
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# XGBoost
xgb_model = xgb.XGBRegressor(objective ='reg:squarederror')
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# Evaluating Model Performance
print("Linear Regression RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))
print("Linear Regression R2 Score:", r2_score(y_test, y_pred_lr))

print("Random Forest RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))
print("Random Forest R2 Score:", r2_score(y_test, y_pred_rf))

print("XGBoost RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_xgb)))
print("XGBoost R2 Score:", r2_score(y_test, y_pred_xgb))

# # Graphing - Comparing Model Performance
rmse_scores = [np.sqrt(mean_squared_error(y_test, y_pred_lr)),
               np.sqrt(mean_squared_error(y_test, y_pred_rf)),
               np.sqrt(mean_squared_error(y_test, y_pred_xgb))]

r2_scores = [r2_score(y_test, y_pred_lr),
             r2_score(y_test, y_pred_rf),
             r2_score(y_test, y_pred_xgb)]

models = ['Linear Regression', 'Random Forest', 'XGBoost']

# RMSE graphing
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.bar(models, rmse_scores, color=['blue', 'green', 'red'])
plt.title('Model Comparison - RMSE')
plt.xlabel('Models')
plt.ylabel('RMSE')

# R2 score graphic
plt.subplot(1, 2, 2)
plt.bar(models, r2_scores, color=['blue', 'green', 'red'])
plt.title('Model Comparison - R2 Score')
plt.xlabel('Models')
plt.ylabel('R2 Score')

plt.tight_layout()
plt.show()

'''
 -RMSE, bir modelin tahmin ettiği değerler ile gerçek değerler arasındaki farkların karelerinin ortalamasının kareköküdür.
 Bu metrik, modelin tahminlerinin ne kadar doğru olduğunu gösterir. RMSE değeri ne kadar düşükse, model o kadar iyi performans göstermektedir.

 -R², modelin bağımsız değişkenlerin bağımlı değişkeni açıklama oranını gösteren bir metriktir.
 0 ile 1 arasında bir değerdir ve 1'e ne kadar yakınsa model o kadar iyi performans göstermektedir.
 R², toplam değişimin ne kadarının model tarafından açıklandığını ifade eder.
 '''

# Assumptions for the year 2050
future_year_2050 = 2050
example_country = df_melted['Country'].mode()[0]

future_data_2050 = {
    'Country': [example_country],
    'Year_sin': [np.sin(2 * np.pi * future_year_2050 / max(df_melted['Year']))],
    'Year_cos': [np.cos(2 * np.pi * future_year_2050 / max(df_melted['Year']))]
}

future_df_2050 = pd.DataFrame(future_data_2050)

# Filling missing columns
for col in numeric_features:
    if col not in future_df_2050.columns:
        future_df_2050[col] = X[col].mean()

# Data scaling
future_df_scaled_2050 = scaler.transform(future_df_2050)

#  Model predictions
future_temp_change_lr_2050 = lr_model.predict(future_df_scaled_2050)
future_temp_change_rf_2050 = rf_model.predict(future_df_scaled_2050)
future_temp_change_xgb_2050 = xgb_model.predict(future_df_scaled_2050)

print(f"Linear Regression model forecast for 2050: {future_temp_change_lr_2050[0]:.4f} °C")
print(f"Random Forest model forecast for 2050: {future_temp_change_rf_2050[0]:.4f} °C")
print(f"XGBoost Model forecast for 2050: {future_temp_change_xgb_2050[0]:.4f} °C")

# Assumptions for the year 3000
future_year_3000 = 3000

future_data_3000 = {
    'Country': [example_country],
    'Year_sin': [np.sin(2 * np.pi * future_year_3000 / max(df_melted['Year']))],
    'Year_cos': [np.cos(2 * np.pi * future_year_3000 / max(df_melted['Year']))]
}

future_df_3000 = pd.DataFrame(future_data_3000)

# Filling missing columns
for col in numeric_features:
    if col not in future_df_3000.columns:
        future_df_3000[col] = X[col].mean()

# Data scaling
future_df_scaled_3000 = scaler.transform(future_df_3000)

# Model predictions
future_temp_change_lr_3000 = lr_model.predict(future_df_scaled_3000)
future_temp_change_rf_3000 = rf_model.predict(future_df_scaled_3000)
future_temp_change_xgb_3000 = xgb_model.predict(future_df_scaled_3000)

print(f"Linear Regression model forecast for 3000: {future_temp_change_lr_3000[0]:.4f} °C")
print(f"Random Forest model forecast for 30000: {future_temp_change_rf_3000[0]:.4f} °C")
print(f"XGBoost Model forecast for 30000: {future_temp_change_xgb_3000[0]:.4f} °C")

# Graphing - Predictions for Future Years
years = ['2050', '3000']
lr_predictions = [future_temp_change_lr_2050[0], future_temp_change_lr_3000[0]]
rf_predictions = [future_temp_change_rf_2050[0], future_temp_change_rf_3000[0]]
xgb_predictions = [future_temp_change_xgb_2050[0], future_temp_change_xgb_3000[0]]

plt.figure(figsize=(12, 6))

plt.plot(years, lr_predictions, marker='o', label='Linear Regression', color='blue')
plt.plot(years, rf_predictions, marker='o', label='Random Forest', color='green')
plt.plot(years, xgb_predictions, marker='o', label='XGBoost', color='red')

plt.title('Predicted Temperature Changes for 2050 and 3000')
plt.xlabel('Year')
plt.ylabel('Predicted Temperature Change (°C)')
plt.legend()
plt.grid(True)
plt.show()

# Findings and business recommendations
'''
Results:
1. Linear Regression, Random Forest and XGBoost models predicted temperature change for the years 2050 and 3000.
2. Both models predict that the temperature change will be positive in 2050.
3. While the Linear Regression model produced an extremely high estimate for the year 3000, the Random Forest and XGBoost models produced more reasonable estimates.

Business Suggestions:
1. By reviewing current climate policies, stronger measures can be taken to reduce the effects of global warming.
2. Investments in renewable energy sources can be increased.
3. Cleaner technologies can be used in industrial processes and transportation to reduce greenhouse gas emissions.
4. By supporting long-term climate forecasts and research, better preparation can be made for future scenarios.
'''
