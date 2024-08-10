import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# LOAD DATA
@st.cache_data
def load_data():
    data_url = 'https://opendata.arcgis.com/datasets/4063314923d74187be9596f10d034914_0.csv'
    return pd.read_csv(data_url)

df = load_data()

# EXPLORATORY DATA ANALYSIS

st.title(":rainbow[ClimateChange Prediction]")

st.markdown(
    """
    <h1 style="text-align: center; margin-top: -30px;">☀️🌎</h1>
    """,
    unsafe_allow_html=True
)
st.write("### 1.Dataset Preview")
st.write(df.head())

st.write("### 2.Basic Statistics")
st.write(df.describe())

st.write("### 3.Missing Values")
st.write(df.isnull().sum())

# Correlation Matrix
st.write("### 4.Correlation Matrix")
numeric_df = df.select_dtypes(include=['float64', 'int64'])
corr_matrix = numeric_df.corr()
fig, ax = plt.subplots(figsize=(30, 20))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Average Temperature Change by Years
st.write("### 5.Average Temperature Change Over Years")
year_columns = [col for col in df.columns if col.startswith('F')]
df_melted = df.melt(id_vars=['Country'], value_vars=year_columns, var_name='Year', value_name='Temperature')
df_melted['Year'] = df_melted['Year'].str.extract('(\d+)').astype(int)

fig, ax = plt.subplots(figsize=(12, 6))
df_melted.groupby('Year')['Temperature'].mean().plot(ax=ax)
ax.set_title('Average Temperature Change Over Years')
ax.set_xlabel('Year')
ax.set_ylabel('Average Temperature Change (°C)')
st.pyplot(fig)

#DATA PREPROCESSING AND FEATURE ENGINEERING

# Outlier Functions
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outliers(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    return dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None)

def grab_outliers(dataframe, col_name):
    low, up = outlier_thresholds(dataframe, col_name)
    return dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)]

def replace_with_threshold(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[dataframe[variable] < low_limit, variable] = low_limit
    dataframe.loc[dataframe[variable] > up_limit, variable] = up_limit

# Handling Outliers
st.write("### 6. Outlier Detection and Replacement")

# Creating a List to Store Results
results = []

for col in df.columns:
    if col.startswith('F'):
        result_before = check_outliers(df, col)
        results.append((col, result_before))

# Replace Outliers with Threshold Values
for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:
        replace_with_threshold(df, col)

# Checking for Outliers After Changes
for col, was_outlier in results:
    result_after = check_outliers(df, col)
    if was_outlier:
        results_text = f"▪️For {col}: Outliers present before replacement: **:red[True]** -> after replacement: **{result_after}**. "
        if not result_after:
            results_text += " Outliers were successfully replaced with threshold values."
    else:
        results_text = f"For {col}: Outliers present before replacement: **False** -> after replacement: **{result_after}**."
    st.write(results_text)

# Filling Missing Values
st.write("### 7.Filling Missing Values")
def missing_values_table(dataframe):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    return pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])

# Table with Missing Values
st.write("### Missing Values Table")
st.write(missing_values_table(df))

st.write(":red[n_miss:] Shows the number of missing values in each column.")
st.write(":red[ratio:] Shows the ratio of missing values in each column to the total row as a percentage.")

# Show Only Rows Containing Missing Values
missing_rows = df[df.isnull().any(axis=1)]
st.write(" **:blue[Rows with Missing Values]**")
st.dataframe(missing_rows)

# Filling in Missing Values
df_filled = df.copy()
df_filled = df_filled.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)
df_filled = df_filled.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 225) else x, axis=0)

# Show Rows Containing Missing Values ​​as Filled
filled_missing_rows = df_filled[df.isnull().any(axis=1)]
st.write("**:blue[Filled Rows with Missing Values]**")
st.dataframe(filled_missing_rows)

# Coding of Categorical Data
label_encoder = LabelEncoder()
df_melted['Country'] = label_encoder.fit_transform(df_melted['Country'])

# Time Series Feature Engineering
df_melted['Year_sin'] = np.sin(2 * np.pi * df_melted['Year'] / max(df_melted['Year']))
df_melted['Year_cos'] = np.cos(2 * np.pi * df_melted['Year'] / max(df_melted['Year']))

# Separating Data into Training and Test Sets
X = df_melted.drop(columns=['Temperature', 'Year'])
y = df_melted['Temperature']
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
X = X[numeric_features]
X = X.apply(lambda x: x.fillna(x.mean()), axis=0)
y = y.fillna(y.mean())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standart Scaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# MODELLING

st.write("### 8.Model Training and Evaluation")

# Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Random Forest Model
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# XGBoost Model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror')
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# Evaluating Model Performance
st.write(f"- Linear Regression RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lr)):.3f}")
st.write(f"- Linear Regression R2 Score: {r2_score(y_test, y_pred_lr):.3f}")

st.write(f"- Random Forest RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_rf)):.3f}")
st.write(f"- Random Forest R2 Score: {r2_score(y_test, y_pred_rf):.3f}")

st.write(f"- XGBoost RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_xgb)):.3f}")
st.write(f"- XGBoost R2 Score: {r2_score(y_test, y_pred_xgb):.3f}")

# Future Predictions
st.write("### 9.Future Predictions")
future_year = st.slider("Select Future Year for Prediction", min_value=2025, max_value=3000, value=2050)
example_country = df_melted['Country'].mode()[0]

future_data = {
    'Country': [example_country],
    'Year_sin': [np.sin(2 * np.pi * future_year / max(df_melted['Year']))],
    'Year_cos': [np.cos(2 * np.pi * future_year / max(df_melted['Year']))]
}

future_df = pd.DataFrame(future_data)
for col in numeric_features:
    if col not in future_df.columns:
        future_df[col] = X[col].mean()

future_df_scaled = scaler.transform(future_df)

# Model predictions
future_temp_change_lr = lr_model.predict(future_df_scaled)
future_temp_change_rf = rf_model.predict(future_df_scaled)
future_temp_change_xgb = xgb_model.predict(future_df_scaled)

st.write(f"For {future_year} Linear Regression Model Estimation: {future_temp_change_lr[0]:.3f} °C")
st.write(f"For {future_year} Random Forest Model Estimation: {future_temp_change_rf[0]:.3f} °C")
st.write(f"For {future_year} XGBoost Model Estimation: {future_temp_change_xgb[0]:.3f} °C")