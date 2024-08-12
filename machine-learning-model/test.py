import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df1 = pd.read_csv('ratio.csv')
df2 = pd.read_csv('quality.csv')
df3 = pd.read_csv('attendance.csv')
merged_df = df1.merge(df2, on='DBN').merge(df3, on='DBN')
#print(merged_df)

# Preprocess the data
# Drop rows with missing values (or you can handle them differently)
merged_df.dropna(inplace=True)

# Select relevant columns
columns_of_interest = ['School Pupil-Teacher Ratio', '% Attendance', 'Percent English Language Learners', 'Average Grade 8 English Proficiency', 'Average Grade 8 Math Proficiency']
data = merged_df[columns_of_interest]

#print(data)

# for column in columns_of_interest:
#     non_numeric_rows = data[pd.to_numeric(data[column], errors='coerce').isna()]
#     if not non_numeric_rows.empty:
#         print(f"\nNon-numeric values found in column '{column}':")
#         print(non_numeric_rows)

# Drop rows with non-numeric values
data = data.apply(pd.to_numeric, errors='coerce').copy()
data.dropna(inplace=True)

# Select features and target variable for regression
X = data[['School Pupil-Teacher Ratio']]  # Features
y = data['Average Grade 8 English Proficiency']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"English Proficiency Mean Squared Error: {mse}")
print(f"English Proficiency R-squared: {r2}")

# Select features and target variable for regression
X = data[['School Pupil-Teacher Ratio']]  # Features
y = data['Average Grade 8 Math Proficiency']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Math Proficiency Mean Squared Error: {mse}")
print(f"Math Proficiency R-squared: {r2}")