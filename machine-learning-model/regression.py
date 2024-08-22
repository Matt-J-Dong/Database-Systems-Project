import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

df1 = pd.read_csv('ratio.csv')
df2 = pd.read_csv('quality.csv')
df3 = pd.read_csv('attendance.csv')
merged_df = df1.merge(df2, on='DBN').merge(df3, on='DBN')
# print(merged_df)
merged_df.dropna(inplace=True)

columns_of_interest = ['School Pupil-Teacher Ratio', '% Attendance', 'Percent English Language Learners', 'Average Grade 8 English Proficiency', 'Average Grade 8 Math Proficiency']
data = merged_df[columns_of_interest]

# print(data)

# for column in columns_of_interest:
#     non_numeric_rows = data[pd.to_numeric(data[column], errors='coerce').isna()]
#     if not non_numeric_rows.empty:
#         print(f"\nNon-numeric values found in column '{column}':")
#         print(non_numeric_rows)

data = data.apply(pd.to_numeric, errors='coerce').copy()
data.dropna(inplace=True)
X = data[
    ["School Pupil-Teacher Ratio", "% Attendance", "Percent English Language Learners"]
]
y = data["Average Grade 8 English Proficiency"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"English Proficiency Mean Squared Error: {mse}")
print(f"English Proficiency R-squared: {r2}")

X = data[
    ["School Pupil-Teacher Ratio", "% Attendance", "Percent English Language Learners"]
]
y = data["Average Grade 8 Math Proficiency"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Math Proficiency Mean Squared Error: {mse}")
print(f"Math Proficiency R-squared: {r2}")

plt.figure(figsize=(10, 6))
sns.regplot(x='School Pupil-Teacher Ratio', y='Average Grade 8 English Proficiency', data=data)
plt.title('School Pupil-Teacher Ratio vs. Average Grade 8 English Proficiency')
plt.xlabel('School Pupil-Teacher Ratio')
plt.ylabel('Average Grade 8 English Proficiency')
plt.savefig('./data/School_Pupil_Teacher_Ratio_vs_Average_Grade_8_English_Proficiency.png')

plt.figure(figsize=(10, 6))
sns.regplot(x='% Attendance', y='Average Grade 8 English Proficiency', data=data)
plt.title('% Attendance vs. Average Grade 8 English Proficiency')
plt.xlabel('% Attendance')
plt.ylabel('Average Grade 8 English Proficiency')
plt.savefig('./data/Attendance_vs_Average_Grade_8_English_Proficiency.png')

plt.figure(figsize=(10, 6))
sns.regplot(x='Percent English Language Learners', y='Average Grade 8 English Proficiency', data=data)
plt.title('Percent English Language Learners vs. Average Grade 8 English Proficiency')
plt.xlabel('Percent English Language Learners')
plt.ylabel('Average Grade 8 English Proficiency')
plt.savefig('./data/Percent_English_Language_Learners_vs_Average_Grade_8_English_Proficiency.png')

plt.figure(figsize=(10, 6))
sns.regplot(x='School Pupil-Teacher Ratio', y='Average Grade 8 Math Proficiency', data=data)
plt.title('School Pupil-Teacher Ratio vs. Average Grade 8 Math Proficiency')
plt.xlabel('School Pupil-Teacher Ratio')
plt.ylabel('Average Grade 8 Math Proficiency')
plt.savefig('./data/School_Pupil_Teacher_Ratio_vs_Average_Grade_8_Math_Proficiency.png')

plt.figure(figsize=(10, 6))
sns.regplot(x='% Attendance', y='Average Grade 8 Math Proficiency', data=data)
plt.title('% Attendance vs. Average Grade 8 Math Proficiency')
plt.xlabel('% Attendance')
plt.ylabel('Average Grade 8 Math Proficiency')
plt.savefig('./data/Attendance_vs_Average_Grade_8_Math_Proficiency.png')

plt.figure(figsize=(10, 6))
sns.regplot(x='Percent English Language Learners', y='Average Grade 8 Math Proficiency', data=data)
plt.title('Percent English Language Learners vs. Average Grade 8 Math Proficiency')
plt.xlabel('Percent English Language Learners')
plt.ylabel('Average Grade 8 Math Proficiency')
plt.savefig('./data/Percent_English_Language_Learners_vs_Average_Grade_8_Math_Proficiency.png')
