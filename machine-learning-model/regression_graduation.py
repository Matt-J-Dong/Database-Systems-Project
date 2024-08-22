import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

attendance_summary = pd.read_csv('attendance_summary.csv')
quality_all = pd.read_csv('quality_all.csv')
graduation = pd.read_csv('graduation.csv')
graduation.drop(columns=['Cohort'], inplace=True)
graduation.rename(columns={'Cohort Year': 'Cohort'}, inplace=True)
attendance_summary['Cohort'] = attendance_summary['Cohort'].astype(int)
quality_all['Cohort'] = quality_all['Cohort'].astype(int)
graduation['Cohort'] = graduation['Cohort'].astype(int)

graduation["Cohort_numeric"] = pd.to_numeric(
    graduation["Cohort"], errors="coerce"
)  # Converting to numerical values
non_numeric_cohort_count = graduation["Cohort_numeric"].isna().sum()
non_numeric_cohort_rows = graduation[graduation['Cohort_numeric'].isna()]
# print(non_numeric_cohort_rows)
# print(graduation)

merged_df = attendance_summary.merge(quality_all, on=['DBN', 'Cohort']).merge(graduation, on=['DBN', 'Cohort'])
merged_df = merged_df[['Average Attendance', 'Overall Score', 'Total Grads % of cohort', 'Total Regents % of cohort', 'Advanced Regents % of cohort', 'Dropped Out % of cohort']]
print(merged_df)
merged_df.dropna(inplace=True)

X = merged_df[["Average Attendance", "Overall Score"]]
y_variables = ['Total Grads % of cohort', 'Total Regents % of cohort', 'Advanced Regents % of cohort', 'Dropped Out % of cohort']

for y_var in y_variables:
    y = merged_df[y_var]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"{y_var} Mean Squared Error: {mse}")
    print(f"{y_var} R-squared: {r2}")

    for feature in ['Average Attendance', 'Overall Score']:
        plt.figure(figsize=(10, 6))
        sns.regplot(x=feature, y=y_var, data=merged_df)
        plt.title(f'{feature} vs. {y_var}')
        plt.xlabel(feature)
        plt.ylabel(y_var)
        plt.savefig(f'./data/graduation/{feature}_vs_{y_var}.png')
        plt.close()
