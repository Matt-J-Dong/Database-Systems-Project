import pandas as pd

# Load the data
df1 = pd.read_csv('attendance_graduation.csv')
df2 = pd.read_csv('progress_2006.csv')
df3 = pd.read_csv('progress_2007.csv')

# Rename the 'School' column to 'DBN'
df1.rename(columns={'School': 'DBN'}, inplace=True)
df1['SchoolYear'] = df1['SchoolYear'].astype(str).str[:4]
df1.rename(columns={'SchoolYear': 'Cohort'}, inplace=True)

# Group by 'DBN' and 'SchoolYear', and calculate the average attendance
attendance_summary = df1.groupby(['DBN', 'Cohort']).apply(
    lambda x: pd.Series({
        'Total Enrolled': x['Enrolled'].sum(),
        'Total Present': x['Present'].sum(),
        'Average Attendance': x['Present'].sum() / x['Enrolled'].sum()
    })
).reset_index()

df2['Cohort'] = 2006  # Assuming the cohort year is 2006 for all entries
df3['Cohort'] = 2007  # Assuming the cohort year is 2006 for all entries

# Select only the columns 'DBN', 'Cohort', and 'OVERALL SCORE' for the merge
df2 = df2[['DBN', 'Cohort', 'OVERALL SCORE']]
df3 = df3[['DBN', 'Cohort', 'OVERALL SCORE']]
df2.dropna(inplace=True)
df3.dropna(inplace=True)
df2.rename(columns={'OVERALL SCORE': 'Overall Score'}, inplace=True)
df3.rename(columns={'OVERALL SCORE': 'Overall Score'}, inplace=True)
print(df2)
print(df3)
# Merge df2 and df3 on the compound key ('DBN' and 'Cohort')
quality_all = pd.concat([df2, df3], ignore_index=True)

# Display the resulting DataFrame
print(attendance_summary)

# Save the results to a new CSV file
attendance_summary.to_csv('attendance_summary.csv', index=False)
quality_all.to_csv('quality_all.csv', index=False)