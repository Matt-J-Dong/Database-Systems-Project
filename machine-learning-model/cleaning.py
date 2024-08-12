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

def quality_cleaning():
    columns_to_keep = [
        'DBN', 
        'School Name', 
        'Percent English Language Learners', 
        'Average Grade 8 English Proficiency', 
        'Average Grade 8 Math Proficiency'
    ]
    df2 = df2[columns_to_keep]
    df2.columns = df2.columns.str.replace(' ', '_')
#print(df2.head)
df2.to_csv('quality_cleaned.csv', index=False)
def attendance_cleaning():
    columns_to_convert = ['# Days Absent', '# Days Present', '% Attendance', '# Contributing 20+ Total Days', '# Chronically Absent', '% Chronically Absent']
    for column in columns_to_convert:
        df3[column] = pd.to_numeric(df3[column], errors='coerce')

    df3.dropna(subset=columns_to_convert, inplace=True)
    #print(df3)
    df3.to_csv('attendance_recent_cleaned.csv', index=False)