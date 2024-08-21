import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

df1 = pd.read_csv('sat.csv')
df2 = pd.read_csv('quality.csv')
merged_df = df1.merge(df2, on="DBN")
merged_df.dropna(inplace=True)
predictors = ['Critical Reading Mean', 'Mathematics Mean', 'Writing Mean']
target_variables = ['Rigorous Instruction Rating', 'Collaborative Teachers Rating', 'Supportive Environment Rating', 'Effective School Leadership Rating', 'Strong Family-Community Ties Rating', 'Trust Rating', 'Student Achievement Rating']

for target in target_variables:
    # Map the ratings to binary values: 1 if 'Meeting Target' or 'Exceeding Target', 0 otherwise
    merged_df[f'{target}_binary'] = merged_df[target].apply(lambda x: 1 if x in ['Meeting Target', 'Exceeding Target'] else 0)

X = merged_df[predictors]
results = []
for target in target_variables:
    y = merged_df[f"{target}_binary"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, target_names=['Not Meeting/Approaching Target', 'Meeting/Exceeding Target'])

    results.append((target, accuracy, cm, cr))

    print(f"\n{target} Classification Report:\n")
    print(cr)
    print(f"Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n{cm}\n")

for predictor in predictors:
    plt.figure(figsize=(12, 8))
    for target in target_variables:
        plt.subplot(4, 2, target_variables.index(target) + 1)
        sns.boxplot(x=f'{target}_binary', y=predictor, data=merged_df)
        plt.title(f'{predictor} by {target}')
        plt.xlabel('Binary Target (0: Not Meeting/Approaching, 1: Meeting/Exceeding)')
        plt.ylabel(predictor)
    plt.tight_layout()
    plt.savefig(f'./data/sat/{predictor}_vs_Metrics.png')
    plt.show()
