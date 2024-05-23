############################################
#
#  Data Science:  Decision Tree Information Gain
#
#  Written By : BARA AHMAD MOHAMMED
#
#############################################

# TODO: IMPORT ALL NEEDED LIBRARIES
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mutual_info_score
import matplotlib.pyplot as plt
import os

# Create images directory if it doesn't exist
if not os.path.exists('images'):
    os.makedirs('images')

# Load data
df = pd.read_csv('hw6.data.csv')

# Split dataset into features and labels
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

root_info_gain = float('-inf')
root_col_number = 0

information_gains = []

for idx, column in enumerate(X.columns, start=1):
    # Print column number
    print(f"Processing Column {idx}...")

    col = X[[column]]
    # Initialize decision tree classifier
    dtc = DecisionTreeClassifier()

    # Fit DTC
    dtc.fit(col, y)

    # Get split value
    split_val = dtc.tree_.threshold[0]
    print("split value = ", split_val)

    # Calculate information gain
    information_gain = mutual_info_score(y, dtc.predict(col))
    print("information gain = ", information_gain)
    information_gains.append((idx, information_gain))

    if information_gain > root_info_gain:
        root_info_gain = information_gain
        root_col_number = idx
    print()

print("Max information gain is", root_info_gain, 
      "from column #", root_col_number)

# Plotting Information Gain for each feature
columns = [f'Column {idx}' for idx, _ in information_gains]
gains = [gain for _, gain in information_gains]

plt.figure(figsize=(10, 6))
plt.bar(columns, gains, color='skyblue')
plt.xlabel('Feature')
plt.ylabel('Information Gain')
plt.title('Information Gain for Each Feature')
plt.xticks(rotation=90)
plt.tight_layout()

# Save the plot
plt.savefig('images/information_gain.png')
plt.show()
