# -*- coding: utf-8 -*-

#plots to compare the shift direction and duration of the shift class commuters using their socio economic features

#also initial arrival time distribution plot




"""
Created on Tue Jul 30 18:38:57 2024

@author: shams
"""




import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import csv
import textwrap
from datetime import datetime
import re
from collections import defaultdict
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns
#from betareg import Beta
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from scipy.stats import f_oneway
from scipy.stats import chi2
from statsmodels.discrete.discrete_model import MNLogit
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import KFold, cross_val_score
from sklearn.feature_selection import RFECV
from sklearn.metrics import f1_score, make_scorer
from datetime import datetime, timedelta
import random


from preprocess_data import preprocess_data
from sklearn.datasets import make_classification


# Specify the file path
file_path = 'enter file path'

# Read the CSV file into a DataFrame
data2 = pd.read_csv(file_path, quotechar='"', delimiter=';', na_values=[
                    "", "\"\""], encoding="UTF-8", engine='python')

data2.rename(columns=lambda x: x.split('.')[0] if '.' in x else x, inplace=True)

# Preprocess
data2 = preprocess_data(data2)

# Display the first few rows of the DataFrame to check
print(data2.head())


# Function to extract the maximum time from the interval string
def extract_max_time(interval_str):
    match = re.findall(r'\d+', str(interval_str))  # Ensure interval_str is a string
    if match:
        return max(map(int, match))
    else:
        return 0

# Function to determine the shift value and direction based on the specified rules
def determine_shift_value_with_direction(delay_str, advance_str):
    
    if pd.isna(delay_str) and pd.isna(advance_str):
        return np.nan
    
    if delay_str == advance_str:
        #return delay_str + ' (Del)'
        return advance_str #+ ' (Adv)'
    if pd.isna(delay_str):
        return advance_str + ' (Adv)'
    elif pd.isna(advance_str):
        return delay_str + ' (Del)'
    else:
     #   return None #advance_str
        delay_max = extract_max_time(delay_str)
        advance_max = extract_max_time(advance_str)
        if delay_max >= advance_max:
             return delay_str + ' (Del)'
        else:
             return advance_str + ' (Adv)'

# Create a new column with the shift value and direction determined by the specified rules
data2['Shift Value Max'] = data2.apply(
    lambda row: determine_shift_value_with_direction(row['Delay Time By'], row['Advance Time By']), axis=1)

# Identify the rows where advance_str equals delay_str
equal_shift_indices =  data2[data2['Delay Time By'] == data2['Advance Time By']].index
#data2[(data2['Delay Time By'] != 0) & (data2['Advance Time By'] != 0)].index
#data2[data2['Delay Time By'] == data2['Advance Time By']].index

# Randomly assign 'Adv' to 70% and 'Del' to 30% of these rows
num_adv = int(0.75 * len(equal_shift_indices))
adv_indices = np.random.choice(equal_shift_indices, num_adv, replace=False)
del_indices = equal_shift_indices.difference(adv_indices)

# Update the 'Shift Value Max' column with directions
data2.loc[adv_indices, 'Shift Value Max'] += ' (Adv)'
data2.loc[del_indices, 'Shift Value Max'] += ' (Del)'



################################################################################"



def extract_shift_info(shift_str):
   if pd.isnull(shift_str):
        return 0, 0, None
    
    # Match for the "More than" case
   more_than_match = re.search(r'More than (\d+)\s*minutes\s*\((Adv|Del)\)', shift_str)
   if more_than_match:
       min_shift = 60
       max_shift = 90
       direction = more_than_match.group(2)
       return min_shift, max_shift, direction
   
   # Match for the regular case
   regular_match = re.search(r'(\d+)\s*-\s*(\d+)\s*minutes\s*\((Adv|Del)\)', shift_str)
   if regular_match:
       min_shift = int(regular_match.group(1))
       max_shift = int(regular_match.group(2))
       direction = regular_match.group(3)
       return min_shift, max_shift, direction
   
    
   return np.nan, np.nan, np.nan

# Apply the function to create new columns with the shift limits and direction
data2[['Min_Shift', 'Max_Shift', 'Shift_Direction']] = data2['Shift Value Max'].apply(extract_shift_info).apply(pd.Series)

df=data2[data2['Shift'].isin(['Shift Everyday', 'Shift but Not Everyday'])]


############### shift duration, direction analysis based on chosen socio-eco characterstics

# Filter the subpopulations based on the shift criteria

#to print shift direction case
subpop1 = df[(df['Min_Shift'] >= 60)]
subpop2 = df[(df['Max_Shift'] <= 15)]  # Modify as needed


#to print shift duration graphs
#subpop1 = df[(df['Min_Shift'] >= 45) & (df['Shift_Direction'] == 'Del')]
#subpop2 = df[(df['Min_Shift'] >= 45) & (df['Shift_Direction'] == 'Adv')]
#for direction case, the graph values are different every time because for shift length equal in adv and del, every run assigns such users with 70% to be adv and rest del. so this randomness causes percentages to slightly change in every run


# Calculate the sizes of the subpopulations
size_total = len(df)
size_subpop1 = len(subpop1)
size_subpop2 = len(subpop2)

# Calculate the relative sizes
relative_size_subpop1 = (size_subpop1 / size_total) * 100
relative_size_subpop2 = (size_subpop2 / size_total) * 100

# Print the results
print(f"Total size of the dataframe: {size_total}")
print(f"Size of Subpopulation 1: {size_subpop1} ({relative_size_subpop1:.2f}% of total)")
print(f"Size of Subpopulation 2: {size_subpop2} ({relative_size_subpop2:.2f}% of total)")

# # Define features to compare
features = ['Age Group', 'Income', 'Theoretical Arrival Time Contract', 'Child Drop Off Frequency']



# Calculate percentages for each feature
def calculate_percentages(dff, feature):
    return dff[feature].value_counts(normalize=True) * 100

# Create the figure and subplots
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(25, 20))

# Define colors
color_subpop1 = 'skyblue'
color_subpop2 = 'salmon'

# Set font sizes
fontsize_labels = 25
fontsize_ticks = 24
legend_fontsize = 25
fontsize_numbers = 24  # New parameter for numbers on bars

for i, feature in enumerate(features):
    # Calculate percentages
    percentages_subpop1 = calculate_percentages(subpop1, feature)
    percentages_subpop2 = calculate_percentages(subpop2, feature)

    # Align indices
    all_indices = percentages_subpop1.index.union(percentages_subpop2.index)
    percentages_subpop1 = percentages_subpop1.reindex(all_indices, fill_value=0)
    percentages_subpop2 = percentages_subpop2.reindex(all_indices, fill_value=0)

    # Sort by subpop1
    percentages_subpop1 = percentages_subpop1.sort_values(ascending=False)
    percentages_subpop2 = percentages_subpop2[percentages_subpop1.index]

    # Chi-square test
    contingency_table = pd.DataFrame({
        'Delayed Arrival': subpop1[feature].value_counts().reindex(all_indices, fill_value=0),
        'Advanced Arrival': subpop2[feature].value_counts().reindex(all_indices, fill_value=0)
    })
    chi2, p, _, _ = chi2_contingency(contingency_table.T)

    # Axis setup
    indices = np.arange(len(all_indices))
    width = 0.4
    ax = axs[i // 2, i % 2]

    # Bar plot
    bars1 = ax.bar(indices - width / 2, percentages_subpop1, width=width,
                   color=color_subpop1, edgecolor='black', label='Delayed Arrival')
    bars2 = ax.bar(indices + width / 2, percentages_subpop2, width=width,
                   color=color_subpop2, edgecolor='black', label='Advanced Arrival')

    # Annotate bars with collision handling
    for idx, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        h1, h2 = bar1.get_height(), bar2.get_height()
        x1 = bar1.get_x() + bar1.get_width() / 2
        x2 = bar2.get_x() + bar2.get_width() / 2

        if abs(h1 - h2) < 5:  # close heights → avoid overlap
            if h1 >= h2:
                ax.text(x1, h1 + 2, f'{h1:.1f}%', ha='center', fontsize=fontsize_numbers)
                ax.text(x2, h2 + 0.2, f'{h2:.1f}%', ha='center', fontsize=fontsize_numbers)
            else:
                ax.text(x1, h1 + 0.2, f'{h1:.1f}%', ha='center', fontsize=fontsize_numbers)
                ax.text(x2, h2 + 2, f'{h2:.1f}%', ha='center', fontsize=fontsize_numbers)
        else:
            ax.text(x1, h1 + 0.2, f'{h1:.1f}%', ha='center', fontsize=fontsize_numbers)
            ax.text(x2, h2 + 0.2, f'{h2:.1f}%', ha='center', fontsize=fontsize_numbers)

    # Labels and ticks
    ax.set_title(f'{feature} Distribution', fontsize=fontsize_labels)
    ax.set_xlabel(feature, fontsize=fontsize_labels)
    ax.set_ylabel('Percentage', fontsize=fontsize_labels)
    ax.set_xticks(indices)
    ax.set_xticklabels(percentages_subpop1.index, rotation=45, ha='right', fontsize=fontsize_ticks)
    ax.tick_params(axis='y', labelsize=fontsize_ticks)
    # Adjust y-axis to avoid clipping top labels
    ymax = max([b.get_height() for b in bars1] + [b.get_height() for b in bars2])
    ax.set_ylim(0, ymax * 1.15)  # 15% headroom above tallest bar

    # Legend and p-value below it inside subplot
    legend = ax.legend(fontsize=legend_fontsize)
    box = legend.get_window_extent().transformed(ax.transAxes.inverted())
    pval_y = box.y0 - 0.05
    ax.text(0.95, pval_y, f'p = {p:.4e}', transform=ax.transAxes,
            fontsize=fontsize_labels, ha='right', va='top',
            bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2'))
# Layout
plt.tight_layout()
plt.show()



