# -*- coding: utf-8 -*-

#plots wrt gender wise distribution of various features and shift direction and shift duration 

"""
Created on Mon Sep 30 10:37:52 2024

@author: shams
"""

# Import necessary libraries
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
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel



from preprocess_data import preprocess_data
from sklearn.datasets import make_classification


# Specify the file path
file_path = 'C:/Users/Cosmos/Desktop/hawking/hawking_edited.csv'

# Read the CSV file into a DataFrame
data2 = pd.read_csv(file_path, quotechar='"', delimiter=';', na_values=[
                    "", "\"\""], encoding="UTF-8", engine='python')

data2.rename(columns=lambda x: x.split('.')[0] if '.' in x else x, inplace=True)

# Preprocess
data2 = preprocess_data(data2)

# Display the first few rows of the DataFrame to check
print(data2.head())

########################## distances ###############################



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

# Assign 'Adv' and 'Del' to the corresponding indices in the 'Shift Value Max' column
# data2.loc[adv_indices, 'Shift Value Max'] = data2.loc[adv_indices].apply(lambda row: str(row['Advance Time By']) + ' (Adv)', axis=1)
# data2.loc[del_indices, 'Shift Value Max'] = data2.loc[del_indices].apply(lambda row: str(row['Delay Time By']) + ' (Del)', axis=1)


################################################################################"

def convert_to_hh_mm(time_str):

        if time_str is None:
            return None

        # Extract numbers using regex
        numbers = re.findall(r'\d+', time_str)

        # Ensure we have at least one number
        if not numbers:
            #print(time_str)
            return None

            #raise ValueError(f"Error: No numbers found: {time_str}")

        # Extract hour and minute
        hour = numbers[0].zfill(2)  # Add leading zero if necessary
        minute = numbers[1].zfill(2) if len(
            numbers) > 1 else '00'  # Add leading zero if necessary

        # Ensure hour and minute are in correct range
        hour = str(int(hour) % 24).zfill(2)
        minute = str(int(minute) % 60).zfill(2)

        # Return formatted time
        return f"{hour}:{minute}"

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



# Calculate Avg_Shift
data2['Avg_Shift'] = (data2['Min_Shift'] + data2['Max_Shift']) / 2



############################################################################### difference in drop off and start off time

# Apply the function to both columns
data2['b10a'] = data2['b10a'].apply(lambda x: convert_to_hh_mm(str(x)))
data2['b10b'] = data2['b10b'].apply(lambda x: convert_to_hh_mm(str(x)))



# Function to convert time to minutes after midnight
def convert_to_minutes_after_midnight(time_str):
    try:
        time_obj = pd.to_datetime(time_str, format='%H:%M')
        return time_obj.hour * 60 + time_obj.minute
    except ValueError:
        return None

# Apply the function to both columns
data2['b10a_minutes'] = data2['b10a'].apply(lambda x: convert_to_minutes_after_midnight(str(x)))
data2['b10b_minutes'] = data2['b10b'].apply(lambda x: convert_to_minutes_after_midnight(str(x)))


# Calculate the difference in minutes
data2['diff_minutes'] = data2['b10b_minutes'] - data2['b10a_minutes']

# Define the intervals
bins = [0, 5, 10, 15, float('inf')]
labels = ['<5 min', '5-10 min', '10-15 min', '>15 min']

# Categorize the differences into intervals
data2['Interval'] = pd.cut(data2['diff_minutes'], bins=bins, labels=labels)

# Filter the data for 'Femme' and 'Homme' genders
filtered_data = data2[data2['Gender'].isin(['Femme', 'Homme'])]

# Calculate the count of each interval bin by gender for chi-square test
interval_gender_counts = filtered_data.groupby(['Gender', 'Interval']).size().unstack(fill_value=0)

# Perform the chi-square test
chi2, p_value, _, _ = chi2_contingency(interval_gender_counts)

# Calculate the percentage of each interval bin by gender
interval_gender_percentage = interval_gender_counts.div(interval_gender_counts.sum(axis=1), axis=0) * 100

# Plotting the percentage distribution using a bar plot
ax = interval_gender_percentage.T.plot(kind='bar', figsize=(10, 6), width=0.8, color=['skyblue', 'lightcoral'])

# Set font sizes for title and labels
fontsize_title = 16
fontsize_labels = 15
fontsize_ticks = 16

# Add title with the p-value
plt.title(f'p = {p_value:.3e}', fontsize=fontsize_title)
plt.ylabel('Percentage (%)', fontsize=fontsize_labels)
plt.xlabel('Minutes Before School Start Time', fontsize=fontsize_labels)

# Add percentage labels on top of each bar
for p in ax.patches:
    ax.annotate(f'{p.get_height():.1f}%', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='baseline', fontsize=fontsize_ticks, color='black', xytext=(0, 5), textcoords='offset points')

# Show grid
#plt.grid(axis='y')

# Manually set the legend labels to "Women" and "Men"
plt.legend(labels=['Women', 'Men'], title='Gender', fontsize=fontsize_labels)

# Adjust tick parameters
ax.tick_params(axis='x', labelsize=fontsize_ticks)
ax.tick_params(axis='y', labelsize=fontsize_ticks)

# Display the plot
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()







############################################################### shift duration and direction by gender

# Define the income categories in the desired order
income_order = [
    'Moins de 1200 €', 
    '1200 - 1350 €',
    '1350 - 1600 €',
    '1600 - 1950 €',
    '1950 - 2650 €',
    '2650 - 3550 €',
    'Plus de 3550 €'
]

# Filter the data for Avg_Shift >= 30 min and 'Shift_Direction' as 'Del', and include only Femme and Homme
filtered_data_comparison = data2[(data2['Avg_Shift'] >= 30) & (data2['Shift_Direction'] == 'Adv')]
filtered_data_comparison = filtered_data_comparison[filtered_data_comparison['Gender'].isin(['Femme', 'Homme'])]
filtered_data_comparison['Group'] = filtered_data_comparison['Gender']

# Apply the income order
filtered_data_comparison['Income'] = pd.Categorical(filtered_data_comparison['Income'], categories=income_order, ordered=True)

# Define features to plot
features = ['Income', 'Theoretical Arrival Time Contract', 'Child Drop Off Frequency', 'Age Group']

# Set font sizes
fontsize_labels = 25
fontsize_ticks = 24
legend_fontsize = 25

# Set up the figure with 2 rows and 2 columns for subplots
fig, axes = plt.subplots(2, 2, figsize=(25, 20))  # Larger figure for readability
axes = axes.flatten()  # Flatten axes for easy iteration

for i, feature in enumerate(features):
    # Compute feature percentages
    feature_counts = pd.crosstab(filtered_data_comparison[feature], filtered_data_comparison['Group'])
    feature_percentages = feature_counts.div(feature_counts.sum(axis=0), axis=1) * 100

    # Chi-square test
    chi2, p, _, _ = chi2_contingency(feature_counts)

    # Prepare bars
    indices = np.arange(len(feature_percentages))
    width = 0.4

    # Plot bars
    axes[i].bar(indices - width/2, feature_percentages['Femme'], width=width,
                color='salmon', edgecolor='black', label='Women')
    axes[i].bar(indices + width/2, feature_percentages['Homme'], width=width,
                color='skyblue', edgecolor='black', label='Men')

    # Set title, labels
    axes[i].set_title(f'{feature} Distribution', fontsize=fontsize_labels)
    axes[i].set_xlabel(feature, fontsize=fontsize_labels)
    axes[i].set_ylabel('Percentage (%)', fontsize=fontsize_labels)
    axes[i].set_xticks(indices)
    axes[i].set_xticklabels(feature_percentages.index, rotation=45, ha='right', fontsize=fontsize_ticks)
    axes[i].tick_params(axis='y', labelsize=fontsize_ticks)

    # Place legend at fixed upper right position
    legend = axes[i].legend(loc='upper right', fontsize=legend_fontsize, framealpha=0.8)

    # p-value placed just below the legend, inside the plot
    axes[i].text(0.95, 0.70, f'p = {p:.4e}',
                 transform=axes[i].transAxes, ha='right', va='top',
                 fontsize=fontsize_ticks, bbox=dict(facecolor='white', edgecolor='none', alpha=0.6))

    # Add percentage labels
    for pbar in axes[i].patches:
        height = pbar.get_height()
        if height > 0:
            axes[i].annotate(f'{height:.1f}%', (pbar.get_x() + pbar.get_width() / 2., height),
                             ha='center', va='baseline', fontsize=fontsize_ticks,
                             color='black', xytext=(0, 5), textcoords='offset points')

fig.suptitle('Gender-wise Distribution of Socio-economic Features for Shift Duration/Direction', fontsize=fontsize_labels + 2, fontweight='bold')
plt.tight_layout()
plt.show()





######################### comparing female popoulation with completely free arrival to female pop with free within an imposed slot
# Define the income categories in the desired order
income_order = [
    'Moins de 1200 €', 
    '1200 - 1350 €',
    '1350 - 1600 €',
    '1600 - 1950 €',
    '1950 - 2650 €',
    '2650 - 3550 €',
    'Plus de 3550 €'
]

# Filter data
data2_copy = data2[data2['Theoretical Arrival Time Contract'].isin(['Completely Free Arrival', 'Free Arrival with Imposed Time Range'])]
data2_copy['Free Arrival'] = np.where(data2_copy['Theoretical Arrival Time Contract'] == 'Completely Free Arrival', 'Free Arrival', 'Other')
filtered_data_femme = data2_copy[data2_copy['Gender'] == 'Femme'].copy()
filtered_data_femme['Income'] = pd.Categorical(filtered_data_femme['Income'], categories=income_order, ordered=True)

# Count overall Free Arrival types
arrival_counts = filtered_data_femme['Free Arrival'].value_counts()
print('no: women with free arrrival',arrival_counts)

# Features to plot
features = ['Income', 'Free Arrival', 'Child Drop Off Frequency', 'Age Group']
fig, axes = plt.subplots(2, 2, figsize=(20, 16))
axes = axes.flatten()

# Plot each feature
for i, feature in enumerate(features):
    # Cross-tabulation and Chi-square
    ct = pd.crosstab(filtered_data_femme[feature], filtered_data_femme['Free Arrival'])
    chi2, p, _, _ = chi2_contingency(ct)

    # Percentages
    feature_percentages = ct.div(ct.sum(axis=0), axis=1) * 100

    # Plot
    feature_percentages.plot(kind='bar', stacked=False, ax=axes[i], color=['skyblue', 'lightcoral'])

    # Add title with formatted p-value
    axes[i].set_title(f'{feature} vs Free Arrival (Chi² p = {p:.4e})', fontsize=20)

    axes[i].set_xlabel(feature, fontsize=20)
    axes[i].set_ylabel('Percentage (%)', fontsize=20)
    axes[i].legend(fontsize=20)
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right', fontsize=20)

    # Annotate bars
    for pbar in axes[i].patches:
        height = pbar.get_height()
        if height > 0:
            axes[i].annotate(f'{height:.1f}%', 
                             (pbar.get_x() + pbar.get_width() / 2., height), 
                             ha='center', va='baseline', fontsize=16, 
                             xytext=(0, 5), textcoords='offset points')

fig.suptitle('Distribution of Socio-economic Features for Female subpop for Different Arrival Contract', fontsize=fontsize_labels + 2, fontweight='bold')

plt.tight_layout()
plt.show()

################################Genderwise Arrival Time Distributions##############################################################################

# Define the arrival time categories
arrival_columns = ['Before 08:00', '08:00 - 08:29', '08:30 - 08:59', '09:00 - 09:29', 'After 09:30']

# Filter the data for relevant genders
data2_filtered = data2[data2['Gender'].isin(['Homme', 'Femme'])]

# Convert 'Arrival Time' to a categorical variable with the specified order
data2_filtered['Arrival Time'] = pd.Categorical(data2_filtered['Arrival Time'], categories=arrival_columns, ordered=True)

# Create a crosstab normalized by index (percentage)
crosstab = pd.crosstab(data2_filtered['Gender'], data2_filtered['Arrival Time'], normalize='index') * 100

# Perform a chi-square test
chi2, p, _, _ = chi2_contingency(pd.crosstab(data2_filtered['Gender'], data2_filtered['Arrival Time']))

# Plotting the side-by-side bar chart
ax = crosstab.T.plot(kind='bar', width=0.8, figsize=(10, 6), color=['skyblue', 'salmon'])

# Set font sizes for title and labels
fontsize_title = 16
fontsize_labels = 15
fontsize_ticks = 16

# Adding labels and title
plt.title(f"p-value = {p:.3e}", fontsize=fontsize_title)
plt.xlabel('Arrival Time', fontsize=fontsize_labels)
plt.ylabel('Percentage', fontsize=fontsize_labels)
plt.xticks(rotation=45, fontsize=fontsize_ticks)
plt.legend(title='Gender', fontsize=fontsize_labels)

# Annotate bars with percentage labels
for i in ax.patches:
    ax.annotate(f'{i.get_height():.1f}%', 
                (i.get_x() + i.get_width() / 2, i.get_height()), 
                ha='center', va='bottom', fontsize=fontsize_ticks, color='black')

# Show the plot
plt.tight_layout()
plt.show()

#####################################################################################################
#genderwise distribution of socio-eco features


# Define the income categories in the desired order
income_order = [
    'Moins de 1200 €', 
    '1200 - 1350 €',
    '1350 - 1600 €',
    '1600 - 1950 €',
    '1950 - 2650 €',
    '2650 - 3550 €',
    'Plus de 3550 €'
]

# Filter data and keep the income order
filtered_data_comparison = data2.copy()
filtered_data_comparison = filtered_data_comparison[
    filtered_data_comparison['Gender'].isin(['Femme', 'Homme'])
].copy()
filtered_data_comparison['Income'] = pd.Categorical(
    filtered_data_comparison['Income'], 
    categories=income_order, 
    ordered=True
)
filtered_data_comparison['Group'] = filtered_data_comparison['Gender']

# Define features to plot
features = ['Income', 'Job Category', 'Theoretical Arrival Time Contract', 'Child Drop Off Frequency']

# Set font sizes
fontsize_labels = 25
fontsize_ticks = 24
legend_fontsize = 25

# Set up the figure with 2 rows and 2 columns for subplots
fig, axes = plt.subplots(2, 2, figsize=(25, 20))
axes = axes.flatten()  # Flatten axes for easy iteration

# Add space between rows (some top room will be added per-plot later if needed)
plt.subplots_adjust(hspace=0.8, top=0.92)

for i, feature in enumerate(features):
    # Calculate percentage distribution
    feature_counts = pd.crosstab(filtered_data_comparison[feature], filtered_data_comparison['Group'])
    feature_percentages = feature_counts.div(feature_counts.sum(axis=0).replace(0, np.nan), axis=1) * 100
    feature_percentages = feature_percentages.fillna(0)

    # ensure both columns exist
    for col in ['Femme', 'Homme']:
        if col not in feature_percentages.columns:
            feature_percentages[col] = 0.0
    feature_percentages = feature_percentages[['Femme', 'Homme']]

    # Chi-square test
    chi2, p, _, _ = chi2_contingency(feature_counts) if feature_counts.values.sum() > 0 else (np.nan, np.nan, None, None)

    # Prepare indices
    indices = np.arange(len(feature_percentages))
    width = 0.4

    # Plot bars and keep the returned BarContainers
    bars_femme = axes[i].bar(indices - width/2, feature_percentages['Femme'], width=width,
                             edgecolor='black', label='Women', color='salmon')
    bars_homme = axes[i].bar(indices + width/2, feature_percentages['Homme'], width=width,
                             edgecolor='black', label='Men', color='skyblue')

    # Labels and ticks
    axes[i].set_title(f'{feature} Distribution', fontsize=fontsize_labels)
    axes[i].set_xlabel(feature, fontsize=fontsize_labels)
    axes[i].set_ylabel('Percentage (%)', fontsize=fontsize_labels)
    axes[i].set_xticks(indices)
    axes[i].set_xticklabels(feature_percentages.index, rotation=45, ha='right', fontsize=fontsize_ticks)
    axes[i].tick_params(axis='y', labelsize=fontsize_ticks)

    # Add legend and p-value
    legend = axes[i].legend(fontsize=legend_fontsize)
    legend_box = legend.get_frame()
    legend_box.set_alpha(0.8)
    try:
        legend_pos = legend.get_window_extent(axes[i].figure.canvas.get_renderer())
        inv = axes[i].transAxes.inverted()
        legend_coords = inv.transform([legend_pos.x0, legend_pos.y0])
        axes[i].text(legend_coords[0], legend_coords[1] - 0.08, f'p = {p:.4e}',
                     transform=axes[i].transAxes, ha='left', va='top',
                     fontsize=fontsize_ticks, bbox=dict(facecolor='white', edgecolor='none', alpha=0.6))
    except Exception:
        axes[i].text(0.02, 0.98, f'p = {p}', transform=axes[i].transAxes, ha='left', va='top', fontsize=fontsize_ticks)

    # ---------------------------
    # ANNOTATE PERCENTAGE LABELS
    # ---------------------------
    base_offset = 6
    extra_offset = 18
    close_threshold = 5.0  # percent points threshold

    # Compute maximum bar height to ensure some headroom
    max_height = float(feature_percentages.values.max())
    desired_top = max(max_height + 12, axes[i].get_ylim()[1])
    axes[i].set_ylim(0, desired_top)

    # Iterate paired bars by index so we can decide which label to push
    for j in range(len(indices)):
        bar_f = bars_femme[j]
        bar_m = bars_homme[j]
        hf = bar_f.get_height()
        hm = bar_m.get_height()

        if hf <= 0 and hm <= 0:
            continue

        # Default offsets
        off_f = base_offset
        off_m = base_offset

        # If heights are close, push the TALLER one’s label
        if abs(hf - hm) < close_threshold:
            if hf >= hm:
                off_f += extra_offset  # Femme taller
            else:
                off_m += extra_offset  # Homme taller

        # Annotate Femme
        if hf > 0:
            axes[i].annotate(f'{hf:.1f}%',
                             xy=(bar_f.get_x() + bar_f.get_width() / 2., hf),
                             xytext=(0, off_f), textcoords='offset points',
                             ha='center', va='bottom', fontsize=fontsize_ticks, color='black',
                             bbox=dict(facecolor='white', edgecolor='none', alpha=0.0))

        # Annotate Homme
        if hm > 0:
            axes[i].annotate(f'{hm:.1f}%',
                             xy=(bar_m.get_x() + bar_m.get_width() / 2., hm),
                             xytext=(0, off_m), textcoords='offset points',
                             ha='center', va='bottom', fontsize=fontsize_ticks, color='black',
                             bbox=dict(facecolor='white', edgecolor='none', alpha=0.0))
fig.suptitle('Gender-wise Distribution of Socio-economic Features', fontsize=fontsize_labels + 2, fontweight='bold')
plt.tight_layout()
plt.show()

##############################################################################################################################




