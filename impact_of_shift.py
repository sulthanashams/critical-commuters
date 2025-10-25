# -*- coding: utf-8 -*-

#This code illustrates the impact of shifting the 'shift class' commuters on the peak hour slot. Various cases are considered as described in the paper'


"""
Created on Thu Jul 11 13:57:10 2024

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
file_path = 'enter file path here'

# Read the CSV file into a DataFrame
data2 = pd.read_csv(file_path, quotechar='"', delimiter=';', na_values=[
                    "", "\"\""], encoding="UTF-8", engine='python')

data2.rename(columns=lambda x: x.split('.')[0] if '.' in x else x, inplace=True)

# Preprocess
data2 = preprocess_data(data2)

# Display the first few rows of the DataFrame to check
print(data2.head())


#######################################################shift by AT

# # Count the total arrivals per interval
arrival_counts = data2['Arrival Time'].value_counts()

# Reorder the arrival intervals to ensure "Before 08:00" is first
ordered_intervals = ['Before 08:00', '08:00 - 08:29', '08:30 - 08:59', '09:00 - 09:29', 'After 09:30']
arrival_counts = arrival_counts[ordered_intervals]

# Count the arrivals per interval and per shift
arrival_shift_counts = data2.groupby(['Arrival Time', 'Shift']).size().unstack(fill_value=0).reindex(ordered_intervals)

# Determine the top 3 shifts for each arrival interval
top_shifts = arrival_shift_counts.apply(lambda x: x.nlargest(3).index.tolist(), axis=1)

# Define a consistent color mapping for each shift type
shift_colors = {
    'Shift Everyday': sns.color_palette("husl", 3)[0],
    'Shift but Not Everyday': sns.color_palette("husl", 3)[1],
    'Never': sns.color_palette("husl", 3)[2]
}

# Plotting
fig, ax = plt.subplots(figsize=(14, 8))

# Plot total arrival counts
bars = sns.barplot(x=arrival_counts.index, y=arrival_counts.values, color='gray', ax=ax, label='Total')

# Plot the top 3 shift counts within each interval
bottom = [0] * len(arrival_counts)

for i, (interval, shifts) in enumerate(top_shifts.items()):
    for j, shift in enumerate(shifts):
        count = arrival_shift_counts.at[interval, shift]
        percentage = (count / arrival_counts[interval]) * 100
        ax.bar(interval, count, bottom=bottom[i], color=shift_colors[shift], edgecolor='white', width=0.6)
        ax.text(interval, bottom[i] + count / 2, f'{count}\n({percentage:.1f}%)', ha='center', va='center', color='black', fontsize=10)
        bottom[i] += count

# Add legend for shifts
handles = [plt.Rectangle((0, 0), 1, 1, color=shift_colors[shift]) for shift in shift_colors]
labels = list(shift_colors.keys())
ax.legend(handles, labels, title='Shift Type')

# Customizing the plot
ax.set_title('Arrival Time Distribution with Top 3 Shift Labels')
ax.set_xlabel('Arrival Time')
ax.set_ylabel('Count')
ax.set_xticklabels(arrival_counts.index, rotation=45, ha='right')

plt.show()



##################################################################################
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
#equal_shift_indices =  data2[(data2['Delay Time By'] != 0) & (data2['Advance Time By'] != 0)].index
equal_shift_indices = data2[data2['Delay Time By'] == data2['Advance Time By']].index

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

########################################### plot 

# # Calculate the percentage distribution of 'Shift Value Max' values
# shift_value_counts = data2['Shift Value Max'].value_counts()
# shift_value_percentages = (shift_value_counts / shift_value_counts.sum()) * 100

# # Plot the results
# plt.figure(figsize=(10, 6))
# shift_value_percentages.sort_index().plot(kind='bar', color='skyblue')
# plt.xlabel('Shift Value Max')
# plt.ylabel('Percentage')
# plt.title('Percentage Distribution of Shift Value Max')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()


# ######################################################"    
# # Filter the data to only include 'Shift Everyday' or 'Shift but Not Everyday'
# filtered_data = data2[data2['Shift'].isin(['Shift Everyday', 'Shift but Not Everyday'])]

# # Count the total arrivals per interval for the filtered data
# arrival_counts = filtered_data['Arrival Time'].value_counts()

# # Reorder the arrival intervals to ensure "Before 08:00" is first
# ordered_intervals = ['Before 08:00', '08:00 - 08:29', '08:30 - 08:59', '09:00 - 09:29', 'After 09:30']
# arrival_counts = arrival_counts.reindex(ordered_intervals, fill_value=0)

# # Count the arrivals per interval and per 'Shift Value Max'
# arrival_shift_value_counts = filtered_data.groupby(['Arrival Time', 'Shift Value Max']).size().unstack(fill_value=0).reindex(ordered_intervals, fill_value=0)

# # Determine the top 3 'Shift Value Max' for each arrival interval
# top_shift_values = arrival_shift_value_counts.apply(lambda x: x.nlargest(3).index.tolist(), axis=1)

# # Plotting
# fig, ax = plt.subplots(figsize=(14, 8))

# # Plot total arrival counts
# bars = sns.barplot(x=arrival_counts.index, y=arrival_counts.values, color='gray', ax=ax, label='Total')

# # Plot the top 3 shift value counts within each interval
# bottom = [0] * len(arrival_counts)
# colors = sns.color_palette("husl", 3)

# for i, (interval, shift_values) in enumerate(top_shift_values.items()):
#     for j, shift_value in enumerate(shift_values):
#         count = arrival_shift_value_counts.at[interval, shift_value]
#         percentage = (count / arrival_counts[interval]) * 100 if arrival_counts[interval] > 0 else 0
#         ax.bar(interval, count, bottom=bottom[i], color=colors[j], edgecolor='white', width=0.6, label=shift_value if interval == ordered_intervals[0] else "")
#         ax.text(interval, bottom[i] + count / 2, f'{shift_value}\n{count} ({percentage:.1f}%)', ha='center', va='center', color='black', fontsize=10)
#         bottom[i] += count

# # Add total counts at the top of each bar
# for bar in bars.patches:
#     ax.text(
#         bar.get_x() + bar.get_width() / 2,
#         bar.get_height(),
#         int(bar.get_height()),
#         ha='center',
#         va='bottom',
#         color='black',
#         fontsize=12
#     )

# # Customizing the plot
# ax.set_title('Arrival Time Distribution with Top 3 Shift Value Max Labels')
# ax.set_xlabel('Arrival Time')
# ax.set_ylabel('Count')
# handles, labels = ax.get_legend_handles_labels()
# by_label = dict(zip(labels, handles))
# ax.legend(by_label.values(), by_label.keys(), title='Shift Value Max')
# ax.set_xticklabels(arrival_counts.index, rotation=45, ha='right')

# plt.show()
################################################################################"


df=data2#[data2['Shift'] == 'Shift Everyday']


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
df[['Min_Shift', 'Max_Shift', 'Shift_Direction']] = df['Shift Value Max'].apply(extract_shift_info).apply(pd.Series)

def convert_to_minutes(time_str):
    if time_str is None:
        return None

    # Extract hour and minute
    hour, minute = map(int, time_str.split(':'))

    # Convert to minutes
    minutes_after_midnight = hour * 60 + minute

    return minutes_after_midnight

def minutes_after_midnight_to_hh_mm(minutes):
    hours = minutes // 60
    minutes = minutes % 60
    return f"{hours:02d}:{minutes:02d}"

def is_within_interval(time, interval):
    # Split the interval into start and end times
    start_time, end_time = interval.split(' - ')
    
    # Convert start and end times to datetime objects
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)
    
    # Convert the given time to a datetime object
    time = pd.to_datetime(time)
    
    # Check if the given time falls within the interval
    return start_time <= time <= end_time


# Define the shift application function
def apply_shift(arrival_time, min_shift, max_shift, direction, shift_type, random_days, current_day, shift_case):
    
    if pd.isnull(arrival_time) or pd.isnull(min_shift):
        return arrival_time
    
    if shift_type == 'Never':
        return arrival_time
    
    peak_intervals = ['08:30 - 08:59', '09:00 - 09:29']
    
    if any(is_within_interval(arrival_time, interval) for interval in peak_intervals):
        shift_minutes = 0
        
        if shift_type == 'Shift Everyday':
          #if current_day in random_days:
            if shift_case == 'random':
                shift_minutes = np.random.randint(min_shift, max_shift + 1)
            elif shift_case == 'max':
                shift_minutes = int(max_shift)
            elif shift_case == 'mean':
                shift_minutes = int(np.mean([min_shift, max_shift]))
                
        if (shift_type == 'Shift but Not Everyday'):
          if current_day in random_days:
            if shift_case == 'random':
                shift_minutes = np.random.randint(min_shift, max_shift + 1)
            elif shift_case == 'max':
                shift_minutes = int(max_shift)
            elif shift_case == 'mean':
                shift_minutes = int(np.mean([min_shift, max_shift]))            

        if direction == 'Adv':
            shift_minutes = -shift_minutes

        #print(shift_type, shift_minutes)
        arrival_time_minutes = convert_to_minutes(arrival_time)
        adjusted_arrival_time_minutes = arrival_time_minutes + shift_minutes
        adjusted_arrival_time_hh_mm = minutes_after_midnight_to_hh_mm(adjusted_arrival_time_minutes)
        return adjusted_arrival_time_hh_mm
    else:
        return arrival_time


# Number of random days to pick, adjust as needed
n = 2

# Pre-select random days for each row where Shift is 'Shift but Not Everyday'
def select_random_days(row):
    if row['Shift'] == 'Shift but Not Everyday' or row['Shift'] == 'Shift Everyday':
        return random.sample(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'], n)
    return []

df['Random_Days'] = df.apply(select_random_days, axis=1)

# Filter the DataFrame to include only those who can shift
can_shift_df = df[df['Shift'].isin(['Shift Everyday', 'Shift but Not Everyday'])]#isin(['Never'])]
#can_shift_df = df[df['Shift'].isin(['Never'])]

# # Randomly select 50% of the can_shift_df, Percentage of shift class participation (100, 75, 50)
selected_indices = can_shift_df.sample(frac=1, random_state=42).index



# # Calculate the relative size
subpopulation_size = len(selected_indices)
total_can_shift_size = len(can_shift_df)#

# Calculate the proportion relative to the whole population that can shift
proportion = subpopulation_size / total_can_shift_size
percentage = proportion * 100

print(f"Subpopulation size: {subpopulation_size}")
print(f"Percentage relative to whole population that can shift: {percentage:.2f}%")

# Assign np.nan to Max_Shift and Min_Shift for those who were not selected
not_selected_indices = can_shift_df.index.difference(selected_indices)
df.loc[not_selected_indices, ['Min_Shift', 'Max_Shift']] = None

# Apply the shift transformation to the selected rows for each case
shift_cases = ['original', 'random', 'max', 'mean']
#shift_cases = ['original', 'mean']
for shift_case in shift_cases:
    for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
        if shift_case == 'original':
            df[f'Adjusted_{day}_{shift_case}'] = df[f'{day}']  # Copy original times
        else:
            df[f'Adjusted_{day}_{shift_case}'] = df[f'{day}']  # Initialize with original times
            df.loc[selected_indices, f'Adjusted_{day}_{shift_case}'] = df.loc[selected_indices].apply(
                lambda row, current_day=day: apply_shift(row[day], row['Min_Shift'], row['Max_Shift'], row['Shift_Direction'], row['Shift'], row['Random_Days'], current_day, shift_case),
                axis=1
            )

def count_intervals(df, adjusted, case):
    # Initialize the dictionary to store average counts for each interval
    arrival_intervals = {
        f"Adjusted Before 08:00_{case}": [],
        f"Adjusted 08:00 - 08:29_{case}": [],
        f"Adjusted 08:30 - 08:59_{case}": [],
        f"Adjusted 09:00 - 09:29_{case}": [],
        f"Adjusted After 09:30_{case}": []
    }

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Initialize counts for each interval for the current user
        interval_counts = {
            f"Adjusted Before 08:00_{case}": 0,
            f"Adjusted 08:00 - 08:29_{case}": 0,
            f"Adjusted 08:30 - 08:59_{case}": 0,
            f"Adjusted 09:00 - 09:29_{case}": 0,
            f"Adjusted After 09:30_{case}": 0
        }

        # Adjust column names if necessary
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        if adjusted:
            days = [f'Adjusted_{day}_{case}' for day in days]

        # Iterate over each day of the week
        for day in days:
            arrival_time = row[day]

            # Skip if arrival time is missing
            if pd.isnull(arrival_time):
                continue

            # Update the interval counts based on arrival time
            if arrival_time < '08:00':
                interval_counts[f"Adjusted Before 08:00_{case}"] += 1
            elif '08:00' <= arrival_time < '08:30':
                interval_counts[f"Adjusted 08:00 - 08:29_{case}"] += 1
            elif '08:30' <= arrival_time < '09:00':
                interval_counts[f"Adjusted 08:30 - 08:59_{case}"] += 1
            elif '09:00' <= arrival_time < '09:30':
                interval_counts[f"Adjusted 09:00 - 09:29_{case}"] += 1
            elif arrival_time >= '09:30':
                interval_counts[f"Adjusted After 09:30_{case}"] += 1

        # Compute the total number of days with valid arrival times for the current user
        total_days = 5

        # Compute the average counts in each interval category
        avg_counts = {category: (count / total_days)
                      for category, count in interval_counts.items()}

        # Append the average counts for the current user to the corresponding interval columns
        for category, avg_count in avg_counts.items():
            arrival_intervals[category].append(avg_count)

    # Add the arrival interval columns to the DataFrame
    for category, avg_counts in arrival_intervals.items():
        df[category] = avg_counts

    return df

# Count intervals for the original case
df = count_intervals(df.copy(), False, 'original')

# Count intervals for each shift case
for shift_case in shift_cases:
    if shift_case != 'original':
        df = count_intervals(df.copy(), True, shift_case)

# Interval categories
interval_categories = [
    "Adjusted Before 08:00", "Adjusted 08:00 - 08:29",
    "Adjusted 08:30 - 08:59", "Adjusted 09:00 - 09:29",
    "Adjusted After 09:30"
]

# Calculate the total number of arrivals in each interval category for original case
total_arrivals_per_interval_original = df[[
    "Adjusted Before 08:00_original", "Adjusted 08:00 - 08:29_original",
    "Adjusted 08:30 - 08:59_original", "Adjusted 09:00 - 09:29_original", 
    "Adjusted After 09:30_original"]].sum()

# Calculate the total number of arrivals in each interval category for random case
total_arrivals_per_interval_random = df[[
    "Adjusted Before 08:00_random", "Adjusted 08:00 - 08:29_random",
    "Adjusted 08:30 - 08:59_random", "Adjusted 09:00 - 09:29_random", 
    "Adjusted After 09:30_random"]].sum()

# Calculate the total number of arrivals in each interval category for max case
total_arrivals_per_interval_max = df[[
    "Adjusted Before 08:00_max", "Adjusted 08:00 - 08:29_max",
    "Adjusted 08:30 - 08:59_max", "Adjusted 09:00 - 09:29_max", 
    "Adjusted After 09:30_max"]].sum()

# Calculate the total number of arrivals in each interval category for mean case
total_arrivals_per_interval_mean = df[[
    "Adjusted Before 08:00_mean", "Adjusted 08:00 - 08:29_mean",
    "Adjusted 08:30 - 08:59_mean", "Adjusted 09:00 - 09:29_mean", 
    "Adjusted After 09:30_mean"]].sum()

# Calculate the mean counts for each interval and each case
mean_counts_original = (total_arrivals_per_interval_original / total_arrivals_per_interval_original.sum()) * 100
mean_counts_random = (total_arrivals_per_interval_random / total_arrivals_per_interval_random.sum()) * 100
mean_counts_max = (total_arrivals_per_interval_max / total_arrivals_per_interval_max.sum()) * 100
mean_counts_mean = (total_arrivals_per_interval_mean / total_arrivals_per_interval_mean.sum()) * 100

# mean_counts_original = (total_arrivals_per_interval_original)# / total_arrivals_per_interval_original.sum()) * 100
# mean_counts_random = (total_arrivals_per_interval_random)# / total_arrivals_per_interval_random.sum()) * 100
# mean_counts_max = (total_arrivals_per_interval_max)# / total_arrivals_per_interval_max.sum()) * 100
# mean_counts_mean = (total_arrivals_per_interval_mean )#/ total_arrivals_per_interval_mean.sum()) * 100

# Display the mean counts for verification
print("Mean Counts - Original:")
print(mean_counts_original)
print("\nMean Counts - Random:")
print(mean_counts_random)
print("\nMean Counts - Max:")
print(mean_counts_max)
print("\nMean Counts - Mean:")
print(mean_counts_mean)

# Verify the sum of percentages for each case
print("\nSum of Percentages - Original:", mean_counts_original.sum())
print("Sum of Percentages - Random:", mean_counts_random.sum())
print("Sum of Percentages - Max:", mean_counts_max.sum())
print("Sum of Percentages - Mean:", mean_counts_mean.sum())

# Plotting the results
# Interval categories
interval_categories = [
    "Before 08:00", "08:00 - 08:29",
    "08:30 - 08:59", "09:00 - 09:29",
    "After 09:30"
]

# Calculate total arrivals for each interval category for original case
total_arrivals_original = df[[
    "Adjusted Before 08:00_original", "Adjusted 08:00 - 08:29_original",
    "Adjusted 08:30 - 08:59_original", "Adjusted 09:00 - 09:29_original", 
    "Adjusted After 09:30_original"]].sum()

# Calculate total arrivals for each interval category for random shift case
total_arrivals_random = df[[
    "Adjusted Before 08:00_random", "Adjusted 08:00 - 08:29_random",
    "Adjusted 08:30 - 08:59_random", "Adjusted 09:00 - 09:29_random", 
    "Adjusted After 09:30_random"]].sum()

# Calculate total arrivals for each interval category for max shift case
total_arrivals_max = df[[
    "Adjusted Before 08:00_max", "Adjusted 08:00 - 08:29_max",
    "Adjusted 08:30 - 08:59_max", "Adjusted 09:00 - 09:29_max", 
    "Adjusted After 09:30_max"]].sum()

# Calculate total arrivals for each interval category for mean shift case
total_arrivals_mean = df[[
    "Adjusted Before 08:00_mean", "Adjusted 08:00 - 08:29_mean",
    "Adjusted 08:30 - 08:59_mean", "Adjusted 09:00 - 09:29_mean", 
    "Adjusted After 09:30_mean"]].sum()

# Calculate percentage reduction for each interval category
percentage_reduction_random = (( total_arrivals_random.values - total_arrivals_original.values) / total_arrivals_original.values) * 100
percentage_reduction_max = ((total_arrivals_max.values - total_arrivals_original.values) / total_arrivals_original.values) * 100
percentage_reduction_mean = ((total_arrivals_mean.values-total_arrivals_original.values) / total_arrivals_original.values) * 100

####Plotting the results
#colors = sns.color_palette("husl", 4)
colors = sns.color_palette("Set2", 4)
bar_width = 0.2
index = np.arange(len(interval_categories))

fig, ax = plt.subplots(figsize=(12, 8))

bar_original = ax.bar(index - 1.5 * bar_width, total_arrivals_original, bar_width, label='Original', color=colors[0])
bar_random = ax.bar(index - 0.5 * bar_width, total_arrivals_random, bar_width, label='Random Shift', color=colors[1])
bar_max = ax.bar(index + 1.5 * bar_width, total_arrivals_max, bar_width, label='Max Shift', color=colors[2])
bar_mean = ax.bar(index + 0.5 * bar_width, total_arrivals_mean, bar_width, label='Mean Shift', color=colors[3])

# Annotate bars with percentage reduction for each interval category
for i, category in enumerate(interval_categories):
    reduction_random = percentage_reduction_random[i]
    reduction_max = percentage_reduction_max[i]
    reduction_mean = percentage_reduction_mean[i]

    # Annotate bars with percentage reductions lengthwise along the bars
    ax.text(index[i] - 0.5 * bar_width, total_arrivals_random[i] * 0.5, f'{reduction_random:.1f}%', 
             ha='center', va='center', rotation=90, fontsize=18, color='black')
    
    ax.text(index[i] + 1.5 * bar_width, total_arrivals_max[i] * 0.5, f'{reduction_max:.1f}%', 
             ha='center', va='center', rotation=90, fontsize=18, color='black')
    
    ax.text(index[i] + 0.5 * bar_width, total_arrivals_mean[i] * 0.5, f'{reduction_mean:.1f}%', 
            ha='center', va='center', rotation=90, fontsize=18, color='black')

ax.set_xlabel('Time Intervals', fontsize=25)
ax.set_ylabel('Number of Arrivals', fontsize=25)
ax.set_xticks(index)
ax.set_xticklabels(interval_categories, fontsize=25, rotation=30)
ax.tick_params(axis='y', labelsize=25)  # Increase y-ticks font size
ax.legend(fontsize=25)

plt.tight_layout()
plt.show()

