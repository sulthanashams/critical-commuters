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
from sklearn.preprocessing import OneHotEncoder
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
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight
from imblearn.pipeline import Pipeline
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import learning_curve
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras import layers, regularizers, Model, Input
import tensorflow as tf
from sklearn.datasets import make_classification


# Specify the file path
file_path = 'enter dataset file path'

# Read the CSV file into a DataFrame
data2 = pd.read_csv(file_path, quotechar='"', delimiter=';', na_values=[
                    "", "\"\""], encoding="UTF-8", engine='python')

data2.rename(columns=lambda x: x.split('.')[0] if '.' in x else x, inplace=True)

# Display the first few rows of the DataFrame to check
print(data2.head())

###############################  PREPROCESSING THE DATA   ###############################################

# Mapping of columns to their label names
column_to_label = {
    'c15[SQ001]': 'Yes, several times a week',
    'c15[SQ002]': 'Yes, once a week',
    'c15[SQ005]': 'Yes, 1 to 3 times a month',
    'c15[SQ006]': 'No, never'
}

# Function to map the binary columns to their labels
def map_to_label(row):
    for column, label in column_to_label.items():
        if row[column] == 1:
            return label
    return None

# Create a new column 'long_champs_interchange' using the mapping function
data2['long_champs_interchange'] = data2.apply(map_to_label, axis=1)


############################# own cars ##############

# Map the codes to labels
mapping = {
    'SQ001': 'No None',
    'SQ002': 'Yes, 1 vehicle',
    'SQ003': 'Yes, 2 vehicles',
    'SQ004': 'Yes, 3 vehicles and more'
}

# Apply the mapping
data2['e34'] = data2['e34'].map(mapping)

# Rename the column
data2.rename(columns={'e34': 'own cars'}, inplace=True)


############################## response to traffic measures

# Mapping of responses to labels
response_labels = {
    'A1': 'No opinion',
    'A2': 'No, not at all',
    'A3': 'No, rather not',
    'A4': 'Yes, rather',
    'A5': 'Yes, absolutely'
}

# Apply the mapping to the relevant columns
for col in data2.columns:
    if col.startswith('d21'):
        data2[col] = data2[col].map(response_labels)

column_descriptions = {
    'd21[SQ001]': 'Improve traffic information',
    'd21[SQ002]': 'Set up a bonus system',
    'd21[SQ003]': 'Limit meetings before 10 a.m.',
    'd21[SQ004]': 'Adjust business schedules',
    'd21[SQ005]': 'Arrange school/daycare schedules',
    'd21[SQ006]': 'Encourage the practice of teleworking',
    'd21[SQ007]': 'Authorize the practice of teleworking during peak hours',
    'd21[SQ008]': 'Implement a communication campaign',
    'd21[SQ009]': 'Encourage carpooling',
    'd21[SQ010]': 'Define new rules for access to parking spaces',
    'd21[SQ011]': 'Encourage the use of other modes of travel than the personal car'
}


data2.rename(columns=column_descriptions, inplace=True)

############################################################## No Of Kids 

# First, fill NaN values with 0 (indicating zero kids)
data2['b10'] = data2['b10'].fillna(0)

# Next, map the values according to the specified criteria
def map_kid_number(value):
    if value == 'A1':
        return '1'  # '1 and more'#1
    elif value == 'A2':
        return '2'  # '2 and more'
    elif value == 'A3':
        return '3 and more'
    else:
        return int(value)  # For values above A3


data2['Number of Kid to Drop-off'] = data2['b10'].map(map_kid_number)

# Drop the original column 'b10'
data2.drop(columns=['b10'], inplace=True)

########################################################## Commute Time
# Rename the column label
data2.rename(columns={'a5': 'Long Commute Time'}, inplace=True)


bins = [0, 20, 30, 40, float('inf')]
labels = ['0-20', '20-30','30-40', 'more than 40']


# bin the data and create a new column with the labels
data2['Commute Time'] = pd.cut(data2['Long Commute Time'], bins=bins, labels=labels, right=False)

# Show the distribution of 'Commute Bin' to Check
commute_bin_counts = data2['Commute Time'].value_counts(sort=False)

###### Additional checks
# # Create a binary column indicating whether the commute time is 40 minutes or longer
# data2['Long Commute Time'] = data2['Commute Time'] >= 40
# # Convert boolean values to integers (1 for True, 0 for False)
# data2['Long Commute Time'] = data2['Long Commute Time'].astype(int)

############################################################################## Postal Area #######################################

rennes_postal_codes = {35000, 35200, 35700, 35510, 35238, 35051}
# Define a mapping of column names from R to Python-friendly names
column_mapping_e28 = {
    'e28': 'Residence'
}

# Use the mapping to rename columns in your DataFrame
data2.rename(columns=column_mapping_e28, inplace=True)

# Convert Residence column based on rennes_postal_codes set
data2['Residence'] = data2['Residence'].apply(lambda x: 1 if x in rennes_postal_codes else 0)


data2['b10c'] = pd.to_numeric(data2['b10c'], errors='coerce').astype('Int64')




############################################### Do you Experience Traffic  #############################################


data2['Traffic Experience'] = data2['c17'].replace({1: 'Yes', 2: 'No'})  #Map yes to 1 and no to 0

################################################# Willing to Change Travel Habits ######################################

# Combine columns into one
def combine_traffic_experience(row):
    if row['c18[SQ001]'] == 1:
        return 'Regularly'
    elif row['c18[SQ002]'] == 1:
        return 'Occasionally'
    elif row['c18[SQ003]'] == 1:
        return 'Never'
    elif pd.isna(row['c18[SQ001]']) and pd.isna(row['c18[SQ002]']) and pd.isna(row['c18[SQ003]']):
        return 'No Traffic Experienced'
    else:
        return np.nan


data2['Travel Habits Change'] = data2.apply(combine_traffic_experience, axis=1)

#################################################### If Yes to habit, what Measures to Avoid Traffic ################################

# Mapping dictionary for column renaming
column_mapping_habits = {
    'c18a[SQ001]': 'Change my mode of transport',
    'c18a[SQ002]': 'Change my route to avoid traffic jams',
    'c18a[SQ003]': 'Leave my home early',
    'c18a[SQ004]': 'Leave my home later',
    'c18a[SQ005]': 'Cancel my trip (teleworking for example)',
    'c18a[SQ006]': 'Take a break in my journey (in a café for example)',
    'c18a[other]': 'Other_a'
}

# Rename columns using the mapping dictionary
data2 = data2.rename(columns=column_mapping_habits)

#################################################################### If no habit change then, Why not take measures #################################
# Mapping dictionary for column renaming
column_mapping_habits_no = {
    'c18c[SQ001]': 'I have no other mode of transport available',
    'c18c[SQ002]': 'I have no other possible route',
    'c18c[SQ003]': 'It is impossible for me to shift my journey in time',
    'c18c[SQ004]': 'I do not believe that changing my travel habits would improve my situation',
    'c18c[SQ005]': 'I do not want to change my travel habits, despite the traffic jams I experience',
    'c18c[SQ006]': 'Due to the time I want to leave work in the evening',
    'c18c[other]': 'Other_c'
}

# Rename columns using the mapping dictionary
data2 = data2.rename(columns=column_mapping_habits_no)


columns_of_habit_change = [
    'Change my mode of transport',
    'Change my route to avoid traffic jams',
    'Leave my home early',
    'Leave my home later',
    'Cancel my trip (teleworking for example)',
    'Take a break in my journey (in a café for example)',
    'Other_a',
    'I have no other mode of transport available',
    'I have no other possible route',
    'It is impossible for me to shift my journey in time',
    'I do not believe that changing my travel habits would improve my situation',
    'I do not want to change my travel habits, despite the traffic jams I experience',
    'Due to the time I want to leave work in the evening',
    'Other_c'
]

# Replace NaN values with -1 in specified columns based on 'traffic experience' column
data2.loc[data2['Traffic Experience'] == 'No',
          columns_of_habit_change] = data2.loc[data2['Traffic Experience'] == 'No', columns_of_habit_change].fillna(-1)

data2[columns_of_habit_change] = data2[columns_of_habit_change].fillna(0)


# Replace string values with 1 in 'Other_a' and 'Other_c' columns
data2[['Other_a', 'Other_c']] = data2[['Other_a', 'Other_c']].applymap(
    lambda x: 1 if isinstance(x, str) else x)

########################################### Congestion Discomfort Ratings################################

# Create a list of column names for congestion discomfort ratings
rating_columns = ['c17d[SQ001]', 'c17d[SQ002]',
                  'c17d[SQ003]', 'c17d[SQ004]', 'c17d[SQ005]']

# Create a dictionary to map column names to their corresponding ratings
rating_mapping = {
    'c17d[SQ001]': '1',
    'c17d[SQ002]': '2',
    'c17d[SQ003]': '3',
    'c17d[SQ004]': '4',
    'c17d[SQ005]': '5'
}

# List of rating columns
rating_columns = list(rating_mapping.keys())

# Create a new column that avoids IndexError by handling empty lists
data2['Congestion Discomfort Rating'] = data2.apply(
    lambda row: [rating_mapping[col]
                 for col in rating_columns if pd.notna(row[col]) and row[col]],
    axis=1
)

# Provide a default if the list is empty
data2['Congestion Discomfort Rating'] = data2['Congestion Discomfort Rating'].apply(
    # Return np.nan or a default if the list is empty
    lambda x: x[0] if len(x) > 0 else np.nan
)

# Fill NaN values with 1 in the "Congestion Discomfort Rating" column
data2['Congestion Discomfort Rating'] = data2['Congestion Discomfort Rating'].fillna('No Traffic Exp')


# Define a mapping of column names 
column_mapping_c17c = {'c17c': 'Traffic Delay Amount'}

# Use the mapping to rename columns in your DataFrame
data2.rename(columns=column_mapping_c17c, inplace=True)

# Define delay labels
traffic_group_labels = {
    'SQ001': '0 – 15 min',
    'SQ002': '16 – 30 min',
    'SQ003': '31 – 45 min',
    'SQ004': '46 – 60 min',
    'SQ005': 'More than 60 min'
}


data2.replace({'Traffic Delay Amount': traffic_group_labels}, inplace=True)
data2['Traffic Delay Amount'] = data2['Traffic Delay Amount'].fillna(
    'No Traffic Exp')

############################################################################# Modes of Transport #############
# renaming the column names for ease
mode_names = {
    'a1[SQ001]': 'Solo Car',
    'a1[SQ002]': 'Carpooling',
    'a1[SQ003]': 'Motorcycle/Scooter',
    'a1[SQ004]': 'Bike/Electric Bike/Velostar',
    'a1[SQ005]': 'Walking',
    'a1[SQ006]': 'Metro/Bus',
    'a1[SQ007]': 'Car',
    'a1[SQ008]': 'Train',
    'a1[other]': 'Other mode'
}

# Rename columns using the mapping dictionary
data2.rename(columns=mode_names, inplace=True)
# Convert the 'Other' column into a binary column

data2['Other mode'] = data2['Other mode'].notna().astype(int)

mode_names = {
    'Solo Car': 'Solo Car',
    'Carpooling': 'Carpooling',
    'Motorcycle/Scooter': 'Motorcycle/Scooter',
    'Bike/Electric Bike/Velostar': 'Bike/Electric Bike/Velostar',
    'Walking': 'Walking',
    'Active Mode': 'Active Mode',
    'Metro/Bus': 'Metro/Bus',
    'Car': 'Car',
    'Train': 'Train',
    'Other mode': 'Other mode'
}

mode_columns = list(mode_names.keys())
# If either column has a value of 1, 'Active Mode' should be 1; otherwise, 0
data2['Active Mode'] = data2[['Bike/Electric Bike/Velostar', 'Walking']].max(axis=1)
# Apply a lambda function to create the 'Mode of Transport' column based on which mode is True
data2['Mode of Transport'] = data2.apply(
    lambda row: [mode_names[col] for col in mode_columns if row[col]][0], axis=1)




# Define the replacements
replacement_map = {
    'Bike/Electric Bike/Velostar': 'Active Mode',
    'Walking': 'Active Mode',


}


# Replace the specified labels with 'Active Mode' in the 'Mode of Transport' column
data2['Mode of Transport'].replace(replacement_map, inplace=True)


# # Remove the original 'Bike/Electric Bike/Velostar' and 'Walking' columns
data2.drop(['Bike/Electric Bike/Velostar', 'Walking'], axis=1, inplace=True)


###################################################################################### Dropping of Frequencies before work###################


# Define the reversed mapping of options in the column to A1, A2, etc.
response_labels = {
    'A1': 'Yes, several times a week',
    'A2': 'Yes, once a week',
    'A3': 'Yes, 1 to 3 times a month',
    'A4': 'Yes, less than once a month',
    'A5': 'Never'
}


# Rename the column label
data2.rename(columns={'a4[SQ002]': 'Partner Drop Off'}, inplace=True)
# Rename the values in the 'Income' column using the mapping
data2['Partner Drop Off'] = data2['Partner Drop Off'].map(response_labels)




# Define the reversed mapping of options in the column to A1, A2, etc.
response_labels = {
    'A1': 'Yes, several times a week',
    'A2': 'Yes, once a week',
    'A3': 'Yes, 1 to 3 times a month',
    'A4': 'Yes, less than once a month',
    'A5': 'Never'
}


# Rename the column label
data2.rename(columns={'a4[SQ001]': 'Child Drop Off Frequency'}, inplace=True)
# Rename the values in the 'Income' column using the mapping
data2['Child Drop Off Frequency'] = data2['Child Drop Off Frequency'].map(response_labels)

######################### Income ########################
# Define the mapping of options SQ to corresponding labels
income_labels = {
    'SQ001': 'Moins de 1200 €',
    'SQ002': '1200 - 1350 €',
    'SQ003': '1350 - 1600 €',
    'SQ004': '1600 - 1950 €',
    'SQ005': '1950 - 2650 €',
    'SQ006': '2650 - 3550 €',
    'SQ007': 'Plus de 3550 €'
}

# Rename the column label
data2.rename(columns={'e33': 'Income'}, inplace=True)

# Rename the values in the 'Income' column using the mapping
data2['Income'] = data2['Income'].map(income_labels)


##################################################################### Job Category ###################
column_mapping_e27 = {
    'e27[SQ001]': 'Executive',
    'e27[SQ002]': 'Manager',
    'e27[SQ003]': 'Intermediate',
    'e27[SQ004]': 'Employee',
    'e27[SQ005]': 'Worker',
    'e27[other]': 'Other-e27'
}

# Use the mapping to rename columns in your DataFrame
data2.rename(columns=column_mapping_e27, inplace=True)
data2['Other-e27'] = data2['Other-e27'].notna().astype(int)

column_mapping_e27 = {
    'Executive': 'Executive',
    'Manager': 'Manager',
    'Intermediate': 'Intermediate',
    'Employee': 'Employee',
    'Worker': 'Worker',
    'Other-e27': 'Other-e27'
}

# Get the list of column names after renaming
job_columns = list(column_mapping_e27.values())

# Create a new combined column indicating the job category
data2['Job Category'] = data2.apply(
    lambda row: [col for col in job_columns if row[col]],
    axis=1
)

# Extract the first non-empty value or return a default if empty
data2['Job Category'] = data2['Job Category'].apply(
    lambda x: x[0] if x else np.nan
)

######################################################################## Contract Type

contract_type_labels = {
    'e37[SQ001]': 'CDI',
    'e37[SQ002]': 'CDD',
    'e37[SQ003]': 'Prestataire',
    'e37[SQ004]': 'Temp/Interim'
}

# Rename columns in the DataFrame based on the mapping
data2.rename(columns=contract_type_labels, inplace=True)

contract_type_labels = {
    'CDI': 'CDI',
    'CDD': 'CDD',
    'Prestataire': 'Prestataire',
    'Temp/Interim': 'Temp/Interim'
}

# Extract the list of contract type columns
contract_type_columns = list(contract_type_labels.values())

# Create a new "Contract Type" column by mapping the `True` value
data2['Contract Type'] = data2.apply(
    lambda row: [col for col in contract_type_columns if row[col]],
    axis=1
)

# Return the first non-empty value or a default if no value is `True`
data2['Contract Type'] = data2['Contract Type'].apply(
    lambda x: x[0] if x else np.nan
)

#data2 = data2[data2['Contract Type'] != 'Prestataire']
#data2 = data2[data2['Contract Type'] != 'Temp/Interim']

# Define a mapping of column names from R to short names
column_mapping_e31 = {
    'e31[SQ001]': 'Homme',
    'e31[SQ002]': 'Femme',
    'e31[SQ003]': 'Sans réponse'
}

############################################################# Gender
# Use the mapping to rename columns in your DataFrame
data2.rename(columns=column_mapping_e31, inplace=True)

column_mapping_e31 = {
    'Homme': 'Homme',
    'Femme': 'Femme',
    'Sans réponse': 'Sans réponse'
}

gender_columns = list(column_mapping_e31.keys())

# Apply a lambda function to create the 'Gender' column based on which column is True
data2['Gender'] = data2.apply(lambda row: [column_mapping_e31[col]
                              for col in gender_columns if row[col]][0], axis=1)



##########################################Age Group
# Define a mapping of column names
column_mapping_e32 = {
    'e32': 'Age Group'
}

# Use the mapping to rename columns in your DataFrame
data2.rename(columns=column_mapping_e32, inplace=True)

# Define age group labels
age_group_labels = {
    'SQ001': '18 – 24 ans',
    'SQ002': '25 – 34 ans',
    'SQ003': '35 – 44 ans',
    'SQ004': '44 – 55 ans',
    'SQ005': 'Plus de 55 ans'
}


data2.replace({'Age Group': age_group_labels}, inplace=True)


########################################################### Arrival Contract Type
# Define the mapping from original to short names
column_mapping = {
    'b7[SQ001]': 'Free Arrival with Imposed Time Range',
    'b7[SQ002]': 'Completely Free Arrival',
    'b7[SQ003]': 'Fixed Arrival, Identical Times',
    'b7[SQ004]': 'Fixed Arrival, Varies by Day',
    'b7[SQ005]': 'Fixed Arrival, Varies by Week',
    'b7[SQ006]': 'Fixed and Free Arrival',
    'b7[other]': 'Other Arrival Rules for Work'
}

# Use the mapping to rename columns in your DataFrame
data2.rename(columns=column_mapping, inplace=True)


column_mapping = {
    'Free Arrival with Imposed Time Range': 'Free Arrival with Imposed Time Range',
    'Completely Free Arrival': 'Completely Free Arrival',
    'Fixed Arrival, Identical Times': 'Fixed Arrival, Identical Times',
    'Fixed Arrival, Varies by Day': 'Fixed Arrival, Varies by Day',
    'Fixed Arrival, Varies by Week': 'Fixed Arrival, Varies by Week',
    'Fixed and Free Arrival': 'Fixed and Free Arrival',
    'Other Arrival Rules for Work': 'Other Arrival Rules for Work'
}

# Get the list of column names from the mapping
arrival_columns = list(column_mapping.keys())

# Create a new combined column by mapping the first `True` value to its description
data2['Theoretical Arrival Time Contract'] = data2.apply(
    lambda row: [column_mapping[col]
                 for col in arrival_columns if row[col]][0],
    axis=1
)

# # # Define the specific arrival types that should be labeled as 'Free'
# free_arrival_types = {'Free Arrival with Imposed Time Range', 'Completely Free Arrival'}

# # Apply the function to the 'Arrival Type' column
# data2['Arrival Type'] = data2['Arrival Type'].apply(map_arrival_type)

########################################Actual Working Timings 
# Define a mapping of column 
column_mapping_b8 = {
    'b8[SQ001]': 'With regular schedules, identical every day (arrival window within +/- 15 minutes)',
    'b8[SQ002]': 'With regular schedules but varying by day of the week (arrival window within +/- 15 minutes)',
    'b8[SQ003]': 'With variable schedules (arrival window greater than +/- 15 minutes)',
    'b8[SQ004]': 'With fixed hours every day',
    'b8[SQ005]': 'With fixed hours but varying by day of the week'
}


# Use the mapping to rename columns in your DataFrame
data2.rename(columns=column_mapping_b8, inplace=True)

column_mapping_b8 = {
    'With regular schedules, identical every day (arrival window within +/- 15 minutes)':
        # With regular schedules, identical every day (arrival window within +/- 15 minutes)',
        'Regular Hours',

    'With regular schedules but varying by day of the week (arrival window within +/- 15 minutes)':
        # 'With regular schedules but varying by day of the week (arrival window within +/- 15 minutes)',
        'Regular Hours',

    'With variable schedules (arrival window greater than +/- 15 minutes)':
        # 'With variable schedules (arrival window greater than +/- 15 minutes)',
        'Variable Hours',

    'With fixed hours every day':
        'Fixed Hours',  # 'With fixed hours every day',

    'With fixed hours but varying by day of the week':
        'Fixed Hours',  # 'With fixed hours but varying by day of the week'
}
# Extract the keys from telework_labels to use as the reference for column names
work_columns = list(column_mapping_b8.keys())

# Create the 'Schedule Type' column by mapping the True value to its description
data2['Practical Arrival Time'] = data2.apply(
    lambda row: [column_mapping_b8[col] for col in work_columns if row[col]][0], axis=1)


#################################### Telework 
# Define the labels for each option
telework_labels = {
    'd22[SQ001]': 'Yes, one or two days every week',
    'd22[SQ002]': 'Yes, a few days per month depending on constraints',
    'd22[SQ003]': 'Yes, less often',
    'd22[SQ004]': 'No, never'
}

# Rename columns to meaningful labels
data2.rename(columns=telework_labels, inplace=True)

telework_labels = {
    'Yes, one or two days every week': 'Yes, one or two days every week',
    'Yes, a few days per month depending on constraints': 'Yes, a few days per month depending on constraints',
    'Yes, less often': 'Yes, less often',
    'No, never': 'No, never'
}

# Extract the keys from telework_labels to use as the reference for column names
telework_columns = list(telework_labels.keys())


# Apply a lambda function to create the 'Telework Frequency' column based on which column is True
data2['Telework Frequency'] = data2.apply(
    lambda row: [telework_labels[col] for col in telework_columns if row[col]][0], axis=1)

telework_frequency_mapping = {
    'Yes, one or two days every week': 'Frequent',
    'Yes, a few days per month depending on constraints': 'Rarely',
    'Yes, less often': 'Rarely',
    'No, never': 'Never'
}

# Apply the relabelling using the map function
data2['Telework Frequency'] = data2['Telework Frequency'].map(
    telework_frequency_mapping)


############################################################################################################ Arrival Time Distribution ####################################
# Define a mapping of column names from French to English days of the week with the specified format
day_mapping = {
    'b9[SQ001_SQ001]': 'Monday',
    'b9[SQ001_SQ002]': 'Tuesday',
    'b9[SQ001_SQ003]': 'Wednesday',
    'b9[SQ001_SQ004]': 'Thursday',
    'b9[SQ001_SQ005]': 'Friday',
    'b9[SQ001_SQ006]': 'Saturday',
    'b9[SQ001_SQ007]': 'Sunday',
    'b9b': 'fixed'
}
# Use the mapping to rename columns in your DataFrame
data2.rename(columns=day_mapping, inplace=True)

# Iterate over each row in the DataFrame
for index, row in data2.iterrows():
    # Iterate over each day of the week
    for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
        # Check if the value in the current day column is empty
        if pd.isna(row[day]):
            # Replace the empty value with the value from the 'Fixed' column
            data2.at[index, day] = row['fixed']

# Define a function to convert numbers to 'HH:MM' format


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


def convert_min_to_hh_mm(time):
    if pd.isnull(time):
        return '00:00'  # Placeholder value for NaN
    else:
        hours = int(time)
        minutes = int((time - int(hours)) * 60)
        return f"{hours:02d}:{minutes:02d}"


# Iterate over each row in the DataFrame to convert them all to same uniform time format
for index, row in data2.iterrows():
    # Iterate over each day of the week
    for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
        # Get the time value from the current day column
        time_value = row[day]
        # Check if the time value is not NaN
        if not pd.isna(time_value) and row[day] != "teleW":
            # Convert the time value to HH:MM format
            data2.at[index, day] = convert_to_hh_mm(time_value)

        else:
            data2.at[index, day] = np.nan


# Define the arrival time intervals
arrival_intervals = {
    "Before 08:00": [],
    "08:00 - 08:29": [],
    "08:30 - 08:59": [],
    "09:00 - 09:29": [],
    "After 09:30": []
}

# Iterate over each row in the DataFrame
for index, row in data2.iterrows():
    # Initialize counts for each interval for the current user
    interval_counts = {
        "Before 08:00": 0,
        "08:00 - 08:29": 0,
        "08:30 - 08:59": 0,
        "09:00 - 09:29": 0,
        "After 09:30": 0
    }

    # Iterate over each day of the week
    for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
        arrival_time = row[day]

        # Skip if arrival time is missing
        if pd.isnull(arrival_time):
            continue

        # Update the interval counts based on arrival time
        if arrival_time < '08:00':
            interval_counts["Before 08:00"] += 1
        elif '08:00' <= arrival_time < '08:30':
            interval_counts["08:00 - 08:29"] += 1
        elif '08:30' <= arrival_time < '09:00':
            interval_counts["08:30 - 08:59"] += 1
        elif '09:00' <= arrival_time < '09:30':
            interval_counts["09:00 - 09:29"] += 1
        elif arrival_time >= '09:30':
            interval_counts["After 09:30"] += 1

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
    data2[category] = avg_counts


# Step 1: Calculate the total number of arrivals in each interval category
total_arrivals_per_interval = data2[[
    "Before 08:00", "08:00 - 08:29", "08:30 - 08:59", "09:00 - 09:29", "After 09:30"]].sum()

# Step 2: Compute the percentage of arrivals per interval category
percentage_arrivals_per_interval = (
    total_arrivals_per_interval / total_arrivals_per_interval.sum()) * 100

# Step 3: Plot the percentage of arrivals in a bar plot
plt.figure(figsize=(10, 6))
plt.bar(percentage_arrivals_per_interval.index,
        percentage_arrivals_per_interval, color='skyblue')

# Step 4: Add the values (percentages) inside each bar
for i, percentage in enumerate(percentage_arrivals_per_interval):
    plt.text(i, percentage, f'{percentage:.2f}%', ha='center', va='bottom')

# Add labels and title
plt.xlabel('Arrival Time Intervals')
plt.ylabel('Percentage of Arrivals')
plt.title('Percentage of Arrivals per Time Interval')
plt.ylim(0, 100)

# Show plot
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

###############################################  age AT





# List of columns to combine
arrival_columns = ['Before 08:00', '08:00 - 08:29',
                   '08:30 - 08:59', '09:00 - 09:29', 'After 09:30']

# Function to get the first matching column name or return 'Unknown' if none match


def get_arrival_time(row):
    # Find all columns where the value is 1
    matching_columns = [col for col in arrival_columns if row[col] > 0.5]
    # Return the first matching column, or 'Unknown' if there are no matches
    return matching_columns[0] if matching_columns else np.nan


# Create a new column 'Arrival Time' with the name of the column that has the maximum value
data2['Arrival Time'] = data2[arrival_columns].idxmax(axis=1)


data2.rename(columns={
    'b13[SQ001]': 'Advance Arrival Everyday',
    'b13[SQ002]': 'Advance Arrival Not Everyday',
    'b13[SQ003]': 'Never Advance Arrival'
}, inplace=True)

# Mapping of renamed column names to their descriptions for advance arrival pattern
adv_arrival_labels = {
    'Advance Arrival Everyday': 'Advance Arrival Everyday',
    'Advance Arrival Not Everyday': 'Advance Arrival Not Everyday',
    'Never Advance Arrival': 'Never Advance Arrival'}

# Extract the keys to use as the reference for column names
adv_arrival_columns = list(adv_arrival_labels.keys())


# Default value for when no column has a value of 1
default_value = "Never Advance Arrival"

# Create a new 'Delay Arrival' column based on which column has a value of 1
data2['Advance Arrival'] = data2.apply(
    lambda row: [adv_arrival_labels[col]
                 for col in adv_arrival_columns if pd.notna(row[col]) and row[col] == 1],
    axis=1
)

# Return the first matching label or a default value if the list is empty
data2['Advance Arrival'] = data2['Advance Arrival'].apply(
    # If the list is empty, return default
    lambda x: x[0] if x else default_value
)

data2.rename(columns={
    'b12[SQ001]': 'Delay Arrival Everyday',
    'b12[SQ002]': 'Delay Arrival Not Everyday',
    'b12[SQ003]': 'Never Delay Arrival'
}, inplace=True)


# Mapping of the renamed column names to their corresponding labels for delay arrival pattern
delay_arrival_labels = {
    'Delay Arrival Everyday': 'Delay Arrival Everyday',
    'Delay Arrival Not Everyday': 'Delay Arrival Not Everyday',
    'Never Delay Arrival': 'Never Delay Arrival'
}


del_arrival_columns = list(delay_arrival_labels.keys())

# # Create a new 'Delay Arrival' column based on the True column for each row
# data2['Delay Arrival'] = data2.apply(lambda row: [delay_arrival_labels[col] for col in del_arrival_columns if row[col]][0], axis=1)

# Default value for when no column has a value of 1
default_value = "Never Delay Arrival"

# Create a new 'Delay Arrival' column based on which column has a value of 1
data2['Delay Arrival'] = data2.apply(
    lambda row: [delay_arrival_labels[col]
                 for col in del_arrival_columns if pd.notna(row[col]) and row[col] == 1],
    axis=1
)

# Return the first matching label or a default value if the list is empty
data2['Delay Arrival'] = data2['Delay Arrival'].apply(
    # If the list is empty, return default
    lambda x: x[0] if x else default_value
)

#################################################  Shift Type 
def get_shift_value(delay, advance):
    
    # Check for "Not Everyday" in either column
    if 'Never Advance Arrival' in advance and 'Never Delay Arrival' in delay:
        return 'Never' 


    # Check for "Everyday" in either column
    elif 'Advance Arrival Everyday' in advance or 'Delay Arrival Everyday' in delay:
        return 'Shift Everyday'
    
    
    elif 'Advance Arrival Not Everyday' in advance or 'Delay Arrival Not Everyday' in delay:
                return 'Shift but Not Everyday'#np.nan#

    else:
        return 'What'



# Apply the function to create the new "Shift" column
data2['Shift'] = data2.apply(lambda row: get_shift_value(row['Delay Arrival'], row['Advance Arrival']), axis=1)

# Apply the function to the DataFrame
data2['Shift'] = data2.apply(relabel_shift, axis=1)

# # # Calculate value counts and percentages
value_counts = data2['Shift'].value_counts()
percentages = data2['Shift'].value_counts(normalize=True) * 100

# Create a bar plot
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=percentages.index, y=percentages.values, palette='viridis')

# # Add value counts within bars
# for i, count in enumerate(value_counts):
#     plt.text(i, percentages[i], f'{count}',
#              ha='center', va='bottom', fontsize=12)

# # Rotate x-labels at a 45-degree angle
# plt.xticks(rotation=45, ha='right')

# # Add labels and title
# plt.xlabel('Shift')
# plt.ylabel('Percentage (%)')
# plt.title('Distribution of Shift Type')

# # Show plot
# plt.tight_layout()
# plt.show()


# Convert the data in column 'b13a' to string type
data2['b13a'] = data2['b13a'].astype(str)

# Map the values in column 'b13a' to their corresponding labels
data2['b13a'] = data2['b13a'].map({
    "SQ001": "0 - 15 minutes",
    "SQ002": "16 - 30 minutes",
    "SQ003": "31 - 45 minutes",
    "SQ004": "46 - 60 minutes",
    "SQ005": "More than 60 minutes"
})

# Rename the column label
data2.rename(columns={'b13a': 'Advance Time By'}, inplace=True)

# Convert the data in column 'b13a' to string type
data2['b12a'] = data2['b12a'].astype(str)


# Map the values in column 'b13a' to their corresponding labels
data2['b12a'] = data2['b12a'].map({
    "SQ001": "0 - 15 minutes",
    "SQ002": "16 - 30 minutes",
    "SQ003": "31 - 45 minutes",
    "SQ004": "46 - 60 minutes",
    "SQ005": "More than 60 minutes"
})

data2.rename(columns={'b12a': 'Delay Time By'}, inplace=True)

# Fill NaN values in 'Delay Arrival' and 'Advance Arrival' with 0
data2[['Delay Time By', 'Advance Time By']].fillna(0, inplace=False)

column_mapping_b11 = {
    'b11[1]': 'Rank 1',
    'b11[2]': 'Rank 2',
    'b11[3]': 'Rank 3',
    'b11[4]': 'Rank 4',
    'b11[5]': 'Rank 5'
}

# Use the mapping to rename columns in your DataFrame
data2.rename(columns=column_mapping_b11, inplace=True)

# Mapping of short reasons to full descriptions
reason_mapping = {
    'A1': 'Work schedule set by my employer',
    'A2': 'Schedule of meetings or people I work with (colleagues, clients...)',
    'A3': 'Schedule of my children\'s school / daycare',
    'A4': 'Schedule of my children\'s daycare / nanny',
    'A5': 'Schedule of my partner',
    'A6': 'Schedule of my carpool(s)',
    'A7': 'Public transportation schedule',
    'A8': 'Avoiding traffic jams',
    'A9': 'Parking conditions at my workplace',
    'A10': 'Extra-curricular activities',
    'A11': 'Schedule I want to leave work in the evening',
    'A12': 'Personal preferences, habits',
    'A13': 'Implicit norms in the company, perception of colleagues or hierarchy',
    'A14': 'Class schedules',
    'A15': 'Other (please specify in the following field)'
}

data2['Rank 1'] = data2['Rank 1'].map(reason_mapping)
data2['Rank 2'] = data2['Rank 2'].map(reason_mapping)

#data2 = data2[data2['Rank 1'] != 'Other (please specify in the following field)']
#data2 = data2[data2['Rank 2'] != 'Other (please specify in the following field)']


###########################child drop off times###############################"
# Define the arrival time categories
arrival_time_categories = {
    "Before 08:00": 0,
    "08:00 - 08:29": 0,
    "08:30 - 08:59": 0,
    "09:00 - 09:29": 0,
    "After 09:30": 0
}

# Count the number of drop-off times in each arrival time category
for dropoff_time in data2['b10a']:
    # Check if the value is missing or NaN
    if pd.isnull(dropoff_time):
        continue
    
    # Convert the value to string and then to HH:MM format
    dropoff_time_str = str(dropoff_time)
    hh_mm_time = convert_to_hh_mm(dropoff_time_str)
    
    # Proceed with categorization if the conversion is successful
    if hh_mm_time is None:
        continue

    if hh_mm_time < '08:00':
        arrival_time_categories["Before 08:00"] += 1
    elif hh_mm_time < '08:30':
        arrival_time_categories["08:00 - 08:29"] += 1
    elif hh_mm_time < '09:00':
        arrival_time_categories["08:30 - 08:59"] += 1
    elif hh_mm_time < '09:30':
        arrival_time_categories["09:00 - 09:29"] += 1
    else:
        arrival_time_categories["After 09:30"] += 1

# Calculate the total number of drop-off times
total_dropoff_times = sum(arrival_time_categories.values())

# Calculate the percentage of drop-off times in each arrival time category
arrival_time_percentages = {category: (count / total_dropoff_times) * 100 for category, count in arrival_time_categories.items()}

# # Plotting the bar chart
# plt.bar(arrival_time_percentages.keys(), arrival_time_percentages.values(), color='skyblue')
# plt.xlabel('Class Drop Off Time Categories')
# plt.ylabel('Percentage (%)')
# plt.title('Drop-off Times by Arrival Time Categories')
# plt.xticks(rotation=45)
# plt.show()

###########################class start times###############################"
# Count the number of start times for classes in each arrival time category
for class_start_time in data2['b10b']:
    # Check if the value is missing or NaN
    if pd.isnull(class_start_time):
        continue
    
    # Convert the value to string and then to HH:MM format
    class_start_time_str = str(class_start_time)
    hh_mm_time = convert_to_hh_mm(class_start_time_str)
    
    # Proceed with categorization if the conversion is successful
    if hh_mm_time is None:
        continue

    if hh_mm_time < '08:00':
        arrival_time_categories["Before 08:00"] += 1
    elif hh_mm_time < '08:30':
        arrival_time_categories["08:00 - 08:29"] += 1
    elif hh_mm_time < '09:00':
        arrival_time_categories["08:30 - 08:59"] += 1
    elif hh_mm_time < '09:30':
        arrival_time_categories["09:00 - 09:29"] += 1
    else:
        arrival_time_categories["After 09:30"] += 1

# Calculate the total number of start times for classes
total_class_start_times = sum(arrival_time_categories.values())

# Calculate the percentage of start times for classes in each arrival time category
arrival_time_percentages = {category: (count / total_class_start_times) * 100 for category, count in arrival_time_categories.items()}

# # Plotting the bar chart
# plt.bar(arrival_time_percentages.keys(), arrival_time_percentages.values(), color='skyblue')
# plt.xlabel('Class Start Time Categories')
# plt.ylabel('Percentage (%)')
# #plt.title('Start Times for Classes by Arrival Time Categories')
# plt.xticks(rotation=45)
# plt.show()

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
bins = [0, 10, 20, float('inf')]
labels = ['0-10 min', '10-20 min', '>20 min']

# Categorize the differences into intervals
data2['interval'] = pd.cut(data2['diff_minutes'], bins=bins, labels=labels)

# Count the instances in each interval and calculate percentages
interval_counts = data2['interval'].value_counts(normalize=True).sort_index() * 100  # Convert counts to percentages

# # Plot the results
# plt.figure(figsize=(10, 6))
# interval_counts.plot(kind='bar', color='skyblue')
# plt.xlabel('Minutes Before Start Time')
# plt.ylabel('Percentage of Drop-offs')
# #plt.title('Drop-off Times by Minutes Before Start Time')
# plt.xticks(rotation=45)
# plt.ylim(0, 100)  # Ensure the y-axis starts from 0 to 100 for percentage representation
# plt.show()

############################# columns to include for modeling #####################################

columns_to_include = [

   'Income',
   
   'Commute Time',
   
   #'own cars',

   'Job Category',
    
   'Contract Type',

    'Gender',  # Gender types

    'Age Group',

   'Child Drop Off Frequency',

    
    #'Partner Drop Off',

    'Number of Kid to Drop-off',

    'Practical Arrival Time'   ,   #arrival time types

    'Theoretical Arrival Time Contract',

    'Telework Frequency',

    'Mode of Transport',  # Mode of transport types

    'Shift',


    # 'Before 08:00',
    # '08:00 - 08:29',
    # '08:30 - 08:59',
    # '09:00 - 09:29',
    # 'After 09:30',  # 
    #'Arrival Time',#arrival interval

   #'Rank 1', 
    #'Rank 2',

  #'Traffic Experience',
    
   #'Long Commute Time',
    
    'Congestion Discomfort Rating',
     #'Travel Habits Change',
   #'Traffic Delay Amount',

    # 'Change my mode of transport',  # travel habits changes and why not reasons
    # 'Change my route to avoid traffic jams',
    # 'Leave my home early',
    # 'Leave my home later',
    # 'Cancel my trip (teleworking for example)',
    # 'Take a break in my journey (in a café for example)',
    # 'Other_a',
    # 'I have no other mode of transport available',
    # 'I have no other possible route',
    # 'It is impossible for me to shift my journey in time',
    # 'I do not believe that changing my travel habits would improve my situation',
    # 'I do not want to change my travel habits, despite the traffic jams I experience',
    # 'Due to the time I want to leave work in the evening',
    # 'Other_c',
    
#     'Improve traffic information',
#     'Set up a bonus system',
#     'Limit meetings before 10 a.m.',
# 'Adjust business schedules',
# 'Arrange school/daycare schedules',
# 'Encourage the practice of teleworking',
# 'Authorize the practice of teleworking during peak hours',
#   'Implement a communication campaign',
#   'Encourage carpooling',
#   'Define new rules for access to parking spaces',
#   'Encourage the use of other modes of travel than the personal car',

    'Residence'

]


data_filter = data2.filter(columns_to_include)

# Mapping for the new labels
shift_mapping = {
      'Shift Everyday': 1,
      'Never': 0,

    'Shift but Not Everyday': 1
}

# Apply the mapping to assign numeric codes to 'Arrival Time'
data_filter['Shift'] = data_filter['Shift'].map(shift_mapping)
data_filter.dropna(inplace=True)


# Modeling Starts here

Apply SMOTE to handle class imbalance (if necessary)
from imblearn.over_sampling import RandomOverSampler



X = data_filter.drop(['Shift'], axis=1)
y = data_filter['Shift']

# Handle class imbalance with RandomOverSampler
ros = RandomOverSampler(random_state=42)
X, y = ros.fit_resample(X, y)

# Prepare categorical data for XGBoost native categorical input
X_cat = X.copy()
for col in X_cat.select_dtypes(include='object').columns:
    X_cat[col] = X_cat[col].astype('category')

# Prepare one-hot encoded data for other models
X_encoded = pd.get_dummies(X, drop_first=False)

# === Neural Network attention layer ===
class Attention(layers.Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], input_shape[-1]),
                                 initializer="random_normal", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[-1],),
                                 initializer="zeros", trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        a = tf.nn.softmax(e, axis=1)
        return tf.reduce_sum(x * a, axis=1)

def create_attention_model(input_shape):
    inputs = Input(shape=input_shape)
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Reshape((1, 16))(x)
    x = Attention()(x)
    outputs = layers.Dense(2, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# === Prepare cross-validation ===
models = ['RandomForest', 'XGBoost_OHE', 'XGBoost_Cat', 'NeuralNet']
results = {m: {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []} for m in models}

kf = KFold(n_splits=10, shuffle=True, random_state=42)

for train_idx, val_idx in kf.split(X):
    # Split data into train/val for each variant
    X_train_cat, X_val_cat = X_cat.iloc[train_idx], X_cat.iloc[val_idx]
    X_train_ohe, X_val_ohe = X_encoded.iloc[train_idx], X_encoded.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # --- Random Forest ---
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=5,
        min_samples_split=10, min_samples_leaf=5,
        max_features='sqrt', max_samples=0.8, random_state=42
    )
    rf.fit(X_train_ohe, y_train)
    y_pred_rf = rf.predict(X_val_ohe)
    results['RandomForest']['accuracy'].append(accuracy_score(y_val, y_pred_rf))
    results['RandomForest']['precision'].append(precision_score(y_val, y_pred_rf, zero_division=0))
    results['RandomForest']['recall'].append(recall_score(y_val, y_pred_rf, zero_division=0))
    results['RandomForest']['f1_score'].append(f1_score(y_val, y_pred_rf, zero_division=0))

    # --- XGBoost with One-Hot Encoding ---
    xgb_ohe = XGBClassifier(
        objective='binary:logistic', max_depth=5, learning_rate=0.1,
        eval_metric='logloss', alpha=0.1, reg_lambda=0.1, random_state=42
    )
    xgb_ohe.fit(X_train_ohe, y_train)
    y_pred_xgb_ohe = xgb_ohe.predict(X_val_ohe)
    results['XGBoost_OHE']['accuracy'].append(accuracy_score(y_val, y_pred_xgb_ohe))
    results['XGBoost_OHE']['precision'].append(precision_score(y_val, y_pred_xgb_ohe, zero_division=0))
    results['XGBoost_OHE']['recall'].append(recall_score(y_val, y_pred_xgb_ohe, zero_division=0))
    results['XGBoost_OHE']['f1_score'].append(f1_score(y_val, y_pred_xgb_ohe, zero_division=0))

    # --- XGBoost with native categorical input ---
    xgb_cat = XGBClassifier(
        objective='binary:logistic', max_depth=5, learning_rate=0.1,
        eval_metric='logloss', alpha=0.1, reg_lambda=0.1, enable_categorical=True, random_state=42
    )
    xgb_cat.fit(X_train_cat, y_train)
    y_pred_xgb_cat = xgb_cat.predict(X_val_cat)
    results['XGBoost_Cat']['accuracy'].append(accuracy_score(y_val, y_pred_xgb_cat))
    results['XGBoost_Cat']['precision'].append(precision_score(y_val, y_pred_xgb_cat, zero_division=0))
    results['XGBoost_Cat']['recall'].append(recall_score(y_val, y_pred_xgb_cat, zero_division=0))
    results['XGBoost_Cat']['f1_score'].append(f1_score(y_val, y_pred_xgb_cat, zero_division=0))

    # --- Neural Network on scaled one-hot encoded data ---
    # No scaling, but ensure numpy float32 for TF
    X_train_nn = X_train_ohe.values.astype('float32')
    X_val_nn = X_val_ohe.values.astype('float32')
    nn = create_attention_model(input_shape=(X_train_nn.shape[1],))
    nn.fit(X_train_nn, y_train, epochs=10, verbose=0)
    y_pred_nn = np.argmax(nn.predict(X_val_nn), axis=1)
    results['NeuralNet']['accuracy'].append(accuracy_score(y_val, y_pred_nn))
    results['NeuralNet']['precision'].append(precision_score(y_val, y_pred_nn, zero_division=0))
    results['NeuralNet']['recall'].append(recall_score(y_val, y_pred_nn, zero_division=0))
    results['NeuralNet']['f1_score'].append(f1_score(y_val, y_pred_nn, zero_division=0))

# === Print average metrics ===
print("\nCross-Validation Results (10-fold):")
for model in models:
    print(f"\n--- {model} ---")
    for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
        avg = np.mean(results[model][metric])
        print(f"{metric.capitalize():<10}: {avg:.4f}")
        
        
# === Learning Curve Parameters ===
train_sizes = np.linspace(0.1, 1.0, 10)
kf = KFold(n_splits=10, shuffle=True, random_state=42)

models = ['RandomForest', 'XGBoost_OHE', 'XGBoost_Cat', 'NeuralNet']
learning_curve_results = {m: {'train': [], 'val': []} for m in models}

for frac in train_sizes:
    fold_results = {m: {'train': [], 'val': []} for m in models}
    for train_idx, val_idx in kf.split(X_encoded):
        # Split folds
        X_train_cat, X_val_cat = X_cat.iloc[train_idx], X_cat.iloc[val_idx]
        X_train_ohe, X_val_ohe = X_encoded.iloc[train_idx], X_encoded.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Subsample training set
        sub_size = int(len(X_train_ohe) * frac)
        X_sub_ohe, y_sub = X_train_ohe.iloc[:sub_size], y_train.iloc[:sub_size]
        X_sub_cat = X_train_cat.iloc[:sub_size]

        # --- Random Forest ---
        rf = RandomForestClassifier(n_estimators=100, max_depth=5,
                                    min_samples_split=10, min_samples_leaf=5,
                                    max_features='sqrt', max_samples=0.8, random_state=42)
        rf.fit(X_sub_ohe, y_sub)
        fold_results['RandomForest']['train'].append(accuracy_score(y_sub, rf.predict(X_sub_ohe)))
        fold_results['RandomForest']['val'].append(accuracy_score(y_val, rf.predict(X_val_ohe)))

        # --- XGBoost OHE ---
        xgb_ohe = XGBClassifier(objective='binary:logistic', max_depth=5, learning_rate=0.1,
                                eval_metric='logloss', alpha=0.1, reg_lambda=0.1, random_state=42)
        xgb_ohe.fit(X_sub_ohe, y_sub)
        fold_results['XGBoost_OHE']['train'].append(accuracy_score(y_sub, xgb_ohe.predict(X_sub_ohe)))
        fold_results['XGBoost_OHE']['val'].append(accuracy_score(y_val, xgb_ohe.predict(X_val_ohe)))

        # --- XGBoost Cat ---
        xgb_cat = XGBClassifier(objective='binary:logistic', max_depth=5, learning_rate=0.1,
                                eval_metric='logloss', alpha=0.1, reg_lambda=0.1,
                                enable_categorical=True, random_state=42)
        xgb_cat.fit(X_sub_cat, y_sub)
        fold_results['XGBoost_Cat']['train'].append(accuracy_score(y_sub, xgb_cat.predict(X_sub_cat)))
        fold_results['XGBoost_Cat']['val'].append(accuracy_score(y_val, xgb_cat.predict(X_val_cat)))

        # --- Neural Network ---
        X_sub_nn = X_sub_ohe.values.astype('float32')
        X_val_nn = X_val_ohe.values.astype('float32')
        y_sub_nn = y_sub.values.astype('int32')
        y_val_nn = y_val.values.astype('int32')
        nn = create_attention_model((X_sub_nn.shape[1],))
        nn.fit(X_sub_nn, y_sub_nn, epochs=10, batch_size=32, verbose=0)
        train_acc = accuracy_score(y_sub_nn, np.argmax(nn.predict(X_sub_nn, verbose=0), axis=1))
        val_acc = accuracy_score(y_val_nn, np.argmax(nn.predict(X_val_nn, verbose=0), axis=1))
        fold_results['NeuralNet']['train'].append(train_acc)
        fold_results['NeuralNet']['val'].append(val_acc)

    # Average over folds
    for model in models:
        learning_curve_results[model]['train'].append(np.mean(fold_results[model]['train']))
        learning_curve_results[model]['val'].append(np.mean(fold_results[model]['val']))

# === Plot Learning Curves ===
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.ravel()

for i, model in enumerate(models):
    ax = axs[i]
    ax.plot(train_sizes, learning_curve_results[model]['train'], label='Train Accuracy', marker='o')
    ax.plot(train_sizes, learning_curve_results[model]['val'], label='Validation Accuracy', marker='x')
    ax.set_title(f'Learning Curve - {model}')
    ax.set_xlabel('Training Set Size (Fraction)')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0.6, 1.05)
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()
