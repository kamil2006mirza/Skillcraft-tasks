# Imports
import pandas as pd
# import numpy as np # Not used in visible code
# import matplotlib.pyplot as plt
# import seaborn as sns

# File path settings
DATA_PATH = 'accidents.csv' # Change if needed
OUT_PATH = 'data/output/'

# Set display options (if using interactive environment)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)

# Load dataset
df = pd.read_csv(DATA_PATH)
print("Loaded data: {}({})".format(DATA_PATH, df.shape))
# df = df.sample(frac=0.1, random_state=42) # For faster processing/testing
print("Reduced data size: {}".format(df.shape))

# -------------------------------------------------------------
# Analyze Time Column
# -------------------------------------------------------------
print("\nAnalyze 'Time' column")
# The column name is assumed to be 'Time'
df['Time'] = df['Time'].astype(str) # Ensure it's string

# Function to convert time column
def convert_time_column():
    pass # Function not defined in visible code

# 1. Convert Time to Hour
df['Hour'] = pd.NA # Create empty column

def simple_hour_conversion():
    # Try splitting the string: 'hh:mm:ss' or 'hh:mm'
    # Assumes time is the first part before a space or hyphen, if any
    df['TimeStr'] = df['Time'].apply(lambda x: x.split()[0].split('-')[0])
    print("TimeStr format: 'hh:mm:ss' or 'hh:mm'")

    def extract_hour(time_str):
        try:
            return int(time_str.split(':')[0])
        except Exception:
            # Handle cases where the format isn't 'H:M:S'
            return pd.NA

    df['Hour'] = df['TimeStr'].apply(extract_hour)
    print("Time 'Hour' column created")
    
    # Error checking/handling for missing/bad data
    print("Missing 'Hour' values: {}".format(df['Hour'].isna().sum()))
    df.dropna(subset=['Hour'], inplace=True)
    df['Hour'] = df['Hour'].astype(int)

    # Alternate approach: pd.to_datetime (more robust)
    try:
        df['Time_dt'] = pd.to_datetime(df['Time'], errors='coerce')
        df['Hour_dt'] = df['Time_dt'].dt.hour
        print("Using pd.to_datetime for hour extraction.")
        df.dropna(subset=['Hour_dt'], inplace=True)
        df['Hour_dt'] = df['Hour_dt'].astype(int)
    except Exception as e:
        print("Error using pd.to_datetime for 'Time' column: {}".format(e))
        pass # Stick to simple extraction if robust method fails

# Using the simpler extraction method
simple_hour_conversion()

# 2. Extract Day of Week
# Assuming 'Date' column exists or can be derived from 'Time'
try:
    # If using the datetime object from the alternate approach:
    # df['Day_of_Week'] = df['Time_dt'].dt.dayofweek # 0=Monday, 6=Sunday
    
    # If there is a separate 'Date' column, convert it first
    # For now, let's assume the 'Time' column contains date info or we'll skip
    pass
except Exception:
    print("Could not extract Day_of_Week.")

# -------------------------------------------------------------
# Analysis: Accident Severity vs Incidence
# -------------------------------------------------------------
print("\nAnalysis: Accident Severity vs Incidence")

# 1. Group by Severity
severity_map = {1: 'Slight', 2: 'Serious', 3: 'Fatal'}
df['Accident_Severity_Desc'] = df['Accident_Severity'].map(severity_map)

severity_counts = df['Accident_Severity_Desc'].value_counts().reset_index()
severity_counts.columns = ['Severity', 'Total_Accidents']

# Plotting: Severity
# plt.bar(severity_counts['Severity'], severity_counts['Total_Accidents'])
# plt.title('Total Accidents by Severity')
# plt.xlabel('Severity')
# plt.ylabel('Number of Accidents')
# plt.show()

# -------------------------------------------------------------
# Analysis: Weather Conditions
# -------------------------------------------------------------
print("\nAnalysis: Weather Conditions")

# The column name is assumed to be 'Weather_Conditions'
# Get average accident counts per weather condition (Normalization/Rate)
avg_severity_by_weather = df.groupby('Weather_Conditions')['Accident_Severity'].mean().reset_index()
avg_severity_by_weather.columns = ['Weather_Condition', 'Avg_Accident_Severity']
# Note: Mean severity is a weighted average

# Get total number of accidents per weather condition
accidents_by_weather = df['Weather_Conditions'].value_counts().reset_index()
accidents_by_weather.columns = ['Weather_Condition', 'Number_of_Accidents']

# Merge the two results
weather_analysis = pd.merge(accidents_by_weather, avg_severity_by_weather, on='Weather_Condition')

# Plotting: Number of Accidents by Weather
# plt.bar(weather_analysis['Weather_Condition'], weather_analysis['Number_of_Accidents'])
# plt.title('Number of Accidents by Weather Condition')
# plt.xlabel('Weather Condition')
# plt.ylabel('Number of Accidents')
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()
# plt.show()

# -------------------------------------------------------------
# Analysis: Time (Hour)
# -------------------------------------------------------------
print("\nAnalysis: Time (Hour)")

# 1. Group by Hour
accidents_by_hour = df.groupby('Hour')['Accident_Severity'].count().reset_index()
accidents_by_hour.columns = ['Hour', 'Number_of_Accidents']

# 2. Get average severity by hour
avg_severity_by_hour = df.groupby('Hour')['Accident_Severity'].mean().reset_index()
avg_severity_by_hour.columns = ['Hour', 'Avg_Severity']

# Merge the two results
hour_analysis = pd.merge(accidents_by_hour, avg_severity_by_hour, on='Hour')

# Plotting: Accidents by Hour
# plt.bar(hour_analysis['Hour'], hour_analysis['Number_of_Accidents'])
# plt.title('Number of Accidents by Hour of Day')
# plt.xlabel('Hour')
# plt.ylabel('Number of Accidents')
# plt.show()

# Plotting: Average Severity by Hour
# plt.plot(hour_analysis['Hour'], hour_analysis['Avg_Severity'], marker='o')
# plt.title('Average Accident Severity by Hour of Day')
# plt.xlabel('Hour')
# plt.ylabel('Average Severity')
# plt.show()

# -------------------------------------------------------------
# Analysis: Day of Week
# -------------------------------------------------------------
print("\nAnalysis: Day of Week")
# Assuming 'Day_of_Week' was successfully created (0=Mon, 6=Sun)

# 1. Group by Day of Week
accidents_by_day = df.groupby('Day_of_Week')['Accident_Severity'].count().reset_index()
accidents_by_day.columns = ['Day_of_Week', 'Number_of_Accidents']

# 2. Get average severity by day
avg_severity_by_day = df.groupby('Day_of_Week')['Accident_Severity'].mean().reset_index()
avg_severity_by_day.columns = ['Day_of_Week', 'Avg_Severity']

# Merge the two results
day_analysis = pd.merge(accidents_by_day, avg_severity_by_day, on='Day_of_Week')

# Mapping to day names for better plotting
day_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
day_analysis['Day_Name'] = day_analysis['Day_of_Week'].map(day_map)

# Plotting: Accidents by Day
# plt.bar(day_analysis['Day_Name'], day_analysis['Number_of_Accidents'])
# plt.title('Number of Accidents by Day of Week')
# plt.xlabel('Day of Week')
# plt.ylabel('Number of Accidents')
# plt.show()

# Final output (e.g., saving analysis data)
# weather_analysis.to_csv(OUT_PATH + 'weather_analysis.csv', index=False)
# hour_analysis.to_csv(OUT_PATH + 'hour_analysis.csv', index=False)
# day_analysis.to_csv(OUT_PATH + 'day_analysis.csv', index=False)

print("\nAnalysis complete.")
