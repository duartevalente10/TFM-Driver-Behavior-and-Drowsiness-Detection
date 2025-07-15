import pandas as pd

# Load the CSV file into a DataFrame
# Replace 'your_file.csv' with the actual path to your CSV file
df = pd.read_csv('datasets/sim/sim_filered_34.csv')

# Set the threshold for detecting missing instances (1 second)
threshold = 0.5

def compute_adjusted_duration(group):
    # Sort by timer [s] just in case
    group = group.sort_values(by='timer [s]')
    
    # Calculate the time differences between consecutive rows
    group['time_diff'] = group['timer [s]'].diff()

    # Compute total duration assuming no gaps initially
    total_duration = group['timer [s]'].max() - group['timer [s]'].min()

    # Subtract the gaps that exceed the threshold
    missing_time = group[group['time_diff'] > threshold]['time_diff'].sum()

    # Adjusted duration is total duration minus missing time
    adjusted_duration = round(total_duration - missing_time,1)

    return adjusted_duration

# Group by 'Filename' and apply the adjusted duration calculation
duration_df = df.groupby('Filename').apply(compute_adjusted_duration).reset_index(name='Duration')

# Print the result
#print(duration_df)

# save
duration_df.to_csv('datasets/sim/sim_filtered_durations.csv', index=False)
