import pandas as pd

# load df
df = pd.read_csv('datasets/sim/sim_filered_34.csv')

# set the threshold (0.5s)
threshold = 0.5

# detect consecutive instances with a gap > that the threshold and compute the missing duration
def compute_adjusted_duration(group):
    # sort by timer
    group = group.sort_values(by='timer [s]')
    
    # compute the time differences between consecutive rows
    group['time_diff'] = group['timer [s]'].diff()

    # compute total duration
    total_duration = group['timer [s]'].max() - group['timer [s]'].min()

    # subtract the gaps that exceed the threshold
    missing_time = group[group['time_diff'] > threshold]['time_diff'].sum()

    # adjusted duration is total duration minus missing time
    adjusted_duration = round(total_duration - missing_time,1)

    return adjusted_duration

# group by filename and apply the adjusted duration
duration_df = df.groupby('Filename').apply(compute_adjusted_duration).reset_index(name='Duration')

# result
print(duration_df)

# save
#duration_df.to_csv('datasets/sim/sim_filtered_durations.csv', index=False)
