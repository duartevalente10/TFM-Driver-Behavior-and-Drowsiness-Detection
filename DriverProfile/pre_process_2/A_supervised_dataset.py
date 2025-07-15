import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load the csv
df = pd.read_csv('datasets/supervised/supervised_HRV_KSS.csv')

# summary statistics for numerical columns
print("Dataset Statistics:")
print(df.describe())

print("Mean KKS by File:")
# mean kss_answer grouped by Filename
print(df.groupby('Filename')['kss_answer'].mean())

# plot size
plt.figure(figsize=(10, 6))

# loop for filename and plot the KKS over time
for filename, group in df.groupby('Filename'):
    plt.plot(group['Interval_Start'], group['kss_answer'], marker='o', label=filename)

# add title and labels
plt.title('Evolution of KSS Answer Over Time for Each Filename')
plt.xlabel('Time (Interval Start)')
plt.ylabel('KSS Answer')

# add a legend 
plt.legend(title='Filename')

# show
plt.grid(True)
plt.show()