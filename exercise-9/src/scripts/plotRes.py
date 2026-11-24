import csv
import matplotlib.pyplot as plt

# Initialize lists
names = []
avg_times = []
stdev_times = []

# Read CSV file, change the name of the file to read a different file
filename = 'transformVbenchmarkUnroll.txt'
with open(filename, 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        names.append(row['name'])
        avg_times.append(float(row['avg time']))
        stdev_times.append(float(row['stdev time']))

# Create bar plot
plt.figure(figsize=(10, 6))
plt.bar(names, avg_times, yerr=stdev_times, capsize=5, color='lightgreen', edgecolor='black')

# Plot formatting
plt.xticks(rotation=45, ha='right')
plt.ylabel('Average Time (ns)')
plt.title('Benchmark Average Time with Standard Deviation' + filename[7:-4])
plt.tight_layout()


# save the plot, you can also change this
plt.savefig("../figs/" +filename[:-4] + ".png")