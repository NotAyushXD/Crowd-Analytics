import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

df = pd.read_csv("./processed_data/heatmap_scores.csv")


# Create a directory to save the plots
output_dir = "/Users/ayushpant/Desktop/Working/Workstation/Hackathon - Vision/Crowd-Analysis/processed_data/"

# Normalize each column (excluding the Timestamp)
for col in df.columns[1:]:
    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

# Plotting all data on a single graph
plt.figure(figsize=(15, 10))

for col in df.columns[1:]:
    plt.plot(df['Timestamp'], df[col], label=col)

plt.xlabel('Timestamp')
plt.ylabel('Normalized Values')
plt.title('Combined Plot of All Columns')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot
plt.savefig(output_dir)
plt.close()

output_dir
