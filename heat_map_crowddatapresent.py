import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv
import json
import datetime
from math import floor

human_count = []
violate_count = []
with open('processed_data/crowd_data.csv', 'r') as file:
    reader = csv.reader(file, delimiter=',')
    next(reader)
    for row in reader:
        human_count.append(int(row[1]))
        violate_count.append(int(row[2]))

with open('processed_data/video_data.json', 'r') as file:
    data = json.load(file)
    data_record_frame = data["DATA_RECORD_FRAME"]
    vid_fps = data["VID_FPS"]

time_steps = data_record_frame / vid_fps
data_length = len(human_count)

# Reshape data for heatmap (assuming human_count represents a time series)
heatmap_data = [human_count[i:i + int(time_steps)] for i in range(0, data_length, int(time_steps))]

fig, ax = plt.subplots()
cmap = cm.get_cmap('YlGnBu')  # Choose a colormap (adjust as desired)
ax.imshow(heatmap_data, aspect='auto', cmap=cmap)

# Configure heatmap ticks and labels (optional)
# ... (add code to set tick positions and labels based on your data)

plt.title("Crowd Heatmap")
plt.xlabel("Time Step")
plt.ylabel("Count")
plt.colorbar(label='Crowd Count')
plt.show()
