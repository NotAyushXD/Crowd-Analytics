import csv
import imutils
import cv2
import json
import math
import numpy as np
from config import VIDEO_CONFIG
from itertools import zip_longest
from math import ceil
from scipy.spatial.distance import euclidean
from colors import RGB_COLORS, gradient_color_RGB
from tqdm import tqdm

# Function to calculate heatmap score
def calculate_heatmap_score(heatmap):
    return np.sum(heatmap)

# Function to calculate heatmap score for specific grid cells
def calculate_grid_heatmap_scores(heatmap, grid_size):
    heatmap_scores = []
    height, width = heatmap.shape[:2]
    cell_height = height // grid_size
    cell_width = width // grid_size

    for i in range(grid_size):
        for j in range(grid_size):
            cell_heatmap = heatmap[i*cell_height:(i+1)*cell_height, j*cell_width:(j+1)*cell_width]
            score = calculate_heatmap_score(cell_heatmap)
            heatmap_scores.append(score)

    return heatmap_scores

# Function to get cell name (e.g., A1, A2, B1, etc.)
def get_cell_name(row, col):
    return f"{chr(65 + row)}{col + 1}"  # A is 65 in ASCII, +1 for 1-based index

tracks = []
print("Reading movement data...")
with open('processed_data/movement_data.csv', 'r') as file:
    reader = csv.reader(file, delimiter=',')
    for row in reader:
        if len(row[3:]) > 4:
            temp = []
            data = row[3:]
            for i in range(0, len(data), 2):
                temp.append([int(data[i]), int(data[i+1])])
            tracks.append(temp)

print("Reading video data...")
with open('processed_data/video_data.json', 'r') as file:
    data = json.load(file)
    vid_fps = data["VID_FPS"]
    data_record_frame = data["DATA_RECORD_FRAME"]
    frame_size = data["PROCESSED_FRAME_SIZE"]

cap = cv2.VideoCapture(VIDEO_CONFIG["VIDEO_CAP"])
cap.set(1, 100)
(ret, tracks_frame) = cap.read()
tracks_frame = imutils.resize(tracks_frame, width=frame_size)
heatmap_frame = np.copy(tracks_frame)
print("Frame size:", tracks_frame.shape)
stationary_threshold_seconds = 2
stationary_threshold_frame = round(vid_fps * stationary_threshold_seconds / data_record_frame)
stationary_distance = frame_size * 0.05
max_stationary_time = 120
blob_layer = 50
max_blob_size = frame_size * 0.1
layer_size = max_blob_size / blob_layer
color_start = 210
color_end = 0
color_steps = int((color_start - color_end) / blob_layer)
scale = 1.5

# Define grid parameters
grid_size = 5  # Adjust grid size as needed

stationary_points = []
movement_points = []
total = 0
print("Processing movement points...")
for movement in tracks:
    temp_movement_point = [movement[0]]
    stationary = movement[0]
    stationary_time = 0
    for i in movement[1:]:
        if euclidean(stationary, i) < stationary_distance:
            stationary_time += 1
        else:
            temp_movement_point.append(i)
            if stationary_time > stationary_threshold_frame:
                stationary_points.append([stationary, stationary_time])
            stationary = i
            stationary_time = 0
    movement_points.append(temp_movement_point)
    total += len(temp_movement_point)

# Generate heatmap
color1 = (255, 96, 0)
color2 = (0, 28, 255)
for track in movement_points:
    for i in range(len(track) - 1):
        color = gradient_color_RGB(color1, color2, len(track) - 1, i)
        cv2.line(tracks_frame, tuple(track[i]), tuple(track[i+1]), color, 2)

def draw_blob(frame, coordinates, time):
    if time >= max_stationary_time:
        layer = blob_layer
    else:
        layer = math.ceil(time * scale / layer_size)
    for x in reversed(range(layer)):
        color = color_start - (color_steps * x)
        size = x * layer_size
        cv2.circle(frame, coordinates, int(size), (color, color, color), -1)

print("Processing video and calculating heatmap scores...")

# Open CSV file for writing heatmap scores
with open('heatmap_scores.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    header = ["Timestamp"] + [get_cell_name(i // grid_size, i % grid_size) for i in range(grid_size**2)]
    csv_writer.writerow(header)

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for _ in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        timestamp = frame_count / vid_fps

        heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        for points in stationary_points:
            draw_heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            draw_blob(draw_heatmap, tuple(points[0]), points[1])
            heatmap = cv2.add(heatmap, draw_heatmap)

        lo = np.array([color_start])
        hi = np.array([255])
        mask = cv2.inRange(heatmap, lo, hi)
        heatmap[mask > 0] = color_start

        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        lo = np.array([128,0,0])
        hi = np.array([136,0,0])
        mask = cv2.inRange(heatmap, lo, hi)
        heatmap[mask > 0] = (0, 0, 0)

        for row in range(heatmap.shape[0]):
            for col in range(heatmap.shape[1]):
                if (heatmap[row][col] == np.array([0,0,0])).all():
                    heatmap[row][col] = frame[row][col]

        heatmap_frame = cv2.addWeighted(heatmap, 0.75, frame, 0.25, 1)

        # Calculate grid heatmap scores for the current frame
        heatmap_scores = calculate_grid_heatmap_scores(heatmap, grid_size)

        # Write heatmap scores to CSV file
        csv_writer.writerow([timestamp] + heatmap_scores)

cap.release()
cv2.destroyAllWindows()
print("Processing complete.")


# _________________________________

# import csv
# import imutils
# import cv2
# import json
# import math
# import numpy as np
# from config import VIDEO_CONFIG
# from itertools import zip_longest
# from math import ceil
# from scipy.spatial.distance import euclidean
# from colors import RGB_COLORS, gradient_color_RGB

# # Function to calculate heatmap score
# def calculate_heatmap_score(heatmap):
#     return np.sum(heatmap)

# # Function to calculate heatmap score for specific grid cells
# def calculate_grid_heatmap_scores(heatmap, grid_size):
#     heatmap_scores = []
#     height, width = heatmap.shape[:2]
#     cell_height = height // grid_size
#     cell_width = width // grid_size

#     for i in range(grid_size):
#         for j in range(grid_size):
#             cell_heatmap = heatmap[i*cell_height:(i+1)*cell_height, j*cell_width:(j+1)*cell_width]
#             score = calculate_heatmap_score(cell_heatmap)
#             heatmap_scores.append(score)

#     return heatmap_scores

# # Function to get cell name (e.g., A1, A2, B1, etc.)
# def get_cell_name(row, col):
#     return f"{chr(65 + row)}{col + 1}"  # A is 65 in ASCII, +1 for 1-based index

# tracks = []
# with open('processed_data/movement_data.csv', 'r') as file:
#     reader = csv.reader(file, delimiter=',')
#     for row in reader:
#         if len(row[3:]) > 4:
#             temp = []
#             data = row[3:]
#             for i in range(0, len(data), 2):
#                 temp.append([int(data[i]), int(data[i+1])])
#             tracks.append(temp)

# with open('processed_data/video_data.json', 'r') as file:
#     data = json.load(file)
#     vid_fps = data["VID_FPS"]
#     data_record_frame = data["DATA_RECORD_FRAME"]
#     frame_size = data["PROCESSED_FRAME_SIZE"]

# cap = cv2.VideoCapture(VIDEO_CONFIG["VIDEO_CAP"])
# cap.set(1, 100)
# (ret, tracks_frame) = cap.read()
# tracks_frame = imutils.resize(tracks_frame, width=frame_size)
# heatmap_frame = np.copy(tracks_frame)
# print(tracks_frame.shape)
# stationary_threshold_seconds = 2
# stationary_threshold_frame =  round(vid_fps * stationary_threshold_seconds / data_record_frame)
# stationary_distance = frame_size * 0.05
# max_stationary_time = 120
# blob_layer = 50
# max_blob_size = frame_size * 0.1
# layer_size = max_blob_size / blob_layer
# color_start = 210
# color_end = 0
# color_steps = int((color_start - color_end) / blob_layer)
# scale = 1.5

# # Define grid parameters
# grid_size = 5  # Adjust grid size as needed

# stationary_points = []
# movement_points = []
# total = 0
# for movement in tracks:
#     temp_movement_point = [movement[0]]
#     stationary = movement[0]
#     stationary_time = 0
#     for i in movement[1:]:
#         if euclidean(stationary, i) < stationary_distance:
#             stationary_time += 1
#         else:
#             temp_movement_point.append(i)
#             if stationary_time > stationary_threshold_frame:
#                 stationary_points.append([stationary, stationary_time])
#             stationary = i
#             stationary_time = 0
#     movement_points.append(temp_movement_point)
#     total += len(temp_movement_point)

# # Generate heatmap
# color1 = (255, 96, 0)
# color2 = (0, 28, 255)
# for track in movement_points:
#     for i in range(len(track) - 1):
#         color = gradient_color_RGB(color1, color2, len(track) - 1, i)
#         cv2.line(tracks_frame, tuple(track[i]), tuple(track[i+1]), color, 2)
    
# def draw_blob(frame, coordinates, time):
#     if time >= max_stationary_time:
#         layer = blob_layer
#     else:
#         layer = math.ceil(time * scale / layer_size)
#     for x in reversed(range(layer)):
#         color = color_start - (color_steps * x)
#         size = x * layer_size
#         cv2.circle(frame, coordinates, int(size), (color, color, color), -1)

# heatmap = np.zeros((heatmap_frame.shape[0], heatmap_frame.shape[1]), dtype=np.uint8)
# for points in stationary_points:
#     draw_heatmap = np.zeros((heatmap_frame.shape[0], heatmap_frame.shape[1]), dtype=np.uint8)
#     draw_blob(draw_heatmap, tuple(points[0]), points[1])
#     heatmap = cv2.add(heatmap, draw_heatmap)

# lo = np.array([color_start])
# hi = np.array([255])
# mask = cv2.inRange(heatmap, lo, hi)
# heatmap[mask > 0] = color_start

# heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
# lo = np.array([128,0,0])
# hi = np.array([136,0,0])
# mask = cv2.inRange(heatmap, lo, hi)
# heatmap[mask > 0] = (0, 0, 0)

# for row in range(heatmap.shape[0]):
#     for col in range(heatmap.shape[1]):
#         if (heatmap[row][col] == np.array([0,0,0])).all():
#             heatmap[row][col] = heatmap_frame[row][col] 

# heatmap_frame = cv2.addWeighted(heatmap, 0.75, heatmap_frame, 0.25, 1)

# # Calculate grid heatmap scores
# heatmap_scores = calculate_grid_heatmap_scores(heatmap, grid_size)
# print("Grid Heatmap Scores:")
# for i, score in enumerate(heatmap_scores):
#     cell_name = get_cell_name(i // grid_size, i % grid_size)
#     print(f"{cell_name}: {score}")

# # Draw grid lines and annotate cells
# cell_height = tracks_frame.shape[0] // grid_size
# cell_width = tracks_frame.shape[1] // grid_size

# # Draw vertical lines and cell names on tracks_frame
# for i in range(1, grid_size):
#     x = i * cell_width
#     cv2.line(tracks_frame, (x, 0), (x, tracks_frame.shape[0]), (0, 255, 0), 2)
#     cv2.putText(tracks_frame, chr(65 + i), (x + 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# # Draw horizontal lines and cell names on tracks_frame
# for j in range(1, grid_size):
#     y = j * cell_height
#     cv2.line(tracks_frame, (0, y), (tracks_frame.shape[1], y), (0, 255, 0), 2)
#     cv2.putText(tracks_frame, str(j + 1), (10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# # Draw vertical lines and cell names on heatmap_frame
# for i in range(1, grid_size):
#     x = i * cell_width
#     cv2.line(heatmap_frame, (x, 0), (x, heatmap_frame.shape[0]), (0, 255, 0), 2)
#     cv2.putText(heatmap_frame, chr(65 + i), (x + 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# # Draw horizontal lines and cell names on heatmap_frame
# for j in range(1, grid_size):
#     y = j * cell_height
#     cv2.line(heatmap_frame, (0, y), (heatmap_frame.shape[1], y), (0, 255, 0), 2)
#     cv2.putText(heatmap_frame, str(j + 1), (10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# cv2.imshow("Movement Tracks", tracks_frame)
# cv2.imshow("Stationary Location Heatmap", heatmap_frame)
# cv2.waitKey()
# cv2.destroyAllWindows()
# cap.release()
# ____________________


# import cv2
# import numpy as np
# import csv
# import math
# import imutils
# import json
# from config import VIDEO_CONFIG

# # Function to calculate heatmap score
# def calculate_heatmap_score(heatmap):
#     return np.sum(heatmap)

# # Function to calculate heatmap score for specific grid cells
# def calculate_grid_heatmap_scores(heatmap, grid_size):
#     heatmap_scores = []
#     height, width = heatmap.shape[:2]
#     cell_height = height // grid_size
#     cell_width = width // grid_size

#     for i in range(grid_size):
#         for j in range(grid_size):
#             cell_heatmap = heatmap[i*cell_height:(i+1)*cell_height, j*cell_width:(j+1)*cell_width]
#             score = calculate_heatmap_score(cell_heatmap)
#             heatmap_scores.append(score)

#     return heatmap_scores

# # Function to get cell name (e.g., A1, A2, B1, etc.)
# def get_cell_name(row, col):
#     return f"{chr(65 + row)}{col + 1}"  # A is 65 in ASCII, +1 for 1-based index

# # Load video and JSON data
# cap = cv2.VideoCapture(VIDEO_CONFIG["VIDEO_CAP"])
# with open('processed_data/video_data.json', 'r') as file:
#     data = json.load(file)
#     vid_fps = data["VID_FPS"]
#     frame_size = data["PROCESSED_FRAME_SIZE"]

# # Define grid and interval parameters
# grid_size = 5
# interval_seconds = 10
# frames_per_interval = int(vid_fps * interval_seconds)
# total_intervals = int(math.ceil(cap.get(cv2.CAP_PROP_FRAME_COUNT) / frames_per_interval))

# # Prepare data structure for storing scores
# scores_by_time = {i: [0] * (grid_size ** 2) for i in range(total_intervals)}

# # Process video frames by intervals
# current_interval = 0
# frame_count = 0

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Resize and process frame
#     frame = imutils.resize(frame, width=frame_size)
#     # Generate a basic heatmap (for demonstration, replace with actual heatmap logic)
#     heatmap = np.random.randint(0, 256, (frame.shape[0], frame.shape[1]), dtype=np.uint8)

#     # Accumulate heatmap scores for the current interval
#     grid_scores = calculate_grid_heatmap_scores(heatmap, grid_size)
#     for i, score in enumerate(grid_scores):
#         scores_by_time[current_interval][i] += score

#     frame_count += 1
#     if frame_count % frames_per_interval == 0:
#         current_interval += 1

# cap.release()

# # Debug: print some data to see what's being captured
# print("Sample scores from the first interval:", scores_by_time[0])

# # Save results to CSV
# csv_path = './processed_data/heatmap_grid_output.csv'
# with open(csv_path, 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['Time Interval'] + [f"Cell {get_cell_name(i // grid_size, i % grid_size)}" for i in range(grid_size ** 2)])
#     for interval in range(total_intervals):
#         writer.writerow([f"{interval * interval_seconds}-{(interval + 1) * interval_seconds} sec"] + scores_by_time[interval])