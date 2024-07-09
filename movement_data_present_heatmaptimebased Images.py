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
# from tqdm import tqdm
# import os

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

# def transform_perspective(frame, src_points, dst_points):
#     M = cv2.getPerspectiveTransform(src_points, dst_points)
#     return cv2.warpPerspective(frame, M, (frame.shape[1], frame.shape[0]))

# tracks = []
# print("Reading movement data...")
# with open('processed_data/movement_data.csv', 'r') as file:
#     reader = csv.reader(file, delimiter=',')
#     for row in reader:
#         if len(row[3:]) > 4:
#             temp = []
#             data = row[3:]
#             for i in range(0, len(data), 2):
#                 temp.append([int(data[i]), int(data[i+1])])
#             tracks.append(temp)

# print("Reading video data...")
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
# print("Frame size:", tracks_frame.shape)
# stationary_threshold_seconds = 2
# stationary_threshold_frame = round(vid_fps * stationary_threshold_seconds / data_record_frame)
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

# # Perspective transformation points
# src_points = np.float32([[0, 0], [frame_size, 0], [frame_size, frame_size], [0, frame_size]])
# dst_points = np.float32([[0, 0], [frame_size, 0], [frame_size, frame_size], [0, frame_size]])  # Modify this according to the desired perspective

# # Aggregation interval in frames
# aggregation_interval = 10

# stationary_points = []
# movement_points = []
# total = 0
# print("Processing movement points...")
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

# print("Processing video and calculating heatmap scores...")

# # Create output directory for heatmap images
# heatmap_dir = 'heatmap_images'
# os.makedirs(heatmap_dir, exist_ok=True)

# # Open CSV file for writing heatmap scores
# with open('./processed_data/heatmap_scores.csv', 'w', newline='') as csvfile:
#     csv_writer = csv.writer(csvfile)
#     header = ["Timestamp"] + [get_cell_name(i // grid_size, i % grid_size) for i in range(grid_size**2)]
#     csv_writer.writerow(header)

#     frame_count = 0
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     aggregated_heatmap = None
#     interval_count = 0

#     for _ in tqdm(range(total_frames)):
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_count += 1
#         timestamp = frame_count / vid_fps

#         # Apply perspective transformation to the entire frame
#         transformed_frame = transform_perspective(frame, src_points, dst_points)

#         heatmap = np.zeros((transformed_frame.shape[0], transformed_frame.shape[1]), dtype=np.uint8)
#         for points in stationary_points:
#             if frame_count >= points[1]:  # Ensure that only relevant stationary points are considered
#                 draw_heatmap = np.zeros((transformed_frame.shape[0], transformed_frame.shape[1]), dtype=np.uint8)
#                 draw_blob(draw_heatmap, tuple(points[0]), points[1])
#                 heatmap = cv2.add(heatmap, draw_heatmap)

#         lo = np.array([color_start])
#         hi = np.array([255])
#         mask = cv2.inRange(heatmap, lo, hi)
#         heatmap[mask > 0] = color_start

#         heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#         lo = np.array([128,0,0])
#         hi = np.array([136,0,0])
#         mask = cv2.inRange(heatmap, lo, hi)
#         heatmap[mask > 0] = (0, 0, 0)

#         for row in range(heatmap.shape[0]):
#             for col in range(heatmap.shape[1]):
#                 if (heatmap[row][col] == np.array([0,0,0])).all():
#                     heatmap[row][col] = transformed_frame[row][col]

#         heatmap_frame = cv2.addWeighted(heatmap, 0.75, transformed_frame, 0.25, 1)

#         print(aggregation_interval, frame_count)
#         # Aggregate the heatmap over aggregation_interval frames
#         if frame_count % aggregation_interval == 0:
#             if aggregated_heatmap is None:
#                 aggregated_heatmap = np.zeros_like(heatmap, dtype=np.float64)
#             aggregated_heatmap += heatmap.astype(np.float64)
#             interval_count += 1

#             # Save the aggregated heatmap image every aggregation_interval frames
#             if interval_count == aggregation_interval:
#                 aggregated_heatmap /= aggregation_interval  # Normalize the aggregated heatmap
#                 heatmap_scores = calculate_grid_heatmap_scores(aggregated_heatmap.astype(np.uint8), grid_size)
#                 heatmap_image_path = os.path.join(heatmap_dir, f'heatmap_{frame_count:05d}.png')
#                 cv2.imwrite(heatmap_image_path, cv2.addWeighted(aggregated_heatmap.astype(np.uint8), 0.75, transformed_frame, 0.25, 1))

#                 # Write heatmap scores to CSV file
#                 csv_writer.writerow([timestamp] + heatmap_scores)

#                 # Reset aggregation variables
#                 aggregated_heatmap = None
#                 interval_count = 0

# cap.release()
# cv2.destroyAllWindows()
# print("Processing complete.")






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
import os

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

def transform_perspective(frame, src_points, dst_points):
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    return cv2.warpPerspective(frame, M, (frame.shape[1], frame.shape[0]))

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

# Perspective transformation points
src_points = np.float32([[0, 0], [frame_size, 0], [frame_size, frame_size], [0, frame_size]])
dst_points = np.float32([[0, 0], [frame_size, 0], [frame_size, frame_size], [0, frame_size]])  # Modify this according to the desired perspective

# Aggregation interval in frames
aggregation_interval = 10

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

# Create output directory for heatmap images
heatmap_dir = 'heatmap_images'
os.makedirs(heatmap_dir, exist_ok=True)

# Open CSV file for writing heatmap scores
with open('./processed_data/heatmap_scores.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    header = ["Timestamp"] + [get_cell_name(i // grid_size, i % grid_size) for i in range(grid_size**2)]
    csv_writer.writerow(header)

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    aggregated_heatmap = None
    interval_count = 0

    for _ in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        timestamp = frame_count / vid_fps

        # Apply perspective transformation to the entire frame
        transformed_frame = transform_perspective(frame, src_points, dst_points)

        heatmap = np.zeros((transformed_frame.shape[0], transformed_frame.shape[1]), dtype=np.uint8)
        for points in stationary_points:
            if frame_count >= points[1]:  # Ensure that only relevant stationary points are considered
                draw_heatmap = np.zeros((transformed_frame.shape[0], transformed_frame.shape[1]), dtype=np.uint8)
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
                    heatmap[row][col] = transformed_frame[row][col]

        heatmap_frame = cv2.addWeighted(heatmap, 0.75, transformed_frame, 0.25, 1)

        # Aggregate the heatmap over aggregation_interval frames
        if aggregated_heatmap is None:
            aggregated_heatmap = np.zeros_like(heatmap, dtype=np.float64)
        aggregated_heatmap += heatmap.astype(np.float64)
        interval_count += 1

        # Save the aggregated heatmap image every aggregation_interval frames
        if interval_count == aggregation_interval:
            aggregated_heatmap /= aggregation_interval  # Normalize the aggregated heatmap
            heatmap_scores = calculate_grid_heatmap_scores(aggregated_heatmap.astype(np.uint8), grid_size)
            heatmap_image_path = os.path.join(heatmap_dir, f'heatmap_{frame_count:05d}.png')
            cv2.imwrite(heatmap_image_path, cv2.addWeighted(aggregated_heatmap.astype(np.uint8), 0.75, transformed_frame, 0.25, 1))

            # Write heatmap scores to CSV file
            csv_writer.writerow([timestamp] + heatmap_scores)

            # Reset aggregation variables
            aggregated_heatmap = None
            interval_count = 0

cap.release()
cv2.destroyAllWindows()
print("Processing complete.")
