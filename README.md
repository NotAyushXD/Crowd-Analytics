# Crowd-Analysis

The project is dedicated to apply on CCTV and other survailance system for simple crowd monitoring and crowd analysis. 

Current functions implemented includes:

- Crowd movement tracks and flow
- Crowd stationaries point (Heatmap)

## Building

YOLOv4-tiny is used for this documentation. You can use other YOLO variation for desire usage and output.

### Requirements

Install the requirements

```shell
pip3 install requirements.txt
```

## Configuration

`config.py` contains all configurations for this program.

Place the **video source** under `VIDEO_CONFIG.VIDEO_CAP` in `config.py`

Refer to [User Manual](#user-manual) on how to use the `config.py` file.

## Running

Before you run the program, make sure you have input a valid **video source**. You have to provide your own video for the program. Replace the path at `VIDEO_CONFIG.VIDEO_CAP` in `config.py` with the path of your own video.

To process a video, run `main.py`

```shell
python3 main.py
```

`main.py` will yield a set of data from the video source in the form of csv and json. These data will be placed in the directory `processed_data`.

From these data, you can generate movement data, crowd summary and abnormal crowd movement.

```shell
python3 crowd_data_present.py
python3 movement_data_present.py
```