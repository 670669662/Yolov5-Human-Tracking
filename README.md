# Yolov5-Human-Tracking
This project utilizes the YOLOv5 object detection algorithm and the DeepSORT object tracking algorithm to detect and track people in a video. By analyzing the movement trajectories of each person in the video, the project can calculate their dwell time and other related information.

## Method Overview:

1. **Perform frame-by-frame analysis** of the video using the YOLOv5 algorithm to recognize and detect people in the video.
2. **Apply the DeepSORT algorithm** to track the detected people, assigning a unique ID to each person.
3. **Calculate the center point** of each person and add it to the tracking trajectory. Trajectories are stored as dictionaries, with the person's ID as the key.
4. **Draw bounding boxes and trajectory lines** for each person in the video.
5. **Periodically save information** such as the trajectory length, dwell time, and average speed of the people to a CSV file (e.g., every 10 seconds).
6. **Facial Blurring** using MHCNN (Multi-Task Hierarchical Convolutional Neural Network) to blur faces for privacy protection. However, if you are using an infrared camera, this step can be skipped.

7. **Save images** containing the person's trajectory and images containing only the trajectory as image files.

![1-01](https://user-images.githubusercontent.com/70087271/235361681-4d411676-9022-4f80-83ad-e6c7969a4e54.jpg)
![2-01](https://user-images.githubusercontent.com/70087271/235361683-8f57c100-1e85-48bd-9fdc-76ba298e6921.jpg)
## Required Libraries:

- `os`: For operating system-related functions.
- `time`: For handling time-related functions.
- `numpy`: For numerical operations and array manipulation.
- `detect`: Custom module for object detection using YOLOv5.
- `cv2` (OpenCV): For image and video processing.
- `csv`: For reading and writing CSV files.
- `deep_sort`: Custom module for object tracking using the DeepSORT algorithm.
- `collections`: For using `deque` data structure.
- `yaml`: For parsing YAML configuration files.

## Result demonstration
![4_画板 1](https://user-images.githubusercontent.com/70087271/235362239-fe5df122-bb43-4419-9619-64b2372f9e8d.jpg)
**Camera usage advice**
1. **720p Webcam (RGB):** Ample outdoor lighting, allowing for capturing full-body shots.
2. **120p Infrared Camera (Gray):** An indoor environment with low room temperature, or an outdoor environment at night. Avoid brightly lit situations.
3. **120p Infrared Camera (Normal):** An indoor environment with low or medium room temperature, or an outdoor environment at night. Avoid brightly lit situations

![3-01](https://user-images.githubusercontent.com/70087271/235364248-50f863d9-aa36-498c-9a41-803314acabc0.jpg)

PS:Thanks to George Verghese and Jam Kim for their contributions during the video collection



