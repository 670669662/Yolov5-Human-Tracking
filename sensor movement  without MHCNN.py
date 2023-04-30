import os
import time
import numpy as np
import detect
import cv2
import csv
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from collections import deque
import yaml

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

pts = [dict(points=deque(maxlen=None), length=0.0, time=0, speed=[]) for _ in range(9999)]
COLORS = np.random.randint(0, 255, size=(80, 3), dtype="uint8")

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

def compute_color_for_labels(label):
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def draw_boxes(img, bbox, identities=None, offset=(0,0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img
def draw_trails_only(trails, track_ids, img):
    for i in track_ids:
        for j in range(1, len(trails[i]['points'])):
            color = [int(c) for c in COLORS[i % len(COLORS)]]
            thickness = 2
            try:
                cv2.line(img, (trails[i]['points'][j - 1]), (trails[i]['points'][j]), color, thickness)
            except:
                pass
    return img

cfg = get_config()
with open("deep_sort/configs/deep_sort.yaml", 'r') as f:
    deep_sort_cfg = yaml.safe_load(f)
    cfg.update(deep_sort_cfg)
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

input_video_path = "E:\\Master Works\\Urban Sensing\\03.Code\\1st floor of Avery.mp4"
output_video_path = "E:\\Master Works\\Urban Sensing\\03.Code\\result.mp4"
cap = cv2.VideoCapture(input_video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'XVID')


writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

start_time = time.time()
file_index = time.time()
frame_count = 0

os.makedirs("E:\\Master Works\\Urban Sensing\\03.Code\\image", exist_ok=True)
os.makedirs("E:\\Master Works\\Urban Sensing\\03.Code\\csv", exist_ok=True)

last_clear_time = time.time()


while True:
    ret, frame = cap.read()
    if not ret:
        break

    boxes, xywhs, confss = detect.recognition(frame)

    track_id = []
    if len(boxes) > 0:
        outputs = deepsort.update(xywhs, confss, frame)

        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -1]
            draw_boxes(frame, bbox_xyxy, identities)

            for i in range(0, len(outputs)):
                id = outputs[i][4]
                if id not in track_id:
                    track_id.append(id)

                center_x = int(outputs[i][0] + 0.5 * (outputs[i][2] - outputs[i][0]))
                center_y = int(outputs[i][1] + 0.5 * (outputs[i][3] - outputs[i][1]))

                if len(pts[id]['points']) > 1:
                    if np.sqrt((center_x - pts[id]['points'][-1][0]) ** 2 + (center_y - pts[id]['points'][-1][1]) ** 2) < 50:
                        pts[id]['points'].append((center_x, center_y))
                        pts[id]['length'] += np.sqrt((center_x - pts[id]['points'][-2][0]) ** 2 + (center_y - pts[id]['points'][-2][1]) ** 2)
                    else:
                        pts[id]['points'] = deque([(center_x, center_y)], maxlen=None)
                        pts[id]['length'] = 0.0
                else:
                    pts[id]['points'].append((center_x, center_y))

    # Draw the trajectory lines for all the tracked objects, even if they are not currently on screen
    for i in range(len(pts)):
        if len(pts[i]['points']) > 1:
            color = [int(c) for c in COLORS[i % len(COLORS)]]
            for j in range(1, len(pts[i]['points'])):
                thickness = 2
                try:
                    cv2.line(frame, (pts[i]['points'][j - 1]), (pts[i]['points'][j]), color, thickness)
                except:
                    pass

    cv2.imshow('', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    writer.write(frame)

    frame_count += 1

    # save csv and image by 10 seconds
    if frame_count % (fps * 10) == 0:
        file_index = frame_count // (fps * 5)
        csv_filename = f"E:\\Master Works\\Urban Sensing\\03.Code\\csv\\trails_{file_index}.csv"
        with open(csv_filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['ID', 'Trail_Length', 'Exist_Time', 'Avg_Speed'])

            # add data to csv
            for i in track_id:
                exist_time = frame_count / fps
                avg_speed = (pts[i]['length'] / exist_time) if exist_time > 0 else 0
                csv_writer.writerow([i, pts[i]['length'], exist_time, avg_speed])

        # save trail combined with video image
        trail_img = frame.copy()
        trail_img_name = f"E:\\Master Works\\Urban Sensing\\03.Code\\image\\trail_{file_index}_combined.jpg"
        cv2.imwrite(trail_img_name, trail_img)

        # save trail only png
        trail_only_img = np.zeros_like(frame)
        draw_trails_only(pts, track_id, trail_only_img)
        trail_only_img_name = f"E:\\Master Works\\Urban Sensing\\03.Code\\image\\trail_{file_index}_only.png"
        cv2.imwrite(trail_only_img_name, trail_only_img)

    # remove the tack line by 5 mins
    if frame_count % (fps * 60 * 5) == 0:
        for i in track_id:
            pts[i]['points'].clear()
            pts[i]['length'] = 0.0
            pts[i]['speed'] = []

        # save trail combined with video image
        trail_img = frame.copy()
        trail_img_name = f"E:\\Master Works\\Urban Sensing\\03.Code\\image\\trail_{file_index}_combined.jpg"
        cv2.imwrite(trail_img_name, trail_img)

        # save trail only png
        trail_only_img = np.zeros_like(frame)
        for i in track_id:
            for j in range(1, len(pts[i]['points'])):
                color = [int(c) for c in COLORS[i % len(COLORS)]]
                thickness = 2
                try:
                    cv2.line(trail_only_img, (pts[i]['points'][j - 1]), (pts[i]['points'][j]), color, thickness)
                except:
                    pass
        trail_only_img_name = f"E:\\Master Works\\Urban Sensing\\03.Code\\image\\trail_{file_index}_only.png"
        cv2.imwrite(trail_only_img_name, trail_only_img)
        for i in track_id:
            pts[i]['points'].clear()
            pts[i]['length'] = 0.0
            pts[i]['speed'] = []

cap.release()
writer.release()
cv2.destroyAllWindows()