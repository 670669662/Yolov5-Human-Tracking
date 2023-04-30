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
from mtcnn import MTCNN

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

pts = [dict(points=deque(maxlen=None), length=0.0) for _ in range(9999)]
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

cfg = get_config()
with open("deep_sort/configs/deep_sort.yaml", 'r') as f:
    deep_sort_cfg = yaml.safe_load(f)
    cfg.update(deep_sort_cfg)
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)
detector = MTCNN()
input_video_path = "E:\\Master Works\\Urban Sensing\\03.Code\\outpy.avi"
output_video_path = "E:\\Master Works\\Urban Sensing\\03.Code\\result2.mp4"
cap = cv2.VideoCapture(input_video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

start_time = time.time()
current_time = time.time()
frame_count = 0

os.makedirs("E:\\Master Works\\Urban Sensing\\03.Code\\image", exist_ok=True)
os.makedirs("E:\\Master Works\\Urban Sensing\\03.Code\\csv", exist_ok=True)

last_clear_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Add the following code for face detection and blurring
    face_locations = detector.detect_faces(frame)

    for face_location in face_locations:
        x, y, w, h = face_location['box']
        top, right, bottom, left = y, x + w, y + h, x
        face_image = frame[top:bottom, left:right]
        face_image = cv2.GaussianBlur(face_image, (99, 99), 30)
        frame[top:bottom, left:right] = face_image
    # End of face detection and blurring

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

            for i in range(len(pts)):
                color = [int(c) for c in COLORS[i % len(COLORS)]]
                for j in range(1, len(pts[i]['points'])):
                    if pts[i]['points'][j - 1] is None or pts[i]['points'][j] is None:
                        continue
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
    # 每一分钟保存数据
    if frame_count % (fps * 60) == 0:
        current_time = time.time()
        csv_filename = f"E:\\Master Works\\Urban Sensing\\03.Code\\csv\\trails_{int(current_time)}.csv"
        with open(csv_filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['ID', 'Trail_Length'])
            for i in track_id:
                csv_writer.writerow([i, pts[i]['length']])

        # 保存轨迹线结合视频图片
        trail_img = frame.copy()
        trail_img_name = f"E:\\Master Works\\Urban Sensing\\03.Code\\image\\trail_{int(current_time)}_combined.jpg"
        cv2.imwrite(trail_img_name, trail_img)

        # 保存单独的轨迹线png
        trail_only_img = np.zeros_like(frame)
        for i in track_id:
            for j in range(1, len(pts[i]['points'])):
                color = [int(c) for c in COLORS[i % len(COLORS)]]
                thickness = 2
                try:
                    cv2.line(trail_only_img, (pts[i]['points'][j - 1]), (pts[i]['points'][j]), color, thickness)
                except:
                    pass
        trail_only_img_name = f"E:\\Master Works\\Urban Sensing\\03.Code\\image\\trail_{int(current_time)}_only.png"
        cv2.imwrite(trail_only_img_name, trail_only_img)

        # 清空轨迹
        if time.time() - last_clear_time >= 60:
            current_time = int(time.time())
            trail_img_name = f"E:\\Master Works\\Urban Sensing\\03.Code\\image\\trail_{current_time}_combined.jpg"
            cv2.imwrite(trail_img_name, frame)

            trail_only_img = np.zeros_like(frame)
            for i in track_id:
                for j in range(1, len(pts[i]['points'])):
                    color = [int(c) for c in COLORS[i % len(COLORS)]]
                    thickness = 2
                    try:
                        cv2.line(trail_only_img, (pts[i]['points'][j - 1]), (pts[i]['points'][j]), color, thickness)
                    except:
                        pass

            trail_only_img_name = f"E:\\Master Works\\Urban Sensing\\03.Code\\image\\trail_{current_time}_only.png"
            cv2.imwrite(trail_only_img_name, trail_only_img)

            for i in track_id:
                pts[i]['points'].clear()
            last_clear_time = time.time()

            # 暂停一下视频处理，确保保存的轨迹与实际轨迹一致
            time.sleep(1)

            # 重新绘制当前帧的轨迹线
            for i in range(len(pts)):
                color = [int(c) for c in COLORS[i % len(COLORS)]]
                for j in range(1, len(pts[i]['points'])):
                    if pts[i]['points'][j - 1] is None or pts[i]['points'][j] is None:
                        continue
                    thickness = 2
                    try:
                        cv2.line(frame, (pts[i]['points'][j - 1]), (pts[i]['points'][j]), color, thickness)
                    except:
                        pass


cap.release()
writer.release()
cv2.destroyAllWindows()

