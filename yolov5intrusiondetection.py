import cv2
# from yolov5 import YOLOv5
import numpy as np
import torch
from line_draw import person_in_poly_area_dangerous

# 加载预训练的YOLOv5模型
# model = YOLOv5("yolov5s.pt", device='cuda')  # 选择模型
model = torch.hub.load('.',
                      'yolov5s',
                       pretrained=True,
                       source='local')  # or yolov5m, yolov5l, yolov5x, custom
# 打开摄像头
cap = cv2.VideoCapture(0)
while True:
    # 从摄像头读取帧
    ret, frame = cap.read()
    if not ret:
        break
    # 使用YOLOv5进行目标检测
    results = model(frame)
    area_poly = np.array([[100, 100], [300, 100], [400, 300], [200, 400], [50, 250]], np.int32)
    cv2.polylines(frame,
                  [area_poly],  # 顶点列表（注意外层加[]）
                  isClosed=True,  # 闭合多边形
                  color=[0, 0, 255],  # 红色线条（BGR）
                  thickness=2  # 线条粗细
                  )

    # 在帧上绘制检测结果
    for *xyxy, conf, cls in results.xyxy[0]:
        label = f'{model.model.names[int(cls)]} {conf:.2f}'
        if cls == 0 and person_in_poly_area_dangerous(xyxy, area_poly) == True:
            cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 0, 255), 2)
            cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    # 显示帧
    cv2.imshow('YOLOv5 Real-time Object Detection', frame)
    # 按'q'键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# 释放资源并关闭窗口
cap.release()
cv2.destroyAllWindows()