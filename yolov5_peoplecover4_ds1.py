#不需要摄像头参数
import torch
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device
import numpy as np
import cv2
import time
from deep_sort_pytorch.deep_sort import DeepSort
import math
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont

YOLO_WEIGHTS = 'yolov5s.pt'
DEEPSORT_MODEL_PATH = 'deepsort/deep_sort_pytorch/ckpt.t7'

# 聚集判断参数
RELATIVE_DISTANCE_THRESHOLD = 0.7  # 相对距离阈值（基于人体高度）
IOU_THRESHOLD = 0.1  # 重叠阈值（用于检测重叠的人员）
PEOPLE_THRESHOLD = 2  # 聚集人数阈值
TIME_THRESHOLD = 1  # 持续聚集时间阈值（秒）


# -------------------------- 自定义letterbox函数 --------------------------
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    """自定义letterbox：保持宽高比缩放图像，并填充到目标尺寸（YOLOv5标准预处理）"""
    shape = img.shape[:2]  # 原始尺寸 (h, w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # 计算缩放比例
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # 不允许放大，只缩小
        r = min(r, 1.0)

    # 计算缩放后的尺寸和填充
    ratio = r, r  # 宽高缩放比例一致（保持宽高比）
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # (w, h)
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # 填充量
    if auto:  # 自动填充（两侧平均填充）
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # 填充量必须是32的倍数（YOLOv5要求）
    else:
        dw, dh = 0, 0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2  # x方向两侧填充（单侧量）
    dh /= 2  # y方向两侧填充（单侧量）

    # 定义pad变量
    pad = (dw, dh)  # (x方向单侧填充量, y方向单侧填充量)

    if shape[::-1] != new_unpad:  # 缩放图像
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # 填充
    return img, ratio, pad


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # 复制自YOLOv5源码，用于坐标缩放（将模型输出坐标映射到原始图像）
    if ratio_pad is None:  # 计算从img0到img1的缩放比例和 padding
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # 缩放比例
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]  # 这里需要ratio_pad是一个包含(ratio, pad)的元组
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)  # 确保坐标在图像范围内
    return coords


def clip_coords(boxes, img_shape):
    # 裁剪边界框到图像尺寸内
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


# -------------------------- 辅助函数 --------------------------
def calculate_iou(box1, box2):
    """计算两个边界框的交并比(IoU)"""
    # 解包边界框坐标
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # 计算交集区域
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    # 如果没有交集
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # 计算交集面积
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # 计算并集面积
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - intersection_area

    # 计算IoU
    iou = intersection_area / union_area if union_area > 0 else 0

    return iou


def estimate_relative_distance(bbox_height1, bbox_height2, pixel_distance):
    """
    估算两个人之间的相对距离（基于人体高度）
    使用两个人高度的几何平均值作为参考尺度
    """
    # 计算平均高度（使用几何平均以减少极端值的影响）
    avg_height = math.sqrt(bbox_height1 * bbox_height2)

    # 避免除以零
    if avg_height == 0:
        return float('inf')

    # 相对距离 = 像素距离 / 平均高度
    relative_distance = pixel_distance / avg_height

    return relative_distance


def estimate_perspective_corrected_position(bottom_center_x, bottom_center_y, bbox_height, img_height):
    """
    估算透视校正后的位置
    假设图像底部的人更近，顶部的人更远
    通过y坐标进行简单的透视校正
    """
    # 计算y坐标的归一化值（0在顶部，1在底部）
    normalized_y = bottom_center_y / img_height

    # 透视校正因子：底部的人需要更大的权重
    # 使用非线性函数来模拟透视效果
    perspective_factor = 1.0 / (1.0 - 0.5 * normalized_y)  # 简单的透视校正

    # 应用透视校正
    corrected_x = bottom_center_x * perspective_factor
    corrected_y = bottom_center_y * perspective_factor

    return corrected_x, corrected_y, perspective_factor


# -------------------------- 模型初始化 --------------------------
def init_models(yolo_weights=YOLO_WEIGHTS, deepsort_model=DEEPSORT_MODEL_PATH, device='0'):
    """初始化YOLOv5和DeepSort模型"""
    # 初始化YOLOv5
    device = torch.device(f'cuda:{device}' if torch.cuda.is_available() and device != 'cpu' else 'cpu')
    yolo_model = attempt_load(yolo_weights, device=device)
    yolo_model.eval()

    # 初始化DeepSort
    deepsort_tracker = DeepSort(deepsort_model)

    return yolo_model, deepsort_tracker, device


def detect_persons(frame, model, device):
    """YOLOv5检测人员，修复图像预处理步骤"""
    # 关键修复：同时获取缩放比例和填充信息
    img, ratio, pad = letterbox(frame, new_shape=640)  # 获取三个返回值

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0
    img = img.permute(2, 0, 1)  # HWC -> CHW
    if img.ndimension() == 3:
        img = img.unsqueeze(0)  # 增加批次维度

    # 推理与后处理
    pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.45, classes=[0])  # 只检测人

    # 提取检测结果
    persons = []
    for det in pred:
        if len(det):
            # 修复：传递正确格式的ratio_pad参数 (ratio, pad)
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape,
                                      ratio_pad=(ratio, pad)).round()
            for *xyxy, conf, cls in reversed(det):
                x1, y1, x2, y2 = map(int, xyxy)
                persons.append((x1, y1, x2, y2, float(conf)))
    return persons


def track_persons(frame, detections, deepsort_tracker):
    """DeepSort跟踪人员（适配需要classes参数的版本）"""
    if not detections:
        return {}

    # 转换格式：(x1,y1,x2,y2,conf) -> (cx, cy, w, h, conf)
    bbox_xywh = []
    confidences = []
    classes = []
    for (x1, y1, x2, y2, conf) in detections:
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2
        bbox_xywh.append([cx, cy, w, h])
        confidences.append(conf)
        classes.append(0)  # 0是COCO数据集中"person"的类别ID

    # 转换为numpy数组
    bbox_xywh = np.array(bbox_xywh, dtype=np.float32)
    confidences = np.array(confidences, dtype=np.float32)
    classes = np.array(classes, dtype=np.int32)

    # 更新跟踪器
    outputs, _ = deepsort_tracker.update(bbox_xywh, confidences, classes, frame)

    # 整理结果：{track_id: (x1,y1,x2,y2)}
    tracked = {}
    for output in outputs:
        x1, y1, x2, y2, track_cls, track_id = output
        if track_cls == 0:
            tracked[track_id] = (int(x1), int(y1), int(x2), int(y2))
    return tracked


def judge_gathering(tracked_persons, history, gathering_start_time, current_time, frame_shape):
    """判断是否存在人员聚集（基于相对距离、透视校正和IoU）"""
    h, w = frame_shape[:2]

    # 记录当前位置和透视校正后的位置
    person_info = {}  # track_id -> (corrected_x, corrected_y, bbox_height, bbox)
    for track_id, (x1, y1, x2, y2) in tracked_persons.items():
        # 使用底部中心点作为人的位置
        bottom_center_x = (x1 + x2) / 2
        bottom_center_y = y2

        bbox_height = y2 - y1

        # 应用透视校正
        corrected_x, corrected_y, perspective_factor = estimate_perspective_corrected_position(
            bottom_center_x, bottom_center_y, bbox_height, h
        )

        person_info[track_id] = (corrected_x, corrected_y, bbox_height, perspective_factor, (x1, y1, x2, y2))
        history[track_id].append((corrected_x, corrected_y, current_time))

        # 只保留最近100条记录
        if len(history[track_id]) > 100:
            history[track_id].pop(0)

    # 人数不足直接返回
    if len(tracked_persons) < PEOPLE_THRESHOLD:
        # 清除聚集开始时间
        for track_id in tracked_persons.keys():
            if track_id in gathering_start_time:
                del gathering_start_time[track_id]
        return False, []

    # 提取所有人员校正后的位置和边界框
    corrected_positions = [info[:2] for info in person_info.values()]
    bbox_heights = [info[2] for info in person_info.values()]
    bboxes = [info[4] for info in person_info.values()]  # 提取边界框
    track_ids = list(tracked_persons.keys())

    # 计算相对距离矩阵和IoU矩阵，判断聚集簇
    gathering_ids = []
    for i in range(len(corrected_positions)):
        cluster = [i]
        for j in range(i + 1, len(corrected_positions)):
            # 计算像素距离
            dx = corrected_positions[i][0] - corrected_positions[j][0]
            dy = corrected_positions[i][1] - corrected_positions[j][1]
            pixel_distance = math.hypot(dx, dy)

            # 计算相对距离（基于人体高度）
            relative_distance = estimate_relative_distance(
                bbox_heights[i], bbox_heights[j], pixel_distance
            )

            # 计算IoU（用于检测重叠的人员）
            iou = calculate_iou(bboxes[i], bboxes[j])

            # 如果相对距离小于阈值或者IoU大于阈值（表示重叠），则认为是聚集
            if relative_distance < RELATIVE_DISTANCE_THRESHOLD or iou > IOU_THRESHOLD:
                cluster.append(j)

        # 超过阈值人数的簇判定为聚集
        if len(cluster) >= PEOPLE_THRESHOLD:
            gathering_ids.extend([track_ids[k] for k in cluster])

    # 更新聚集开始时间
    # 1. 为新进入聚集的ID设置开始时间
    for track_id in gathering_ids:
        if track_id not in gathering_start_time:
            gathering_start_time[track_id] = current_time  # 记录进入聚集的时间

    # 2. 清除不再聚集的ID的开始时间
    for track_id in list(gathering_start_time.keys()):
        if track_id not in gathering_ids:
            del gathering_start_time[track_id]

    # 判断持续时间
    if gathering_ids and gathering_start_time:
        # 取最早进入聚集的时间
        earliest_time = min(gathering_start_time.values())
        if current_time - earliest_time >= TIME_THRESHOLD:
            return True, gathering_ids

    return False, []


def put_chinese_text(img, text, position, font_size=1, color=(0, 0, 255)):
    """
    在图像上绘制中文文本（修复版）
    :param img: OpenCV图像(numpy array)
    :param text: 要绘制的文本
    :param position: 位置 (x, y)
    :param font_size: 字体大小
    :param color: 颜色 (B, G, R)
    :return: 绘制了文本的图像
    """
    # 转换颜色空间 (BGR -> RGB)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 转换为PIL Image对象（关键修复）
    pil_img = Image.fromarray(rgb_img)
    # 创建绘图对象
    draw = ImageDraw.Draw(pil_img)

    # 加载中文字体
    try:
        # Windows系统默认字体
        font = ImageFont.truetype("simhei.ttf", int(font_size * 20))
    except:
        try:
            # Linux系统默认字体
            font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", int(font_size * 20))
        except:
            #  fallback字体
            font = ImageFont.load_default()

    # 绘制文本 (注意颜色需要转换为RGB格式)
    draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))

    # 转换回OpenCV格式 (RGB -> BGR)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# -------------------------- 主函数 --------------------------
def main(video_path=0, output_path='D:/dataset/Collective Activity/data/ActivityDataset/seq01/output.mp4'):
    # 初始化模型
    yolo_model, deepsort_tracker, device = init_models()
    print(f"使用设备: {device}")

    # 初始化视频流
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频源")
        return
    # 获取视频基本信息用于保存输出视频
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4格式编码
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    # 用于记录跟踪历史（位置）
    track_history = defaultdict(list)
    # 新增：记录每个ID开始聚集的时间
    gathering_start_time = dict()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("视频读取结束")
            break

        current_time = time.time()

        # 1. 检测人员
        detections = detect_persons(frame, yolo_model, device)

        # 2. 跟踪人员
        tracked_persons = track_persons(frame, detections, deepsort_tracker)

        # 3. 判断聚集（传入gathering_start_time记录开始时间和帧形状）
        is_gathering, gathering_ids = judge_gathering(
            tracked_persons, track_history, gathering_start_time, current_time, frame.shape
        )

        # 4. 可视化
        # 绘制所有跟踪目标
        for track_id, (x1, y1, x2, y2) in tracked_persons.items():
            color = (0, 255, 0) if track_id not in gathering_ids else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 绘制底部中心点（用于距离计算）
            bottom_center_x = (x1 + x2) // 2
            bottom_center_y = y2
            cv2.circle(frame, (bottom_center_x, bottom_center_y), 5, (255, 0, 0), -1)

            # 绘制ID（英文，用默认字体即可）
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 显示相对距离信息（仅用于调试）
            if track_id in gathering_ids:
                cv2.putText(frame, "CLOSE", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 显示聚集警告（中文，使用自定义函数）
        if is_gathering:
            frame = put_chinese_text(frame, "警告: 人员聚集!", (50, 50),
                                     font_size=1, color=(0, 0, 255))

            # 在画面底部显示提示信息
            frame = put_chinese_text(frame, "检测到人员过于接近，请保持安全距离",
                                     (50, frame.shape[0] - 50), font_size=0.8, color=(0, 0, 255))

        # 显示调试信息
        frame = put_chinese_text(frame, f"人员数量: {len(tracked_persons)}", (10, 30),
                                 font_size=0.7, color=(255, 255, 255))

        if is_gathering:
            frame = put_chinese_text(frame, f"聚集人数: {len(gathering_ids)}", (10, 60), font_size=0.7,
                                     color=(0, 0, 255))

        # 写入视频帧
        out.write(frame)
        # 显示画面
        cv2.imshow("人员聚集检测", frame)


        # 按q退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main("D:/dataset/Collective Activity/data/ActivityDataset/seq01/yolov5_input.mp4")