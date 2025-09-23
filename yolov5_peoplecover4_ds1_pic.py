import torch
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device
import numpy as np
import cv2
import math
from PIL import Image, ImageDraw, ImageFont

YOLO_WEIGHTS = 'yolov5s.pt'

# 聚集判断参数（保持原有参数不变）
RELATIVE_DISTANCE_THRESHOLD = 0.7  # 相对距离阈值（基于人体高度）
IOU_THRESHOLD = 0.1  # 重叠阈值（用于检测重叠的人员）
PEOPLE_THRESHOLD = 2  # 聚集人数阈值


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
def init_model(yolo_weights=YOLO_WEIGHTS, device='0'):
    """初始化YOLOv5模型（单张图片不需要跟踪，移除DeepSort）"""
    device = torch.device(f'cuda:{device}' if torch.cuda.is_available() and device != 'cpu' else 'cpu')
    model = attempt_load(yolo_weights, device=device)
    model.eval()
    return model, device


def detect_persons(frame, model, device):
    """YOLOv5检测人员，保持原有逻辑"""
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
            # 传递正确格式的ratio_pad参数 (ratio, pad)
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape,
                                      ratio_pad=(ratio, pad)).round()
            for *xyxy, conf, cls in reversed(det):
                x1, y1, x2, y2 = map(int, xyxy)
                persons.append((x1, y1, x2, y2, float(conf)))
    return persons


def judge_gathering(persons, frame_shape):
    """判断是否存在人员聚集（移除时间相关逻辑，保留核心判断）"""
    h, w = frame_shape[:2]

    # 记录当前位置和透视校正后的位置
    person_info = {}  # index -> (corrected_x, corrected_y, bbox_height, bbox)
    for i, (x1, y1, x2, y2, _) in enumerate(persons):
        # 使用底部中心点作为人的位置
        bottom_center_x = (x1 + x2) / 2
        bottom_center_y = y2

        bbox_height = y2 - y1

        # 应用透视校正
        corrected_x, corrected_y, perspective_factor = estimate_perspective_corrected_position(
            bottom_center_x, bottom_center_y, bbox_height, h
        )

        person_info[i] = (corrected_x, corrected_y, bbox_height, perspective_factor, (x1, y1, x2, y2))

    # 人数不足直接返回
    if len(persons) < PEOPLE_THRESHOLD:
        return False, []

    # 提取所有人员校正后的位置和边界框
    corrected_positions = [info[:2] for info in person_info.values()]
    bbox_heights = [info[2] for info in person_info.values()]
    bboxes = [info[4] for info in person_info.values()]

    # 计算相对距离矩阵和IoU矩阵，判断聚集簇
    gathering_indices = []
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
            gathering_indices.extend(cluster)

    # 去重并返回结果
    gathering_indices = list(set(gathering_indices))
    return len(gathering_indices) >= PEOPLE_THRESHOLD, gathering_indices


def put_chinese_text(img, text, position, font_size=1, color=(0, 0, 255)):
    """在图像上绘制中文文本"""
    # 转换颜色空间 (BGR -> RGB)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 转换为PIL Image对象
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
            # fallback字体
            font = ImageFont.load_default()

    # 绘制文本 (注意颜色需要转换为RGB格式)
    draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))

    # 转换回OpenCV格式 (RGB -> BGR)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# -------------------------- 主函数 --------------------------
def main(image_path, output_path='D:/dataset/Collective Activity/data/ActivityDataset/seq01/frame0001output_result.jpg'):
    # 初始化模型（移除DeepSort跟踪器）
    yolo_model, device = init_model()
    print(f"使用设备: {device}")

    # 读取图片
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"无法读取图片: {image_path}")
        return

    # 1. 检测人员
    detections = detect_persons(frame, yolo_model, device)

    # 2. 判断聚集（使用检测到的人员直接判断，无需跟踪）
    is_gathering, gathering_indices = judge_gathering(detections, frame.shape)

    # 3. 可视化
    # 绘制所有检测到的人员
    for i, (x1, y1, x2, y2, conf) in enumerate(detections):
        # 聚集人员标红，其他标绿
        color = (0, 0, 255) if i in gathering_indices else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # 绘制底部中心点（用于距离计算）
        bottom_center_x = (x1 + x2) // 2
        bottom_center_y = y2
        cv2.circle(frame, (bottom_center_x, bottom_center_y), 5, (255, 0, 0), -1)

        # 显示置信度
        cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 显示聚集状态
        if i in gathering_indices:
            cv2.putText(frame, "CLOSE", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # 显示聚集警告（中文）
    if is_gathering:
        frame = put_chinese_text(frame, "警告: 人员聚集!", (50, 50),
                                 font_size=1, color=(0, 0, 255))
    else:
        frame = put_chinese_text(frame, "状态: 正常", (50, 50),
                                 font_size=1, color=(0, 255, 0))

    # 保存处理结果到output_path
    cv2.imwrite(output_path, frame)
    print(f"处理结果已保存至: {output_path}")


if __name__ == "__main__":
    # 示例调用
    import sys
    if len(sys.argv) > 1:
        main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'output_result.jpg')
    else:
        # 默认测试图片路径
        main('D:/dataset/Collective Activity/data/ActivityDataset/seq01/frame0001.jpg')
