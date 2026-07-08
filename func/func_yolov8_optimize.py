# 以下代码改自https://github.com/rockchip-linux/rknn-toolkit2/tree/master/examples/onnx/yolov5
import cv2
import numpy as np
from copy import copy
import time
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None

OBJ_THRESH, NMS_THRESH, IMG_SIZE = 0.25, 0.45, 640
out_win = "output_style_full_screen"
GRID_CACHE = {}
DFL_ACC_CACHE = {}
FONT_CACHE = {}
FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
)
CLASSES = (
    "饮水",
    "进食",
    "手套",
    "护目镜",
    "头罩",
    "实验服",
    "口罩",
    "未戴手套",
    "未戴头罩",
    "未穿实验服",
    "未戴口罩",
    "未戴护目镜",
    "烧杯",
    "布氏漏斗",
    "滴定管架",
    "量热仪",
    "锥形瓶",
    "漏斗",
    "玻璃棒",
    "量筒",
    "机械天平",
    "奈斯勒试剂瓶",
    "移液管",
    "瓷研钵和研杵",
    "精密天平",
    "试剂瓶",
    "单颈圆底烧瓶",
    "双颈圆底烧瓶",
    "三颈圆底烧瓶",
    "分液漏斗",
    "酒精灯",
    "试管夹",
    "试管",
    "容量瓶",
    "容量移液管",
    "洗瓶",
    "称量瓶",
)

DISPLAY_CLASSES = (
    "drinking",
    "eating",
    "gloves",
    "goggles",
    "hood",
    "lab_coat",
    "mask",
    "no_gloves",
    "no_hood",
    "no_lab_coat",
    "no_mask",
    "no_goggles",
    "beaker",
    "buchner_funnel",
    "burette_stand",
    "calorimeter",
    "erlenmeyer_flask",
    "funnel",
    "glass_rod",
    "graduated_cylinder",
    "mechanical_balance",
    "nessler_bottle",
    "pipette",
    "mortar_pestle",
    "analytical_balance",
    "reagent_bottle",
    "round_bottom_flask_1",
    "round_bottom_flask_2",
    "round_bottom_flask_3",
    "separatory_funnel",
    "alcohol_lamp",
    "test_tube_clamp",
    "test_tube",
    "volumetric_flask",
    "volumetric_pipette",
    "wash_bottle",
    "weighing_bottle",
)

INTERESTED_CLASSES = list(CLASSES)



CLASS_INDICES = {cls: idx for idx, cls in enumerate(CLASSES)}
INTERESTED_CLASS_INDICES = [CLASS_INDICES[cls] for cls in INTERESTED_CLASSES]
def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with object threshold.
    """
    box_confidences = box_confidences.reshape(-1)
    candidate, class_num = box_class_probs.shape

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score * box_confidences >= OBJ_THRESH)
    scores = (class_max_score * box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]

    return boxes, classes, scores


def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.
    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


# def dfl(position):
#     # Distribution Focal Loss (DFL)
#     import torch
#     x = torch.tensor(position)
#     n,c,h,w = x.shape
#     p_num = 4
#     mc = c//p_num
#     y = x.reshape(n,p_num,mc,h,w)
#     y = y.softmax(2)
#     acc_metrix = torch.tensor(range(mc)).float().reshape(1,1,mc,1,1)
#     y = (y*acc_metrix).sum(2)
#     return y.numpy()

# def dfl(position):
#     # Distribution Focal Loss (DFL)
#     n, c, h, w = position.shape
#     p_num = 4
#     mc = c // p_num
#     y = position.reshape(n, p_num, mc, h, w)
#     exp_y = np.exp(y)
#     y = exp_y / np.sum(exp_y, axis=2, keepdims=True)
#     acc_metrix = np.arange(mc).reshape(1, 1, mc, 1, 1).astype(float)
#     y = (y * acc_metrix).sum(2)
#     return y

def dfl(position):
    # Distribution Focal Loss (DFL)
    # x = np.array(position)
    n, c, h, w = position.shape
    p_num = 4
    mc = c // p_num
    y = position.reshape(n, p_num, mc, h, w)

    # Vectorized softmax
    e_y = np.exp(y - np.max(y, axis=2, keepdims=True))  # subtract max for numerical stability
    y = e_y / np.sum(e_y, axis=2, keepdims=True)

    if mc not in DFL_ACC_CACHE:
        DFL_ACC_CACHE[mc] = np.arange(mc, dtype=np.float32).reshape(1, 1, mc, 1, 1)
    acc_metrix = DFL_ACC_CACHE[mc]
    y = (y * acc_metrix).sum(2)
    return y


def box_process(position):
    grid_h, grid_w = position.shape[2:4]
    cache_key = (grid_h, grid_w)
    if cache_key not in GRID_CACHE:
        col, row = np.meshgrid(
            np.arange(0, grid_w, dtype=np.float32),
            np.arange(0, grid_h, dtype=np.float32),
        )
        col = col.reshape(1, 1, grid_h, grid_w)
        row = row.reshape(1, 1, grid_h, grid_w)
        grid = np.concatenate((col, row), axis=1)
        stride = np.array(
            [IMG_SIZE // grid_h, IMG_SIZE // grid_w],
            dtype=np.float32,
        ).reshape(1, 2, 1, 1)
        GRID_CACHE[cache_key] = (grid, stride)
    else:
        grid, stride = GRID_CACHE[cache_key]

    position = dfl(position)
    box_xy = grid + 0.5 - position[:, 0:2, :, :]
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
    xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)

    return xyxy


def yolov8_post_process(input_data):
    boxes, scores, classes_conf = [], [], []
    defualt_branch = 3
    pair_per_branch = len(input_data) // defualt_branch
    # Python 忽略 score_sum 输出
    for i in range(defualt_branch):
        boxes.append(box_process(input_data[pair_per_branch * i]))
        classes_conf.append(input_data[pair_per_branch * i + 1])
        scores.append(np.ones_like(input_data[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32))

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0, 2, 3, 1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]

    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)

    # filter according to threshold
    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

    # nms
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)

        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores

def draw_box_corner(draw_img, top, left, right, bottom, length, corner_color):
    # Top Left
    cv2.line(draw_img, (top, left), (top + length, left), corner_color, thickness=3)
    cv2.line(draw_img, (top, left), (top, left + length), corner_color, thickness=3)
    # Top Right
    cv2.line(draw_img, (right, left), (right - length, left), corner_color, thickness=3)
    cv2.line(draw_img, (right, left), (right, left + length), corner_color, thickness=3)
    # Bottom Left
    cv2.line(draw_img, (top, bottom), (top + length, bottom), corner_color, thickness=3)
    cv2.line(draw_img, (top, bottom), (top, bottom - length), corner_color, thickness=3)
    # Bottom Right
    cv2.line(draw_img, (right, bottom), (right - length, bottom), corner_color, thickness=3)
    cv2.line(draw_img, (right, bottom), (right, bottom - length), corner_color, thickness=3)


def get_chinese_font(font_size=28):
    if ImageFont is None:
        return None
    if font_size in FONT_CACHE:
        return FONT_CACHE[font_size]
    for font_path in FONT_CANDIDATES:
        if Path(font_path).exists():
            FONT_CACHE[font_size] = ImageFont.truetype(font_path, font_size)
            return FONT_CACHE[font_size]
    FONT_CACHE[font_size] = ImageFont.load_default()
    return FONT_CACHE[font_size]


def draw_label_type(draw_img, top, left, class_name, label_color, score=None):
    if score is not None:
        label = f"{class_name} {score:.2f}"
    else:
        label = str(class_name)

    if Image is None or ImageDraw is None:
        cv2.putText(draw_img, label, (top, max(20, left - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, thickness=2)
        return draw_img

    font = get_chinese_font(28)
    pil_img = Image.fromarray(cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    x = max(0, top)
    y = max(0, left - 34)

    bbox = draw.textbbox((x, y), label, font=font)
    pad = 4
    bg_box = (
        max(0, bbox[0] - pad),
        max(0, bbox[1] - pad),
        min(draw_img.shape[1], bbox[2] + pad),
        min(draw_img.shape[0], bbox[3] + pad),
    )
    draw.rectangle(bg_box, fill=(label_color[2], label_color[1], label_color[0]))
    draw.text((x, y), label, font=font, fill=(0, 0, 0))

    draw_img[:, :] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return draw_img


# def draw(image, boxes, scores, classes, ratio, padding):
#     for box, score, cl in zip(boxes, scores, classes):
#         top, left, right, bottom = box
#
#         top = (top - padding[0]) / ratio[0]
#         left = (left - padding[1]) / ratio[1]
#         right = (right - padding[0]) / ratio[0]
#         bottom = (bottom - padding[1]) / ratio[1]
#         # print('class: {}, score: {}'.format(CLASSES[cl], score))
#         # print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))
#         top = int(top)
#         left = int(left)
#
#         cv2.rectangle(image, (top, left), (int(right), int(bottom)), (255, 0, 0), 2)
#         cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
#                     (top, left - 6),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.6, (0, 0, 255), 2)

def draw(image, boxes, scores, classes, ratio, padding):
    for box, score, cl in zip(boxes, scores, classes):
            top, left, right, bottom = box

            top = int((top - padding[0]) / ratio[0])
            left = int((left - padding[1]) / ratio[1])
            right = int((right - padding[0]) / ratio[0])
            bottom = int((bottom - padding[1]) / ratio[1])

            cv2.rectangle(image, (top,left), (right, bottom), (255,0,255), 2)
            draw_box_corner(image, top, left, right, bottom, 15, (0, 255, 0))
            label = DISPLAY_CLASSES[int(cl)] if int(cl) < len(DISPLAY_CLASSES) else f"class_{int(cl)}"
            draw_label_type(image, top, left, label, (255,0,255), score)



def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
             new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize——
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=color)  # add border
    # return im
    return im, ratio, (left, top)


def myFunc(rknn_lite, IMG):
    result_img, _, _, _ = infer_image(rknn_lite, IMG)
    return result_img


def infer_image(
    rknn_lite,
    image,
    draw_result=True,
    return_detections=True,
    post_process=True,
    return_profile=False,
):
    profile = {}

    t0 = time.perf_counter()
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_image, ratio, padding = letterbox(rgb_image)
    input_image = np.expand_dims(input_image, 0)
    input_image = np.ascontiguousarray(input_image)
    t1 = time.perf_counter()
    profile["preprocess_ms"] = (t1 - t0) * 1000.0

    t2 = time.perf_counter()
    outputs = rknn_lite.inference(inputs=[input_image], data_format=['nhwc'])
    t3 = time.perf_counter()
    profile["inference_ms"] = (t3 - t2) * 1000.0

    if not post_process:
        result_image = copy(image) if draw_result else image
        profile["postprocess_ms"] = 0.0
        profile["draw_ms"] = 0.0
        if return_profile:
            return result_image, [], None, None, profile
        return result_image, [], None, None

    t4 = time.perf_counter()
    boxes, classes, scores = yolov8_post_process(outputs)
    t5 = time.perf_counter()
    profile["postprocess_ms"] = (t5 - t4) * 1000.0

    result_image = copy(image) if draw_result else image
    detections = [] if return_detections else None

    if boxes is not None:
        if return_detections:
            for box, score, cl in zip(boxes, scores, classes):
                top, left, right, bottom = box
                detections.append({
                    "class_id": int(cl),
                    "class_name": CLASSES[int(cl)],
                    "score": float(score),
                    "box": [
                        int((top - padding[0]) / ratio[0]),
                        int((left - padding[1]) / ratio[1]),
                        int((right - padding[0]) / ratio[0]),
                        int((bottom - padding[1]) / ratio[1]),
                    ],
                })
        if draw_result:
            t6 = time.perf_counter()
            draw(
                result_image,
                boxes,
                scores,
                classes,
                ratio,
                padding,
            )
            t7 = time.perf_counter()
            profile["draw_ms"] = (t7 - t6) * 1000.0
        else:
            profile["draw_ms"] = 0.0
    else:
        profile["draw_ms"] = 0.0

    if return_profile:
        return result_image, detections, boxes, scores, profile
    return result_image, detections, boxes, scores

