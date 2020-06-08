import cv2
import pyclipper
import random
import numpy as np
import augment
data_aug = augment.DataAugment()

def check_and_validate_polys(polys, xxx_todo_changeme):
    '''
    check so that the text poly is in the same direction,
    and also filter some invalid polygons
    :param polys:
    :param tags:
    :return:
    '''
    (h, w) = xxx_todo_changeme
    if polys.shape[0] == 0:
        return polys
    polys[:, :, 0] = np.clip(polys[:, :, 0], 0, w - 1) # x coord not max w-1, and not min 0
    polys[:, :, 1] = np.clip(polys[:, :, 1], 0, h - 1) # x coord not max h-1, and not min 0

    validated_polys = []
    for poly in  polys:
        p_area = cv2.contourArea(poly)
        if abs(p_area) < 1: # 剔除面积过小的
            continue
        validated_polys.append(poly)
    return np.array(validated_polys)

def augmentation(im: np.ndarray, text_polys: np.ndarray, scales: np.ndarray, degrees: int, input_size: int) -> tuple:
    # the images are rescaled with ratio {0.5, 1.0, 2.0, 3.0} randomly
    im, text_polys = data_aug.random_scale(im, text_polys, scales)
    # the images are horizontally fliped and rotated in range [−10◦, 10◦] randomly
    if random.random() < 0.5:
        im, text_polys = data_aug.horizontal_flip(im, text_polys)
    if random.random() < 0.5:
        im, text_polys = data_aug.random_rotate_img_bbox(im, text_polys, degrees)
    # 640 × 640 random samples are cropped from the transformed images
    # im, text_polys = data_aug.random_crop_img_bboxes(im, text_polys)

    # im, text_polys = data_aug.resize(im, text_polys, input_size, keep_ratio=False)
    # im, text_polys = data_aug.random_crop_image_pse(im, text_polys, input_size)

    return im, text_polys

def generate_rbox(im_size, text_polys, i, n, m):
    """
    生成mask图，白色部分是文本，黑色是背景
    :param im_size: 图像的h,w
    :param text_polys: 框的坐标
    :param i, n, m: 比例尺寸超参数
    :return: 生成的mask图
    """
    h, w = im_size
    score_map = np.zeros((h, w), dtype=np.uint8)
    for poly in text_polys:
        poly = poly.astype(np.int)
        r_i = 1 - (1 - m) * (n - i) / (n - 1)
        d_i = cv2.contourArea(poly) * (1 - r_i * r_i) / cv2.arcLength(poly, True)
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        shrinked_poly = np.array(pco.Execute(-d_i))
        cv2.fillPoly(score_map, shrinked_poly, 1)
    return score_map

def gen_image_label(im_fn: str, text_polys: np.ndarray, n:int, m: float, input_size: int,
                defrees: int = 10, scales: np.ndarray = np.array([0.5, 1.0, 2.0, 3.0])) -> tuple:
    '''
    get image's corresponding matrix and ground truth
    return
    images [512, 512, 3]
    score  [128, 128, 1]
    geo    [128, 128, 5]
    '''
    im = cv2.imread(im_fn)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    h, w, _ = im.shape
    # 检查是否越界
    text_polys = check_and_validate_polys(text_polys, (h, w))
    im, text_polys, = augmentation(im, text_polys, scales, defrees, input_size)

    h, w, _ = im.shape
    short_edge = min(h, w)
    if short_edge < input_size:
        # 保证短边 >= inputsize
        scale = input_size / short_edge
        im = cv2.resize(im, dsize=None, fx=scale, fy=scale)
        text_polys *= scale

    # # normal images
    # im = im.astype(np.float32)
    # im /= 255.0
    # im -= np.array((0.485, 0.456, 0.406))
    # im /= np.array((0.229, 0.224, 0.225))

    h, w, _ = im.shape
    score_maps = []
    for i in range(1, n + 1):
        # s1->sn,由小到大
        score_map = generate_rbox((h, w), text_polys, i, n, m)
        score_maps.append(score_map)
    score_map = np.array(score_maps, dtype=np.float32)
    imgs = data_aug.random_crop_author(([im, score_map.transpose((1, 2, 0))]), (input_size, input_size))
    return imgs[0], imgs[1].transpose((2, 0, 1))



