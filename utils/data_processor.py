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
    # im, text_polys = data_aug.random_scale(im, text_polys, scales)
    # the images are horizontally fliped and rotated in range [−10◦, 10◦] randomly
    # if random.random() < 0.5:
    #     im, text_polys = data_aug.horizontal_flip(im, text_polys)
    if random.random() < 0.5:
        im, text_polys = data_aug.random_rotate_img_bbox(im, text_polys, degrees)
    # 640 × 640 random samples are cropped from the transformed images
    # im, text_polys = data_aug.random_crop_img_bboxes(im, text_polys)

    # im, text_polys = data_aug.resize(im, text_polys, input_size, keep_ratio=False)
    # im, text_polys = data_aug.random_crop_image_pse(im, text_polys, input_size)

    return im, text_polys

def generate_rbox(im_size, text_polys, text_tags, training_mask, shrink_ratio):
    """
    生成mask图，白色部分是文本，黑色是背景
    :param im_size: 图像的h,w
    :param text_polys: 框的坐标
    :param i, n, m: 比例尺寸超参数
    :return: 生成的mask图
    """
    h, w = im_size
    score_map = np.zeros((h, w), dtype=np.uint8)
    for i, (poly, tag) in enumerate(zip(text_polys, text_tags)):
        try:
            poly = poly.astype(np.int)
            # d_i = cv2.contourArea(poly) * (1 - shrink_ratio * shrink_ratio) / cv2.arcLength(poly, True)
            d_i = cv2.contourArea(poly) * (1 - shrink_ratio) / cv2.arcLength(poly, True) + 0.5
            pco = pyclipper.PyclipperOffset()
            pco.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            shrinked_poly = np.array(pco.Execute(-d_i))
            cv2.fillPoly(score_map, shrinked_poly, i + 1) # 这里每一个文本实例都有自己的编号，用于loss_aggs, loss_diss的计算，在dice_loss中则统一设为1
            if not tag:
                cv2.fillPoly(training_mask, shrinked_poly, 0)
        except:
            print(poly)
    return score_map, training_mask

def gen_image_label(im_fn: str, text_polys: np.ndarray, text_tags: np.ndarray, input_size: int, shrink_ratio: float = 0.5,
                degrees: int = 10, scales: np.ndarray = np.array([0.5, 1.0, 2.0, 3.0])) -> tuple:
    '''
    get image's corresponding matrix and ground truth
    :param im: 图片
    :param text_polys: 文本标注框
    :param text_tags: 是否忽略文本的标致：true 忽略, false 不忽略
    :param input_size: 输出图像的尺寸
    :param shrink_ratio: gt收缩的比例
    :param degrees: 随机旋转的角度
    :param scales: 随机缩放的尺度
    return
    images [256, 256, 3]
    map  [256, 256, 2]
    '''
    im = cv2.imread(im_fn)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    h, w, _ = im.shape
    # 检查是否越界
    text_polys = check_and_validate_polys(text_polys, (h, w))
    im, text_polys = data_aug.resize(im, text_polys, input_size)

    ## augmentation
    # im, text_polys = augmentation(im, text_polys, scales, degrees, input_size)
    # h, w, _ = im.shape
    # short_edge = min(h, w)
    # if short_edge < input_size:
    #     # 保证短边 >= inputsize
    #     scale = input_size / short_edge
    #     im = cv2.resize(im, dsize=None, fx=scale, fy=scale)
    #     text_polys *= scale

    h, w, _ = im.shape
    training_mask = np.ones((h, w), dtype=np.uint8)
    score_maps = []
    for i in (1, shrink_ratio):
        # s1->sn,由小到大
        score_map, training_mask = generate_rbox((h, w), text_polys, text_tags, training_mask, i)
        score_maps.append(score_map)
    score_map = np.array(score_maps, dtype=np.float32)
    # imgs = data_aug.random_crop_author(([im, score_map.transpose((1, 2, 0))]), (input_size, input_size))
    return im, score_map.transpose((1, 2, 0)), training_mask



