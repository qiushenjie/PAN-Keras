# -*- coding: utf-8 -*-
import numpy as np
import cv2
from numba import jit
import time

@jit
def get_dist(dist, sv_dist):
    for i in range(dist.shape[0]):
        sv_dist[int(dist[i, 0]), int(dist[i, 1])] = dist[i, 2]
    return sv_dist

@jit
def ufunc_4(S1, S2, TAG, dist, dis_threshold):
    # indices 四邻域 x-1 x+1 y-1 y+1，如果等于TAG 则赋值为label

    for h in range(1, S1.shape[0] - 1):
        for w in range(1, S1.shape[1] - 1):
            instance_num = S1[h][w]
            # print(dist.shape)
            if (instance_num != 0):
                if (S2[h][w - 1] == TAG and dist[instance_num, h, w - 1] < dis_threshold):
                    S2[h][w - 1] = instance_num
                if (S2[h][w + 1] == TAG and dist[instance_num, h, w + 1] < dis_threshold):
                    S2[h][w + 1] = instance_num
                if (S2[h - 1][w] == TAG and dist[instance_num, h - 1, w] < dis_threshold):
                    S2[h - 1][w] = instance_num
                if (S2[h + 1][w] == TAG and dist[instance_num, h + 1, w] < dis_threshold):
                    S2[h + 1][w] = instance_num


def scale_expand_kernel(S1, S2, dist, dis_threshold):
    TAG = 10240
    S2[S2 == 255] = TAG
    mask = (S1 != 0)
    S2[mask] = S1[mask]
    cond = True
    while (cond):
        before = np.count_nonzero(S1 == 0)
        ufunc_4(S1, S2, TAG, dist, dis_threshold)
        S1[S2 != TAG] = S2[S2 != TAG]
        after = np.count_nonzero(S1 == 0)
        if (before <= after):
            cond = False

    return S1


def filter_label_by_area(labelimge, num_label, area=5):
    for i in range(1, num_label + 1):
        if (np.count_nonzero(labelimge == i) <= area):
            labelimge[labelimge == i] == 0
    return labelimge


def fit_boundingRect(num_label, labelImage):
    rects = []
    for label in range(1, num_label + 1):
        points = np.array(np.where(labelImage == label)[::-1]).T
        x, y, w, h = cv2.boundingRect(points)
        rect = np.array([x, y, x + w, y + h])
        rects.append(rect)
    return rects

def pse_py(text, similarity_vectors, label, label_num, dis_threshold=0.8, filter=False):

    if (filter == True):
        label = filter_label_by_area(label, label_num)

    # 计算kernel中每个实例的相似向量的均值
    sv_dists = []
    # t1 = time.time()
    similarity_vector_coord = np.array(np.where(text == 1))
    for i in range(label_num):
        sv_dist = np.ones((256, 256), dtype=np.float) * 10240
        kernel_idx = label == i
        kernel_similarity_vector = similarity_vectors[kernel_idx].mean(0)  # shape:(4,)
        dist = np.array([np.linalg.norm(similarity_vectors[similarity_vector_coord[0, :], similarity_vector_coord[1, :], :] - kernel_similarity_vector, axis=1)])
        dist = np.concatenate((similarity_vector_coord, dist), axis=0).transpose((1, 0))
        sv_dist = get_dist(dist, sv_dist)
        sv_dists.append(sv_dist)
    sv_dists = np.array(sv_dists)
    text = text * 255
    # print('dist:{}'.format(time.time() - t1))
    # t1 = time.time()
    labelimage = scale_expand_kernel(label, text, sv_dists, dis_threshold)
    # print('pse:{}'.format(time.time() - t1))

    return labelimage