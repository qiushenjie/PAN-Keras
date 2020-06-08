import cv2
import os
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
from keras.preprocessing import image
import matplotlib.pyplot as plt
from pypse import pse_py, fit_boundingRect
from cal_pixel_precision_recall_f1 import cal_pixel_precision_recall_f1

def get_true_mask(boxes, x_scale, y_scale, mask_shape):
    boxes = np.array(boxes, dtype=np.float)
    mask = np.zeros(mask_shape, dtype=np.float)
    boxes[:, :, 0] /= x_scale
    boxes[:, :, 1] /= y_scale
    for box in boxes:
        cv2.fillPoly(mask, np.int32([box]), 1)
    return mask

def get_preds_mask(boxes, mask_shape):
    mask = np.zeros(mask_shape, dtype=np.float)
    for box in boxes:
        if box.reshape((4, -1)).shape[1] == 1:
            box = np.array([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]])
        cv2.fillPoly(mask, np.int32([box]), 1)
    return mask

def draw_boxes(im, rects, score=1):
    quad_draw = ImageDraw.Draw(im)
    d_wight, d_height = (256, 256)
    scale_ratio_w = d_wight / im.width
    scale_ratio_h = d_height / im.height
    rects = np.array(rects, dtype=np.float).reshape(-1, 2, 2)
    for rect in rects:
        rect[:, 0] /= scale_ratio_w
        rect[:, 1] /= scale_ratio_h
        if np.amin(score) > 0:
            # quad_draw.line([tuple(geo[0]),
            #                 tuple(geo[1]),
            #                 tuple(geo[2]),
            #                 tuple(geo[3]),
            #                 tuple(geo[0])], width=3, fill='red')
            quad_draw.line([(min(rect[:, 0]), min(rect[:, 1])),
                            (max(rect[:, 0]), min(rect[:, 1])),
                            (max(rect[:, 0]), max(rect[:, 1])),
                            (min(rect[:, 0]), max(rect[:, 1])),
                            (min(rect[:, 0]), min(rect[:, 1]))], width=3, fill='red')
    im.save(os.path.join('./test_images/result',
                              'test_{}'.format(image_name)))

def process(res):
    text = res[:, :, 0] > 0.8  # text
    kernel = (res[:, :, 1] > 0.5) * text  # kernel
    similarity_vectors = res[:, :, 2:]

    label_num, label = cv2.connectedComponents(kernel.astype(np.uint8), connectivity=4)

    pred = pse_py(text.astype(np.int8), similarity_vectors, label, label_num, 0.8)
    # pred = pse_py(text.astype(np.uint8), similarity_vectors, label, label_num, 0.8) # pspse_queue.py
    rects = fit_boundingRect(label_num, pred)

    return rects

class Reference():
    def __init__(self, pb_path):
        self.pb_path = pb_path
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.init_model()

    def init_model(self):
        tf.Graph().as_default()
        self.output_graph_def = tf.GraphDef()
        with open(self.pb_path, 'rb') as f:
            self.output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(
                self.output_graph_def,
                input_map=None,
                return_elements=None,
                name=None,
                op_dict=None,
                producer_op_list=None
            )

        self.sess = tf.Session(config=self.config)
        self.input = self.sess.graph.get_tensor_by_name("import/input_1:0")  # 自己定义的输入tensor名称
        self.output = self.sess.graph.get_tensor_by_name("import/output_1:0")  # 自己定义的输出tensor名称

    def predict(self, img):
        img = img[np.newaxis, :, :, :]
        res = self.sess.run(self.output, feed_dict={self.input: img})
        res_mask = np.array(res)
        return res_mask

    def batch_predict(self, img_list):
        res_masks = []
        for img in img_list:
            res_mask = self.predict(img)
            res_masks.append(res_mask)
        return res_masks

def readfile(filename):
    res = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for i in lines:
            res.append(i.strip())
        dic = {}
        for i in res:
            p = i.split(' ', 1) # （图片名称， 文本框坐标）
            dic[p[0]] = p[1]
        return dic

def str_to_poly(coor_str): # coor_str: xmin1 ymin1 xmax1 ymax1 xmin2 ymin2 xmax2 ymax2......
    coor_list = coor_str.split(' ')
    coor_list = [int(i) for i in coor_list]
    assert (len(coor_list) % 8 == 0), 'wrong label length'
    coor_list = np.array(coor_list).reshape(-1, 4, 2)
    return coor_list

if __name__ == '__main__':
    val_data_file = 'val.txt'
    img_path = './tese_images/'
    # img_path = '../../location_data/vietnam_test'
    images_name = readfile(val_data_file)
    # images_name = os.listdir(img_path)
    img_list = []
    img_ploys = []
    reference = Reference(pb_path='./models/PAN.pb')
    for name, ploys in images_name.items():
        img_list.append(name)
        img_ploys.append(str_to_poly(ploys))
    # for name in images_name:
    #     if name.endswith('.jpg'):
    #         img_list.append(name)

    pixel_precision_recall_f1 = cal_pixel_precision_recall_f1()
    import time
    t_predict = 0.0
    t_postpro = 0.0
    for image_name, ploys in zip(img_list[:], img_ploys[:]):
        img = Image.open(os.path.join(img_path, image_name))
        W, H = img.width, img.height
        src_image = img.copy()
        img = img.resize((256, 256), Image.NEAREST).convert('RGB')
        img = image.img_to_array(img)
        t1 = time.time()
        outputs_map = reference.predict(img)
        t_predict += time.time() - t1
        t2 = time.time()
        rects = process(outputs_map)
        t_postpro += time.time() - t2

        x_scale = W / 256
        y_scale = H / 256
        true_mask = get_true_mask(ploys, x_scale, y_scale, [256, 256])
        preds_mask = get_preds_mask(rects, [256, 256])
        pixel_precision_recall_f1.update(true_mask, preds_mask)

        draw_boxes(src_image.copy(), rects)


    precision, recall, f1, iou, iou_count = pixel_precision_recall_f1.get_scores()
    print('precision:{:.3f}%, recall:{:.3f}%, f1:{:.3f}%, iou:{:.3f}%, iou_count:{:.3f}%'.format(precision * 100, recall * 100, f1 * 100, iou * 100, iou_count * 100))
    print('time for predict:{}'.format(t_predict / len(img_list)))
    print('time for postprocess:{}'.format(t_postpro / len(img_list)))