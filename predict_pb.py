import cv2
import os
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
from keras.preprocessing import image
import matplotlib.pyplot as plt
from pypse import pse_py, fit_boundingRect

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
    plt.figure()
    plt.imshow(im)
    plt.show()
    # im.save(os.path.join('./test_images/result',
    #                           'test_{}'.format(image_name)))

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

if __name__ == '__main__':
    val_data_file = 'val.txt'
    img_path = './tese_images/'
    # img_path = '../../location_data/vietnam_test'
    images_name = readfile(val_data_file)
    # images_name = os.listdir(img_path)
    img_list = []
    reference = Reference(pb_path='./models/PAN.pb')
    for name, _ in images_name.items():
        img_list.append(name)
    # for name in images_name:
    #     if name.endswith('.jpg'):
    #         img_list.append(name)
    import time
    t_predict = 0.0
    t_postpro = 0.0
    # img_list = img_list[:50]
    for image_name in img_list[:]:
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

        draw_boxes(src_image.copy(), rects)



    print('time for predict:{}'.format(t_predict / len(img_list)))
    print('time for postprocess:{}'.format(t_postpro / len(img_list)))