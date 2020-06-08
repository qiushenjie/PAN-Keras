import numpy as np
import keras
from data_processor import gen_image_label
import config
import os

def readfile(filename):
    res = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for i in lines:
            res.append(i.strip())
        dic = {}
        for i in res:
            p = i.split(' ', 1) # （图片名称，文本框坐标）
            dic[p[0]] = p[1]
        return dic

def str_to_poly(coor_str): # coor_str: xmin1 ymin1 xmax1 ymax1 xmin2 ymin2 xmax2 ymax2......
    coor_list = coor_str.split(' ')
    coor_list = [int(i) for i in coor_list]
    assert (len(coor_list) % config.point_num * 2 == 0), 'wrong label length'
    coor_list = np.array(coor_list, dtype=np.float32).reshape((-1, 4, 2))
    return coor_list

class Generator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, name_list, image_label_dict, image_path, batch_size=32, size=(256, 256), shuffle=True):
        'Initialization'
        self.size = size
        self.image_label_dict = image_label_dict
        self.image_path = image_path
        self.batch_size = batch_size
        self.name_list = name_list
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.name_list) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_name_temp = [self.name_list[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_name_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.name_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, images_name_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        inputs = []
        gt_maps = []
        training_masks = []

        # Generate data
        for i, item in enumerate(images_name_temp):
            # Store sample
            text_polys = str_to_poly(self.image_label_dict[item])
            text_tags = np.array([False] * len(text_polys))  # 所有文本框都有效
            input, gt_map, _ = gen_image_label(os.path.join(self.image_path, item), text_polys, text_tags,
                                                                       self.size[0], shrink_ratio=0.5)

            inputs.append(input)
            gt_maps.append(gt_map)

        return np.array(inputs), np.array(gt_maps)

if __name__ == '__main__':
    train_data_file = 'train.txt'
    val_data_file = 'val.txt'
    image_path = './'
    data_file = ''
    image_label_dict = readfile(train_data_file)
    _name_list = [i for i, j in image_label_dict.items()]
    training_generator = Generator(_name_list, image_label_dict, image_path, batch_size=2)
    for i, j in training_generator:
        print(i[1].shape)
        print(j[1].shape)