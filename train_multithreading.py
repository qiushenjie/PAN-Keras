import os
import numpy as np
from PIL import Image
import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard
from imp import reload
from PANNet import PANNET
import config
from utils.DataGenerator import Generator
from loss import PAN_LOSS
from utils.metrics import build_iou, mean_iou

batch_size = 2
train_data_file = 'train.txt'
val_data_file = 'val.txt'
image_path = './datasets'
pretrainedPath = './base/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5'
# pretrainedPath = './models/weights_-10-0.07.h5'

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

def get_session(gpu_fraction=1.0):

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def gen(data_file, image_path, batch_size=64, imagesize=(256, 256)):
    image_label_dict = readfile(data_file)
    _name_list = [i for i, j in image_label_dict.items()]
    data_generator = Generator(_name_list, image_label_dict, image_path, batch_size=batch_size, size=imagesize)
    return data_generator


def train(input_shape, max_epoch, frozen=True):
    K.set_session(get_session())
    # reload(PSENet.PSENet)

    model = PANNET(input_shape)

    if os.path.exists(pretrainedPath):
        print('Loading model weights...')
        model.load_weights(pretrainedPath, by_name=True)
        print('done')

    if frozen:
        for layer in model.layers:
            if layer.name in config.base_layer_name:
                layer.trainable = False
            else:
                layer.trainable = True
        print('freeze layers')
    else:

        for layer in model.layers:
            layer.trainable = True
        print('unfreeze layers')

    train_loader = gen(train_data_file,
                       image_path,
                       batch_size=batch_size,
                       imagesize=input_shape)
    val_loader = gen(val_data_file,
                     image_path,
                     batch_size=batch_size,
                     imagesize=input_shape)

    checkpoint = ModelCheckpoint(filepath='./models/weights_-{epoch:02d}-{val_loss:.2f}.h5',
                                 monitor='val_loss',
                                 save_best_only=True,
                                 save_weights_only=True)
    iou = build_iou([0,1],['background','txt'])
    model.compile(loss=PAN_LOSS, optimizer='adam', metrics=iou)
    lr_schedule = lambda epoch: 0.005 * 0.4 ** epoch
    learning_rate = np.array([lr_schedule(i) for i in range(max_epoch)])
    changelr = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))
    earlystop = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
    tensorboard = TensorBoard(log_dir='./models/logs', write_graph=True)

    print('-----------Start training-----------')
    model.fit_generator(train_loader,
                        steps_per_epoch=421 // batch_size,
                        epochs=max_epoch,
                        initial_epoch=0,
                        validation_data=val_loader,
                        validation_steps=75 // batch_size,
                        callbacks=[checkpoint, earlystop, changelr, tensorboard])

train((256, 256, 3), 10, frozen=True)
# for x, y in gen(train_data_file,
#                 image_path,
#                 batch_size=batch_size,
#                 n=config.n,
#                 m=config.m):
#     print('y的一个批次:',y.shape)
#     print('x批次:',x.shape)

