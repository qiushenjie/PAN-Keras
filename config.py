
point_num = 4  # b标注框顶点数

rate_ohem = 3       #positive ：negtive = 1:rate_ohem

alpha = 0.5
beta = 0.25
delta_agg = 0.5
delta_dis = 3

batch_size = 2
reduction = 'mean'

max_text_nums = 11 # 数据集中单张图片最多的文本实例数量
data_gen_min_scales = 0.8
data_gen_max_scales = 2.0
data_gen_itter_scales = 0.3


#dice loss
batch_loss = True

#metric iou
metric_iou_batch = True

base_layer_name = ['input_1', 'conv2d_1', 'batch_normalization_1',
                   'activation_1', 'conv2d_2', 'batch_normalization_2',
                   'activation_2', 'depthwise_conv2d_1', 'batch_normalization_3',
                   'activation_3', 'conv2d_3', 'batch_normalization_4',
                   'conv2d_4', 'batch_normalization_5', 'activation_4',
                   'depthwise_conv2d_2', 'batch_normalization_6', 'activation_5',
                   'conv2d_5', 'batch_normalization_7', 'conv2d_6',
                   'batch_normalization_8', 'activation_6', 'depthwise_conv2d_3',
                   'batch_normalization_9', 'activation_7', 'conv2d_7',
                   'batch_normalization_10', 'add_1', 'conv2d_8',
                   'batch_normalization_11', 'activation_8', 'depthwise_conv2d_4',
                   'batch_normalization_12', 'activation_9', 'conv2d_9',
                   'batch_normalization_13', 'conv2d_10', 'batch_normalization_14',
                   'activation_10', 'depthwise_conv2d_5', 'batch_normalization_15',
                   'activation_11', 'conv2d_11', 'batch_normalization_16',
                   'add_2', 'conv2d_12', 'batch_normalization_17',
                   'activation_12', 'depthwise_conv2d_6', 'batch_normalization_18',
                   'activation_13', 'conv2d_13', 'batch_normalization_19', 'add_3',
                   'conv2d_14', 'batch_normalization_20', 'activation_14',
                   'depthwise_conv2d_7', 'batch_normalization_21', 'activation_15',
                   'conv2d_15', 'batch_normalization_22', 'conv2d_16',
                   'batch_normalization_23', 'activation_16', 'depthwise_conv2d_8',
                   'batch_normalization_24', 'activation_17', 'conv2d_17',
                   'batch_normalization_25', 'add_4', 'conv2d_18',
                   'batch_normalization_26', 'activation_18', 'depthwise_conv2d_9',
                   'batch_normalization_27', 'activation_19', 'conv2d_19',
                   'batch_normalization_28', 'add_5', 'conv2d_20',
                   'batch_normalization_29', 'activation_20', 'depthwise_conv2d_10',
                   'batch_normalization_30', 'activation_21', 'conv2d_21',
                   'batch_normalization_31', 'add_6', 'conv2d_22',
                   'batch_normalization_32', 'activation_22', 'depthwise_conv2d_11',
                   'batch_normalization_33', 'activation_23', 'conv2d_23',
                   'batch_normalization_34', 'conv2d_24', 'batch_normalization_35',
                   'activation_24', 'depthwise_conv2d_12', 'batch_normalization_36',
                   'activation_25', 'conv2d_25', 'batch_normalization_37',
                   'add_7', 'conv2d_26', 'batch_normalization_38',
                   'activation_26', 'depthwise_conv2d_13', 'batch_normalization_39',
                   'activation_27', 'conv2d_27', 'batch_normalization_40',
                   'add_8', 'conv2d_28', 'batch_normalization_41',
                   'activation_28', 'depthwise_conv2d_14', 'batch_normalization_42',
                   'activation_29', 'conv2d_29', 'batch_normalization_43',
                   'conv2d_30', 'batch_normalization_44', 'activation_30',
                   'depthwise_conv2d_15', 'batch_normalization_45', 'activation_31',
                   'conv2d_31', 'batch_normalization_46', 'add_9',
                   'conv2d_32', 'batch_normalization_47', 'activation_32',
                   'depthwise_conv2d_16', 'batch_normalization_48', 'activation_33',
                   'conv2d_33', 'batch_normalization_49', 'add_10',
                   'conv2d_34', 'batch_normalization_50', 'activation_34',
                   'depthwise_conv2d_17', 'batch_normalization_51', 'activation_35',
                   'conv2d_35', 'batch_normalization_52', 'conv2d_36',
                   'batch_normalization_53', 'activation_36', 'global_average_pooling2d_1',
                   'reshape_1', 'Dropout', 'conv2d_37',
                   'softmax', 'reshape_2']
