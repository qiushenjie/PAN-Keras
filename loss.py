import tensorflow as tf
from keras import backend as K
import config
import itertools

assert K.image_data_format() == 'channels_last', 'backend should be tensorflow'

def PAN_LOSS(y_true, y_pred):
    '''
    build psenet loss refer to  section 3.4

    Arg:
        y_true: the ground truth label. [batchsize,h,w,config.SN]
        y_pred : the predict label

    return total loss
    '''
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # complete instance
    y_true_Lt = y_true[:, :, :, 0]
    y_pred_Lt = y_pred[:, :, :, 0]
    # shrinked instance
    y_true_Lk = y_true[:, :, :, 1]
    y_pred_Lk = y_pred[:, :, :, 1]
    #
    similarity_vectors = y_pred[:, :, :, 2:]

    # 计算 text loss
    # adopt ohem to Lt
    M = ohem_batch(y_true_Lt, y_pred_Lt)
    loss_texts = 1 - dice_loss(y_true_Lt * M, y_pred_Lt * M)

    # 计算 kernel loss
    #ignore the pixels of non-text region
    #in the segmentation result Sn to avoid a certain redundancy.
    W = y_pred_Lt > 0.5
    pos_mask = tf.cast(y_true_Lt, tf.bool)
    W = tf.logical_or(pos_mask, W) # 将y_pred中值大于0.5的样本以及y_true中的正样本一起保留下来，忽略非文本区域像素，参与计算损失，其他的样本位置不参与计算, 因为负样本的损失已经在Lc_loss中被计算了，这里就不用再计算了
    W = tf.cast(W, tf.float32) # 有效的loss_mask
    loss_kernels = 1.0 - dice_loss(y_true_Lk * W, y_pred_Lk * W)

    # 计算 agg loss 和 dis loss
    loss_aggs, loss_diss = agg_dis_loss(y_pred_Lt, y_pred_Lk, y_true_Lt, y_true_Lk, similarity_vectors)

    # mean or sum
    if config.reduction == 'mean':
        loss_text = tf.reduce_mean(loss_texts)
        loss_kernel = tf.reduce_mean(loss_kernels)
        loss_agg = tf.reduce_mean(loss_aggs)
        loss_dis = tf.reduce_mean(loss_diss)
    elif config.reduction == 'sum':
        loss_text = tf.reduce_sum(loss_texts)
        loss_kernel = tf.reduce_sum(loss_kernels)
        loss_agg = tf.reduce_sum(loss_aggs)
        loss_dis = tf.reduce_sum(loss_diss)

    loss_all = loss_text + config.alpha * loss_kernel + config.beta * (loss_agg + loss_dis)
    return loss_all

def ohem_batch(y_true_Lt, y_pred_Lt):
    M = tf.map_fn(ohem_single, (y_true_Lt, y_pred_Lt), dtype=tf.float32) # tf.map_fn():高阶函数，分别把(y_true_Lc, y_pred_Lc)的每一个batch传进ohem_single，输出的batch个结果进行map处理。
    return tf.stack(M)

def ohem_single(s_Lt):
    s_y_true_Lt, s_y_pred_Lt = s_Lt
    n_pos = K.sum(s_y_true_Lt)

    def has_pos():
        n_max_neg = K.sum(tf.cast(s_y_true_Lt > -1.0, tf.int32)) # 所有样本点个数，s_y_true_Lt中的值均大于-1.0
        n_neg = n_pos * config.rate_ohem
        n_neg = tf.cast(n_neg, tf.int32)
        n_neg = K.minimum(n_neg, n_max_neg)

        pos_mask = tf.cast(s_y_true_Lt, tf.bool)
        neg_mask = tf.cast(tf.equal(pos_mask, False), tf.float32) # tf.equal() 判断是否相等，相等返回True
        neg = neg_mask * s_y_pred_Lt # neg中只保留y_true中为负样本点的对应位置的值

        vals, _ = tf.nn.top_k(K.reshape(neg, (1, -1)), k = n_neg) # 取neg的前k个最大值,负样本中值越大说明越为模棱两可的样本，也算是越容易被分错的困难样本点
        threshold = vals[0][-1]
        mask = tf.logical_or(pos_mask, neg > threshold) # 将标签y_true中的正例（+1）位置以及预测结果y_pred中负样本的值大于threshold的样本的位置一起记为1（二者不相交），作为mask，之后计算损失时只计算mask中值为1的位置的损失值，其他位置损失抛弃不算，即OHEM
        return tf.cast(mask, tf.float32)

    def no_pos(): #
        mask = K.zeros_like(s_y_true_Lt) # 若标签y_true中均为负样本，则此时的y_pred不计算损失
        return tf.cast(mask, tf.float32)

    return tf.cond(n_pos > 0, has_pos, no_pos) # tf.cond(): 用于有条件的执行函数，当(n_pos > 0)为True时，执行has_pos函数，否则执行no_pos函数

def dice_loss(y_true, y_pred, smooth=1.0):
    y_true = tf.cast(y_true > 0.5, dtype=tf.float32)
    if config.batch_loss:
        intersection = K.sum(y_true * y_pred) #
        loss = K.mean((2.0 * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth))
    else:
        intersection = K.sum(y_true * y_pred, axis=(1, 2, 3)) #
        loss = K.mean((2.0 * intersection + smooth) / (K.sum(y_true, axis=(1, 2, 3)) + K.sum(y_pred, axis=(1, 2, 3)) + smooth))
    return loss


def get_agg_loss(single_kernel_mask, single_text_mask, similarity_vector):
    # G_Ki, shape: 4
    G_kernel = tf.reduce_mean(tf.boolean_mask(similarity_vector, single_kernel_mask, axis=1),
                              axis=1)  # 求单个收缩核的相似向量(学习得出)的聚类中心，按行(channel)求文本实例像素值均值，得shape:4
    # 文本像素的相似向量矩阵 F(p) shape: 4 * nums (num of text pixel, nums <= inputsize * inputsize)
    text_similarity_vector = tf.boolean_mask(similarity_vector, single_text_mask,
                                             axis=1)  # 实例编号为text_idx的文本实例像素值存入text_similarity_vector, 总数小于等于256*256
    # ||F(p) - G(K_i)|| - delta_agg, shape: nums, 求每个文本的单个像素的相似向量矩阵与对应通道的收缩核相似向量聚类中心的距离，越小相似度越大。取每一列(通道)四个点的2范数，最后得到的shape为1*nums
    text_G_ki = tf.norm(tf.subtract(text_similarity_vector, tf.reshape(G_kernel, (4, 1))), ord=2,
                        axis=0) - config.delta_agg
    # D(p,K_i), shape: nums，保证上式(||F(p) - G(K_i)||)小于delta_agg(小于delta_agg的像素点就没必要参与loss计算了，关注那些距离远的值即可)，(并通过loss求最小)
    D_text_kernel = tf.pow(tf.maximum(text_G_ki, 0.0), 2)
    # 计算单个文本实例的loss_agg_single_text, shape: 1
    loss_agg_single_text = tf.reduce_mean(tf.log(D_text_kernel + 1))
    return G_kernel, loss_agg_single_text


def agg_dis_loss(texts, kernels, gt_texts, gt_kernels, similarity_vectors):
    """
    计算 loss agg
    :param texts: 文本实例的分割结果 batch_size * (w*h)
    :param kernels: 缩小的文本实例的分割结果 batch_size * (w*h)
    :param gt_texts: 文本实例的gt batch_size * (w*h)
    :param gt_kernels: 缩小的文本实例的gt batch_size*(w*h)
    :param similarity_vectors: 相似度向量的分割结果 batch_size * 4 *(w*h)
    :return:
    """

    texts = tf.reshape(texts, (config.batch_size, -1))
    kernels = tf.reshape(kernels, (config.batch_size, -1))
    gt_texts = tf.reshape(gt_texts, (config.batch_size, -1))
    gt_kernels = tf.reshape(gt_kernels, (config.batch_size, -1))
    similarity_vectors = tf.transpose(tf.reshape(similarity_vectors, (config.batch_size, -1, 4)), perm=[0, 2, 1])

    loss_aggs = []
    loss_diss = []
    for i in range(config.batch_size):
        text_i, kernel_i, gt_text_i, gt_kernel_i, similarity_vector =(texts[i], kernels[i], gt_texts[i], gt_kernels[i],
                                                                           similarity_vectors[i])
        # text_num = tf.reduce_max(gt_text_i) + 1
        loss_agg_single_sample = []
        G_kernel_list = []  # 存储计算好的G_Ki,用于计算loss dis
        # 求解每一个文本实例的loss agg

        for text_idx in range(1, int(config.max_text_nums)):
            # 计算 D_p_Ki
            single_kernel_mask = tf.equal(gt_kernel_i, text_idx)
            single_text_mask = tf.equal(gt_text_i, text_idx)

            G_kernel, loss_agg_single_text = tf.cond(
                tf.reduce_sum(tf.cast(tf.logical_and(single_kernel_mask, single_text_mask), dtype=tf.int32)) > 0,
                lambda: get_agg_loss(single_kernel_mask, single_text_mask, similarity_vector),
                lambda: (tf.constant([0.0, 0.0, 0.0, 0.0]), tf.constant(0.0)))

            G_kernel_list.append(G_kernel)
            # 累计单个图片的loss_agg_single_sample, shape: text_nums
            loss_agg_single_sample.append(loss_agg_single_text)

        if len(G_kernel_list)> 0:
            loss_agg_single_sample = tf.reduce_mean(tf.stack(loss_agg_single_sample))
        else:
            loss_agg_single_sample = 0.0
        # 累计单个batch的loss_aggs, shape: batch_size
        loss_aggs.append(loss_agg_single_sample)
        # 求解每一个文本实例的loss dis，其作用是保证任意两个kernel聚类中心之间的距离>delta_dis
        loss_dis_single_sample = 0
        for G_kernel_i, G_kernel_j in itertools.combinations(G_kernel_list, 2):
            # delta_dis - ||G(K_i) - G(K_j)||，求任意两个聚类中心的二范数
            kernel_ij = config.delta_dis - tf.norm(tf.subtract(G_kernel_i, G_kernel_j), ord=2)
            # D(K_i,K_j)，求大于0的任意两个聚类中心的二范数矩阵
            D_kernel_ij = tf.pow(tf.maximum(kernel_ij, 0.0), 2)  # 类间距离>delta_dis的kernal就没必要参与loss计算了，关注那些距离近的值即可
            loss_dis_single_sample += tf.log(D_kernel_ij + 1)
        if len(G_kernel_list) > 1:
            loss_dis_single_sample /= (len(G_kernel_list) * (len(G_kernel_list) - 1))
        else:
            loss_dis_single_sample = 0.0
        loss_diss.append(loss_dis_single_sample)
    return tf.stack(loss_aggs), tf.stack(loss_diss)


if __name__ == '__main__':
    import numpy as np
    y_random = np.zeros((2, 20, 20, 6))

    y_true  = np.copy(y_random)
    y_true[:, 2:6, 2:6, :] = 2
    y_true[:, 12:16, 12:16, :] = 3

    y_pred  = np.copy(y_random)
    y_pred[:, 2:7, 2:7, :] = 2


    sess = tf.InteractiveSession()

    loss = PAN_LOSS(y_true,y_pred)

    print('loss:',sess.run(loss))