import os
import tensorflow as tf
from keras import backend as K

def h5_to_pb(h5_model, output_dir, model_name, out_prefix="output_", log_tensorboard=True):
    """.h5模型文件转换成pb模型文件
    Argument:
        h5_model: h5模型文件,这里的模型需包含权重和模型结构信息
        output_dir: pb模型文件保存路径
        model_name: pb模型文件名称
        out_prefix: 根据训练，需要修改
        log_tensorboard: bool,是否生成日志文件
    Return:
        pb模型文件
    """
    if os.path.exists(output_dir) == False:
        os.mkdir(output_dir)
    out_nodes = []
    for i in range(len(h5_model.outputs)):
        out_nodes.append(out_prefix + str(i + 1))
        tf.identity(h5_model.output[i], out_prefix + str(i + 1))
    sess = K.get_session()

    from tensorflow.python.framework import graph_util, graph_io
    # 写入pb模型文件
    init_graph = sess.graph.as_graph_def()
    main_graph = graph_util.convert_variables_to_constants(sess, init_graph, out_nodes)
    graph_io.write_graph(main_graph, output_dir, name=model_name, as_text=False)

    # 输出日志文件, 通过tensorboard --logdir=log目录，查看.ph模型文件的输入输出tensor名称，在之后的推理中会使用到，一般为‘import/input_1’和‘import/output_1’
    if log_tensorboard:
        from tensorflow.python.tools import import_pb_to_tensorboard
        import_pb_to_tensorboard.import_to_tensorboard(os.path.join(output_dir, model_name), output_dir)

if __name__ == '__main__':
    from PANNet import PANNET
    model = PANNET((256, 256, 3))
    model.load_weights('./models/weights_-09-0.13.h5')
    h5_to_pb(model, './models/', 'PAN.pb')

