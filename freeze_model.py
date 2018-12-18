import tensorflow as tf
import os
from inpaint_model import InpaintCAModel
import argparse


def freeze(ckpt_path):
    image = tf.placeholder(tf.float32, shape=(1, None, None, 3), name='image')
    mask = tf.placeholder(tf.float32, shape=(1, None, None, 3), name='mask')
    model = InpaintCAModel()
    input = tf.concat([image, mask],axis=2)

    output = model.build_server_graph(input)
    output = (output + 1.) * 127.5
    output = tf.reverse(output, [-1])
    output = tf.saturate_cast(output, tf.uint8)
    output = tf.add(output, 0, name='output')
 
    init_op = tf.global_variables_initializer()
 
    restore_saver = tf.train.Saver()
    
    with tf.Session() as sess:
        sess.run(init_op)
        restore_saver.restore(sess, ckpt_path)
        frozen_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                        output_node_names=['output'])
        
        path = os.path.dirname(ckpt_path)
        with open(path + '/deepfill.pb', 'wb') as f:
            f.write(frozen_graph_def.SerializeToString())
    print("frozen model path: {}".format( path + '/deepfill.pb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", dest='model_path', type=str)  # e.g. model_logs/epoch_8/snap-80000
    args = parser.parse_args()
    freeze(args.model_path)
