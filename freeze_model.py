import tensorflow as tf
import os
from inpaint_model import InpaintCAModel

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


if __name__ == '__main__':
    ckpt_path = 'model_logs/release_places2_256/snap-0'
    freeze(ckpt_path)
