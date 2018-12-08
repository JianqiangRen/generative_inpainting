# coding=utf-8
# summary:
# author: jianqiang ren
# date:
import tensorflow as tf
import scipy.misc
import numpy as np
from PIL import Image
import cv2

def image_reader(filename):
    """help fn that provides numpy image coding utilities"""
    img = scipy.misc.imread(filename).astype(np.float)
    if len(img.shape) == 2:
        img = np.dstack((img, img, img))
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def single_img_test(model_path,image_path, mask_path):
    f = tf.gfile.FastGFile(model_path, 'rb')
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    persisted_graph = tf.import_graph_def(graph_def, name='')
    sess = tf.InteractiveSession(graph=persisted_graph)

    image = tf.get_default_graph().get_tensor_by_name("image:0")
    mask = tf.get_default_graph().get_tensor_by_name("mask:0")
    output = tf.get_default_graph().get_tensor_by_name("output:0")
    
    image_feed = image_reader(image_path)
    # image_feed = cv2.resize(image_feed, (512, 512))
    
    image_feed = np.expand_dims(image_feed,0)
    mask_feed = image_reader(mask_path)
    # mask_feed = cv2.resize(mask_feed, (512, 512))
    mask_feed = np.expand_dims(mask_feed, 0)
    
    output_value = sess.run(output, feed_dict={image: image_feed,mask: mask_feed})

    print('output_value size:', np.shape(output_value))
    inpainted_img = output_value[0]
    cv2.imwrite("inpainted.png", inpainted_img)
    print("Done")
 

if __name__ == "__main__":
    single_img_test("model_logs/release_places2_256/deepfill.pb","examples/banner/a1.png","examples/banner/a_mask.png" )





