# coding=utf-8
# summary:
# author: jianqiang ren
# date:
import tensorflow as tf
import scipy.misc
import numpy as np
from PIL import Image
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", dest='model', type = str)
parser.add_argument("--image", dest='image',type=str)
parser.add_argument("--mask", dest='mask',type=str)
parser.add_argument("--output", dest='output',type=str)

args=parser.parse_args()

def image_reader(filename):
    """help fn that provides numpy image coding utilities"""
    img = scipy.misc.imread(filename).astype(np.float)
    if len(img.shape) == 2:
        img = np.dstack((img, img, img))
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def single_img_test(model_path,image_path, mask_path, out_path):
    f = tf.gfile.FastGFile(model_path, 'rb')
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    persisted_graph = tf.import_graph_def(graph_def, name='')
    sess = tf.InteractiveSession(graph=persisted_graph)

    image = tf.get_default_graph().get_tensor_by_name("image:0")
    mask = tf.get_default_graph().get_tensor_by_name("mask:0")
    output = tf.get_default_graph().get_tensor_by_name("output:0")
    
    image_feed = image_reader(image_path)
    width = np.shape(image_feed)[1]
    height = np.shape(image_feed)[0]
    
    print(np.shape(image_feed))
    
    if np.shape(image_feed)[0] > 800 or np.shape(image_feed)[1] > 800:
        if np.shape(image_feed)[0] > np.shape(image_feed)[1]:
            image_feed = cv2.resize(image_feed, (int(np.shape(image_feed)[1]* 800/ np.shape(image_feed)[0])//8*8, 800))
        else:
            image_feed = cv2.resize(image_feed, (800,int( np.shape(image_feed)[0] * 800/np.shape(image_feed)[1])//8*8))
    
    image_feed = np.expand_dims(image_feed,0)
    print(np.shape(image_feed))
    
    mask_feed = image_reader(mask_path)
    mask_feed = cv2.resize(mask_feed, (np.shape(image_feed)[2],np.shape(image_feed)[1]))
    mask_feed = np.expand_dims(mask_feed, 0)
    
    output_value = sess.run(output, feed_dict={image: image_feed,mask: mask_feed})

    print('output_value size:', np.shape(output_value))
    inpainted_img = output_value[0]
    inpainted_img = cv2.resize(inpainted_img, (width, height))
    cv2.imwrite(out_path, inpainted_img)
    print("Done")
 

if __name__ == "__main__":
    single_img_test(args.model, args.image,args.mask,args.output)
    print("write to "+ args.output)





