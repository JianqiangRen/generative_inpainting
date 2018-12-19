# coding=utf-8
# summary:
# author: xueluo
# date:

import tensorflow as tf
import logging
import glob
import os
import cv2
import scipy
import numpy as np


MODEL_PATH = "model_logs/epoch_8/deepfill.pb"
INPUT_IMG_FOLDER = "jpg_img"
INPUT_MASK_FOLDER = "jpg_mask"
OUTPUT_FOLDER = "jpg_result"


def image_reader(filename):
    """help fn that provides numpy image coding utilities"""
    img = scipy.misc.imread(filename).astype(np.float)
    if len(img.shape) == 2:
        img = np.dstack((img, img, img))
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img


if __name__ == "__main__":
    images = glob.glob(os.path.join(INPUT_IMG_FOLDER, "*.*"))
    print("images number: {}".format(len(images)))
    for i, image_path in enumerate(images):
        print("processing #{} image".format(i))
        img_name = image_path.split('/')[-1]
        mask_name = 'mask_'+ img_name
  
        f = tf.gfile.FastGFile(MODEL_PATH, 'rb')
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        persisted_graph = tf.import_graph_def(graph_def, name='')
        sess = tf.InteractiveSession(graph=persisted_graph)

        image = tf.get_default_graph().get_tensor_by_name("image:0")
        mask = tf.get_default_graph().get_tensor_by_name("mask:0")
        output = tf.get_default_graph().get_tensor_by_name("output:0")
        
        if not os.path.exists(image_path):
            continue
        image_feed = image_reader(image_path)
        width = np.shape(image_feed)[1]
        height = np.shape(image_feed)[0]

        print(np.shape(image_feed))
        length = 800
        if np.shape(image_feed)[0] > length or np.shape(image_feed)[1] > length:
            if np.shape(image_feed)[0] > np.shape(image_feed)[1]:
                image_feed = cv2.resize(image_feed, (
                int(np.shape(image_feed)[1] * length / np.shape(image_feed)[0]) // 8 * 8, length))
            else:
                image_feed = cv2.resize(image_feed, (
                length, int(np.shape(image_feed)[0] * length / np.shape(image_feed)[1]) // 8 * 8))
        else:
            image_feed = cv2.resize(image_feed,
                                    (int(np.shape(image_feed)[1] // 8 * 8), int(np.shape(image_feed)[0] // 8 * 8)))

        print(np.shape(image_feed))
        
        if not os.path.exists(INPUT_MASK_FOLDER + "/" + mask_name):
            continue

        mask_feed = image_reader(INPUT_MASK_FOLDER + "/" + mask_name)
        mask_feed = cv2.resize(mask_feed, (np.shape(image_feed)[1], np.shape(image_feed)[0]))
        image_feed = np.expand_dims(image_feed, 0)
        mask_feed = np.expand_dims(mask_feed, 0)
        output_value = sess.run(output, feed_dict={image: image_feed, mask: mask_feed})

        print('output_value size:', np.shape(output_value))
        inpainted_img = output_value[0]
        inpainted_img = cv2.resize(inpainted_img, (width, height))
        cv2.imwrite(OUTPUT_FOLDER + "/filter_" + img_name, inpainted_img)