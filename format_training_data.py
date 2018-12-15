# coding=utf-8
# summary:
# author: xueluo
# date:
import argparse
import os
from random import shuffle

parser = argparse.ArgumentParser()
parser.add_argument('--folder_path', default='./training_data', type=str,
                    help='The folder path')
parser.add_argument('--train_filename', default='./data_flist/train_shuffled.flist', type=str,
                    help='The train filename.')
parser.add_argument('--validation_filename', default='./data_flist/validation_shuffled.flist', type=str,
                    help='The validation filename.')
parser.add_argument('--is_shuffled', default='1', type=int,
                    help='Needed to be shuffled')

if __name__ == "__main__":

    args = parser.parse_args()

    # get the list of directories and separate them into 2 types: training and validation
    training_dirs = os.listdir(args.folder_path + "/training")
    validation_dirs = os.listdir(args.folder_path + "/validation")

    # make 2 lists to save file paths
    training_file_names = []
    validation_file_names = []

 
    training_folder = os.listdir(args.folder_path + "/training" )
    for training_item in training_folder:
        # modify to full path -> directory
        if training_item.startswith("."):
            continue
        training_item = args.folder_path + "/training" + "/"+training_item
        training_file_names.append(training_item)

 
    validation_folder = os.listdir(args.folder_path + "/validation" )
    for validation_item in validation_folder:
        # modify to full path -> directory
        if validation_item.startswith("."):
            continue
        validation_item = args.folder_path + "/validation" + "/"  + validation_item
        validation_file_names.append(validation_item)


    # print all file paths
    for i in training_file_names:
        print(i)
    for i in validation_file_names:
        print(i)
    print('training sample number :{}'.format(len(training_file_names)))
    print('validation sample number :{}'.format(len(validation_file_names)))

    # shuffle file names if set
    if args.is_shuffled == 1:
        shuffle(training_file_names)
        shuffle(validation_file_names)

    # make output file if not existed
    # if not os.path.exists(args.train_filename):
    #     os.mknod(args.train_filename)
    #
    # if not os.path.exists(args.validation_filename):
    #     os.mknod(args.validation_filename)

    # write to file
    fo = open(args.train_filename, "w")
    fo.write("\n".join(training_file_names))
    fo.close()

    fo = open(args.validation_filename, "w")
    fo.write("\n".join(validation_file_names))
    fo.close()

    # print process
    print("Written file is: ", args.train_filename, ", is_shuffle: ", args.is_shuffled)