# coding=utf-8
import os
import shutil
import random


def mkdir(folder):
    
    try:
        shutil.rmtree(folder)
    except OSError:
        pass
    os.mkdir(folder)


def add_txt(dataset_path_list, label_list, train_test_rate):
    for dataset_path in dataset_path_list:
        
        with open(dataset_path + '/labels.txt', 'w') as f:
            for label in label_list:
                f.write(label)
                f.write(os.linesep)

        files = os.listdir(dataset_path + '/Images')
        random.shuffle(files)
        split_index = int(len(files) * train_test_rate)

        
        with open(dataset_path + '/train_list.txt', 'w') as f:
            for i in range(split_index):
                f.write('Images/{} Annotations/{}.xml'.format(files[i], files[i].split('.')[0]))
                f.write(os.linesep)

        
        with open(dataset_path + '/test_list.txt', 'w') as f:
            for i in range(split_index, len(files)):
                f.write('Images/{} Annotations/{}.xml'.format(files[i], files[i].split('.')[0]))
                f.write(os.linesep)
