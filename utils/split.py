import os
import numpy as np
import cv2
from PIL import Image
import random
from tqdm import tqdm

num_test = 10

def no_people_split(src,no,num):
    no_test = 5
    no_list = os.listdir(src)
    no_img_list = []
    no_index_list = []

    for item in no_list:
        if item.find('.JPG') != -1:
            if item[:10] not in no_index_list:
                no_index_list.append(item[:10])
            no_img_list.append(item)

    temp = random.sample(no_index_list,no_test)

    index = 0
    no_s = []
    for i in temp :    
        for item in no_img_list:
            if item.find(i) != -1:
                no_s.append(item)
                index = index + 1
        if index >= no/10:
            break
    tatal_len = len(no_s)
    index = 0

    f = open("./result/{}/data/normal_val.txt".format(num), 'w')
    with tqdm(total=tatal_len, desc='Spliting normal validation', unit='img', leave=False) as pbar:
        for name in sorted(no_s) :
            no_img_list.remove(name)
            data = '{}\n'.format(name)
            f.write(data)
            pbar.update(index)
            index = index + 1
    f.close()

    temp = random.sample(no_index_list,no_test)

    index = 0
    no_s = []
    for i in temp :    
        for item in no_img_list:
            if item.find(i) != -1:
                no_s.append(item)
                index = index + 1
        if index >= no/10:
            break
    tatal_len = len(no_s)
    index = 0

    f = open("./result/{}/data/normal_test.txt".format(num), 'w')
    with tqdm(total=tatal_len, desc='Spliting normal test', unit='img', leave=False) as pbar:
        for name in sorted(no_s) :
            no_img_list.remove(name)
            data = '{}\n'.format(name)
            f.write(data)
            pbar.update(index)
            index = index + 1
    f.close()

    tatal_len = len(no_img_list)
    index = 0
    f = open("./result/{}/data/normal_train.txt".format(num), 'w')
    with tqdm(total=tatal_len, desc='Spliting normal train', unit='img', leave=False) as pbar:
        for name in sorted(no_img_list) :
            data = '{}\n'.format(name)
            f.write(data)
            pbar.update(index)
            index = index + 1
    f.close()

def ul_people_split(src,ul,num):
    ul_test = 5
    ul_list = os.listdir(src)
    ul_img_list = []
    ul_index_list = []

    for item in ul_list:
        if item.find('.JPG') != -1:
            if item[:10] not in ul_index_list:
                ul_index_list.append(item[:10])
            ul_img_list.append(item)

    temp = random.sample(ul_index_list,ul_test)

    index = 0
    ul_s = []
    for i in temp :    
        for item in ul_img_list:
            if item.find(i) != -1:
                ul_s.append(item)
                index = index + 1
        if index >= ul/10:
            break
    tatal_len = len(ul_s)
    index = 0

    f = open("./result/{}/data/ulcer_val.txt".format(num), 'w')
    with tqdm(total=tatal_len, desc='Spliting ulcer validation', unit='img', leave=False) as pbar:
        for name in sorted(ul_s) :
            ul_img_list.remove(name)
            data = '{}\n'.format(name)
            f.write(data)
            pbar.update(index)
            index = index + 1
    f.close()

    temp = random.sample(ul_index_list,ul_test)

    index = 0
    ul_s = []
    for i in temp :    
        for item in ul_img_list:
            if item.find(i) != -1:
                ul_s.append(item)
                index = index + 1
        if index >= ul/10:
            break
    tatal_len = len(ul_s)
    index = 0
    f = open("./result/{}/data/ulcer_test.txt".format(num), 'w')
    with tqdm(total=tatal_len, desc='Spliting ulcer test', unit='img', leave=False) as pbar:
        for name in sorted(ul_s) :
            ul_img_list.remove(name)
            data = '{}\n'.format(name)
            f.write(data)
            pbar.update(index)
            index = index + 1
    f.close()

    tatal_len = len(ul_img_list)
    index = 0
    f = open("./result/{}/data/ulcer_train.txt".format(num), 'w')
    with tqdm(total=tatal_len, desc='Spliting ulcer train', unit='img', leave=False) as pbar:
        for name in sorted(ul_img_list) :
            data = '{}\n'.format(name)
            f.write(data)
            pbar.update(index)
            index = index + 1
    f.close()

def no_instance_split(src,no,num):
    no_test = int(no/num_test)
    no_list = os.listdir(src)
    no_img_list = []

    for item in no_list:
        if item.find('.JPG') != -1:
            no_img_list.append(item)

    no_s = random.sample(no_img_list,no_test)

    tatal_len = len(no_s)
    index = 0
    f = open("./result/{}/data/normal_val.txt".format(num), 'w')
    with tqdm(total=tatal_len, desc='Spliting normal validation', unit='img', leave=False) as pbar:
        for name in sorted(no_s) :
            no_img_list.remove(name)
            data = '{}\n'.format(name)
            f.write(data)
            pbar.update(index)
            index = index + 1
    f.close()
    no_s = random.sample(no_img_list,no_test)
    tatal_len = len(no_s)
    index = 0
    f = open("./result/{}/data/normal_test.txt".format(num), 'w')
    with tqdm(total=tatal_len, desc='Spliting normal test', unit='img', leave=False) as pbar:
        for name in sorted(no_s) :
            no_img_list.remove(name)
            data = '{}\n'.format(name)
            f.write(data)
            pbar.update(index)
            index = index + 1
    f.close()
    tatal_len = len(no_img_list)
    index = 0
    f = open("./result/{}/data/normal_train.txt".format(num), 'w')
    with tqdm(total=tatal_len, desc='Spliting normal train', unit='img', leave=False) as pbar:
        for name in sorted(no_img_list) :
            data = '{}\n'.format(name)
            f.write(data)
            pbar.update(index)
            index = index + 1
    f.close()

def ul_instance_split(src,ul,num):
    ul_test = int(ul/num_test)
    ul_list = os.listdir(src)
    ul_img_list = []

    for item in ul_list:
        if item.find('.JPG') != -1:
            ul_img_list.append(item)

    ul_s = random.sample(ul_img_list,ul_test)

    tatal_len = len(ul_s)
    index = 0
    f = open("./result/{}/data/ulcer_val.txt".format(num), 'w')
    with tqdm(total=tatal_len, desc='Spliting ulcer validation', unit='img', leave=False) as pbar:
        for name in sorted(ul_s) :
            ul_img_list.remove(name)
            data = '{}\n'.format(name)
            f.write(data)
            pbar.update(index)
            index = index + 1
    f.close()
    ul_s = random.sample(ul_img_list,ul_test)
    tatal_len = len(ul_s)
    index = 0
    f = open("./result/{}/data/ulcer_test.txt".format(num), 'w')
    with tqdm(total=tatal_len, desc='Spliting ulcer test', unit='img', leave=False) as pbar:
        for name in sorted(ul_s) :
            ul_img_list.remove(name)
            data = '{}\n'.format(name)
            f.write(data)
            pbar.update(index)
            index = index + 1
    f.close()
    tatal_len = len(ul_img_list)
    index = 0
    f = open("./result/{}/data/ulcer_train.txt".format(num), 'w')
    with tqdm(total=tatal_len, desc='Spliting ulcer train', unit='img', leave=False) as pbar:
        for name in sorted(ul_img_list) :
            data = '{}\n'.format(name)
            f.write(data)
            pbar.update(index)
            index = index + 1
    f.close()