# pylint: enable=line-too-long

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import re
import sys
import csv
import cv2
import numpy as np
import logging
import random as rnd
import tensorflow as tf

emotion_table = {'neutral'  : 0, 
                 'happiness': 1, 
                 'surprise' : 2, 
                 'sadness'  : 3, 
                 'anger'    : 4, 
                 'disgust'  : 5, 
                 'fear'     : 6, 
                 'contempt' : 7}

def _process_data(emotion_raw):        
    size = len(emotion_raw)
    emotion_unknown     = [0.0] * size
    emotion_unknown[-2] = 1.0

    # remove emotions with a single vote (outlier removal) 
    for i in range(size):
        if emotion_raw[i] < 1.0 + sys.float_info.epsilon:
            emotion_raw[i] = 0.0

    sum_list = sum(emotion_raw)
    emotion = [0.0] * size 

    # find the peak value of the emo_raw list 
    maxval = max(emotion_raw) 
    if maxval > 0.5*sum_list: 
        emotion[np.argmax(emotion_raw)] = maxval 
    else: 
        emotion = emotion_unknown   # force setting as unknown                             
    return [float(i)/sum(emotion) for i in emotion]
                 
def create_dataset(folder_path, label_file_name='label.csv', image_size=64, emotion_count=8):
    images = []
    labels = []

    in_label_path = os.path.join(folder_path, label_file_name)
    with open(in_label_path) as csvfile: 
        emotion_label = csv.reader(csvfile) 
        for row in emotion_label: 
            # image path
            image_path = os.path.join(folder_path, row[0])

            emotion_raw = list(map(float, row[2:len(row)]))
            emotion = _process_data(emotion_raw) 
            idx = np.argmax(emotion)
            if idx < emotion_count: # not unknown or non-face 
                emotion = emotion[:-2]
                emotion = [float(i)/sum(emotion) for i in emotion]
                image = cv2.imread(image_path,0)
                image = np.array(image,dtype=np.float32)
                image = np.resize(image,[image_size,image_size])
                images.append(image)
                labels.append(np.argmax(emotion))

    assert len(images) == len(labels), "Files and labels lists are not same length."

    images = np.array(images)
    labels = np.array(labels,dtype=np.int32)

    indices = np.arange(len(images))
    np.random.shuffle(indices)
    
    return images[indices], labels[indices]



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
      'image_dir',
      type=str,
      default='',
      help='Path to folders of labeled images.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    files, labels = create_dataset(FLAGS.image_dir)
    print(len(files))
    print(len(labels))
    
