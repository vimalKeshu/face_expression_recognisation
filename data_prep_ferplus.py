# pylint: enable=line-too-long

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import re
import sys
import csv
import numpy as np
import logging
import random as rnd
import tensorflow as tf

def read_image(filename, label, image_size=64):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_png(image_string)
    image_resized = tf.image.resize_images(image_decoded, [image_size, image_size])
    return image_resized, label

emotion_table = {'neutral'  : 0, 
                 'happiness': 1, 
                 'surprise' : 2, 
                 'sadness'  : 3, 
                 'anger'    : 4, 
                 'disgust'  : 5, 
                 'fear'     : 6, 
                 'contempt' : 7}
                 
def create_dataset(folder_path, label_file_name='label.csv', mode='majority', emotion_count=8):
    images = []
    labels = []
    #per_emotion_count = np.zeros(emotion_count, dtype=np.int)

    in_label_path = os.path.join(folder_path, label_file_name)
    with open(in_label_path) as csvfile: 
        emotion_label = csv.reader(csvfile) 
        for row in emotion_label: 
            # image path
            image_path = os.path.join(folder_path, row[0])

            # face rectangle 
            #box = list(map(int, row[1][1:-1].split(',')))
            #face_rc = Rect(box)

            emotion_raw = list(map(float, row[2:len(row)]))
            emotion = _process_data(emotion_raw, mode) 
            idx = np.argmax(emotion)
            if idx < emotion_count: # not unknown or non-face 
                emotion = emotion[:-2]
                emotion = [float(i)/sum(emotion) for i in emotion]
                images.append(image_path)
                labels.append(np.argmax(emotion))
                #per_emotion_count[idx] += 1

    assert len(images) == len(labels), "Files and labels lists are not same length."

    images = np.array(images)
    labels = np.array(labels,dtype=np.int32)

    indices = np.arange(len(images))
    np.random.shuffle(indices)
    
    return images[indices], labels[indices]

def _process_data(emotion_raw, mode):        
    size = len(emotion_raw)
    emotion_unknown     = [0.0] * size
    emotion_unknown[-2] = 1.0

    # remove emotions with a single vote (outlier removal) 
    for i in range(size):
        if emotion_raw[i] < 1.0 + sys.float_info.epsilon:
            emotion_raw[i] = 0.0

    sum_list = sum(emotion_raw)
    emotion = [0.0] * size 

    if mode == 'majority': 
        # find the peak value of the emo_raw list 
        maxval = max(emotion_raw) 
        if maxval > 0.5*sum_list: 
            emotion[np.argmax(emotion_raw)] = maxval 
        else: 
            emotion = emotion_unknown   # force setting as unknown 
    elif (mode == 'probability') or (mode == 'crossentropy'):
        sum_part = 0
        count = 0
        valid_emotion = True
        while sum_part < 0.75*sum_list and count < 3 and valid_emotion:
            maxval = max(emotion_raw) 
            for i in range(size): 
                if emotion_raw[i] == maxval: 
                    emotion[i] = maxval
                    emotion_raw[i] = 0
                    sum_part += emotion[i]
                    count += 1
                    if i >= 8:  # unknown or non-face share same number of max votes 
                        valid_emotion = False
                        if sum(emotion) > maxval:   # there have been other emotions ahead of unknown or non-face
                            emotion[i] = 0
                            count -= 1
                        break
        if sum(emotion) <= 0.5*sum_list or count > 3: # less than 50% of the votes are integrated, or there are too many emotions, we'd better discard this example
            emotion = emotion_unknown   # force setting as unknown 
    elif mode == 'multi_target':
        threshold = 0.3
        for i in range(size): 
            if emotion_raw[i] >= threshold*sum_list: 
                emotion[i] = emotion_raw[i] 
        if sum(emotion) <= 0.5 * sum_list: # less than 50% of the votes are integrated, we discard this example 
            emotion = emotion_unknown   # set as unknown 
                            
    return [float(i)/sum(emotion) for i in emotion]

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
    
