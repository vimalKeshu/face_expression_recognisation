# pylint: enable=line-too-long

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
from model import extract_features


tf.logging.set_verbosity(tf.logging.FATAL)
CLASSES = 8
IMAGE_SIZE = 64
LABEL_DICT = {0:'neutral',1:'happy',2:'surprise',3:'sad',4:'angry',5:'disgust',6:'fear',7:'contempt'}

class Fer:
    
    def __init__(self,model):
        self.model = model

    def __cnn_model_fn(self,features, labels, mode):
        # Make predictions
        logits = extract_features(features,CLASSES,IMAGE_SIZE,mode)
        outputs = {
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=outputs)

    def build(self):
        self.classifier = tf.estimator.Estimator(model_fn=self.__cnn_model_fn,warm_start_from=self.model)
        
    def find(self,images):
        predict_results = self.classifier.predict(input_fn=lambda: np.array([images]))
        result = []
        for predict in predict_results:
                expression = LABEL_DICT[np.argmax(predict['probabilities'])]
                tf.logging.info(expression)
                result.append(expression)
        return result

if __name__ == "__main__":

    from data_prep_ferplus import create_dataset
    parser = argparse.ArgumentParser()

    parser.add_argument(
      'test_dir',
      type=str,
      default='',
      help='Path to image directory.'
    )
    parser.add_argument(
      'model',
      type=str,
      default='',
      help='Path to pre trained model directory.'
    )
    FLAGS, unparsed = parser.parse_known_args()

    fer = Fer(FLAGS.model)
    fer.build()
    predict_files, predict_labels = create_dataset(FLAGS.test_dir,image_size=IMAGE_SIZE)
    result = fer.find(predict_files)
    for exp in result:
        print(exp)