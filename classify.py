# pylint: enable=line-too-long

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import cv2
from model import extract_features

#tf.logging.set_verbosity(tf.logging.INFO)
classes = 8
image_size = 64

label_dict =   {4:'angry',
                5:'disgust',
                6:'fear',
                1:'happy',
                0:'neutral',
                2:'surprise',
                3:'sad',
                7:'contempt'}

def create_dataset(image_folder):
    file_list = []
    extensions = sorted(set(os.path.normcase(ext) for ext in ['JPEG', 'JPG', 'jpeg', 'jpg','png', 'PNG']))
    for extension in extensions:
        file_glob = os.path.join(image_folder, '*.' + extension)
        file_list.extend(tf.gfile.Glob(file_glob))
    assert len(file_list)!=0, 'Not able to find the image files.'
    images = []
    for file in file_list:
        image = cv2.imread(file,cv2.IMREAD_GRAYSCALE)
        if image is not None:
            image = np.resize(image,(image_size,image_size))
            image = np.array(image,dtype=np.float32)
            images.append(image)
    images = np.array(images)
    return images

def cnn_model_fn(features, mode):
    # Make predictions
    logits = extract_features(features,classes,image_size,mode)
    outputs = {
        "predicted_label": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    return tf.estimator.EstimatorSpec(mode=mode, predictions=outputs)

def input_fn(features, batch_size=1):
    features = tf.constant(features)
    #features = tf.image.rgb_to_grayscale(features)
    dataset = tf.data.Dataset.from_tensor_slices(features)
    #dataset = dataset.map(read_image)
    return dataset.batch(batch_size)

def read_image(image):
    image = tf.image.rgb_to_grayscale(image)
    image = tf.reshape(image,[64,64])
    return image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
      'test_dir',
      type=str,
      default='',
      help='Path to image directory.'
    )
    parser.add_argument(
      '--pre_trained_model',
      type=str,
      default='',
      help='Path to image directory.'
    )
  
    FLAGS, unparsed = parser.parse_known_args()

    # Build the Estimator
    face_expression_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn,warm_start_from=FLAGS.pre_trained_model)

    predict_files = create_dataset(FLAGS.test_dir)
    predict_results = face_expression_classifier.predict(input_fn=lambda: input_fn(predict_files))
    
    template = ('\nPrediction is "{}" ({:.1f}%)')
    for predict in predict_results:
        class_id = np.argmax(predict['probabilities'])
        probability = predict['probabilities'][class_id]
        print(predict['probabilities'])
        print(template.format(label_dict[class_id],100 * probability))

    