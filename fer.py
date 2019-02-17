# pylint: enable=line-too-long

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import cv2
from datetime import datetime
from model import extract_features
from data_prep_ferplus import create_dataset

tf.logging.set_verbosity(tf.logging.INFO)
CLASSES = 8
IMAGE_SIZE = 64
label_dict = {0:'neutral',
              1:'happy',
              2:'surprise',
              3:'sad',
              4:'angry',
              5:'disgust',
              6:'fear',
              7:'contempt'}
              
def cnn_model_fn(features, labels, mode):
    # Make predictions
    logits = extract_features(features,CLASSES,IMAGE_SIZE,mode)
    predicted_classes = tf.argmax(input=logits, axis=1)

    # Predict the model
    if mode == tf.estimator.ModeKeys.PREDICT:
        outputs = {
            'class_ids': predicted_classes[:, tf.newaxis],
            "predicted_label": predicted_classes,
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=outputs)
    
    # If not in mode.PREDICT, compute the loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Train the model
    if mode == tf.estimator.ModeKeys.TRAIN:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    
    # Eval the model
    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                               predictions=predicted_classes,
                               name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=metrics)

def input_fn(features, labels, batch_size=64, is_eval=False):
    # Convert the inputs to a Dataset.
    features = tf.constant(features)
    labels = tf.constant(labels)
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))

    # Shuffle, repeat, and batch the examples.
    if is_eval:
        return dataset.batch(batch_size)
    else:
        return dataset.shuffle(100).repeat().batch(batch_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
      'train_dir',
      type=str,
      default='',
      help='Path to train folder.'
    )

    parser.add_argument(
      'eval_dir',
      type=str,
      default='',
      help='Path to eval folder.'
    )

    parser.add_argument(
      'test_dir',
      type=str,
      default='',
      help='Path to test folder.'
    )

    parser.add_argument(
      '--output_dir',
      type=str,
      default='~/fer/output',
      help='Path to test folder.'
    )    

    FLAGS, unparsed = parser.parse_known_args()

    # Build the Estimator
    model_name = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    output_dir = os.path.join(FLAGS.output_dir,model_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fer_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn,
                                            model_dir=output_dir)

    # Train the model
    train_files, train_labels = create_dataset(FLAGS.train_dir,image_size=IMAGE_SIZE)
    '''
    tensors_to_log = {"loss": "loss"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=1000)
    fer_classifier.Train(input_fn=lambda: input_fn(train_files, train_labels), hooks=[logging_hook], max_steps=20000)
    '''
    fer_classifier.train(input_fn=lambda: input_fn(train_files, train_labels), max_steps=20000)
    
    # Eval the model 
    eval_files, eval_labels = create_dataset(FLAGS.eval_dir,image_size=IMAGE_SIZE)
    eval_result = fer_classifier.evaluate(input_fn=lambda: input_fn(eval_files, eval_labels, 64, True))
    print('\nEval set accuracy: {}\n'.format(eval_result))

    # Test the model
    predict_files, predict_labels = create_dataset(FLAGS.test_dir,image_size=IMAGE_SIZE)
    predict_results = fer_classifier.predict(input_fn=lambda: input_fn(predict_files, predict_labels, 1, True))
    
    template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')
    count = 0
    correct_prediction = 0
    wrong_prediction = 0
    for predict, label in zip(predict_results,predict_labels):
        count = count + 1
        class_id = predict['predicted_label']
        probability = predict['probabilities'][class_id]
        #index = np.argmax(label,axis=0)
        if label_dict[class_id] == label_dict[label]:
            correct_prediction += 1
            #print(template.format(label_dict[class_id],100 * probability, label_dict[index]))
        else:
            wrong_prediction += 1
    pred = (correct_prediction/count)*100
    print("Correct prediction {} out of {}. Accuracy: {}".format(correct_prediction,count,pred))
    pred1 = (wrong_prediction/count)*100
    print("Wrong prediction {} out of {}. Accuracy: {}".format(wrong_prediction,count,pred1))