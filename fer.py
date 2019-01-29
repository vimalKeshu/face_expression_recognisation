# pylint: enable=line-too-long

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from datetime import datetime
import argparse
from model import extract_features
from data_prep_ferplus import create_dataset, read_image

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
    # Make predictions
    classes = 8
    image_size = 64
    logits = extract_features(features,classes,image_size,mode)

    # We just want the predictions
    if mode == tf.estimator.ModeKeys.PREDICT:
        outputs = {
            "predicted_label": tf.argmax(input=logits, axis=1),
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=outputs)
    
    # If not in mode.PREDICT, compute the loss (mean squared error)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    outputs = {
        "predicted_label": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
        "loss" : loss
    }

    # Single optimization step
    if mode == tf.estimator.ModeKeys.TRAIN:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    
    # If not PREDICT or TRAIN, then we are evaluating the model
    eval_metric_ops = {
      "rmse": tf.metrics.accuracy(
          labels=labels, predictions=outputs["probabilities"])}
    return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def input_fn(features, labels, batch_size, is_eval=False):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    features = tf.constant(features)
    labels = tf.constant(labels)
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.map(read_image)

    # Shuffle, repeat, and batch the examples.
    if is_eval:
        return dataset.batch(batch_size)
    else:
        return dataset.shuffle(100).repeat().batch(batch_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
      '--train_dir',
      type=str,
      default='',
      help='Path to folders of labeled images.'
    )

    parser.add_argument(
      '--eval_dir',
      type=str,
      default='',
      help='Path to folders of labeled images.'
    )

    parser.add_argument(
      '--test_dir',
      type=str,
      default='',
      help='Path to folders of labeled images.'
    )
    
    FLAGS, unparsed = parser.parse_known_args()

    # Build the Estimator
    model_name = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    model_dir = "/Users/vimal/git/fer/output/"+model_name
    face_expression_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, 
                                                        model_dir=model_dir,
                                                        warm_start_from="/Users/vimal/git/fer/output/20190125-110715/"
                                                        )

    # Train the model
    # Set up logging for train
    if FLAGS.train_dir:
        train_files, train_labels = create_dataset(FLAGS.train_dir)
        #tensors_to_log = {"loss": "loss"}
        #logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)    
        face_expression_classifier.train(input_fn=lambda: input_fn(train_files, train_labels, 32), steps=20000)

    # Eval the model
    if FLAGS.eval_dir:
        eval_files, eval_labels = create_dataset(FLAGS.eval_dir)
        eval_result = face_expression_classifier.evaluate(input_fn=lambda: input_fn(eval_files, eval_labels, 64, True))
        #print('\nEval set accuracy: {}\n'.format(**eval_result))

    # Eval the model
    if FLAGS.test_dir:
        predict_files, predict_labels = create_dataset(FLAGS.test_dir)
        predict_results = face_expression_classifier.predict(input_fn=lambda: input_fn(predict_files, predict_labels, 64, True))
        label_dict = {4:'angry',5:'disgust',6:'fear',1:'happy',0:'neutral',2:'surprise',3:'sad',7:'contempt'}
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