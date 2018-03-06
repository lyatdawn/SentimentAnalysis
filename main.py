# -*- coding:utf-8 -*-
"""
Implementation of Sentiment Analysis using TensorFlow (work in progress).
"""
import os
import glob
import tensorflow as tf
import numpy as np
from model import SA_model

def main(_):
    tf_flags = tf.app.flags.FLAGS
    # gpu config.
    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.8
    config.gpu_options.allow_growth = True

    if tf_flags.phase == "train":
        with tf.Session(config=config) as sess: 
            train_model = SA_model.SA(sess, tf_flags)
            train_model.train(tf_flags.training_steps, tf_flags.summary_steps, tf_flags.checkpoint_steps)
    else:
        with tf.Session(config=config) as sess:
            # val on a image pair.
            val_model = SA_model.SA(sess, tf_flags)
            val_model.load(tf_flags.checkpoint)
            # val
            val_model.val()

if __name__ == '__main__':
    tf.app.flags.DEFINE_string("output_dir", "model_output", 
                               "checkpoint and summary directory.")
    tf.app.flags.DEFINE_string("phase", "train", 
                               "model phase: train/val.")
    tf.app.flags.DEFINE_string("datasets", "./datasets", 
                               "the root path of data.")
    tf.app.flags.DEFINE_integer("batch_size", 64, 
                                "batch size for training.")
    tf.app.flags.DEFINE_integer("num_Classes", 2, 
                                "the number of classes.")
    tf.app.flags.DEFINE_integer("maxSeqLength", 250, 
                                "max sequence length.")
    tf.app.flags.DEFINE_integer("lstmUnits", 64, 
                                "the number of lstm units.")
    tf.app.flags.DEFINE_integer("training_steps", 100000, 
                                "total training steps.")
    tf.app.flags.DEFINE_integer("summary_steps", 100, 
                                "summary period.")
    tf.app.flags.DEFINE_integer("checkpoint_steps", 1000, 
                                "checkpoint period.")
    tf.app.flags.DEFINE_string("checkpoint", None, 
                                "checkpoint name for restoring.")
    tf.app.run(main=main)