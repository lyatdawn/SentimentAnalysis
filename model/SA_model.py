# -*- coding:utf-8 -*-
"""
Load data.
Train and val Sentiment Analysis.
"""
import os
import logging
from datetime import datetime
import time
import tensorflow as tf
import numpy as np
from random import randint

class SA(object):
    def __init__(self, sess, tf_flags):
    	# sess
        self.sess = sess

        # checkpoint and summary.
        self.output_dir = tf_flags.output_dir
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoint")
        self.checkpoint_prefix = "model"
        self.saver_name = "checkpoint"
        self.summary_dir = os.path.join(self.output_dir, "summary")

        self.is_training = (tf_flags.phase == "train") # train or val.

        # data.
        self.wordsVector = np.load(os.path.join(tf_flags.datasets, "wordsVector.npy"))
        self.ids = np.load(os.path.join(tf_flags.datasets, "indexsMatrix.npy"))

        # parameters
        self.batch_size = tf_flags.batch_size
        self.num_Classes = tf_flags.num_Classes
        self.maxSeqLength = tf_flags.maxSeqLength
        self.lstmUnits = tf_flags.lstmUnits
        # placeholder, input data and labels.
        # input data. Here, input data is the index matrix, not the word embeddings.
        # The column of index matrix is maxSeqLength, it will set 250 in this experiment.
        self.input_data = tf.placeholder(tf.int32, [None, self.maxSeqLength])
        # The type of input data is tf.int32, since index is a integer.
        self.labels = tf.placeholder(tf.float32, [None, self.num_Classes])
        # The type of labels is tf.float32.

        # Build model
        self._build_model() # Build model.

        # train
        if self.is_training:
            # makedir aux dir
            self._make_aux_dirs()
            # compute and define loss
            self._build_training()
            # logging, only use in training
            log_file = os.path.join(self.output_dir, "SentimentAnalysis.log")
            logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                                filename=log_file,
                                level=logging.DEBUG,
                                filemode='a+')
            logging.getLogger().addHandler(logging.StreamHandler())
        else:
            # val. Use val() to val directly.
            self.saver = tf.train.Saver(max_to_keep=10, name=self.saver_name) 
            # define saver, after the network!

    '''
    Loading data. return the index matrix and labels. 
    The labels is numpy array, every element is a array, like [1, 0], [0, 1].
    [1, 0] represents positive samples, [0, 1] represents negative samples.

    Because index matrix and labels are small, so we can return these data directly. 
    Or we can use yield to return data.
    indexsMatrix.npy. 0-12499 is positive sample, 12500-24999 is negative sample.
    '''
    def getTrainBatch(self, batch_size, maxSeqLength):
        labels = []
        arr = np.zeros([batch_size, maxSeqLength])
        for i in range(batch_size):
            if (i % 2 == 0): 
                num = randint(1, 11499) # 0-11499, positive samples.
                labels.append([1,0])
            else:
                num = randint(13499, 24999) # 13499-24999, negative samples.
                labels.append([0,1])
            arr[i] = np.reshape(self.ids[num-1:num], (maxSeqLength))
            # In a batch size data, including batch_size/2 positive samples and batch_size/2 negative samples.
        return arr, labels

    def getValBatch(self, batch_size, maxSeqLength):
        # Sample 200 samples from positive and negative datastes.
        # Include 100 positive samples, 100 negative samples.
        labels = []
        arr = np.zeros([batch_size, maxSeqLength])
        for i in range(batch_size):
            num = randint(11499, 13499) # 11499-12499, positive samples; 12500-13499, negative samples.
            if (num <= 12499):
                labels.append([1,0])
            else:
                labels.append([0,1])
            arr[i] = np.reshape(self.ids[num-1:num], (maxSeqLength))
        return arr, labels

    def _build_model(self):
        # word embeddings. Through index matrix and tf.nn.embedding_lookup to find corresponding word 
        # embeddings.
        word_emb = tf.nn.embedding_lookup(self.wordsVector, self.input_data)
        # The shape of input data(index matrix) is [batch_size, maxSeqLength].
        # The shape of wordsVector is [25000, 50]. So the shape of word_emb is [batch_size, 250, 50].
        # i.e., the input of LSTM is 3D Tensor.

        # LSTM network.
        # Call the tf.nn.rnn_cell.BasicLSTMCell() to build LSTM unit. This method takes in an integer for
        # the number of LSTM units. tf.nn.rnn_cell.BasicLSTMCell and tf.contrib.rnn.BasicLSTMCell.
        # Here, we only use one LSTM cell, you can stack multiple LSTM cells. 
        # A LSTM cell like this:
        '''
                        y^(t)
                         |
                         |
                      --------
           c^(t-1) -> |      | -> c^(t)
                      | LSTM | 
           h^(t-1) -> |      | -> h^(t)
                      --------
                         |
                         |
                       x^(t)
        '''
        lstmCell = tf.contrib.rnn.BasicLSTMCell(self.lstmUnits)
        '''
        BasicLSTMCell, class, define in tensorflow/python/ops/rnn_cell_impl.py. 
        The arguments of __init__() are:
        num_units: The number of units in the LSTM cell.
        forget_bias: float, The bias added to forget gates. Default is 1.0.
        activation:  Activation function. Default: tanh.
        reuse.
        '''
        # wrap that LSTM cell in a dropout layer to prevent the network from overfitting.
        lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
        '''
        tf.contrib.rnn.DropoutWrapper and tf.nn.rnn_cell.DropoutWrapper. 
        Operator adding dropout to inputs and outputs of the given cell. 
        The arguments of __init__() are:
        cell: an RNNCell, a projection to output_size is added to it.
        output_keep_prob: output keep probability.
        '''

        # Finally, we’ll feed both the LSTM cell and the 3-D tensor into tf.nn.dynamic_rnn. 
        # This function is in charge of unrolling the whole network and creating a pathway for the data 
        # to flow through the RNN graph.
        # Utlize tf.nn.dynamic_rnn to get the output according the input data, i.e. word embeddings.
        output, _ = tf.nn.dynamic_rnn(lstmCell, word_emb, dtype=tf.float32)
        '''
        tf.nn.dynamic_rnn, define in tensorflow/python/ops/rnn.py. Creates RNN. The arguments are:
        cell: An instance of RNNCell.
        inputs: The RNN inputs.
        dtype: The data type for the initial state and expected output.

        returns (outputs, state):
            outputs: The RNN output Tensor.
            state: The final state.
        '''

        # The first output of the dynamic RNN function can be thought of as the last hidden state vector. 
        # This vector will be reshaped and then multiplied by a final weight matrix and a bias term to 
        # obtain the final output values.
        weight = tf.Variable(tf.truncated_normal(shape=[self.lstmUnits, self.num_Classes]))
        # Outputs random values from a truncated normal distribution. shape is [self.lstmUnits, self.num_Classes].
        # The mean of the truncated normal distribution is 0.0, standard deviation is 1.0. 
        bias = tf.Variable(tf.constant(0.1, shape=[self.num_Classes]))
        # tf.constant, define a constant Tensor.
        output = tf.transpose(output, [1, 0, 2])
        last = tf.gather(output, int(output.get_shape()[0]) - 1)

        # Compute the prediction. The shape of prediction is [batch_size, num_Classes].
        self.prediction = (tf.matmul(last, weight) + bias)
        # When use Tensorflow to classify, the process like this:
        # utlize tf.equal() + tf.argmax() + tf.cast().
        # Define correct prediction and accuracy metrics.
        correctPred = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.labels, 1))
        # Utlize tf.argmax(a, axis=1) to generate the prediction label, the shape is [batch_size].
        # tf.equal(), Returns the truth value of (x == y) element-wise.
        self.accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
        # tf.cast(), transform the Tensor's type to float32.

    def _build_training(self):
        # loss and optimizer.
        # Loss is softmax cross entropy.
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.prediction, labels=self.labels))
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

        # summary
        tf.summary.scalar('Loss', self.loss) # When training, it will output loss.
        tf.summary.scalar('Accuracy', self.accuracy) # When training, it will output accuracy.

        self.summary = tf.summary.merge_all()
        # summary and checkpoint
        self.writer = tf.summary.FileWriter(
            self.summary_dir, graph=self.sess.graph)
        self.saver = tf.train.Saver(max_to_keep=10, name=self.saver_name)
        self.summary_proto = tf.Summary()

    def train(self, training_steps, summary_steps, checkpoint_steps):
        step_num = 0
        # restore last checkpoint
        latest_checkpoint = tf.train.latest_checkpoint("path/checkpoint") 
        # use pretrained model, it can be self.checkpoint_dir, "", or you can appoint the saved checkpoint path.
        # e.g., path/checkpoint
        
        if latest_checkpoint:
            step_num = int(os.path.basename(latest_checkpoint).split("-")[1])
            assert step_num > 0, "Please ensure checkpoint format is model-*.*."
            self.saver.restore(self.sess, latest_checkpoint)
            logging.info("{}: Resume training from step {}. Loaded checkpoint {}".format(datetime.now(), 
                step_num, latest_checkpoint))
        else:
            self.sess.run(tf.global_variables_initializer()) # init all variables
            logging.info("{}: Init new training".format(datetime.now()))

        # train
        c_time = time.time()
        for c_step in xrange(step_num + 1, training_steps + 1):
            # load data. Utlize getTrainBatch(), getValBatch() to load data directly.
            nextBatch, nextBatchLabels = self.getTrainBatch(self.batch_size, self.maxSeqLength);

            c_feed_dict = {
                # numpy ndarray
                self.input_data: nextBatch,
                self.labels: nextBatchLabels
            }

            # Train network.
            self.sess.run(self.optimizer, feed_dict=c_feed_dict)

            # save summary
            if c_step % summary_steps == 0:
                c_summary = self.sess.run(self.summary, feed_dict=c_feed_dict)
                self.writer.add_summary(c_summary, c_step)

                e_time = time.time() - c_time
                time_periter = e_time / summary_steps
                logging.info("{}: Iteration_{} ({:.4f}s/iter) {}".format(
                    datetime.now(), c_step, time_periter,
                    self._print_summary(c_summary)))
                c_time = time.time() # update time

            # save checkpoint
            if c_step % checkpoint_steps == 0:
                self.saver.save(self.sess,
                    os.path.join(self.checkpoint_dir, self.checkpoint_prefix),
                    global_step=c_step)
                logging.info("{}: Iteration_{} Saved checkpoint".format(
                    datetime.now(), c_step))

        logging.info("{}: Done training".format(datetime.now()))

    def load(self, checkpoint_name=None):
        # restore checkpoint
        print("{}: Loading checkpoint...".format(datetime.now())),
        if checkpoint_name:
            checkpoint = os.path.join(self.checkpoint_dir, checkpoint_name)
            self.saver.restore(self.sess, checkpoint)
            print(" loaded {}".format(checkpoint_name))
        else:
            # restore latest model
            latest_checkpoint = tf.train.latest_checkpoint(
                self.checkpoint_dir)
            if latest_checkpoint:
                self.saver.restore(self.sess, latest_checkpoint)
                print(" loaded {}".format(os.path.basename(latest_checkpoint)))
            else:
                raise IOError(
                    "No checkpoints found in {}".format(self.checkpoint_dir))

    def val(self):
        iterations = 10 # Test iterations batch data.
        for i in range(iterations):
            nextBatch, nextBatchLabels = self.getValBatch(self.batch_size, self.maxSeqLength)
            c_feed_dict = {
                # numpy ndarray
                self.input_data: nextBatch,
                self.labels: nextBatchLabels
            }
            # Test 
            accuracy = self.sess.run(self.accuracy, feed_dict=c_feed_dict)
            print("Accuracy for this batch is: {}".format(accuracy))

    def _make_aux_dirs(self):
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def _print_summary(self, summary_string):
        self.summary_proto.ParseFromString(summary_string)
        result = []
        for val in self.summary_proto.value:
            result.append("({}={})".format(val.tag, val.simple_value))
        return " ".join(result)