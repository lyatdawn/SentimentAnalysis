# -*- coding:utf-8 -*-
"""
Testing Sentiment Analysis with a specified sentence.
"""
import os
import re
import numpy as np
import tensorflow as tf

# global parameters
# word list and word vectors.
wordsList = np.load(os.path.join("./datasets", "wordsList.npy")).tolist()
wordsVector = np.load(os.path.join("./datasets", "wordsVector.npy"))
# data
batch_size = 1
num_classes = 2
maxSeqLength = 250

# Removes punctuation, parentheses, question marks, etc., and leaves only alphanumeric characters
def cleanSentences(string):
    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
    string = string.lower().replace("<br />", " ")

    return re.sub(strip_special_chars, "", string.lower())

def getSentenceMatrix(sentence):
    sentenceMatrix = np.zeros([batch_size, maxSeqLength], dtype='int32') + 1343
    cleanedSentence = cleanSentences(sentence)

    split = cleanedSentence.split()
    for indexCounter, word in enumerate(split):
        try:
            sentenceMatrix[0, indexCounter] = wordsList.index(word)
        except ValueError:
            sentenceMatrix[0, indexCounter] = 201534 # Vector for unkown words

    return sentenceMatrix

if __name__ == '__main__':
    input_data = tf.placeholder(tf.int32, [None, maxSeqLength])

    # build model
    # word embeddings.
    word_emb = tf.nn.embedding_lookup(wordsVector, input_data)

    # LSTM network.
    lstmUnits = 64
    lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
    lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
    output, _ = tf.nn.dynamic_rnn(lstmCell, word_emb, dtype=tf.float32)
    
    weight = tf.Variable(tf.truncated_normal(shape=[lstmUnits, num_classes]))
    bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))
    output = tf.transpose(output, [1, 0, 2])
    last = tf.gather(output, int(output.get_shape()[0]) - 1)

    # Compute the prediction.
    prediction = (tf.matmul(last, weight) + bias)

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint("model_output_20180305210852/checkpoint"))
    print("Successfully loading trained model.")

    # inputText = "That movie was terrible." # negative sample
    inputText = "That movie was the best one I have ever seen." # positive sample
    inputMatrix = getSentenceMatrix(inputText)

    predictedSentiment = sess.run(prediction, {input_data: inputMatrix})[0]
    # The first dim of output is batch size.
    # predictedSentiment[0] represents output score for positive sentiment.
    # predictedSentiment[1] represents output score for negative sentiment.

    if (predictedSentiment[0] > predictedSentiment[1]):
        print "Positive Sentiment."
    else:
        print "Negative Sentiment."