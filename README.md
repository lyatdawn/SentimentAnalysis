# Sentiment Analysis
* Tensorflow implement of Sentiment Analysis.
* Borrowed code and ideas from adeshpande3's LSTM-Sentiment-Analysis: https://github.com/adeshpande3/LSTM-Sentiment-Analysis.

## Install Required Packages
First ensure that you have installed the following required packages:
* TensorFlow1.4.0 ([instructions](https://www.tensorflow.org/install/)). Maybe other version is ok.

See requirements.txt for details.

## Datasets
* To implement Sentiment Analysis with Tensorflow, we will use IMDB movie reviews dataset. This set has train and test dataset, each includes 25000 movie reviews. In this experiment, we only use train datasets, then split it to train and val set. This train set has 25,000 movie reviews, with 12,500 positive 
reviews and 12,500 negative reviews. You can put all these movie reviews in datasets floder, the positive reviews are stored in one directory and the negative reviews are stored in another.
* In the process of experiment, we will use the pretrained word vectors with Glove. The version of pretrained word vector is glove.6B.50d.txt. The glove.6B.50d.txt includes 40K words, every word is expressed to 50d vector. You can down those txt files in https://nlp.stanford.edu/projects/glove/. First, use **scripts/generate_word_list_and_vectors.py** to generate word list and word vectors. We will save the word list and word vectors to numpy array.
* Before transforming a sentence to a word vectors, we should get the best value for setting our maximum sequence length. The max sequence length value is largely dependent on the inputs you have. Run **scripts/generate_maxSeqLength.py** to generate the max sequence length. From the histogram we have plotted and the average number of words per file, we can safely say that most reviews will fall under 250 words, which is the max sequence length value we will set.
* Run **scripts/generate_indexMatrix.py** creating the ids matrix for the whole training set. Use the indexs matrix and tensorflow.nn.embedding_lookup(), we can look up the word embeddings through the indexs.

## Training and Valing Model
* Run the following script to train the model, in the process of training, we will output the loss and classification accuracy every 500 steps. See the **model/SA_model.py** for details.
```shell
sh train.sh
```
You can change the arguments in train.sh depend on your machine config.
* Run the following script to val the trained model.
```shell
sh val.sh
```
This script will load the trained SA model to classify a specified sentence. You could change the arguments in val.sh depend on your machine config.

## Testing
* Run the following script to test the trained model. In the process of testing, we will specify a positive sentence and a negative sentence, then use the trained model to classify.
```
python test.py
```

## Downloading trained model
* Pretrained model: [[download]](https://drive.google.com/open?id=1FJ__XNfu9rVH2CQQ8sU81taVXhCktenK).