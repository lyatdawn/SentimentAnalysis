# -*- coding:utf-8 -*-
"""
Load in the IMDB movie training set and integerize it to get a 25000 x 250 matrix. 
This was a computationally expensive process, so instead of having you run the whole process, 
we’re going to load in a pre-computed IDs matrix.
"""
import os
import glob
import re
import numpy as np

# Removes punctuation, parentheses, question marks, etc., and leaves only alphanumeric characters.
def cleanSentences(string):
    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
    '''
    The arguments of compile() function are:
    pattern: a regular expression with the type is string. We use [^A-Za-z0-9 ]+ to represent all the
         alphanumeric characters.
    flags: math mode.
    '''
    string = string.lower().replace("<br />", " ")
    # string object call lower() method to convert capital letters to lowercase letters. The Glove word
    # vectors are all lowercase letters, so there must use lower().
    # Then call replace() method to replace the specified characters.
    # In eachIMDB movie reviews, there will include <br />. <br /> is HTML grammar, represent wrap.
    # Here, replace <br /> to " ".

    return re.sub(strip_special_chars, "", string.lower())
    '''
    re.sub(), replace the specified item. Usage: re.sub(pattern, repl, string, count=0, flags=0)
    '''

if __name__ == '__main__':
    # word list.
    wordsList = np.load("../datasets/wordsList.npy")
    wordsList = wordsList.tolist() # Originally loaded as numpy array.

    # numFiles. The total number of files, only use training dataset of IMDB movie reviews.
    numFiles = 25000
    # maxSeqLength. max sequence length value. Utlize scripts/generate_maxSeqLength.py to get max sequence length.
    maxSeqLength = 250
    ids = np.zeros((numFiles, maxSeqLength), dtype='int32') + 1343
    # Since "0" locates at 1343, so initial element of the index matrix is 1343. "0" represent the empty
    # element.
    fileCounter = 0 # Final fileCounter is 25000.

    # Utlize glob module to generate all txt files' path.
    positiveFiles = glob.glob(os.path.join("../datasets", "positiveReviews", "*.txt"))
    negativeFiles = glob.glob(os.path.join("../datasets", "negativeReviews", "*.txt"))

    for pos_file in positiveFiles:
        with open(pos_file, "r") as f:
            indexCounter = 0 # Represent the column of index matrix, range form 0 to 249.
            line = f.readline() # A txt file includes a sentence, i.e. the movie reviews.
            cleanedLine = cleanSentences(line)
            split = cleanedLine.split()
            for word in split:
                # Address each word in sentence.
                try:
                    ids[fileCounter][indexCounter] = wordsList.index(word)
                    # wordsList.index(word), find the word index through index().
                except ValueError:
                    ids[fileCounter][indexCounter] = 201534 
                    # Vector for unkown words. If the sentence has the word which not in Vocabulary,
                    # then this word regard to "UNK".
                indexCounter = indexCounter + 1 # Atfer address a word of sentence, indexCounter + 1.

                if indexCounter >= maxSeqLength:
                    break
            fileCounter = fileCounter + 1 # Atfer address a txt file, fileCounter + 1.
        
        if fileCounter % 100 == 0:
            print("Have addressed {} files...".format(fileCounter))

    for neg_file in negativeFiles:
        with open(neg_file, "r") as f:
            indexCounter = 0 # Represent the column of index matrix, range form 0 to 249.
            line = f.readline() # A txt file includes a sentence, i.e. the movie reviews.
            cleanedLine = cleanSentences(line)
            split = cleanedLine.split()
            for word in split:
                try:
                    ids[fileCounter][indexCounter] = wordsList.index(word)
                except ValueError:
                    ids[fileCounter][indexCounter] = 201534
                    # Vector for unkown words. If the sentence has the word which not in Vocabulary,
                    # then this word regard to "UNK".
                indexCounter = indexCounter + 1

                if indexCounter >= maxSeqLength:
                    break
            fileCounter = fileCounter + 1

        if fileCounter % 100 == 0:
            print("Have addressed {} files...".format(fileCounter))

    print("The total number of files is", fileCounter) # 25000.
    np.save('../datasets/indexsMatrix.npy', ids)
    # indexsMatrix.npy. 0-11499 is positive sample, 12500-24999 is negative sample.