# -*- coding:utf-8 -*-
"""
In this code, we will use the pretrained word vectors using Glove. The version of pretrained word vector
is glove.6B.50d.txt. You can down those txt files in https://nlp.stanford.edu/projects/glove/.
First, use generate_word_list_and_vectors.py to generate word list and word vectors.
"""
import os
import numpy as np

if __name__ == '__main__':
    with open("../datasets/glove.6B.50d.txt", "r") as f:
        lines = f.readlines()
        # Use .readlines() return a list, the element oflist is the every line of datasets/glove.6B.50d.txt.
        # The end of line is \r\n. use strip() or split() to get rid of it.
        
        # Use a loop to read each line in txt file. The format of each line is:
        # word vector
        # e.g., the 0.418 0.24968 .... The dim of vector is 50. 
        # The type of each line is string, can use slit().
        
        wordsList = np.zeros((len(lines)), dtype="S68")
        # define a string array, then assign with words. The default stype of string include only a 
        # character, e.g a = np.zeros((len(lines), 1), dtype=np.str), a[0] = "the", then a[0] is "t".
        # If we want get a specified length of character, we can use "S10", "S68" and so on.
        # "S10", "S68" is a type of numpy. They are the type code of string type.
        wordsVector = np.zeros((len(lines), 50), dtype=np.float32)

        # define a float32 array, then assign with word vectors. Default type is float32. Standard float type.
        for i in range(len(lines)):
            wordsList[i] = lines[i].split(" ")[0].strip()
            wordsVector[i] = np.array([float(x) for x in lines[i].split(" ")[1:]])
        
        np.save("../datasets/wordsList.npy", wordsList)
        np.save("../datasets/wordsVector.npy", wordsVector)