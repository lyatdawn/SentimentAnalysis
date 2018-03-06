# -*- coding:utf-8
"""
Before creating the ids matrix for the whole training set, let’s first take some time to visualize the 
type of data that we have. This will help us determine the best value for setting our maximum sequence 
length. The max sequence length value is largely dependent on the inputs you have.  

The training set we're going to use is the Imdb movie review dataset. This set has train and test 
dataset, each includes 25000 movie reviews. In this experiment, we only use train datasets, then split
it to train and val set. This train set has 25,000 movie reviews, with 12,500 positive 
reviews and 12,500 negative reviews. 

Each of the reviews is stored in a txt file that we need to parse through. 
The positive reviews are stored in one directory and the negative reviews are stored in another. 
The following piece of code will determine total and average number of words in each review. 
"""
import os
import glob
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Utlize glob module to generate all txt files' path.
    positiveFiles = glob.glob(os.path.join("../datasets", "positiveReviews", "*.txt"))
    negativeFiles = glob.glob(os.path.join("../datasets", "negativeReviews", "*.txt"))

    numWords = [] # Then we will generate the length of each sentence, i.e. counter.
    # A txt file includes a sentence, i.e. the movie reviews.

    for pos_file in positiveFiles:
        # positive reviews txt file.
        with open(pos_file, "r") as f:
            line = f.readline() # A txt file includes a sentence, i.e. the movie reviews.
            counter = len(line.split()) # default split is " ".
            print("The lenght of sequence is {}".format(counter))
            numWords.append(counter)       
    print("Positive files finished")

    for neg_file in negativeFiles:
        # negative reviews txt file.
        with open(neg_file, "r") as f:
            line = f.readline() # A txt file includes a sentence, i.e. the movie reviews.
            counter = len(line.split())
            print("The lenght of sequence is {}".format(counter))
            numWords.append(counter)  
    print("Negative files finished")

    numFiles = len(numWords) # The number of files.
    print("The total number of files is", numFiles) # 25000.
    print("The total number of words in the files is", sum(numWords)) # 5844464.
    print("The average number of words in the files is", sum(numWords) / len(numWords)) # 233.


    # We can also use the Matplot library to visualize this data in a histogram format. 
    plt.hist(numWords, 50)
    plt.xlabel('Sequence Length')
    plt.ylabel('Frequency')
    plt.axis([0, 1200, 0, 8000])
    plt.show()

    # From the histogram as well as the average number of words per file, we can safely say that most 
    # reviews will fall under 250 words, which is the max sequence length value we will set. 
    # maxSeqLength = 250