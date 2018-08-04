from __future__ import division
import sys
import collections
import sklearn.naive_bayes
import sklearn.linear_model
import nltk
import random
import datetime
from collections import Counter
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
random.seed(0)
from sklearn.naive_bayes import BernoulliNB
import numpy
import matplotlib.pyplot as plt

path_to_data = "C:\Python27\data\witter"



def main():
    train_pos, train_neg, test_pos, test_neg = load_data(path_to_data)


    train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_NLP(train_pos, train_neg, test_pos,
                                                                                    test_neg)

    nb_model = build_models_NLP(train_pos_vec, train_neg_vec)

    print "Naive Bayes"
    print "-----------"
    evaluate_model(nb_model, test_pos_vec, test_neg_vec, True)



def load_data(path_to_dir):
 
    train_pos = []
    train_neg = []
    test_pos = []
    test_neg = []
    with open(path_to_dir +"\\"+ "train-pos.txt", "r") as f:
        for i, line in enumerate(f):
            words = [w.lower() for w in line.strip().split() if len(w) >= 3]
            train_pos.append(words)
    with open(path_to_dir +"\\" "train-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w) >= 3]
            train_neg.append(words)
    with open(path_to_dir +"\\" "test-pos.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w) >= 3]
            test_pos.append(words)
    with open(path_to_dir +"\\" "test-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w) >= 3]
            test_neg.append(words)

    return train_pos, train_neg, test_pos, test_neg





def feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg):

    stopwords = set(nltk.corpus.stopwords.words('english'))
 

    pos_list = []
    for tweet in train_pos:
        pos_list.append(list(set(tweet)))
    wordList = [item for sublist in pos_list for item in sublist]
    filteredWords = [w for w in wordList if w not in stopwords]
    wordCount = Counter(filteredWords)
    length = len(train_pos) / 100
    positive = dict((key, value) for key, value in wordCount.items() if value >= length)

    neg_list = []
    for tweet in train_neg:
        neg_list.append(list(set(tweet)))
    wordList = [item for sublist in neg_list for item in sublist]
    #print wordlist
    filteredWords = [word for word in wordList if word not in stopwords]
    wordCount = Counter(filteredWords)
    length = len(train_neg) / 100
    negative = dict((key, value) for key, value in wordCount.items() if value >= length)
    features = []
    for key in positive.keys():
        if key in negative.keys():
            if positive[key] >= 2 * negative[key]:
                features.append(key)
        else:
            features.append(key)
    for key in negative.keys():
        if key in positive.keys():
            if negative[key] >= 2 * positive[key]:
                features.append(key)
        else:
            features.append(key)

    train_pos_vec = []
    train_neg_vec = []
    test_pos_vec = []
    test_neg_vec = []

    for text in train_pos:
        pos_list = []
        for word in features:
            if word in text:
                pos_list.append(1)
            else:
                pos_list.append(0)
        train_pos_vec.append(pos_list)

    for text in train_neg:
        neg_list = []
        for word in features:
            if word in text:
                neg_list.append(1)
            else:
                neg_list.append(0)
        train_neg_vec.append(neg_list)

    for text in test_pos:
        pos_list = []
        for word in features:
            if word in text:
                pos_list.append(1)
            else:
                pos_list.append(0)
        test_pos_vec.append(pos_list)

    for text in test_neg:
        neg_list = []
        for word in features:
            if word in text:
                neg_list.append(1)
            else:
                neg_list.append(0)
        test_neg_vec.append(neg_list)
  
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec







def build_models_NLP(train_pos_vec, train_neg_vec):

    Y = ["pos"] * len(train_pos_vec) + ["neg"] * len(train_neg_vec)
  
    final_vec = train_pos_vec
    final_vec.extend(train_neg_vec)
    nb_model = BernoulliNB(alpha=1.0, binarize=None, class_prior=None, fit_prior=True)
    nb_model.fit(final_vec, Y)
   

    return nb_model



def evaluate_model(model, test_pos_vec, test_neg_vec, print_confusion=False):
 
    final_vec = test_pos_vec
    final_vec.extend(test_neg_vec)
    confusion_matrix = model.predict(test_pos_vec)
    pos_count = len(test_pos_vec)
    neg_count = len(test_neg_vec)
    pos_list = confusion_matrix[:pos_count]
    print Counter(pos_list)
    tp = 0.0
    fp = 0.0
    tn = 0.0
    fn = 0.0

    test_pos_vec = numpy.array(test_pos_vec)
    test_neg_vec = numpy.array(test_neg_vec)

    for var in test_pos_vec:
        if model.predict(var.reshape(1, -1)) == ['pos']:
            tp = tp + 1
        else:
            fn = fn + 1
    for var in test_neg_vec:
        if model.predict(var.reshape(1, -1)) == ['neg']:
            tn = tn + 1
        else:
            fp = fp + 1
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    Accuracy="Accuracy="+str(round(accuracy*100,2))+"%"
    if print_confusion:
     
        print "pos\t\t%d" % (tp)
        print "neg\t\t%d" % (tn)

    labels = 'Liking', 'Not Liking'
    

    sizes = [tp,tn]
    colors = ['gold', 'red']
    explode =(0,0.1)
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,autopct='%1.1f%%', shadow=True, startangle=140)
    plt.text(1.1,0.9,Accuracy)
   # ax = plt.add_subplot(111)
    #ax.text(3,8,Accuracy)
    plt.axis('equal')
    plt.show()

if "true":
    main()
