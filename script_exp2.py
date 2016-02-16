# Compute the weight vector of linear SVM based on the model file
# Original Perl Author: Thorsten Joachims (thorsten@joachims.org)
# Python Version: Ori Cohen (orioric@gmail.com)
# Call: python svm2weights.py svm_model

import sys
from operator import itemgetter
import numpy as np
import random
import os
import subprocess
import matplotlib.pyplot as plt
import random


try:
    import psyco
    psyco.full()
except ImportError:
    print 'Psyco not installed, the program will just run slower'


fold_num = 10
c_parameter = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
num_features = 57
C = 0.5
acc_list = []
m_list = []

def LoadSpamData(filename = "spambase.data"):
    # """
    # Each line in the datafile is a csv with features values, followed by a single label (0 or 1),
    # per sample; one sample per line
    # """

    # "The file function reads the filename from the current directory, unless you provide an absolute path
    # e.g. /path/to/file/file.py or C:\\path\\to\\file.py"

    unprocessed_data_file = file(filename,'r')

    "Obtain all lines in the file as a list of strings."

    unprocessed_data = unprocessed_data_file.readlines()

    labels = []
    features = []

    for line in unprocessed_data:
        feature_vector = []

        "Convert the String into a list of strings, being the elements of the string separated by commas"
        split_line = line.split(',')

        "Iterate across elements in the split_line except for the final element "
        for element in split_line[:-1]:
            feature_vector.append(float(element))

        "Add the new vector of feature values for the sample to the features list"
        features.append(feature_vector)

        "Obtain the label for the sample and add it to the labels list"
        labels.append(int(split_line[-1]))

    "Return List of features with its list of corresponding labels"
    return features, labels


def BalanceDataset(features, labels):
    # """
    # Assumes the lists of features and labels are ordered such that all like-labelled samples are
    # together (all the zeros come before all the ones, or vice versa)
    # """

    count_0 = labels.count(0)
    count_1 = labels.count(1)
    balanced_count = min(count_0, count_1)

    # Indexing with a negative value tracks from the end of the list
    return features[:balanced_count] + features[-balanced_count:], labels[:balanced_count] + labels[-balanced_count:]



def ConvertDataToArrays(the_features):
    """
    conversion to a numpy array is easy if you're starting with a List of lists.
    The returned array has dimensions (M,N), where M is the number of lists and N is the number of

    """

    return np.asarray(the_features)


def get_mean_and_std(the_training):
    the_means = np.mean(the_training, axis = 0)
    the_std = np.std(the_training, axis = 0)

    return the_means, the_std


def NormalizeFeatures(the_data, the_means, the_std):
    the_data -= the_means
    the_data /= the_std

    return the_data


def attachLabel(the_features, the_labels):
    for i in range(len(the_features)):
        the_features[i].append(the_labels[i])
    return the_features


def split(the_dataSet):
    half_size = len(dataSet)/2
    the_training = dataSet[:half_size]
    the_testing = dataSet[half_size:]

    return the_training, the_testing

# delete the label from feature lists
def detachLabel(the_training):
    the_training_label = []
    for element in the_training:
        the_training_label.append(element[-1])
        element.pop()
    return the_training, the_training_label


def PrintDataToSvmLightFormat(the_features, the_labels, filename):
  # """
  # Readable format for SVM Light should be, with
  # lable 0:feature0, 1:feature1, 2:feature2, etc...
  # where label is -1 or 1.
  # """

    if len(the_features) != len(the_labels):
        raise Exception("Number of samples and labels must match")
    dat_file = file(filename,'w')
    for s in range(len(the_features)):
        if the_labels[s]==0:
            line="-1 "
        else:
            line="1 "

        for f in range(len(the_features[s])):
            line +="%i:%f " % (f+1 , the_features[s][f])
        line += "\n"
        dat_file.write(line)
    dat_file.close()


def splitToTenSubsets(the_training, the_training_label):
    subset_len = len(the_training) / fold_num
    remainder = len(the_training) % fold_num
    the_tenSubsets = []
    the_tenSubsets_label = []
    for i in range(fold_num):
        one_subset = []
        one_subset_label = []
        for j in range(subset_len):
            one_subset.append(the_training[j + i*subset_len])
            one_subset_label.append(the_training_label[j + i*subset_len])
        the_tenSubsets.append(one_subset)
        the_tenSubsets_label.append(one_subset_label)
    for i in range(remainder):
        the_tenSubsets[i].append(the_training[-i])
        the_tenSubsets_label[i].append(the_training_label[-i])
    return the_tenSubsets, the_tenSubsets_label


def separate_train_and_test(the_tenSubsets, the_tenSubsets_label, the_validation_index):
    the_validation_set = the_tenSubsets[the_validation_index]
    the_validation_set_label = the_tenSubsets_label[the_validation_index]
    the_true_training = []
    the_true_training_label = []
    for i in range(fold_num):
        if i != the_validation_index:
            for j in range(len(the_tenSubsets[i])):
                the_true_training.append(the_tenSubsets[i][j])
                the_true_training_label.append(the_tenSubsets_label[i][j])
    return the_true_training, the_true_training_label, the_validation_set, the_validation_set_label


def sortbyvalue(d,reverse=True):
    ''' proposed in PEP 265, using  the itemgetter this function sorts a dictionary'''
    return sorted(d.iteritems(), key=itemgetter(1), reverse=True)

def sortbykey(d,reverse=True):
    ''' proposed in PEP 265, using  the itemgetter this function sorts a dictionary'''
    return sorted(d.iteritems(), key=itemgetter(0), reverse=False)

def get_file():
    """
    Tries to extract a filename from the command line.  If none is present, it
    assumes file to be svm_model (default svmLight output).  If the file
    exists, it returns it, otherwise it prints an error message and ends
    execution.
    """
    # Get the name of the data file and load it into
    if len(sys.argv) < 2:
        # assume file to be svm_model (default svmLight output)
        print "Assuming file as svm_model"
        filename = 'svm_model'
        #filename = sys.stdin.readline().strip()
    else:
        filename = sys.argv[1]


    try:
        f = open(filename, "r")
    except IOError:
        print "Error: The file '%s' was not found on this system." % filename
        sys.exit(0)

    return f


def abs_for_weight(the_list):
    for i in range(len(the_list)):
        the_list[i] = (i + 1, abs(the_list[i][1]))
    return the_list


def get_key_list(the_list, the_num_keys):
    the_key_list = []
    for i in range(the_num_keys):
        the_key_list.append(the_list[i][0])
    return the_key_list


def PrintDataToSvmLightFormat_exp2(the_features, the_labels, filename, the_key_list):
  # """
  # Readable format for SVM Light should be, with
  # lable 0:feature0, 1:feature1, 2:feature2, etc...
  # where label is -1 or 1.
  # """

    if len(the_features) != len(the_labels):
        raise Exception("Number of samples and labels must match")
    dat_file = file(filename,'w')
    for s in range(len(the_features)):
        if the_labels[s]==0:
            line="-1 "
        else:
            line="1 "

        for f in range(len(the_key_list)):
            line +="%i:%f " % (the_key_list[f] , the_features[s][the_key_list[f] - 1])
        line += "\n"
        dat_file.write(line)
    dat_file.close()


def get_accuracy(the_testing_file_name, the_model_file_name):
    the_cmd = "./svm_classify " + the_testing_file_name + " " + the_model_file_name
    proc = subprocess.Popen([the_cmd], stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
    start_index = out.find("Accuracy on test set: ") + len("Accuracy on test set: ")
    end_index = out.find("%")
    return float(out[start_index:end_index])/100

# def cut_off_training_set(the_training, the_key_list):
#     temp_training_list = []
#     for i in range(len(the_training)):
#         temp_feature_list = []
#         for j in range(len(the_key_list)):
#             temp_feature_list.append(the_training[i][the_key_list[j]])




if __name__ == "__main__":
    f = get_file()
    i=0
    lines = f.readlines()
    printOutput = True
    w = {}
    for line in lines:
        if i>10:
            features = line[:line.find('#')-1]
            comments = line[line.find('#'):]
            alpha = features[:features.find(' ')]
            feat = features[features.find(' ')+1:]
            for p in feat.split(' '): # Changed the code here.
                a,v = p.split(':')
                if not (int(a) in w):
                    w[int(a)] = 0
            for p in feat.split(' '):
                a,v = p.split(':')
                w[int(a)] +=float(alpha)*float(v)
        elif i==1:
            if line.find('0')==-1:
                print 'Not linear Kernel!\n'
                printOutput = False
                break
        elif i==10:
            if line.find('threshold b')==-1:
                print "Parsing error!\n"
                printOutput = False
                break

        i+=1
    f.close()

    #if you need to sort the features by value and not by feature ID then use this line intead:
    # ws = sortbyvalue(w)

    ws = sortbykey(w)
    ws = abs_for_weight(ws)
    w = dict(ws)
    ws = sortbyvalue(w)

    # if printOutput == True:
    #     for (i,j) in ws:
    #         print i,':',j
    #         i+=1
    # random.shuffle(ws)
    #
    features, labels = LoadSpamData()
    features, labels = BalanceDataset(features, labels)
    dataSet = attachLabel(features, labels)
    random.shuffle(dataSet)
    training, testing = split(dataSet)
    training, training_label = detachLabel(training)
    testing, testing_label = detachLabel(testing)
    training = ConvertDataToArrays(training)
    testing = ConvertDataToArrays(testing)
    means, std = get_mean_and_std(training)
    training = NormalizeFeatures(training, means, std)
    testing = NormalizeFeatures(testing, means, std)

    testing_file_name = "entire_testing.data"
    PrintDataToSvmLightFormat(testing, testing_label, testing_file_name)
    # print len(training)
    # print len(training[0])
    # print ws
    for i in range(2,num_features):
        key_list = get_key_list(ws, i)
        key_list.sort()
        training_file_name = "training"+str(i)+".data"
        model_file_name = "svm_model" + str(i) + ".data"
        PrintDataToSvmLightFormat_exp2(training, training_label, training_file_name, key_list)
        while not os.path.exists(training_file_name):
            pass
        training_cmd = "./svm_learn "+"-c " + str(C) + " " + training_file_name + " " + model_file_name
        os.system(training_cmd)
        while not os.path.exists(model_file_name):
            pass
        acc = get_accuracy(testing_file_name, model_file_name)
        acc_list.append(acc)
        m_list.append(i)

    plt.plot(m_list, acc_list)
    plt.xlabel('m')
    plt.ylabel('Accuracy')

    plt.show()




