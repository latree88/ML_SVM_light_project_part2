import numpy as np
import random
import os

fold_num = 10
c_parameter = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

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
PrintDataToSvmLightFormat(training, training_label, "entire_training.data")
PrintDataToSvmLightFormat(testing, testing_label, "entire_testing.data")

tenSubsets, tenSubsets_label = splitToTenSubsets(training, training_label)


# for i in range(fold_num):
#
#     PrintDataToSvmLightFormat(training, training_label, "svm_features"+str(i)+".data")
for j in range(len(c_parameter)):
    for i in range(fold_num):
        validation_index = i
        C = c_parameter[j]

        # separate ten sets of data file into two part
        # one part is for validation set
        # another part is for training
        true_training, true_training_label, validation_set, validation_set_label \
            = separate_train_and_test(tenSubsets, tenSubsets_label, validation_index)

        # print the data as svm light format
        training_file_name = "training"+str(j)+str(i)+".data"
        testing_file_name = "testing"+str(j)+str(i)+".data"
        PrintDataToSvmLightFormat(true_training, true_training_label, training_file_name)
        PrintDataToSvmLightFormat(validation_set, validation_set_label, testing_file_name)



        # execute svm_learn by feeding into training data
        # model_file_name = "model" + str(j) + str(i)
        # cmd = "./svm_learn "+"-c " + str(C) + training_file_name + " " + model_file_name
        # os.system(cmd)


# features, labels = ConvertDataToArrays(features, labels)
# features = NormalizeFeatures(features)

# print len(features)
# print len(training)
# print len(testing)
# print len(tenSubsets)
# print len(tenSubsets[0])
# print len(training[0])
# print len(training_label)

