#!/usr/bin/env python
import os
import subprocess
import matplotlib.pyplot as plt

fold_num = 10
c_parameter = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
C = 0
num_thresholds = 200
model_file_name = "svm_model"
prediction_file_name = "svm_predictions"

def get_accuracy():
    proc = subprocess.Popen(["./svm_classify testing00.data svm_model"], stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
    start_index = out.find("Accuracy on test set: ") + len("Accuracy on test set: ")
    end_index = out.find("%")
    return float(out[start_index:end_index])/100


def find_max(the_list):
    the_max = 0
    the_index = 0
    for element in range(len(the_list)):
        if the_list[element] > max:
            the_max = the_list[element]
            the_index = element
    return the_max, the_index


def read_output_file(the_filename):
    the_output = file(the_filename, 'r')
    the_predicted_res = the_output.readlines()
    the_predicted_data = []
    for line in the_predicted_res:
        line = line.replace("\n", "")
        line = float(line)
        the_predicted_data.append(line)

    return the_predicted_data


def get_actual_class(the_filename):
    the_testing_actual_class = file(the_filename, 'r')
    the_testing_actual_class_res = the_testing_actual_class.readlines()
    the_testing_actual_class_data = []
    for line in the_testing_actual_class_res:
        split_line = line.split(' ')
        one_actual_class = split_line[0]
        one_actual_class = int(one_actual_class)
        the_testing_actual_class_data.append(one_actual_class)

    return the_testing_actual_class_data


def get_TPR_and_FPR(the_threshold, the_predicted_data, the_actual_class):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(the_predicted_data)):
        if the_actual_class[i] == 1 and the_predicted_data[i] >=the_threshold:
            tp += 1
        elif the_actual_class[i] == -1 and the_predicted_data[i] >=the_threshold:
            fp += 1
        elif the_actual_class[i] == 1 and the_predicted_data[i] < the_threshold:
            fn += 1
        elif the_actual_class[i] == -1 and the_predicted_data[i] < the_threshold:
            tn += 1

    fpr = fp / (float(fp + tn))
    tpr = tp / (float(tp + fn))
    return tpr, fpr


def get_thresholds(the_list):
    the_max = max(the_list)
    the_min = min(the_list)
    the_range = the_max - the_min
    one_step = the_range / num_thresholds
    the_thresholds = []
    for i in range(num_thresholds):
        the_max -= one_step
        the_thresholds.append(the_max)
    return the_thresholds


def wrapper_get_TPR_and_FPR(the_thresholds, the_predicted_data, the_actual_class):
    the_tpr_list = []
    the_fpr_list = []
    for i in range(num_thresholds):
        one_tpr, one_fpr = get_TPR_and_FPR(the_thresholds[i], the_predicted_data, the_actual_class)
        the_tpr_list.append(one_tpr)
        the_fpr_list.append(one_fpr)

    return the_tpr_list, the_fpr_list


for j in range(len(c_parameter)):
    acc_list = []
    acc = 0.0
    for i in range(fold_num):
        validation_set_index = i
        C = c_parameter[j]
        # execute svm_learn by feeding into training data
        training_file_name = "training"+str(j)+str(i)+".data"
        testing_file_name = "testing"+str(j)+str(i)+".data"
        learn_cmd = "./svm_learn "+"-c " + str(C) + " " + training_file_name
        os.system(learn_cmd)
        while not os.path.exists(model_file_name):
            pass
        classify_cmd = "./svm_classify " + testing_file_name + " " + model_file_name
        os.system(classify_cmd)
        acc += get_accuracy()
        if os.path.exists(model_file_name):
            os.system("rm svm_model")
        if os.path.exists(prediction_file_name):
            os.system("rm svm_predictions")

    ave_acc = acc/fold_num
    acc_list.append(ave_acc)
    max_acc, max_acc_index = find_max(acc_list)
    C = c_parameter[max_acc_index]


if os.path.exists(model_file_name):
    os.system("rm svm_model")
if os.path.exists(prediction_file_name):
    os.system("rm svm_predictions")
learn_cmd_2 = "./svm_learn "+"-c " + str(C) + " " + "entire_training.data"
os.system(learn_cmd_2)
while not os.path.exists(model_file_name):
    pass
classify_cmd_2 = "./svm_classify " + "entire_testing.data" + " " + model_file_name
os.system(classify_cmd_2)
while not os.path.exists(prediction_file_name):
    pass

predicted_data = read_output_file("svm_predictions")
# print predicted_data

actual_class = get_actual_class("entire_testing.data")
# print actual_class
thresholds = get_thresholds(predicted_data)

tpr_list, fpr_list = wrapper_get_TPR_and_FPR(thresholds, predicted_data, actual_class)

plt.plot(fpr_list, tpr_list)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.show()


# proc = subprocess.Popen(["./svm_classify testing00.data svm_model"], stdout=subprocess.PIPE, shell=True)
# (out, err) = proc.communicate()
# start_index = out.find("Accuracy on test set: ") + len("Accuracy on test set: ")
# end_index = out.find("%")
# print out[start_index:end_index]




