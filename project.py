#!/usr/bin/env python
# Usage: python project.py algebra_2005_2006
import sys
import os.path
import operator
import math
import random

def read_file(file_name):
    f = open(file_name, 'rb')
    return [ line.rstrip().split('\t') for line in f ]

def write_file(file_name, result):
    f = open(file_name, 'w')
    for i in result:
        f.write(str(i)+'\n')
    return

def training(training_data):
    #print training_data[0]
    #for i in range(1,15):
    #    print training_data[i][0:6], training_data[i][13]
    #    print training_data[i][6:12]
    #    print training_data[i][12:17]
    #    print training_data[i][17:19]
    #    print "------------------------------------"
    student_result = {}
    overall_result = {}
    for i in range(1,len(training_data)):
        studentId = training_data[i][1]
        problem_hierarchy = training_data[i][2]
        problem_name = training_data[i][3]

        is_first_correct = training_data[i][13]

        student = student_result.setdefault(studentId, {})
        problem_result = student.setdefault(problem_hierarchy,{}).setdefault(problem_name,{})
        current_result = problem_result.setdefault(is_first_correct, 0)
        problem_result[is_first_correct] = current_result + 1

        problem_result = overall_result.setdefault(problem_hierarchy,{}).setdefault(problem_name,{})
        current_result = problem_result.setdefault(is_first_correct, 0)
        problem_result[is_first_correct] = current_result + 1
    return student_result, overall_result

def get_result_by_question(student, problem_hierarchy, problem_name):
    correct = student[problem_hierarchy][problem_name].setdefault('1',0.0)
    incorrect = student[problem_hierarchy][problem_name].setdefault('0',0.0)
    return correct, incorrect

def get_result_by_topic(student, problem_hierarchy):
    questions = student[problem_hierarchy]
    correct = sum([ questions[name].setdefault('1',0.0) for name in questions])
    incorrect = sum([ questions[name].setdefault('0',0.0) for name in questions])
    return correct, incorrect

def predict(student_result, overall_result, testing_data):
    predict_result = []
    for i in range(1,len(testing_data)):

        studentId = testing_data[i][1]
        problem_hierarchy = testing_data[i][2]
        problem_name = testing_data[i][3]

        try:
            student = student_result[studentId]
            correct, incorrect = get_result_by_question(student,problem_hierarchy,problem_name)
        except KeyError:
            try:
                student = student_result[studentId]
                correct, incorrect = get_result_by_topic(student,problem_hierarchy)
            except KeyError:
                try:
                    correct, incorrect = get_result_by_question(overall_result, problem_hierarchy, problem_name)
                except KeyError:
                    correct, incorrect = get_result_by_topic(overall_result, problem_hierarchy)

        predict = float(correct) / (correct + incorrect)

        # a = min(correct_count,incorrect_count)
        # b = max(correct_count,incorrect_count)
        #predict = 1.0 * random.randint(a, b) / (correct_count + incorrect_count)

        predict_result.append(predict)
    return predict_result

def rmse(predict_result, expected_result):
    n = len(predict_result)
    mse = sum([(predict - expected)**2 for predict, expected in zip(predict_result, expected_result)]) / n
    return mse ** 0.5

def main(arg):
    folder_prefix = 'dataset/Development/'
    dataset = arg[1] #'algebra_2005_2006'
    training_file_name = folder_prefix + dataset + '/' +dataset+'_train.txt'
    testing_file_name = folder_prefix + dataset + '/' +dataset+'_test.txt'
    testing_result_name = folder_prefix + dataset + '/' +dataset+'_master.txt'

    training_data = read_file(training_file_name)
    testing_data = read_file(testing_file_name)
    testing_result_data = read_file(testing_result_name)

    student_result, overall_result = training(training_data)

    predict_result = predict(student_result, overall_result, testing_data)
    expected_result = [float(i[13]) for i in testing_result_data[1:]]
    print rmse(predict_result, expected_result)

    predict_result = predict(student_result, overall_result, training_data)
    expected_result = [float(i[13]) for i in training_data[1:]]
    print rmse(predict_result, expected_result)

    return

if __name__ == "__main__":
    main(sys.argv)
