#!/usr/bin/env python
# Usage: python project.py algebra_2005_2006
import sys
import os.path
import operator
import math
import random
import re

def read_file(file_name):
    f = open(file_name, 'rb')
    return [ line.rstrip().split('\t') for line in f ]

def write_file(file_name, result):
    f = open(file_name, 'w')
    for i in result:
        f.write(str(i)[1:-1]+'\n')
    return

def load_data(dataset):
    folder_prefix = 'dataset/Development/'
    training_file_name = folder_prefix + dataset + '/' +dataset+'_train.txt'
    testing_file_name = folder_prefix + dataset + '/' +dataset+'_test.txt'
    testing_result_name = folder_prefix + dataset + '/' +dataset+'_master.txt'

    training_data = read_file(training_file_name)
    testing_data = read_file(testing_file_name)
    testing_result_data = read_file(testing_result_name)
    return training_data, testing_data, testing_result_data

def process_step_name(step_name):
    step_name = re.sub(r'\s', '', step_name)
    step_name = re.sub(r'[a-z]+', '{var}', step_name)
    step_name = re.sub(r'[0-9]+\.[0-9]+', '{d}', step_name)
    step_name = re.sub(r'[0-9]+', '{d}', step_name)
    step_name = re.sub(r'^-\{d\}', '{d}', step_name)
    step_name = re.sub(r'^-\{var\}', '{var}', step_name)
    step_name = re.sub(r'\/-\{var\}}', '/{var}', step_name)
    step_name = re.sub(r'\/-\{d\}', '/{d}', step_name)
    step_name = re.sub(r'\*-\{d\}', '*{d}', step_name)
    step_name = re.sub(r'\*-\{var\}', '*{var}', step_name)
    step_name = re.sub(r'=-\{d\}', '={d}', step_name)
    step_name = re.sub(r'\(-\{d\}', '({d}', step_name)
    step_name = re.sub(r'\(-\{var\}', '({var}', step_name)
    return step_name

def process_problem_name(problem_name):
    problem_name = re.sub(r'[^ a-zA-Z]', '', problem_name)
    return problem_name

def show_data(data):
    #print training_data[0]
    for i in range(1,15):
        print data[i][0:6] #, training_data[i][13]
        print data[i][2].split(", ")
        print data[i][len(data[i])-2].split("~~")
        print data[i][len(data[i])-1].split("~~")
        #print training_data[i][6:12]
        #print training_data[i][12:17]
        #print training_data[i][17:19]
        print "------------------------------------"
    return

def rmse(predict_result, expected_result):
    n = len(predict_result)
    mse = sum([(predict - expected)**2 for predict, expected in zip(predict_result, expected_result)]) / n
    return mse ** 0.5
