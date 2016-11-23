#!/usr/bin/env python
# Usage: python project.py algebra_2005_2006
import sys
import random
import gc
import time
from project import write_file, load_data, show_data, rmse, process_step_name
from sklearn import svm
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import os
import psutil

def get_feature_vector(element_set, value_set, w=1.0):
    result = [0.0] * len(element_set)
    for val in value_set:
        if val in element_set :
            result[element_set.index(val)] = w
    return result

def get_feature_vectors(dataset, N, studentId_list, unit_list, section_list, problem_name_list, step_name_list, kc_list):
    rows = []
    for i in range(1,N):
        student_id_feature = get_feature_vector(studentId_list,[dataset[i][1]],10)
        studentId_size = len(student_id_feature)

        unit, section = dataset[i][2].split(", ")

        unit_feature = get_feature_vector(unit_list,[unit],1)
        unit_size = len(unit_feature)
        section_feature = get_feature_vector(section_list,[section], 1)
        section_size = len(section_feature)


        #problem_name_feature = get_feature_vector(problem_name_list,[dataset[i][3]])
        #problem_name_size = len(problem_name_feature)

        problem_view_feature = [float(dataset[i][4])/(float(dataset[i][4])+1)]
        problem_view_size = len(problem_view_feature)

        p_step = process_step_name(dataset[i][5])
        step_name_feature = [ p_step.count('+'),p_step.count('-'),p_step.count('*'),p_step.count('/'),p_step.count('{var}'),p_step.count('{d}') ]
        #step_name_feature = [float(x)/sum(step_name_feature) for x in step_name_feature]
        step_name_size = len(step_name_feature)  

        #print step_name_feature
        kc_feature = get_feature_vector(kc_list, dataset[i][len(dataset[i])-2].split("~~"),1 )
        kc_feature_size = len(kc_feature)        
        #o = dataset[i][len(dataset[i])-1].split("~~")

        #print problem_hierarchy_feature
        rows.append(student_id_feature +  unit_feature + section_feature +problem_view_feature + step_name_feature + kc_feature)
    print "feature vector composition: ", studentId_size, unit_size, section_size,problem_view_size, step_name_size, kc_feature_size
    return rows

def process_data(training_data, testing_data):
    #show_data(training_data)
    studentId_list = []
    section_list = []
    unit_list = []
    problem_name_list = []
    step_name_list = []
    is_first_correct_list = []
    kc_list = []
    testing_rows = []
    N = 50000#len(training_data)
    print "Num Of Lines to train: ", N
    for i in range(1,N):
        studentId = training_data[i][1]
        unit, section = training_data[i][2].split(", ")
        problem_name = training_data[i][3]
        step_name = process_step_name(training_data[i][5])

        kcs = training_data[i][len(training_data[i])-2].split("~~")
        o = training_data[i][len(training_data[i])-1].split("~~")

        is_first_correct_list.append(training_data[i][13])

        if studentId not in studentId_list:
            studentId_list.append(studentId)
        if unit not in unit_list:
            unit_list.append(unit)
        if section not in section_list:
            section_list.append(section)            
        if problem_name not in problem_name_list:
            problem_name_list.append(problem_name)
        if step_name not in step_name_list:
            step_name_list.append(step_name)
        for kc in kcs:
            if kc not in kc_list:
                kc_list.append(kc)

    print "#of unique item in each categories: ",len(studentId_list), len(unit_list), len(section_list), len(problem_name_list), len(step_name_list), len(kc_list)
    # Create matrix...
    training_data_rows = get_feature_vectors(training_data, N, studentId_list, unit_list, section_list, problem_name_list, step_name_list, kc_list)

    testing_data_rows = get_feature_vectors(testing_data, len(testing_data), studentId_list, unit_list, section_list, problem_name_list, step_name_list, kc_list)
    return training_data_rows, is_first_correct_list, testing_data_rows


def main(arg):
    dataset = arg[1] #'algebra_2005_2006'
    start = time.time()
    training_data, testing_data, testing_result_data = load_data(dataset)
    end = time.time()
    print "Time to load data", end-start, " sec"

    start = time.time()
    rows, is_first_correct_list, testing_rows = process_data(training_data, testing_data)
    end = time.time()
    print "Time to process data", end-start , " sec"   
    print len(rows),len(is_first_correct_list),len(testing_rows),len(rows[0])

    #print rows[:200]
    #print testing_rows[:200]

    del training_data
    del testing_data
    gc.collect()

    process = psutil.Process(os.getpid())
    print "RAM usage (MB):", process.memory_info().rss/1024/1024

    #write_file("preprocessed.txt", rows)

    start = time.time()
    #clf = linear_model.SGDClassifier(n_jobs=-1,n_iter=500)
    #clf = linear_model.LogisticRegressionCV(n_jobs=-1, verbose=True)
    clf = KNeighborsClassifier(n_jobs=-1, weights='distance', n_neighbors=5)
    #clf = GaussianNB()
    #clf = RandomForestClassifier(n_estimators=20,n_jobs=-1, verbose=True)
    #clf = svm.SVC(verbose=True, cache_size=5000, kernel='linear', C=0.5)
    #clf = tree.DecisionTreeClassifier()

    clf.fit(rows, is_first_correct_list)
    print clf

    end = time.time()
    print "Time to train classifier", end-start, " sec"

    process = psutil.Process(os.getpid())
    print "RAM usage (MB):", process.memory_info().rss/1024/1024

    start = time.time()
    predict_result = clf.predict(rows)
    end = time.time()
    print "Time to do prediction of self-test", end-start, " sec"

    #print "Mean accuracy" , clf.score(rows, is_first_correct_list)
    print "first 50 items of predict: ", predict_result[:50]
    print "first 50 items of GT: ", is_first_correct_list[:50]
    predict_result = [ float(i) for i in predict_result]
    training_error = rmse(predict_result, [ float(i) for i in is_first_correct_list])
    print "rmse of first 10 items ", rmse([ float(i) for i in predict_result[:10]], [ float(i) for i in is_first_correct_list[:10]])


    predict_result = clf.predict(testing_rows)
    predict_result = [ float(i) for i in predict_result]
    predict_error =  rmse(predict_result, [float(i[13]) for i in testing_result_data[1:]])

    print '|', dataset, '|', training_error, '|', predict_error ,'|'
    return

if __name__ == "__main__":
    main(sys.argv)

