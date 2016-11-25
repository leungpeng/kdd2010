#!/usr/bin/env python
# Usage: python project.py algebra_2005_2006
import sys
import random
import gc
import time
from project import write_file, load_data, show_data, rmse, process_step_name, process_problem_name
from sklearn import svm
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import os
import psutil
from sklearn.metrics import jaccard_similarity_score

def get_feature_vector_opp(element_set, kc_value, opp, w=1.0):
    result = [0.0] * len(element_set)
    i=0
    for val in kc_value:
        if val in element_set :
            try:
                result[element_set.index(val)] = float(opp[i])/200.0
            except ValueError:
                result[element_set.index(val)] = 0
            i=i+1
    return result

def get_feature_vector(element_set, value_set, w=1.0):
    result = [0.0] * len(element_set)
    for val in value_set:
        if val in element_set :
            result[element_set.index(val)] = w
    return result


def get_feature_vectors_nb(dataset, N, studentId_list, unit_list, section_list,
                         problem_name_list, step_name_list, kc_list, kc_list_raw):

    step_name_dict = []
    for i in range(1,N):
        p_step = process_step_name(dataset[i][5])
        step_name_hist = [ p_step.count('+'),p_step.count('-'),p_step.count('*'),p_step.count('/'),p_step.count('{var}'),p_step.count('{d}'),p_step.count('(') ]        
        if step_name_hist not in step_name_dict:
            step_name_dict.append(step_name_hist)  

    rows = []
    for i in range(1,N):
        if dataset[i][1] in studentId_list:
            student_id_feature = studentId_list.index(dataset[i][1])
        else:
            student_id_feature = len(studentId_list)+1
        
        unit, section = dataset[i][2].split(", ")
        if unit in unit_list:
            unit_feature = unit_list.index(unit)
        else:
            unit_feature = len(unit_list)+1        
        if section in section_list:
            section_feature = section_list.index(section)
        else:
            section_feature = len(section_list)+1  

        if dataset[i][3] in problem_name_list:
            problem_name_feature = problem_name_list.index(dataset[i][3])
        else:
            problem_name_feature = len(problem_name_list)+1  

        step = process_step_name(dataset[i][5])
        step_name_processed = [ step.count('+'),step.count('-'),step.count('*'),step.count('/'),step.count('{var}'),step.count('{d}'),p_step.count('(') ]          
        if step_name_processed in step_name_dict:
            step_name_feature = step_name_dict.index(step_name_processed)
        else:
            step_name_feature = len(step_name_dict)+1          

        if dataset[i][len(dataset[i])-2] in kc_list_raw:
            kc_feature = kc_list_raw.index(dataset[i][len(dataset[i])-2])
        else:
            kc_feature = len(kc_list_raw)+1               
        #o = dataset[i][len(dataset[i])-1].split("~~")

        #print problem_hierarchy_feature
        rows.append([student_id_feature] + [unit_feature] + [section_feature]  + [problem_name_feature] + [step_name_feature] + [kc_feature])

    return rows




def get_feature_vectors(dataset, N, studentId_list, unit_list, section_list,
                         problem_name_list, step_name_list, kc_list, kc_list_raw):

    rows = []
    for i in range(1,N):
        student_id_feature = get_feature_vector(studentId_list,[dataset[i][1]],10)
        studentId_size = len(student_id_feature)

        unit, section = dataset[i][2].split(", ")
        unit_feature = get_feature_vector(unit_list,[unit], 1)
        unit_size = len(unit_feature)        
        section_feature = get_feature_vector(section_list,[section], 1)
        section_size = len(section_feature)

        ppname = process_problem_name(dataset[i][3])
        problem_name_feature = get_feature_vector(problem_name_list,[ppname],1 )
        problem_name_size = len(problem_name_feature)

        problem_view_feature = [float(dataset[i][4])/(float(dataset[i][4])+1)]
        problem_view_size = len(problem_view_feature)

        p_step = process_step_name(dataset[i][5])
        step_name_feature = [ p_step.count('+'),p_step.count('-'),p_step.count('*'),p_step.count('/'),p_step.count('{var}'),p_step.count('{d}'),p_step.count('(') ]
        #step_name_feature = [float(x)/sum(step_name_feature) for x in step_name_feature]
        step_name_size = len(step_name_feature)  

        #print step_name_feature
        kc_feature = get_feature_vector(kc_list, dataset[i][len(dataset[i])-2].split("~~"),1 )
        kc_feature_size = len(kc_feature)  

        opp_feature = get_feature_vector_opp(kc_list, dataset[i][len(dataset[i])-2].split("~~"), dataset[i][len(dataset[i])-1].split("~~"), 1 )
        opp_size = len(opp_feature)  

        #o = dataset[i][len(dataset[i])-1].split("~~")

        #print problem_hierarchy_feature
        rows.append(student_id_feature + unit_feature + section_feature + problem_name_feature+ problem_view_feature + step_name_feature + kc_feature + opp_feature)
    print "feature vector composition: ", studentId_size, unit_size, section_size, problem_name_size, problem_view_size, step_name_size, kc_feature_size, opp_size
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
    kc_list_raw = []
    testing_rows = []

    N = 50000#len(training_data)
    print "Num Of Lines to train: ", N
    for i in range(1,N):
        studentId = training_data[i][1]
        unit, section = training_data[i][2].split(", ")
        problem_name = training_data[i][3]
        step_name = process_step_name(training_data[i][5])

        kcraw = training_data[i][len(training_data[i])-2]
        kcs = training_data[i][len(training_data[i])-2].split("~~")
        #opp = training_data[i][len(training_data[i])-1].split("~~")

        is_first_correct_list.append(training_data[i][13])

        if studentId not in studentId_list:
            studentId_list.append(studentId)
        if unit not in unit_list:
            unit_list.append(unit)
        if section not in section_list:
            section_list.append(section)            

        ppname = process_problem_name(problem_name)     
        if ppname not in problem_name_list:
            problem_name_list.append(ppname)  

        if step_name not in step_name_list:
            step_name_list.append(step_name)
        if kcraw not in kc_list_raw:
            kc_list_raw.append(kcraw)            
        for kc in kcs:
            if kc not in kc_list:
                kc_list.append(kc)

    #print problem_name_list
    print "#of unique item in each categories: ",len(studentId_list), len(unit_list),len(section_list), len(problem_name_list), len(step_name_list), len(kc_list), len(kc_list_raw)

    # Create matrix...
    training_data_rows = get_feature_vectors(training_data, N, studentId_list, unit_list,section_list, problem_name_list, step_name_list, kc_list, kc_list_raw)
    testing_data_rows = get_feature_vectors(testing_data, len(testing_data), studentId_list,unit_list, section_list, problem_name_list, step_name_list, kc_list, kc_list_raw)

    return training_data_rows, is_first_correct_list, testing_data_rows

def myknndist(x, y):
    return np.sum((x-y)**2)

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
    #clf = linear_model.SGDClassifier(n_jobs=-1,n_iter=1000)
    #clf = linear_model.LogisticRegressionCV(n_jobs=-1, verbose=True)

    #clf = KNeighborsClassifier(n_jobs=-1, weights='distance', n_neighbors=5, metric='pyfunc', func=myknndist)
    clf = KNeighborsClassifier(n_jobs=-1, weights='distance', n_neighbors=200, p=2)

    #clf = RandomForestClassifier(n_estimators=50,n_jobs=-1, verbose=True)
    #clf = svm.SVC(verbose=True, cache_size=5000, kernel='linear', C=1.0)
    #clf = tree.DecisionTreeClassifier()

    #clf = GaussianNB()
    #clf = MultinomialNB(alpha=1.0)
    #clf = BernoulliNB(alpha=1.0, binarize=1.0)

    clf.fit(rows, is_first_correct_list)
    print clf
    #print clf.feature_importances_ 

    end = time.time()
    print "Time to train classifier", end-start, " sec"

    process = psutil.Process(os.getpid())
    print "RAM usage (MB):", process.memory_info().rss/1024/1024

    start = time.time()
    predict_result = clf.predict(rows)
    end = time.time()
    print "Time to do prediction of self-test", end-start, " sec"

    #print "Mean accuracy" , clf.score(rows, is_first_correct_list)
    print "first 50 items of predict: ", predict_result[:100]
    print "first 50 items of GT: ", is_first_correct_list[:100]
    predict_result = [ float(i) for i in predict_result]
    training_error = rmse(predict_result, [ float(i) for i in is_first_correct_list])
    print "rmse of first 50 items ", rmse([ float(i) for i in predict_result[:50]], [ float(i) for i in is_first_correct_list[:50]])
    print "rmse of first 150 items ", rmse([ float(i) for i in predict_result[:150]], [ float(i) for i in is_first_correct_list[:150]])
    print "rmse of first 500 items ", rmse([ float(i) for i in predict_result[:500]], [ float(i) for i in is_first_correct_list[:500]])
    print "rmse of first 1500 items ", rmse([ float(i) for i in predict_result[:1500]], [ float(i) for i in is_first_correct_list[:1500]])
    print "rmse of first 5000 items ", rmse([ float(i) for i in predict_result[:5000]], [ float(i) for i in is_first_correct_list[:5000]])
    print "rmse of first 15000 items ", rmse([ float(i) for i in predict_result[:15000]], [ float(i) for i in is_first_correct_list[:15000]])

    predict_result = clf.predict(testing_rows)
    predict_result = [ float(i) for i in predict_result]
    predict_error =  rmse(predict_result, [float(i[13]) for i in testing_result_data[1:]])

    print '|', dataset, '|', training_error, '|', predict_error ,'|'
    return

if __name__ == "__main__":
    main(sys.argv)
