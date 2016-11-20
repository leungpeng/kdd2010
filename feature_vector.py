#!/usr/bin/env python
# Usage: python project.py algebra_2005_2006
import sys
import random
from project import load_data, show_data, rmse, process_step_name
from sklearn import svm
from sklearn import linear_model

def get_feature_vector(element_set, value_set):
    result = [0.0] * len(element_set)
    for val in value_set:
        if val in element_set :
            result[element_set.index(val)] = 1.0
    return result

def get_feature_vectors(dataset, N, studentId_list, section_list, unit_list, problem_name_list, step_name_list, kc_list):
    rows = []
    for i in range(1,N):
        student_id_feature = get_feature_vector(studentId_list,[dataset[i][1]])

        section, unit = dataset[i][2].split(", ")

        section_feature = get_feature_vector(section_list,[section])
        unit_feature = get_feature_vector(unit_list,[unit])

        problem_name_feature = get_feature_vector(problem_name_list,[dataset[i][3]])
        problem_view_feature = [float(dataset[i][4])]

        step_name_feature = get_feature_vector(step_name_list,[process_step_name(dataset[i][5])])

        kc_feature = get_feature_vector(kc_list, dataset[i][len(dataset[i])-2].split("~~") )
        o = dataset[i][len(dataset[i])-1].split("~~")

        #print problem_hierarchy_feature
        rows.append(student_id_feature + section_feature + unit_feature + problem_view_feature + step_name_feature + kc_feature)
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
    N = 60000 #len(training_data)
    for i in range(1,N):
        studentId = training_data[i][1]
        section, unit = training_data[i][2].split(", ")
        problem_name = training_data[i][3]
        step_name = process_step_name(training_data[i][5])

        kcs = training_data[i][len(training_data[i])-2].split("~~")
        o = training_data[i][len(training_data[i])-1].split("~~")

        is_first_correct_list.append(training_data[i][13])

        if studentId not in studentId_list:
            studentId_list.append(studentId)
        if section not in section_list:
            section_list.append(section)
        if unit not in unit_list:
            unit_list.append(unit)
        if problem_name not in problem_name_list:
            problem_name_list.append(problem_name)
        if step_name not in step_name_list:
            step_name_list.append(step_name)
        for kc in kcs:
            if kc not in kc_list:
                kc_list.append(kc)

    print len(studentId_list), len(section_list), len(unit_list), len(problem_name_list), len(step_name_list), len(kc_list)
    # Create matrix...
    training_data_rows = get_feature_vectors(training_data, N, studentId_list, section_list, unit_list, problem_name_list, step_name_list, kc_list)

    testing_data_rows = get_feature_vectors(testing_data, len(testing_data), studentId_list, section_list, unit_list, problem_name_list, step_name_list, kc_list)
    return training_data_rows, is_first_correct_list, testing_data_rows


def main(arg):
    dataset = arg[1] #'algebra_2005_2006'
    training_data, testing_data, testing_result_data = load_data(dataset)

    rows, is_first_correct_list, testing_rows = process_data(training_data, testing_data)
    print len(rows),len(is_first_correct_list),len(testing_rows),len(rows[0])

    clf = linear_model.SGDClassifier(n_jobs=2)

    #clf = svm.SVC(gamma=0.001, C=100., verbose=True)
    clf.fit(rows, is_first_correct_list)
    print clf
    predict_result = clf.predict(rows)
    predict_result = [ float(i) for i in predict_result]
    training_error = rmse(predict_result, [ float(i) for i in is_first_correct_list])

    predict_result = clf.predict(testing_rows)
    predict_result = [ float(i) for i in predict_result]
    predict_error =  rmse(predict_result, [float(i[13]) for i in testing_result_data[1:]])

    print '|', dataset, '|', training_error, '|', predict_error ,'|'
    return

if __name__ == "__main__":
    main(sys.argv)
