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
import numpy
from multiprocessing.pool import ThreadPool

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


def get_feature_vectors_nb(training_data, maxtrainID, dataset, N, studentId_list, unit_list, section_list,
                         problem_name_list, step_name_list, kc_list, kc_list_raw,
                          student_dict, step_dict, problem_name_dict, kc_dict,
                          problem_step_dict, student_problem_dict, student_unit_dict,
                           student_kc_dict, student_kc_temporal_dict, day_list):

    step_name_dict = []
    for i in range(1,N):
        p_step = process_step_name(dataset[i][5])
        step_name_hist = [ p_step.count('+'),p_step.count('-'),p_step.count('*'),
        p_step.count('/'),p_step.count('{var}'),p_step.count('{d}'),p_step.count('(') ]  

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
        step_name_processed = [ step.count('+'),step.count('-'),step.count('*'),
        step.count('/'),step.count('{var}'),step.count('{d}'),p_step.count('(') ]    

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
        rows.append([student_id_feature] + [unit_feature] + [section_feature]  + 
        [problem_name_feature] + [step_name_feature] + [kc_feature])

    return rows




def get_feature_vectors(training_data, maxtrainID, dataset, N, studentId_list, unit_list, section_list,
                         problem_name_list, step_name_list, kc_list, kc_list_raw,
                          student_dict, step_dict, problem_name_dict, kc_dict, 
                         problem_step_dict, student_problem_dict, student_unit_dict,
                          student_kc_dict, student_kc_temporal_dict, day_list):

    rows = []
    for i in range(1,N):
        #skip those rows if not yet trained
        if int(dataset[i][0]) > maxtrainID+1:
            continue

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
        step_name_feature = [ p_step.count('+'),p_step.count('-'),p_step.count('*'),
            p_step.count('/'),p_step.count('{var}'),p_step.count('{d}'),p_step.count('(') ]
        #step_name_feature = [float(x)*2 for x in step_name_feature]
        step_name_size = len(step_name_feature)  

        #print step_name_feature
        kc_feature = get_feature_vector(kc_list, dataset[i][len(dataset[i])-2].split("~~"),1 )
        kc_feature_size = len(kc_feature)  

        opp_feature = get_feature_vector_opp(kc_list, dataset[i][len(dataset[i])-2].split("~~"),
         dataset[i][len(dataset[i])-1].split("~~"), 1 )
        opp_size = len(opp_feature)  
        
        #CFAR       
        if student_dict.has_key(dataset[i][1]):
            student_cfar = student_dict[dataset[i][1]]
        else:
            student_cfar = numpy.mean(student_dict.values())

        if step_dict.has_key(p_step):
            step_cfar = step_dict[p_step]
        else:
            step_cfar = numpy.mean(step_dict.values())
						
        if problem_name_dict.has_key(dataset[i][3]):
            problem_name_cfar = problem_name_dict[dataset[i][3]]
        else:
            problem_name_cfar = numpy.mean(problem_name_dict.values())
        
        if kc_dict.has_key(dataset[i][len(dataset[i])-2]):
            kc_cfar = kc_dict[dataset[i][len(dataset[i])-2]]
        else:
            kc_cfar = numpy.mean(kc_dict.values())
						
        if problem_step_dict.has_key((dataset[i][3], p_step)):
            problem_step_cfar = problem_step_dict[(dataset[i][3], p_step)]
        else:
			problem_step_cfar = numpy.mean(problem_step_dict.values())
			
        if student_problem_dict.has_key((dataset[i][1], dataset[i][3])):
            student_problem_cfar = student_problem_dict[(dataset[i][1], dataset[i][3])]
        else:
			student_problem_cfar = numpy.mean(student_problem_dict.values())
			
        if student_unit_dict.has_key((dataset[i][1], unit)):
            student_unit_cfar = student_unit_dict[(dataset[i][1], unit)]
        else:
            student_unit_cfar = numpy.mean(student_unit_dict.values())
		
        student_kc = (dataset[i][1], dataset[i][len(dataset[i])-2])
        student_kc_temporal = [0,0,0] 
        memory=[0,0,0,0] #[1day, 1week, 1 month, >1 month]

        if student_kc_dict.has_key(student_kc):
            student_kc_cfar = student_kc_dict[student_kc]

            itemlist=student_kc_temporal_dict[student_kc]
            # extract the historyitemlist
            historyitemlist =[]
            currid = dataset[i][0]
            for rowindex in itemlist:
                rowid = training_data[rowindex][0]
                if int(rowid) <= int(currid):
                    historyitemlist.append(rowindex)
                    currday = day_list[rowindex] #Find the best possible day of today
                    
            #Perform the memory check
            for rowindex in historyitemlist:
                testday = day_list[rowindex]
                if testday>currday:
                    continue
                elif testday==currday:
                    memory[0]=1
                elif testday+7>=currday:
                    memory[1]=1
                elif testday+30>=currday:
                    memory[2]=1
                else:
                    memory[3]=1

            # Take the last 6 or if any CFA and hint of this (student, kc) pairs            
            historyitemlist=historyitemlist[-6:]
            if len(historyitemlist)>0:
                cfa_mean=0
                hint_mean=0
                for rowindex in historyitemlist:
                    cfa_mean=cfa_mean+ int(training_data[rowindex][13])
                    hint_mean=hint_mean+int(training_data[rowindex][15])
                cfa_mean = float(cfa_mean)/len(historyitemlist)
                hint_mean = float(hint_mean)/len(historyitemlist)
                student_kc_temporal=[cfa_mean, hint_mean, 1]
        else:
            student_kc_cfar = numpy.mean(student_kc_dict.values())
            					

        o = dataset[i][len(dataset[i])-1].split("~~")
        oppsum=0
        for opp in o:
            try:
                oppsum=oppsum+int(opp)
            except ValueError:
                oppsum=oppsum
        oppsum=float(oppsum)/200.0

        #print problem_hierarchy_feature
        rows.append(student_id_feature + unit_feature + section_feature + problem_name_feature+
         problem_view_feature + step_name_feature + kc_feature + opp_feature +
         [student_cfar] + [step_cfar] + [problem_name_cfar] + [kc_cfar] +
         [problem_step_cfar] + [student_problem_cfar] + [student_unit_cfar] + [student_kc_cfar] +
         student_kc_temporal + memory)

        # rows.append(problem_view_feature +
        #  [student_cfar] + [step_cfar] + [problem_name_cfar] + [kc_cfar] +
        #  [problem_step_cfar] + [student_problem_cfar] + [student_unit_cfar] + [student_kc_cfar] +
        #  student_kc_temporal + memory + [oppsum])

    print "feature vector composition: ", studentId_size, unit_size, section_size, \
        problem_name_size, problem_view_size, step_name_size, kc_feature_size, opp_size

    return rows

def process_data(training_data, testing_data):
    #show_data(training_data)
    studentId_list = []
    section_list = []
    unit_list = []
    problem_name_list = []
    step_name_list = []
    CFA_list = []
    kc_list = []
    kc_list_raw = []
    testing_rows = []

    #CFAR
    student_dict={}
    student_dict_sum={}    
    step_dict={}
    step_dict_sum={}
    problem_name_dict={}
    problem_name_dict_sum={}    
    kc_dict={}
    kc_dict_sum={}    
    problem_step_dict={}
    problem_step_dict_sum={}
    student_problem_dict={}
    student_problem_dict_sum={}
    student_unit_dict={}
    student_unit_dict_sum={}
    student_kc_dict={}
    student_kc_dict_sum={}
    student_kc_temporal={}
    day_list = [0]

    N = 50000#len(training_data)
    print "Num Of Lines to train: ", N
    for i in range(1,N):
        studentId = training_data[i][1]
        unit, section = training_data[i][2].split(", ")
        problem_name = training_data[i][3]
        step_name = process_step_name(training_data[i][5])
        step_name_raw = training_data[i][5]

        kcraw = training_data[i][len(training_data[i])-2]
        kcs = training_data[i][len(training_data[i])-2].split("~~")
        #opp = training_data[i][len(training_data[i])-1].split("~~")

        cfa=training_data[i][13];
        CFA_list.append(cfa)

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

        #CFAR
        problem_step = (problem_name, step_name)
        student_problem = (studentId, problem_name)
        student_unit = (studentId, unit)
        student_kcs = (studentId, training_data[i][len(training_data[i])-2])   

        if student_dict.has_key(studentId):
            student_dict[studentId]=student_dict[studentId]+int(cfa)
            student_dict_sum[studentId]=student_dict_sum[studentId]+1
        else:
            student_dict[studentId]=int(cfa)
            student_dict_sum[studentId]=1

        if step_dict.has_key(step_name):
            step_dict[step_name]=step_dict[step_name]+int(cfa)
            step_dict_sum[step_name]=step_dict_sum[step_name]+1
        else:
            step_dict[step_name]=int(cfa)
            step_dict_sum[step_name]=1

        if problem_name_dict.has_key(problem_name):
            problem_name_dict[problem_name]=problem_name_dict[problem_name]+int(cfa)
            problem_name_dict_sum[problem_name]=problem_name_dict_sum[problem_name]+1
        else:
            problem_name_dict[problem_name]=int(cfa)
            problem_name_dict_sum[problem_name]=1

        if kc_dict.has_key(kcraw):
            kc_dict[kcraw]=kc_dict[kcraw]+int(cfa)
            kc_dict_sum[kcraw]=kc_dict_sum[kcraw]+1
        else:
            kc_dict[kcraw]=int(cfa)
            kc_dict_sum[kcraw]=1

        if problem_step_dict.has_key(problem_step):
            problem_step_dict[problem_step]=problem_step_dict[problem_step]+int(cfa)
            problem_step_dict_sum[problem_step]=problem_step_dict_sum[problem_step]+1
        else:
            problem_step_dict[problem_step]=int(cfa)
            problem_step_dict_sum[problem_step]=1


        if student_problem_dict.has_key(student_problem):
            student_problem_dict[student_problem]=student_problem_dict[student_problem]+int(cfa)
            student_problem_dict_sum[student_problem]=student_problem_dict_sum[student_problem]+1
        else:
            student_problem_dict[student_problem]=int(cfa)
            student_problem_dict_sum[student_problem]=1
 
        if student_unit_dict.has_key(student_unit):
            student_unit_dict[student_unit]=student_unit_dict[student_unit]+int(cfa)
            student_unit_dict_sum[student_unit]=student_unit_dict_sum[student_unit]+1
        else:
            student_unit_dict[student_unit]=int(cfa)
            student_unit_dict_sum[student_unit]=1 

        if student_kc_dict.has_key(student_kcs):
            student_kc_dict[student_kcs]=student_kc_dict[student_kcs]+int(cfa)
            student_kc_dict_sum[student_kcs]=student_kc_dict_sum[student_kcs]+1
            student_kc_temporal[student_kcs].append(i)
        else:
            student_kc_dict[student_kcs]=int(cfa)
            student_kc_dict_sum[student_kcs]=1    
            student_kc_temporal[student_kcs]=[i]  

        if float(training_data[i][10]) >= 0:
            day_list.append(day_list[-1])
        else:
            day_list.append(day_list[-1]+1)
    
    #print day_list

    #CFAR
    for key in student_dict:
        student_dict[key] = float(student_dict[key])/student_dict_sum[key]
    for key in step_dict:
        step_dict[key] = float(step_dict[key])/step_dict_sum[key]
    for key in problem_name_dict:
        problem_name_dict[key] = float(problem_name_dict[key])/problem_name_dict_sum[key]
    for key in kc_dict:
        kc_dict[key] = float(kc_dict[key])/kc_dict_sum[key]
                                
    for key in problem_step_dict:
        problem_step_dict[key] = float(problem_step_dict[key])/problem_step_dict_sum[key]
    for key in student_problem_dict:
        student_problem_dict[key] = float(student_problem_dict[key])/student_problem_dict_sum[key]
    for key in student_unit_dict:
        student_unit_dict[key] = float(student_unit_dict[key])/student_unit_dict_sum[key]
    for key in student_kc_dict:
        student_kc_dict[key] = float(student_kc_dict[key])/student_kc_dict_sum[key]
    
    maxtrainID = int(training_data[N-1][0])

    #print problem_name_list
    print "#of unique item in each categories: ",len(studentId_list), len(unit_list),\
        len(section_list), len(problem_name_list), len(step_name_list), len(kc_list), len(kc_list_raw)

    # do it in multi-thread
    # NumOfCore=4
    # partsize = N/NumOfCore
    # thread_list = []
    # training_data_rows= []
    # trainpartresult = [0, 0, 0, 0]
    # pool = ThreadPool(processes=NumOfCore)
    # for i in range(0, NumOfCore):
    #     trainpartresult[i] = pool.apply_async(get_feature_vectors, (training_data, maxtrainID, training_data[i*partsize:(i+1)*partsize], partsize, studentId_list, unit_list,\
    #     section_list, problem_name_list, step_name_list, kc_list, kc_list_raw, student_dict,\
    #      step_dict, problem_name_dict, kc_dict, problem_step_dict, student_problem_dict, \
    #      student_unit_dict, student_kc_dict, student_kc_temporal, day_list))

    # for i in range(0, NumOfCore):
    #     training_data_rows.append(trainpartresult[i].get())

    # Create matrix...
    training_data_rows = get_feature_vectors(training_data, maxtrainID, training_data, N, studentId_list, unit_list,
        section_list, problem_name_list, step_name_list, kc_list, kc_list_raw, student_dict,
         step_dict, problem_name_dict, kc_dict, problem_step_dict, student_problem_dict, 
         student_unit_dict, student_kc_dict, student_kc_temporal, day_list)

    testing_data_rows = get_feature_vectors(training_data, maxtrainID, testing_data, len(testing_data), studentId_list,
        unit_list, section_list, problem_name_list, step_name_list, kc_list, kc_list_raw,
         student_dict, step_dict, problem_name_dict, kc_dict, problem_step_dict,
          student_problem_dict, student_unit_dict, student_kc_dict, student_kc_temporal, day_list)

    return training_data_rows, CFA_list, testing_data_rows

def myknndist(x, y):
    return np.sum((x-y)**2)

def main(arg):
    dataset = arg[1] #'algebra_2005_2006'
    start = time.time()
    training_data, testing_data, testing_result_data = load_data(dataset)
    end = time.time()
    print "Time to load data", end-start, " sec"

    start = time.time()
    rows, CFA_list, testing_rows = process_data(training_data, testing_data)
    end = time.time()
    print "Time to process data", end-start , " sec"   
    print len(rows),len(CFA_list),len(testing_rows),len(rows[0])

    #print rows[:200]
    #print testing_rows[:200]

    del training_data
    del testing_data
    gc.collect()

    process = psutil.Process(os.getpid())
    print "RAM usage (MB):", process.memory_info().rss/1024/1024

    #write_file("preprocessed_train.txt", rows)
    #write_file("preprocessed_test.txt", testing_rows)

    start = time.time()
    #clf = linear_model.SGDClassifier(n_jobs=-1,n_iter=1000)
    #clf = linear_model.LogisticRegressionCV(n_jobs=-1, verbose=True)

    #clf = KNeighborsClassifier(n_jobs=-1, weights='distance', n_neighbors=5, metric='pyfunc', func=myknndist)
    clf = KNeighborsClassifier(n_jobs=-1, weights='distance', n_neighbors=2000, p=2)

    #clf = RandomForestClassifier(n_estimators=100,n_jobs=-1, verbose=True)
    #clf = svm.SVC(verbose=True, cache_size=5000, kernel='linear', C=1.0)
    #clf = tree.DecisionTreeClassifier()

    #clf = GaussianNB()
    #clf = MultinomialNB(alpha=1.0)
    #clf = BernoulliNB(alpha=1.0, binarize=1.0)

    clf.fit(rows, CFA_list)
    print clf
    #print clf.feature_importances_ 

    end = time.time()
    print "Time to train classifier", end-start, " sec"

    process = psutil.Process(os.getpid())
    print "RAM usage (MB):", process.memory_info().rss/1024/1024

    start = time.time()
    predict_result = clf.predict(rows[:1500])
    end = time.time()
    print "Time to do prediction of 1.5k self-test", end-start, " sec"

    #print "Mean accuracy" , clf.score(rows, CFA_list)
    print "first 50 items of predict: ", predict_result[:100]
    print "first 50 items of GT: ", CFA_list[:100]
    predict_result = [ float(i) for i in predict_result]
    training_error = rmse(predict_result, [ float(i) for i in CFA_list[:50000]])
    print "rmse of first 50 items ", rmse([ float(i) for i in predict_result[:50]], [ float(i) for i in CFA_list[:50]])
    print "rmse of first 150 items ", rmse([ float(i) for i in predict_result[:150]], [ float(i) for i in CFA_list[:150]])
    print "rmse of first 500 items ", rmse([ float(i) for i in predict_result[:500]], [ float(i) for i in CFA_list[:500]])
    print "rmse of first 1500 items ", rmse([ float(i) for i in predict_result[:1500]], [ float(i) for i in CFA_list[:1500]])
    #print "rmse of first 5000 items ", rmse([ float(i) for i in predict_result[:5000]], [ float(i) for i in CFA_list[:5000]])
    #print "rmse of first 15000 items ", rmse([ float(i) for i in predict_result[:15000]], [ float(i) for i in CFA_list[:15000]])
    #print "rmse of first 45000 items ", rmse([ float(i) for i in predict_result[:45000]], [ float(i) for i in CFA_list[:45000]])

    start = time.time()
    predict_result = clf.predict(testing_rows)
    end = time.time()
    print "Time to do prediction of testing rows", end-start, " sec"    
    predict_result = [ float(i) for i in predict_result]
    predict_error =  rmse(predict_result, [float(i[13]) for i in testing_result_data[1:]])

    print '|', dataset, '|', training_error, '|', predict_error ,'|'
    return

if __name__ == "__main__":
    main(sys.argv)
