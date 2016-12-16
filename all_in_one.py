
from project import write_file, load_data, show_data, rmse, process_step_name,\
 process_problem_name, plotroc, plotrocmany, Classifier_Eval
from sklearn import svm, linear_model, tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier
import os, numpy, sys, random, time
from feature_vector import process_data
from cf import training, predict_from_matrix

def main(arg):
    dataset = arg[1] #'algebra_2005_2006'
    training_data, testing_data, testing_result_data = load_data(dataset)
    NumOfLineToTrain = 300000 #len(training_data)

    # mode0: normal, 1: normal+condensed, 2: only condensed
    Feature_vector_mode = 0
    rows, CFA_list, testing_rows, test_CFA = process_data(training_data,\
     testing_data, testing_result_data, NumOfLineToTrain, False, Feature_vector_mode)
    print 'Training rows:', len(rows),'Testing rows:', len(testing_rows), \
    '# of features:', len(rows[0])


    clf=[]; y_pred_list=[]; M=3 #number of classifier
    name_list=['KNN', 'RandomForest', 'LinearSVM', \
    'Collabrative filtering']
    #y_pred_list.append([random.randint(0,1) for i in testing_rows])

    ##############################################################

    #clf = KNeighborsClassifier(n_jobs=-1, weights='distance', n_neighbors=5, metric='pyfunc', func=myknndist)
    clf.append(KNeighborsClassifier(n_jobs=-1, weights='distance', n_neighbors=10, p=2))
    clf.append(RandomForestClassifier(n_estimators=100,n_jobs=-1, verbose=False))
    clf.append(svm.LinearSVC(verbose=False, C=4.0))
    #clf = tree.DecisionTreeClassifier()

    #clf.append(GaussianNB())
    #clf = MultinomialNB(alpha=1.0)
    #clf.append(BernoulliNB(alpha=2.0, binarize=1.0))

    #clf.append(linear_model.SGDClassifier(n_jobs=-1,n_iter=1000))
    #clf.append(linear_model.LogisticRegressionCV(n_jobs=-1, verbose=False))

    #############################################################

    #Train and do prediction for each method
    for i in range(M):
        start = time.time()
        print 'Training', name_list[i], '...'
        clf[i].fit(rows, CFA_list)
        print 'Predicting', name_list[i], '...'
        y_pred_list.append(clf[i].predict(testing_rows))
        end = time.time()
        print "Time elapse: ", end-start, " sec"

    start = time.time()
    learnrate = 0.01; regular = 0.02; numofstep = 100
    matrix, students, problems, testing_sample = training(training_data[:NumOfLineToTrain], learnrate, regular, numofstep)
    y_pred_list.append(predict_from_matrix(matrix, students, problems,\
        [ (data[1].upper(), process_step_name(data[5].upper())) for data in testing_data[1:]]))
    end = time.time()
    print "Time elapse: ", end-start, " sec"

    for i in range(len(name_list)):
        print name_list[i], ' rmse= ', rmse(y_pred_list[i], test_CFA)
        print "first 30 items of prediction: ",[int(round(float(i))) for i in y_pred_list[i][:30]]

    print "first 30 items of test GT: ", [int(i) for i in test_CFA[:30]]
    print 'Please close the ROC curve plot'
    plotrocmany(test_CFA, y_pred_list, name_list)
    return

if __name__ == "__main__":
    main(sys.argv)
