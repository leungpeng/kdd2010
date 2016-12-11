
from project import write_file, load_data, show_data, rmse, process_step_name,\
 process_problem_name, plotroc, plotrocmany, Classifier_Eval
from sklearn import svm, linear_model, tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier
import os, numpy, sys, random
from feature_vector import process_data
from cf import training, predict_from_matrix

def main(arg):
    dataset = arg[1] #'algebra_2005_2006'
    NumOfLineToTrain = 50000
    training_data, testing_data, testing_result_data = load_data(dataset)

    rows, CFA_list, testing_rows, test_CFA = process_data(training_data,\
     testing_data, testing_result_data, NumOfLineToTrain, False, False)
    print len(rows),len(CFA_list),len(testing_rows),len(rows[0])


    clf=[]
    y_pred_list=[]
    name_list=['KNN', 'RandomForest','Collabrative filtering']
    #y_pred_list.append([random.randint(0,1) for i in testing_rows])

    ##############################################################
    M=2 # number of classifier

    #clf = linear_model.SGDClassifier(n_jobs=-1,n_iter=1000)
    #clf = linear_model.LogisticRegressionCV(n_jobs=-1, verbose=True)

    #clf = KNeighborsClassifier(n_jobs=-1, weights='distance', n_neighbors=5, metric='pyfunc', func=myknndist)
    clf.append(KNeighborsClassifier(n_jobs=-1, weights='distance', n_neighbors=5, p=2))
    clf.append(RandomForestClassifier(n_estimators=50,n_jobs=-1, verbose=False))

    #clf = svm.SVC(verbose=True, cache_size=5000, C=1.0)
    #clf = tree.DecisionTreeClassifier()

    #clf = GaussianNB()
    #clf = MultinomialNB(alpha=1.0)
    #clf = BernoulliNB(alpha=2.0, binarize=1.0)

    #############################################################


    #Train and do prediction for each method
    for i in range(M):
        print 'Training', name_list[i], '...'
        clf[i].fit(rows, CFA_list)
        print 'Predicting', name_list[i], '...'
        y_pred_list.append(clf[i].predict(testing_rows))

    matrix, students, problems, testing_sample = training(training_data[:NumOfLineToTrain])
    y_pred_list.append(predict_from_matrix(matrix, students, problems,\
        [ (data[1].upper(), data[3].upper()) for data in testing_data[1:]]))

    for i in range(len(name_list)):
        print name_list[i], ' rmse= ', rmse(y_pred_list[i], test_CFA)
        print "first 20 items of prediction: ",[int(i) for i in y_pred_list[i][:20]]

    print "first 20 items of test GT: ", [int(i) for i in test_CFA[:20]]
    print 'Please close the ROC curve plot'
    plotrocmany(test_CFA, y_pred_list, name_list)
    return

if __name__ == "__main__":
    main(sys.argv)
