
from project import write_file, load_data, show_data, rmse, process_step_name,\
 process_problem_name, plotroc, plotrocmany, Classifier_Eval
from sklearn import svm
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import os
import numpy
import sys
from feature_vector import process_data

def main(arg):
    dataset = arg[1] #'algebra_2005_2006'
    training_data, testing_data, testing_result_data = load_data(dataset)

    rows, CFA_list, testing_rows, test_CFA = process_data(training_data,\
     testing_data, testing_result_data, 10000)
    print len(rows),len(CFA_list),len(testing_rows),len(rows[0])

    ##############################################################
    clf=[]
    y_pred_list=[]
    N = 2 # number of methods
    name_list=['KNN', 'RandomForest']

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
    for i in range(N):
        clf[i].fit(rows, CFA_list)
        y_pred_list.append(clf[i].predict(testing_rows))
        print name_list[i], ' rmse= ', rmse(y_pred_list[i], test_CFA)

    plotrocmany(test_CFA, y_pred_list, name_list, N)
    return

if __name__ == "__main__":
    main(sys.argv)