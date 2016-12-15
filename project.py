#!/usr/bin/env python
# Usage: python project.py algebra_2005_2006
import sys, os.path, operator, math, random, re, numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from itertools import cycle

def Classifier_Eval(y_true, y_pred, IsSelfTest=True):
    if IsSelfTest==True:
        print '\n|||Self test Result|||'
    else:
        print '\n|||Test case Result|||'

    truearray = [int(round(float(i))) for i in y_true]
    predarray = [int(round(float(i))) for i in y_pred]
    print 'rmse: ', rmse(predarray, truearray)
    print 'R2 coeff: ', r2_score(truearray, predarray)
    fpr, tpr, thresholds = roc_curve(truearray, predarray)
    print 'roc area: ', auc(fpr, tpr)
    #print classification_report(truearray, predarray)


def plotrocmany(y_true, y_pred_list, name_list):
    plt.figure()
    lw = 2
    fpr=dict()
    tpr=dict()
    thrd=dict()
    roc_auc=dict()
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red','green','blue', 'Yellow'])
    N=len(name_list)

    for i, color in zip(range(N), colors):
        j=int(i)
        fpr[j], tpr[j], thrd[j] = roc_curve([int(k) for k in y_true],\
         [ int(round(float(k))) for k in y_pred_list[j]])
        roc_auc[j] = auc(fpr[j], tpr[j])

        plt.plot(fpr[j], tpr[j], color=color,
                 lw=lw, label='%s (area = %0.2f)' % (name_list[j], roc_auc[j]))

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--', label='Random Baseline')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.legend(loc="lower right")
    plt.show()



def plotroc(train_gt, train_predict, test_gt, test_predict):
    fpr, tpr, thresholds = roc_curve([int(i) for i in train_gt],
     [ int(round(float(i))) for i in train_predict])
    roc_auc = auc(fpr, tpr)
    fpr2, tpr2, thresholds = roc_curve([int(i) for i in test_gt],
     [ int(round(float(i))) for i in test_predict])
    roc_auc2 = auc(fpr2, tpr2)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='deeppink',
             lw=lw, label='Self test ROC curve (area = %0.2f)' % roc_auc)
    plt.plot(fpr2, tpr2, color='darkorange',
             lw=lw, label='Test ROC curve (area = %0.2f)' % roc_auc2)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--', label='Random Baseline')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.legend(loc="lower right")
    plt.show()

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

def rmse(y_pred, y_true):
    #n = len(y_pred)
    #mse = sum([(predict - expected)**2 for predict, expected in zip(y_pred, y_true)]) / n
    truearray = [int(round(float(i))) for i in y_true]
    predarray = [int(round(float(i))) for i in y_pred]
    return np.sqrt(mean_squared_error(truearray, predarray))


'''
Latent Factor Matrix Factorization
'''
def get_avg_rankings(ranking):
    avg_ranking = { i: float(sum(r))/float(len(r)) for i, r in ranking.items() }
    overall_ranking = sum([ ar for i,ar in avg_ranking.items()]) / len(avg_ranking)
    return avg_ranking, overall_ranking

def create_matrix(training_data, user_rankings, item_rankings):
    users, items = [ i for i in user_rankings.keys()], [ i for i in item_rankings.keys()]
    avg_user_rankings , overall_user_ranking = get_avg_rankings(user_rankings)
    avg_item_rankings , overall_item_ranking = get_avg_rankings(item_rankings)
    matrix = []
    print "create_matrix :", len(users), len(items)

    for i in users:
        matrix.append([0.0] * len(items))

    for user, item, ranking in training_data:
        user_bias = avg_user_rankings[user] - overall_user_ranking if user in avg_user_rankings else 0.0
        item_bias = avg_item_rankings[item] - overall_item_ranking if item in avg_item_rankings else 0.0
        predict_value = overall_item_ranking + user_bias + item_bias
        matrix[users.index(user)][items.index(item)] = float(predict_value)

    return matrix

def latent_factor(training_data, matrix, user_rankings, item_rankings, learn=0.001,\
 regular=0.02, steps=50):
    #write_file("cf_matrix.txt", matrix)
    users, items = [ i for i in user_rankings.keys()], [ i for i in item_rankings.keys()]
    U, s, V = np.linalg.svd(matrix, full_matrices=False)
    Q, P = U, np.transpose(np.dot(np.diag(s), V))

    print "Start Latent Factor...", Q.shape, P.shape
    #Q, P = np.random.rand(Q.shape[0],Q.shape[1]), np.random.rand(P.shape[0],P.shape[1])
    for t in range(1, steps):
        for user, item, ranking in training_data:
            #current_matrix = np.dot(Q, np.transpose(P))
            user_idx, item_idx = users.index(user), items.index(item)
            q_i, p_x = Q[user_idx], P[item_idx]
            r_x_i = matrix[user_idx][item_idx]
            q_i_p_x = np.dot(q_i, p_x)
            error = 2 * (r_x_i - q_i_p_x)
            #print 'hahaha', r_x_i, q_i_p_x, error

            error_p_x, error_q_i = np.multiply(error, p_x), np.multiply(error, q_i)
            learn_p_x, learn_q_i = np.multiply(regular, p_x), np.multiply(regular, q_i)

            # q_i = q_i + learn_1 * ( error * p_x - learn_2 * q_i )
            Q[user_idx] = np.add(q_i,np.multiply(learn, np.subtract(error_p_x,learn_q_i)))
            # p_x = p_x + learn_2 * ( error * q_i - learn_1 * p_x )
            P[item_idx] = np.add(p_x,np.multiply(learn, np.subtract(error_q_i,learn_p_x)))

        new_matrix = np.dot(Q, np.transpose(P))

        if t%5 ==0:
            predict_result = predict_from_matrix(new_matrix, user_rankings, item_rankings,[ data[:2] for data in training_data])
            #print predict_result
            print 'rmse of this epoch', t, rmse(predict_result,[ data[2] for data in training_data])

    #write_file("cf_recovered_matrix.txt", np.asarray(new_matrix).reshape(len(users), len(items)))
    return new_matrix

def predict_from_matrix(matrix, user_rankings, item_rankings, data):
    result = []
    users, items = [ i for i in user_rankings.keys()], [ i for i in item_rankings.keys()]
    avg_user_rankings, overall_user_ranking = get_avg_rankings(user_rankings)
    avg_item_rankings, overall_item_ranking = get_avg_rankings(item_rankings)

    for target_user, target_item in data:
        if target_user in users and target_item in items:
            predict_value = matrix[users.index(target_user)][items.index(target_item)]
        else:
            user_bias = avg_user_rankings[target_user] - overall_user_ranking if target_user in avg_user_rankings else 0.0
            item_bias = avg_item_rankings[target_item] - overall_item_ranking if target_item in avg_item_rankings else 0.0
            predict_value = overall_item_ranking + user_bias + item_bias
        result.append(predict_value)

    return result
