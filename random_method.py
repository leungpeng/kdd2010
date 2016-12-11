import sys
import random
from project import load_data, show_data, rmse, plotroc, Classifier_Eval

def main(arg):
    dataset = arg[1] #'algebra_2005_2006'
    training_data, testing_data, testing_result_data = load_data(dataset)

    predict_test_result = [random.randint(0,1) for i in training_data[1:]]
    test_CFA = [float(i[13]) for i in training_data[1:]]
    #training_error = rmse(predict_test_result, test_CFA)
    Classifier_Eval(test_CFA, predict_test_result,True)

    predict_result = [random.randint(0,1) for i in testing_result_data[1:]]
    train_CFA = [float(i[13]) for i in testing_result_data[1:]]
    #predict_error =  rmse(predict_result, train_CFA)
    Classifier_Eval(train_CFA, predict_result, False)   

    #print '|', dataset, '|', training_error, '|', predict_error ,'|'
    plotroc(train_CFA, predict_result, test_CFA, predict_test_result)
    return

if __name__ == "__main__":
    main(sys.argv)

