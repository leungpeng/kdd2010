import sys
import random
from project import load_data, show_data, rmse

def main(arg):
    dataset = arg[1] #'algebra_2005_2006'
    training_data, testing_data, testing_result_data = load_data(dataset)

    predict_result = [random.random() for i in testing_result_data[1:]]
    predict_error =  rmse(predict_result, [float(i[13]) for i in testing_result_data[1:]])

    predict_result = [random.random() for i in training_data[1:]]
    training_error = rmse(predict_result, [float(i[13]) for i in training_data[1:]])

    print '|', dataset, '|', training_error, '|', predict_error ,'|'
    return

if __name__ == "__main__":
    main(sys.argv)
