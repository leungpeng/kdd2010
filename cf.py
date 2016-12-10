import sys
import random
from project import load_data, show_data, rmse, create_matrix, latent_factor, predict_from_matrix

def data_key(data):
    student = data[1].upper()
    problem_hierarchy = data[2].upper()
    problem_name = data[3].upper()
    step_name = data[5].upper()
    return student, problem_hierarchy, problem_name, step_name

def get_avg_rankings(ranking):
    avg_ranking = { i: float(sum(r))/float(len(r)) for i, r in ranking.items() }
    overall_ranking = sum([ ar for i,ar in avg_ranking.items()]) / len(avg_ranking)
    return avg_ranking, overall_ranking

def get_bias_with_key(avg_data, key):
    return avg_data[0][key] - avg_data[1] if key in avg_data[0] else 0.0

def training(data):
    students = {}
    problems = {}
    testing_sample = []
    N = len(data)
    for i in range(1,N):
        student, hierarchy, problem_name, step_name = data_key(data[i])
        is_first_correct = float(data[i][13])
        students.setdefault(student, []).append(is_first_correct)
        problems.setdefault(problem_name, []).append(is_first_correct)
        testing_sample.append((student, problem_name, is_first_correct))
    matrix = create_matrix(testing_sample, students, problems)
    matrix = latent_factor(testing_sample, matrix, students, problems)
    print "Training Done..."
    return matrix, students, problems, testing_sample

def main(arg):
    dataset = arg[1] #'algebra_2005_2006'
    training_data, testing_data, testing_result_data = load_data(dataset)

    matrix, students, problems, testing_sample = training(training_data)
    predict_result = predict_from_matrix(matrix, students, problems,[ (data[0].upper(), data[1].upper()) for data in testing_sample])
    training_error = rmse(predict_result, [float(i[2]) for i in testing_sample[1:]])

    predict_result = predict_from_matrix(matrix, students, problems,[ (data[1].upper(), data[3].upper()) for data in testing_data])
    predict_error = rmse(predict_result, [float(i[13]) for i in testing_result_data[1:]])

    print '|', dataset, '|', training_error, '|', predict_error ,'|'
    return

if __name__ == "__main__":
    main(sys.argv)
