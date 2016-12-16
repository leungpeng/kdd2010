import sys, random
from project import load_data, show_data, rmse, create_matrix, latent_factor,\
 predict_from_matrix, process_problem_name, process_step_name, plotroc

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

def training(data, learnrate, regular, numofstep):
    students = {}
    problems = {}
    testing_sample = []
    N = len(data)

    print "Num Of Lines to train: ", N
    for i in range(1,N):
        student, hierarchy, problem_name, step_name = data_key(data[i])
        is_first_correct = float(data[i][13])
        item_key = process_step_name(step_name)
        students.setdefault(student, []).append(is_first_correct)
        problems.setdefault(item_key, []).append(is_first_correct)
        testing_sample.append((student, item_key, is_first_correct))

    matrix = create_matrix(testing_sample, students, problems)
    matrix = latent_factor(testing_sample, matrix, students, problems, learnrate, regular, numofstep)

    print "Training Done..."
    return matrix, students, problems, testing_sample

def main(arg):
    dataset = arg[1] #'algebra_2005_2006'
    training_data, testing_data, testing_result_data = load_data(dataset)

    #shuffle the training data
    #training_data = random.shuffle(training_data)

    learnrate = 0.01
    regular = 0.02
    numofstep = 30
    matrix, students, problems, testing_sample = training(training_data, learnrate, regular, numofstep)
    predict_result = predict_from_matrix(matrix, students, problems,[ (data[0].upper(), data[1].upper()) for data in testing_sample])
    training_error = rmse(predict_result, [float(i[2]) for i in testing_sample])

    predict_test_result = predict_from_matrix(matrix, students, problems,[ (data[1].upper(), process_step_name(data[5].upper())) for data in testing_data[1:]])
    predict_error = rmse(predict_test_result, [float(i[13]) for i in testing_result_data[1:]])

    print "first 50 items of prediction before rounding: ",[float(i) for i in predict_test_result[:50]]
    print "first 50 items of prediction: ",[int(round(float(i))) for i in predict_test_result[:50]]
    print "first 50 items of test GT: ", [int(i[13]) for i in testing_result_data[1:50]]
    print '|', dataset, '|', training_error, '|', predict_error ,'|'
    plotroc([float(i[2]) for i in testing_sample], predict_result,\
     [float(i[13]) for i in testing_result_data[1:]], predict_test_result)
    return

if __name__ == "__main__":
    main(sys.argv)
