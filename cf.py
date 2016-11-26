import sys
import random
from project import load_data, show_data, rmse

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

def training(training_data):
    students = {}
    student_hierarchy = {}
    student_hierarchy_problem = {}
    for i in range(1,len(training_data)):
        student, hierarchy, problem_name, step_name = data_key(training_data[i])
        is_first_correct = float(training_data[i][13])

        students.setdefault(student, []).append(is_first_correct)
        student_hierarchy.setdefault(student, {}).setdefault(hierarchy, []).append(is_first_correct)
        student_hierarchy_problem.setdefault(student, {}).setdefault(hierarchy, {}).setdefault(problem_name, []).append(is_first_correct)
    avg_student = get_avg_rankings(students)
    avg_student_hierarchy = { student:get_avg_rankings(hierarchy) for student, hierarchy in student_hierarchy.items()}
    avg_student_hierarchy_problem = {}
    #avg_student_hierarchy_problem = { student:{ { hierarchy:get_avg_rankings(problem) for hierarchy, problem in hierarchies.items()} } for student, hierarchies in student_hierarchy_problem.items()}

    print "Training Done..."
    return avg_student, avg_student_hierarchy, avg_student_hierarchy_problem

def predict(avg_student, avg_student_hierarchy, avg_student_hierarchy_problem, testing_data):
    result = []
    for i in range(1,len(testing_data)):
        student, problem_hierarchy, problem, step = data_key(testing_data[i])

        student_bias = get_bias_with_key(avg_student, student)
        hierarchy_bias = get_bias_with_key(avg_student_hierarchy[student], problem_hierarchy) if student in avg_student_hierarchy else 0.0
        #problem_bias = avg_problem_rankings[problem] - overall_problem_ranking if problem in avg_problem_rankings else 0.0


        predict_value = avg_student[1] + student_bias# + hierarchy_bias
        result.append(predict_value)
    return result

def main(arg):
    dataset = arg[1] #'algebra_2005_2006'
    training_data, testing_data, testing_result_data = load_data(dataset)

    avg_student, avg_student_hierarchy, avg_student_hierarchy_problem = training(training_data)

    predict_result = predict(avg_student, avg_student_hierarchy, avg_student_hierarchy_problem, testing_data)
    predict_error =  rmse(predict_result, [float(i[13]) for i in testing_result_data[1:]])

    predict_result = predict(avg_student, avg_student_hierarchy, avg_student_hierarchy_problem, training_data)
    training_error = rmse(predict_result, [float(i[13]) for i in training_data[1:]])

    print '|', dataset, '|', training_error, '|', predict_error ,'|'
    return

if __name__ == "__main__":
    main(sys.argv)
