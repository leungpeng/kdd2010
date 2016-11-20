import sys
import random
from project import load_data, show_data, rmse

def training(training_data):
    #show_data(training_data)
    student_result = {}
    overall_result = {}
    for i in range(1,len(training_data)):
        studentId = training_data[i][1]
        problem_hierarchy = training_data[i][2]
        problem_name = training_data[i][3]
        step_name = training_data[i][5]

        is_first_correct = training_data[i][13]

        student = student_result.setdefault(studentId, {})
        problem_result = student.setdefault(problem_hierarchy,{}).setdefault(problem_name,{}).setdefault(step_name, {})
        current_result = problem_result.setdefault(is_first_correct, 0)
        problem_result[is_first_correct] = current_result + 1

        problem_result = overall_result.setdefault(problem_hierarchy,{}).setdefault(problem_name,{}).setdefault(step_name, {})
        current_result = problem_result.setdefault(is_first_correct, 0)
        problem_result[is_first_correct] = current_result + 1
    return student_result, overall_result

def get_result_by_stepname(student, problem_hierarchy, problem_name, step_name):
    if problem_hierarchy in student:
        problems = student[problem_hierarchy]
        if problem_name in problems:
            steps = problems[problem_name]
            if step_name in steps:
                correct = steps[step_name].setdefault('1',0.0)
                incorrect = steps[step_name].setdefault('0',0.0)
            else:
                correct = sum([ steps[name].setdefault('1',0.0) for name in steps])
                incorrect = sum([ steps[name].setdefault('0',0.0) for name in steps])
        else:
            correct = sum([ sum([ problems[name][step].setdefault('1',0.0) for step in problems[name]]) for name in problems])
            incorrect = sum([ sum([ problems[name][step].setdefault('0',0.0) for step in problems[name]]) for name in problems])
    else:
        raise KeyError(problem_hierarchy)
    return correct, incorrect

def predict(student_result, overall_result, testing_data):
    predict_result = []
    for i in range(1,len(testing_data)):

        studentId = testing_data[i][1]
        problem_hierarchy = testing_data[i][2]
        problem_name = testing_data[i][3]
        step_name = testing_data[i][5]

        try:
            student = student_result[studentId]
            correct, incorrect = get_result_by_stepname(student, problem_hierarchy, problem_name, step_name)
        except KeyError:
            correct, incorrect = get_result_by_stepname(overall_result, problem_hierarchy, problem_name, step_name)
        # print studentId, problem_hierarchy, problem_name, step_name
        predict = float(correct) / (correct + incorrect)

        predict_result.append(predict)
    return predict_result

def main(arg):
    dataset = arg[1] #'algebra_2005_2006'
    training_data, testing_data, testing_result_data = load_data(dataset)

    student_result, overall_result = training(training_data)

    predict_result = predict(student_result, overall_result, testing_data)
    predict_error =  rmse(predict_result, [float(i[13]) for i in testing_result_data[1:]])

    predict_result = predict(student_result, overall_result, training_data)
    training_error = rmse(predict_result, [float(i[13]) for i in training_data[1:]])

    print '|', dataset, '|', training_error, '|', predict_error ,'|'
    return

if __name__ == "__main__":
    main(sys.argv)
