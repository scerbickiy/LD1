import os
import copy
import random
import tabulate

#  1: apple
# -1: pear

learning_rate = 0.01
threshold = 0
file_path = os.path.join(os.getcwd(), 'IS-Lab1-master', 'Data.txt')
fruit_features = ["color", "roundness"]

fruits_feature_data = {
    "apple": [],
    "pear": []
}

result_table = []
headers = [
    "Test Parameters",
    "Expected Output",
    "Perceptron Output",
    "Test Result"
]
test_results = {
    "test_param": [],
    "expected_result": [],
    "perceptron_output": [],
    "test_result": []
}

# setting random weight values 
weights = {
    "color": random.random(),
    "roundness": random.random(),
    "bias": random.random()
}

def sort_input_data(file_path):

    training_data = copy.deepcopy(fruits_feature_data)
    testing_data = copy.deepcopy(fruits_feature_data)

    with open(file_path, 'r') as file:
        data = file.readlines()
        for line in data:
            fruit_feature_values = {}
            fruit_id = int(line.split(',')[-1])
            fruit_name = "apple" if fruit_id == 1 else "pear"
            for i, feature in enumerate(line.split(',')[:-1]):
                fruit_feature_values[fruit_features[i]] = float(feature)
            fruits_feature_data[fruit_name].append(fruit_feature_values)

    for fruit in fruits_feature_data:
        training_data[fruit] = fruits_feature_data[fruit][:int(len(fruits_feature_data[fruit])*0.8)]
        testing_data[fruit] = fruits_feature_data[fruit][int(len(fruits_feature_data[fruit])*0.8):]

    return training_data, testing_data

def perceptron(object_features_data: dict):

    wheighted_sum = weights["bias"]
    for feature, feature_value in object_features_data.items():
        wheighted_sum += feature_value * weights[feature]
    return 1 if wheighted_sum > threshold else -1


def main():

    train_data, test_data = sort_input_data(file_path)

    print("Weight values before training:")
    for weight_type, weight_value in weights.items():
        print(f"{weight_type}: {weight_value}")
    

    #region Perceptron training part
    for fruit in train_data:
        for fruit_features in train_data[fruit]:
            perceptron_output = perceptron(fruit_features)
            desired_output = 1 if fruit == "apple" else -1
            error = desired_output - perceptron_output 
            
            if error != 0: 
                for weight_type, weight_value in weights.items():
                    if weight_type != "bias":
                        weights[weight_type] = weight_value + learning_rate * error * fruit_features[weight_type]
                    else:
                        weights[weight_type] = weight_value + learning_rate * error
    
    #endregion

    print("\n\nWeight values after training:")
    for weight_type, weight_value in weights.items():
        print(f"{weight_type}: {weight_value}")

    #region Perceptron testing part
    for fruit in test_data:
        for fruit_features in test_data[fruit]:
            perceptron_output = perceptron(fruit_features)
            desired_output = (1,"apple") if fruit == "apple" else (-1,"pear")
            if perceptron_output == desired_output[0]:
                test_result = "PASSED"
            else:
                test_result = "FAILED"
            perceptron_output = "apple" if perceptron_output == 1 else "pear"
            test_results["test_param"].append(fruit_features)
            test_results["expected_result"].append(desired_output[1])
            test_results["perceptron_output"].append(perceptron_output)
            test_results["test_result"].append(test_result)
    #endregion
    
    for i in range(len(test_results["test_param"])):
        result_table.append(
            [
                test_results["test_param"][i],
                test_results["expected_result"][i],
                test_results["perceptron_output"][i],
                test_results["test_result"][i]
            ]
        )
    
    print("\n\nPerceptron test results:")
    print(tabulate.tabulate(result_table, headers=headers, tablefmt='grid'))

main()








