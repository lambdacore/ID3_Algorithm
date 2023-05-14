# Jonathan Harrington CS 460 A2
# 10/17/2022

import pandas as pd
import numpy as np
import os
import pickle as p
from graphviz import Digraph

# This class contains the ID3 alg
class ID3:

    # Uses the ID3 algorithm to generates the decision tree
    @staticmethod
    def generate_tree (dataFrame, attribute_dict, use_pruning) -> {}:
        if len(dataFrame) == 0:
            return {}
        list_of_features = list(dataFrame.columns.values [:-1])
        ID3.__instance_threshold = len(dataFrame) // 100
        return ID3.generate_sub_tree(dataFrame, list_of_features, attribute_dict, use_pruning)

    # Helps generates the tree by recursively creating dictionaries
    @staticmethod
    def generate_sub_tree (dataFrame, list_of_features, attribute_dict, use_pruning) -> {}:
        class_label = dataFrame.columns [-1]
        # BC: No more attributes to go over.
        if not list_of_features:
            return dataFrame [class_label].value_counts().idxmax()
        # BC: Entropy == 0. All D in the DataFrame have the same c label
        if not ID3.calc_entropy_of_individual(dataFrame):
            return dataFrame [class_label].iloc [0]
        # BC: before we prun
        if use_pruning and len(dataFrame) < ID3.__instance_threshold:
            return dataFrame [class_label].value_counts().idxmax()
        # make the tree by find the nex best attribute to go over
        tree = {}
        attribute_tree = {}

        next_feature = ID3.find_highest_ig(dataFrame, list_of_features)
        updated_list_of_features = list_of_features.copy()
        updated_list_of_features.remove(next_feature)
        for feature_value in attribute_dict [next_feature]:
            partitioned_df = dataFrame [dataFrame [next_feature] == feature_value]
            # continue only if the DataFrame is not empty
            if len(partitioned_df) != 0:  # main case to exit for play tennis
                attribute_tree [feature_value] = ID3.generate_sub_tree(partitioned_df, updated_list_of_features,
                                                                       attribute_dict, use_pruning)
            # If is empty, return
            else:
                attribute_tree [feature_value] = dataFrame [class_label].value_counts().idxmax()
        tree [next_feature] = attribute_tree
        return tree

    # Get the feature that yields the highest gain
    @staticmethod
    def find_highest_ig (df, list_of_features: list) -> str:
        list_of_gains = {}
        for features in list_of_features:
            list_of_gains [features] = ID3.calc_information_gain(df, features)
        return max(list_of_gains, key = list_of_gains.get)

    # Calculates the information gain acquired from partitioning over attribute
    @staticmethod
    def calc_information_gain (dataFrame, feature) -> float:
        if len(dataFrame) == 0:
            return 0
        # Calculate entropy before partitioning
        entropy_before_split = ID3.calc_entropy_of_individual(dataFrame)
        # Calculate entropy after partitioning
        entropy_after_split = 0
        # value of the feature as in - feature would be age and value of it would be age3
        for value_of_the_feature in dataFrame [feature].unique():
            list_of_data_based_on_feature = dataFrame [feature] == value_of_the_feature
            splited_df = dataFrame [list_of_data_based_on_feature]
            entropy_after_split += len(splited_df) / len(dataFrame) * ID3.calc_entropy_of_individual(splited_df)
        # Information gain = entropy before decision - entropy after decision
        return entropy_before_split - entropy_after_split

    # Calculates the entropy of DataFrame
    @staticmethod
    def calc_entropy_of_individual (dataFrame) -> float:
        # Get the class ratios by dividing the number of class instances by the total number of instances
        class_ratios = []
        class_label = dataFrame.columns [-1]
        total_instances = len(dataFrame.index)
        for class_value in dataFrame [class_label].unique():
            mask = dataFrame [class_label] == class_value
            partitioned_df = dataFrame [mask]
            partitioned_instances = len(partitioned_df.index)
            class_ratios.append(partitioned_instances / total_instances)
        # Entropy equation
        return sum([-class_ratio * np.log2(class_ratio) for class_ratio in class_ratios])

    # split the data and calculate it
    @staticmethod
    def split_data (dataFrame, feature) -> float:
        attribute_ratios = []
        total_instances = len(dataFrame)
        for attribute_value in dataFrame [feature].unique():
            # rename mask to something more descriptive
            mask = dataFrame [feature] == attribute_value
            partitioned_instances = len(dataFrame [mask])
            attribute_ratios.append(partitioned_instances / total_instances)
        return sum([-attribute_ratio * np.log2(attribute_ratio) for attribute_ratio in attribute_ratios])

# this class holds everything to test our model
class ModelTester:

    # will test the accuracy of the model that we made with a given testing data set
    @staticmethod
    def test_accuracy (testing_data_df, passed_in_tree) -> None:
        # make sure the dataframe neither are empty
        if len(testing_data_df) == 0:
            raise ValueError("Sir your dataframe is empty")
        if not passed_in_tree:
            raise ValueError("Sir your tree is empty")
        num_of_correct_guesses = 0
        num_of_incorrect_guesses = 0
        values_not_evaluated = 0
        # now we evalute each datapoint
        for index, instance in testing_data_df.iterrows():
            classification = ModelTester.evaluate_each_data_point(instance, passed_in_tree)
            if classification == True:
                num_of_correct_guesses += 1
            elif classification == False:
                num_of_incorrect_guesses += 1
            else:
                values_not_evaluated += 1
        ModelTester.print_results(num_of_correct_guesses, num_of_incorrect_guesses, values_not_evaluated)

    # Returns indicators whether the classification is true, false, or unknown
    @staticmethod
    def evaluate_each_data_point (instance, decision_tree) -> bool:
        # Traverses down the decision tree until a leaf node has been hit
        while isinstance(decision_tree, dict):
            attribute = next(iter(decision_tree))
            instance_attribute_value = instance [attribute]
            """ If the instance is at an attribute node that does not offer the attribute value of that instance, 
            the instance has hit a dead end. Hence, it can not be classified """
            if instance_attribute_value not in decision_tree [attribute]:
                print("error")
                return "UNKNOWN" # used for debugging
            decision_tree = decision_tree [attribute][instance_attribute_value]
        # returns if the true i.e correct or false if inncorrect
        return True if decision_tree == instance[-1] else False

    # Prints accuracy report
    @staticmethod
    def print_results (num_of_correct_matches, num_of_incorrect_matches, unknowns) -> None:
        total_amount_of_data = num_of_correct_matches + num_of_incorrect_matches + unknowns
        evaluated_data = num_of_correct_matches + num_of_incorrect_matches
        if evaluated_data == 0:
            raise ValueError("None of the instances were classified using the model")
        model_accuracy = num_of_correct_matches / evaluated_data * 100
        print("==================== The Test Has Started")
        print("Number of testing examples =", total_amount_of_data)

        # these next two lines were for debugging
        # print("Number of testing examples found =", evaluated_data)
        # print("Number of testing examples not found =", unknowns)

        print("correct_classification_count =", num_of_correct_matches)
        print("incorrect_classification_count =", num_of_incorrect_matches)
        print("Accuracy =", model_accuracy, "%")
        print("===================== Test has Ended")


# uncomment it out the .view() in test_model to get the result in PDF for the tree
# will draw the tree with Digraph
def draw_decision_tree_dictionary (tree_dictionary):
    if not isinstance(tree_dictionary, dict):
        raise TypeError("Argument must be of type dictionary")
    if not tree_dictionary:
        raise ValueError("Dictionary tree_dictionary is empty")
    dot = Digraph(strict = True)
    draw_tree(dot, tree_dictionary, None)
    return dot


# helper function for draw decision tree
def draw_tree (dot, tree_dictionary, parent_node_name):
    if isinstance(tree_dictionary, dict):
        for key in tree_dictionary:
            no_space_key = str(key).replace(" ", "")
            dot.node(no_space_key, str(key), shape = "ellipse")
            if parent_node_name != None:
                dot.edge(parent_node_name, no_space_key)
            draw_tree(dot, tree_dictionary [key], no_space_key)
    else:
        val = str(tree_dictionary)
        dot.node(val, val, shape = "plaintext")
        dot.edge(parent_node_name, val)

# This runs the ID3 and tests it
def test_model (main_df, test_df, to_pruning):

    title = ["Testing model using Information Gain"]
    pruning = ["", "with Pruning"][to_pruning]
    print("====================", title, pruning, "==================== ")

    # makes a dictionary of our attributes or list of features ie columns
    attribute_dict = {attribute: main_df [attribute].unique() for attribute in main_df.columns.values [:-1]}

    # this will run the ID3 algorith and return a dictionary as the tree
    # this will take 2 min if running without pruning
    tree = ID3.generate_tree(main_df, attribute_dict, to_pruning)

    # this is the start of the pickle saving the dictionary comment it out if you want
    p.dump(tree, open(os.path.join(folder_name, file_name), 'wb'), protocol = 4)
    print("loading model...")
    saved_tree_model = p.load(open(os.path.join(folder_name, file_name), 'rb'))

    # draw_decision_tree_dictionary(saved_tree_model).view()
    ModelTester.test_accuracy(test_df, saved_tree_model)


files = "assets/census_training.csv", "assets/census_training_test.csv"
main_file, test_file = files
main_df = pd.read_csv(main_file)
test_df = pd.read_csv(test_file)

# this code below will creat the folder and set the name of the folder and file name for the dictionary
folder_name = "model"

# this is for non pruning dictionary
# file_name = "copy_of_dictionary"

# this is for pruning dictionary
file_name = "pruning_copy_of_dictionary"

# we then create the folder if it does not exisist
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# this is the main function that will run the program
# if you want pruning set to_pruning=True or False if you don't want it
test_model(main_df, test_df, to_pruning = True)

