import copy
import time

# from binary_tree import create_tree, classify
from multi_branch_tree import create_tree, classify
from metric import output_fusion_matrix
from metric import output_classification_report as output_cls_report
from plottree import createPlot

# from process_data import read_iris_data as read_data
# from process_data import read_wine_data as read_data
from process_data import read_bank_data as read_data
# from process_data import read_car_data as read_data
# from process_data import read_abalone_data as read_data

# from process_data import read_diagnosis_data as read_data
# from process_data import read_balance_scale_data as read_data
# from process_data import read_breast_cancer_data as read_data
# from process_data import read_chees_data as read_data

# from process_data import read_adult_data as read_data

def other_transform(dataset):
    new_dataset = []
    for data in dataset:
        new_data = []
        new_data.append(data[0] * data[1])
        new_data.append(data[2] * data[3])
        new_data.append(data[-1])
        new_dataset.append(new_data)
    return new_dataset

if __name__ == "__main__":
    train_dataset, test_dataset, attributes = read_data()
    # train_dataset = other_transform(train_dataset)
    # test_dataset = other_transform(test_dataset)
    print(train_dataset[0])
    attributes_tmp = copy.deepcopy(attributes)
    train_dataset_tmp = copy.deepcopy(train_dataset)

    create_tree_start_time = time.process_time()

    decision_tree = create_tree(train_dataset_tmp, attributes_tmp, criterion="entrophy")

    create_tree_end_time = time.process_time()
    print("构建决策树的耗时为： ", (create_tree_end_time - create_tree_start_time), "s")
    
    # test train dataset
    train_res, train_gt = classify(decision_tree, attributes, train_dataset)
    output_fusion_matrix(train_res, train_gt)
    output_cls_report(train_res, train_gt)

    # using test dataset validation
    classify_start_time = time.time()
    test_res, test_gt = classify(decision_tree, attributes, test_dataset)
    classify_end_time = time.time()

    print("对决策树进行验证的耗时为: ", (classify_end_time - classify_end_time), "s")
    print("对决策树进行验证的平均耗时为: ", float(classify_end_time - classify_end_time) / len(test_dataset), "s")
    output_fusion_matrix(test_res, test_gt)
    output_cls_report(preict=test_res, ground_truth=test_gt)
    # createPlot(decision_tree)