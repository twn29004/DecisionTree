# coding: utf-8
'''
Author: twn
Date: 2021/10/10
Describe:
使用三种不纯度度量构建决策树
    (1) 熵
    (2) gini指数
    (3) 分类错误率
使用四种评价指标比较决策树的分类性能：
    (1) 
'''

from math import log
import sys
import copy
import random

from metric import cal_acc

random.seed(10)

def cal_entropy(dataset):
    '''计算数据集的熵
    args:
        dataset: 为list类型的数据，是一个二维的list。每一行表示一个对象，每一列表示一个属性，其中最后一列为类别
    return:
        res: 该数据的熵
    
    '''
    if 0 == len(dataset):
        return 0
    
    assert type(dataset).__name__ == "list", "dataset的类型因为list类型"
    assert type(dataset[0]).__name__ == "list", "dataset中的元素的类型应为list类型"

    length_dataset = len(dataset)
    # 用于存储同一属性下不同值的数目
    labelCounts = {}
    for object in dataset:
        # 该对象对应的类别存储在数据集的最后一列
        label = object[-1]
        if label not in labelCounts.keys():
            labelCounts[label] = 1
        else:
            labelCounts[label] += 1

    res = 0.0
    for value in labelCounts.values():
        rate = float(value) / length_dataset
        res -= rate * log(rate, 2)
    return res

def cal_gini(dataset):
    '''计算数据集的gini指数
    args:
        dataset: 为list类型的数据，是一个二维的list。每一行表示一个对象，每一列表示一个属性，其中最后一列为类别
    return:
        gini：该数据集的gini指数
    '''
    if 0 == len(dataset):
        return 0
    
    assert type(dataset).__name__ == "list", "dataset的类型因为list类型"
    assert type(dataset[0]).__name__ == "list", "dataset中的元素的类型应为list类型"

    labelCount = {}
    for object in dataset:
        if type(object).__name__ == 'float':
            print(object)
        label = object[-1]
        if label not in labelCount.keys():
            labelCount[label] = 1
        else:
            labelCount[label] += 1
    
    ret = 0.0
    length_dataset = len(dataset)
    for count in labelCount.values():
        ret += pow(float(count) / length_dataset, 2)
    return 1 - ret

def cal_classification_error(dataset):
    '''计算分类错误率
    args:
        dataset: 为list类型的数据，是一个二维的list。每一行表示一个对象，每一列表示一个属性，其中最后一列为类别
    return:
        error_rate: 分类错误率
    '''
    # 如果是一个空的dataset
    if 0 == len(dataset):
        return 0
    
    assert type(dataset).__name__ == "list", "dataset的类型因为list类型"
    assert type(dataset[0]).__name__ == "list", "dataset中的元素的类型应为list类型"

    labelCount = {}
    for object in dataset:
        label = object[-1]
        if label not in labelCount.keys():
            labelCount[label] = 1
        else:
            labelCount[label] += 1

    cnt = max(list(labelCount.values()))
    return 1 - float(cnt) / len(dataset)


def split_discrete_dataset(dataset, idx, value):
    '''根据离散属性的值划分数据集，这里的划分都是二分的
    args:
        dataset: 为list类型的数据，是一个二维的list。每一行表示一个对象，每一列表示一个属性，其中最后一列为类别
        idx: 划分属性的下标,注意这里的每次划分，即使相同的下标也对应不同的属性，因为属性的list是变化的
        value: 划分属性的值
    return:
        true_dataset, false_dataset: 选择出符合条件的数据集和不符合的数据集,结构与原始数据集一致
    '''
    assert len(dataset) != 0, "数据集的长度为0"
    true_dataset = []
    false_dataset = []
    for object in dataset:
        if object[idx] == value:
            true_object = object[:idx]
            true_object.extend(object[idx + 1: ])
            true_dataset.append(true_object)
        else:
            false_object = object[:idx]
            false_object.extend(object[idx + 1 : ])
            false_dataset.append(false_object)
    return true_dataset, false_dataset

def split_continuous_dataset(dataset, idx, value):
    '''根据连续属性的值划分数据集，这里的划分也是二分的
    args:
        dataset: 为list类型的数据，是一个二维的list。每一行表示一个对象，每一列表示一个属性，其中最后一列为类别
        idx: 划分属性的下标,注意这里的每次划分，即使相同的下标也对应不同的属性，因为属性的list是变化的
        value: 划分属性的值
    return:
        less_dataset, more_dataset 小于value的数据集和大于value的数据集
    '''
    assert len(dataset) != 0, "the length of dataset is zero"
    less_dataset = []
    more_dataset = []
    for object in dataset:
        if object[idx] <= value:
            less_object = object[:idx]
            less_object.extend(object[idx + 1 :])
            less_dataset.append(less_object)
        else:
            more_object = object[:idx]
            more_object.extend(object[idx + 1: ])
            more_dataset.append(more_object)
    return less_dataset, more_dataset

# 使用熵这一不纯度度量选择最优的划分属性
def using_entropy_choose_best_attribute(dataset):
    '''使用熵这一不纯度度量来作为指标，选择最后的属性进行划分
    args:
        dataset: 为list类型的数据，是一个二维的list。每一行表示一个对象，每一列表示一个属性，其中最后一列为类别
    return:
        best_attribute_index: 最优划分属性的下标
        best_attribute_value: 最优划分呢属性的最优划分的值
    '''
    # 属性的数目
    length_attribute = len(dataset[0]) - 1
    # 这个表示的最优属性的下标
    best_attribute_idx = -1
    best_attribute_value = None
    # 计算未划分前的熵
    base_entropy = cal_entropy(dataset)
    best_info_gain = 0.0
    # 逐个遍历属性,找出最优的属性
    for idx in range(length_attribute):
        # 该属性对应的值的集合,python中集合中的树自动排序
        setValue = {object[idx] for object in dataset}
        for value in setValue:
            # 根据value的类型决定使用连续值的划分还是使用离散值的划分
            # 如果类型是小数或者整数，则使用连续性划分
            if type(value).__name__ == "float" or type(value).__name__ == "int":
                less_dataset, more_dataset = split_continuous_dataset(dataset, idx, value)
                tmp_entropy = float(len(less_dataset)) / len(dataset) * cal_entropy(less_dataset) \
                                + float(len(more_dataset)) / len(dataset) * cal_entropy(more_dataset)
            # 使用离散型划分
            else:
                true_dataset, false_dataset = split_discrete_dataset(dataset, idx, value)
                tmp_entropy = float(len(true_dataset)) / len(dataset) * cal_entropy(true_dataset) \
                                + float(len(false_dataset)) / len(dataset) * cal_entropy(false_dataset)
            
            # 计算熵的增益
            tmp_info_gain = base_entropy - tmp_entropy
            # 更新最优解
            if(tmp_info_gain >= best_info_gain):
                best_info_gain = tmp_info_gain
                best_attribute_idx = idx
                best_attribute_value = value
    print("本次划分的信息增益为：", best_info_gain)
    return best_attribute_idx, best_attribute_value


def using_gini_choose_best_attribute(dataset):
    '''使用基尼指数这一不纯度度量来作为指标，选择最后的属性进行划分
    args:
        dataset: 为list类型的数据，是一个二维的list。每一行表示一个对象，每一列表示一个属性，其中最后一列为类别
    return:
        best_attribute_index: 最优划分属性的下标
        best_attribute_value: 最优划分呢属性的最优划分的值
    '''
    # -1的目的是去掉label列
    length_attribute = len(dataset[0]) - 1
    best_gini = float(sys.maxsize)
    best_attribute_idx = -1
    best_attribute_value = None

    for idx in range(length_attribute):
        setValue = {object[idx] for object in dataset}

        for value in setValue:
            tmp_gini = 0.0
            if type(value).__name__ == "float" or type(value).__name__ == "int":
                less_dataset, more_dataset = split_continuous_dataset(dataset, idx, value)
                tmp_gini = float(len(less_dataset)) / len(dataset) * cal_gini(less_dataset) \
                            + float(len(more_dataset)) / len(dataset) * cal_gini(more_dataset)
            else:
                true_dataset, false_dataset = split_discrete_dataset(dataset, idx, value)
                tmp_gini = float(len(true_dataset)) / len(dataset) * cal_gini(true_dataset) \
                            + float(len(false_dataset)) / len(dataset) * cal_gini(false_dataset)
            if tmp_gini < best_gini:
                best_gini = tmp_gini
                best_attribute_idx = idx
                best_attribute_value = value
    print("本次划分最小的gini指数为: ", best_gini)
    return best_attribute_idx, best_attribute_value                       


def using_classification_error_choose_best_attribute(dataset):
    '''使用基尼指数这一不纯度度量来作为指标，选择最后的属性进行划分
    args:
        dataset: 为list类型的数据，是一个二维的list。每一行表示一个对象，每一列表示一个属性，其中最后一列为类别
    return:
        best_attribute_index: 最优划分属性的下标
        best_attribute_value: 最优划分呢属性的最优划分的值
    '''
    length_attribute = len(dataset[0]) - 1
    best_classification_error = 1
    best_attribute_idx = -1
    best_attribute_value = None

    for idx in range(length_attribute):
        setValue = {object[idx] for object in dataset}

        for value in setValue:
            tmp_cls_error = 1
            if type(value).__name__ == 'float' or type(value).__name__ == "int":
                less_dataset, more_dataset = split_continuous_dataset(dataset, idx, value)
                tmp_cls_error = float(len(less_dataset)) / len(dataset) * cal_classification_error(less_dataset) \
                                    + float(len(more_dataset)) / len(dataset) * cal_classification_error(more_dataset)
            else:
                true_dataset, false_dataset = split_discrete_dataset(dataset, idx, value)
                tmp_cls_error = float(len(true_dataset)) / len(dataset) * cal_classification_error(true_dataset) \
                                    + float(len(false_dataset)) / len(dataset) * cal_classification_error(false_dataset)
            
            # 更新最佳选择
            if tmp_cls_error < best_classification_error:
                best_classification_error = tmp_cls_error
                best_attribute_idx = idx
                best_attribute_value = value
    print("本次划分最小的分类错误率为: ", best_classification_error)
    return best_attribute_idx, best_attribute_value

def majorCnt(classList):
    '''数据集已经处理了所有的属性，但是类标签不唯一
        使用该叶节点中较多的类别作为该叶子节点的类别
        args:
            classList: dataset中的最后一列
        return:
            res: 给节点中数目最多的类别
    '''
    classCont = {}
    for label in classList:
        if label not in classCont.keys():
            classCont[label] = 1
        else:
            classCont[label] += 1
    
    # 寻找个数最多的类别
    res = None
    cnt = -1
    for label in classCont.keys():
        if classCont[label] > cnt:
            cnt = classCont[label]
            res = label
    return res


def create_tree(dataset, attributes, criterion="gini"):
    '''根据数据集创建决策树
    args:
        datset: 为list类型的数据，是一个二维的list。每一行表示一个对象，每一列表示一个属性，其中最后一列为类别
        attributes: 属性对应的字符串,每次划分都会从中删除使用的属性
        attributes_full: 所有的属性
        test_dataset: 用于剪枝的测试数据集
        criterion: 选择属性的不纯度度量方法. 默认为"gini",可以设置为"entrophy","cls error"
        pre_pruning: 预剪枝，防止过拟合
    returns:
        decision_tree: 一个数据类型为字典的决策树
    '''
    label_list = [object[-1] for object in dataset]
    # 一个中止条件：该节点都是一个类的时候，返回该类别
    if label_list.count(label_list[0]) == len(label_list):
        # 这里如果出现空的情况可能有点问题，在思考一下
        return label_list[0]
    # 另一个中止条件：所有的属性都分完了，只能结束了的情况
    # 这个时候选择叶子节点中数目较多的类作为该节点的类
    if len(dataset[-1]) == 1:
        return majorCnt(label_list)

    # 根据criterion选择合适的算法选择最优的属性和值进行划分
    choose_best_attribute = None
    if criterion == "gini":
        choose_best_attribute = using_gini_choose_best_attribute
    elif criterion == "entrophy":
        choose_best_attribute = using_entropy_choose_best_attribute
    elif criterion == "cls error":
        choose_best_attribute = using_classification_error_choose_best_attribute
    else:
        assert False, "ciriter error, please input valid args"
    
    best_attribute_idx, best_attribute_value = choose_best_attribute(dataset)
    best_attribute = attributes[best_attribute_idx]

    print("最优属性的下标为: ", best_attribute_idx)
    print("最优属性为： ", best_attribute)
    print("使用该属性最优的分裂值为: ", best_attribute_value)

    # 从属性列表中将该属性移除
    del attributes[best_attribute_idx]

    decision_tree = {best_attribute : {}}
    # 如果是连续值的话
    if type(best_attribute_value).__name__ == "float" or type(best_attribute_value).__name__ == "int":
        left_dataset, right_dataset = split_continuous_dataset(dataset, best_attribute_idx, best_attribute_value)
        left_value = tuple([best_attribute_value, "less"])
        right_value = tuple([best_attribute_value, "more"])
    # 如果是离散值的话
    else:
        left_dataset, right_dataset = split_discrete_dataset(dataset, best_attribute_idx, best_attribute_value)
        left_value = tuple([best_attribute_value, True])
        right_value = tuple([best_attribute_value, False])

    # 这里处理的情况是当进行划分时，某一个子树上划分得到的实例[对象]数为0，此时我们将这个子树的类别设置为数据集中出现频率最高的类别
    left_attributes = copy.deepcopy(attributes)
    if 0 != len(left_dataset):
        decision_tree[best_attribute][left_value] = create_tree(left_dataset, left_attributes, criterion)
    else:
        decision_tree[best_attribute][left_value] = majorCnt(label_list)
    
    right_attributes = copy.deepcopy(attributes)
    if 0 != len(right_dataset):
        decision_tree[best_attribute][right_value] = create_tree(right_dataset, right_attributes, criterion)
    else:
        decision_tree[best_attribute][right_value] = majorCnt(label_list)
    return decision_tree

def classify_single(decision_tree, attributes, object):
    '''使用决策树计算对象的类别
    agrs:
        decision_tree: 以dict形式存储的决策树
        attributes: 属性列表
        object: 对象的各个属性的值
    '''
    attribute = list(decision_tree.keys())[0]
    attribute_idx = attributes.index(attribute)

    # 得到子树
    subTree = decision_tree[attribute]

    value, _ = list(subTree.keys())[0]
    # 判断value的类型，连续值还是离散值
    if type(value).__name__ == "float" or type(value).__name__ == "int":
        # 连续值
        if object[attribute_idx] < value:
            flag = "less"
        else:
            flag = "more"
    else:
        if object[attribute_idx] == value:
            flag = True
        else:
            flag = False
    
    subTree_key = tuple([value, flag])
    # 子树的key值一定在子树的key值列表中
    assert subTree_key in subTree.keys(), "File: tree.py \nFunction: classify_single()\nsubTree key error"
    
    # 如果不是叶子节点的话，继续搜索
    if type(subTree[subTree_key]).__name__ == "dict":
        return classify_single(subTree[subTree_key], attributes, object)
    else:
        return subTree[subTree_key]

def classify(decision_tree, attributes, test_dataset):
    '''根据生成的决策树和测试数据集生成分类结果
    args:
        decision_tree: 以字典形式存储的决策树
        attributes: 属性列表
        test_dataset: 测试数据集
    returns:
        res: 每个对象对应的预测的类别
        ground_truth: 每个对象对应的基准值
    '''
    res = []
    ground_truth = [object[-1] for object in test_dataset]

    for test_object in test_dataset:
        res.append(classify_single(decision_tree, attributes, test_object))    
    return res, ground_truth








    
