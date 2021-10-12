import random

from pandas.core.base import DataError
from scipy.sparse import data

def read_csv(path, splitflag=','):
    with open(path) as f:
        data = f.readlines()
        attributes = data[0].strip('\n').split(splitflag)  # 去除换行符
        dataset = []
        for object in data[1:]:
            object = object.strip('\n')
            object = object.split(splitflag)
            dataset.append(object)
    return dataset, attributes

def transfrom_str_to_float(dataset, setIdx):
    '''将读取的str类型的数据转化为float类型
    '''
    for data in dataset:
        assert len(data) == len(dataset[0]), "数据集各个对象长度应一致"
        for idx in setIdx:
            data[idx] = float(data[idx])
    return dataset

def using_idx_split_dataset(dataset, idx_list):
    '''传入idx_list对数据集进行划分
    '''
    new_dataset = []
    for data in dataset:
        tmp = []
        for idx in idx_list:
            tmp.append(data[idx])
        new_dataset.append(tmp)
    return new_dataset

def combine_dataset(left_dataset, right_dataset):
    '''将两个数据集进行合并
    '''
    assert len(left_dataset) == len(right_dataset), "左右两边的数据集长度应相等"
    for idx in range(len(left_dataset)):
        left_dataset[idx].extend(right_dataset[idx])
    return left_dataset

def split_train_test(dataset, rate=0.7):
    '''
    按照一定比比列划分数据集，生成方式为随机生成
    args:
        dataset: 为list类型的数据，是一个二维的list。每一行表示一个对象，每一列表示一个属性，其中最后一列为类别
        rate: 训练集在整个数据集所占的比列
    return:
        train_dataset, test_dataset 训练数据集和测试数据集
    '''
    # 将数据集随机打乱
    random.seed(100)
    random.shuffle(dataset) # 注意这个函数的返回为None
    random.shuffle(dataset)
    length = int(rate * len(dataset)) + 1
    return dataset[:length], dataset[length:]

# 这个也是好的
def read_iris_data(path):
    dataset, attributes = read_csv(path)
    for data in dataset:
        for idx in range(len(data) - 1):
            data[idx] = float(data[idx])
    train_dataset, test_dataset = split_train_test(dataset)
    return train_dataset, test_dataset, attributes

# 这个是好的
def read_wine_data(path="./data/wine/wine.csv"):
    dataset, attributes = read_csv(path)
    # 该数据集中，所有的属性值都是float类型。
    # 首先将label分离出来
    labels = using_idx_split_dataset(dataset, [0])
    dataset = transfrom_str_to_float(dataset, range(1, len(dataset[0])))
    dataset = combine_dataset(dataset, labels)
    train_dataset, test_dataset = split_train_test(dataset)
    return train_dataset, test_dataset, attributes

def read_wine_quality_data(path="./data/winequality/winequality-{}.csv", type="white"):
    path = path.format(type)
    dataset, attributes = read_csv(path,splitflag=';')
    labels = using_idx_split_dataset(dataset, [0])
    dataset = using_idx_split_dataset(dataset, range(1, len(dataset[0])))
    dataset = transfrom_str_to_float(dataset)
    combine_dataset(dataset, labels)
    train_dataset, test_dataset = split_train_test(dataset)
    return train_dataset, test_dataset, attributes


def read_bank_data(path="./data/bank/bank.csv"):
    # 是连续的的属性: 1, 6, 10, 12, 13, 14, 15
    # 最后一列是类别
    continueSet = [0, 5, 9, 11, 12, 13, 14]
    dataset, attributes = read_csv(path, splitflag=';')
    dataset = transfrom_str_to_float(dataset, continueSet)
    train_dataset, test_dataset = split_train_test(dataset)
    return train_dataset, test_dataset, attributes

