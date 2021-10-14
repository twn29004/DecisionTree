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

def trasfrom_dis_to_int(dataset, idx):
    setValue = {data[idx] for data in dataset}
    listValue = list(setValue)
    mapValue = {}
    cnt = 0
    for value in listValue:
        mapValue[value] = cnt
        cnt += 1
    
    for data in dataset:
        data[idx] = mapValue[data[idx]]
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

def split_data_label(dataset, idx=-1):
    '''将标签数据从数据集中分离出来
    args:
        dataset: list形式的数据集
        idx: 标签所在的下标0或-1
    '''
    datas = []
    labels = []
    if idx == -1:
        for data in dataset:
            datas.append(data[:-1])
            labels.append(data[-1])
    else:
        for data in dataset:
            datas.append(data[1:])
            labels.append(data[0])
    return datas, labels

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
def read_iris_data(path="./data/iris/iris.csv"):
    dataset, attributes = read_csv(path)
    for data in dataset:
        for idx in range(len(data) - 1):
            data[idx] = float(data[idx])
    train_dataset, test_dataset = split_train_test(dataset)
    return train_dataset, test_dataset, attributes

def read_adult_data(path="./data/adult/adult.csv"):
    dataset, attributes = read_csv(path)
    continueList = [0, 2, 4, 10, 11, 12]
    dataset = transfrom_str_to_float(dataset, continueList)
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

def read_car_data(path="./data/car/car.csv"):
    # 这个数据集没有连续值的属性
    dataset, attributes = read_csv(path)
    train_dataset, test_dataset = split_train_test(dataset)
    return train_dataset, test_dataset, attributes

def read_abalone_data(path="./data/abalone/abalone.csv"):
    dataset, attributes = read_csv(path)
    continue_idx_list = [1,2,3,4,5,6,7, 8]
    dataset = transfrom_str_to_float(dataset, continue_idx_list)
    train_dataset, test_dataset = split_train_test(dataset)
    return train_dataset, test_dataset, attributes

def read_diagnosis_data(path='./data/diagnosis/diagnosis.txt'):
    dataset, attributes = read_csv(path, splitflag='\t')

    result_map = {}
    cnt = 0
    
    setValue = {object[-1] for object in dataset}
    for a in setValue:
        for b in setValue:
            result_map[tuple([a, b])] = cnt
            cnt += 1
    # 对最后两列进行转化
    new_dataset = []
    for data in dataset:
        tmp = result_map[tuple([data[-2], data[-1]])]
        data = data[:-2]
        data.append(tmp)
        new_dataset.append(data)
    dataset = new_dataset
    # 同时更新attributes
    attributes = attributes[:-2]
    attributes.append("Class")
    train_dataset, test_dataset = split_train_test(dataset)
    
    attributes = ["Temperature", "Occurrence", "Lumbar", "Urine", "Micturition", "Burning", "Class"]
    assert len(attributes) == len(dataset[0]), "属性的长度和数据集的长度不一致"
    return train_dataset, test_dataset, attributes

def read_balance_scale_data(path="./data/balance-scale/balance-scale.csv"):
    dataset,attributes = read_csv(path)
    new_dataset = []
    for data in dataset:
        tmp = data[1:]
        tmp.append(data[0])
        new_dataset.append(tmp)
    dataset = new_dataset
    dataset = transfrom_str_to_float(dataset, range(len(dataset[0]) - 1))
    train_dataset, test_dataset = split_train_test(dataset)
    return train_dataset, test_dataset, attributes


def read_breast_cancer_data(path="./data/breast-cancer/breast-cancer.csv"):
    dataset, attributes = read_csv(path)
    dataset = transfrom_str_to_float(dataset, range(len(dataset[0]) - 1))
    train_dataset, test_dataset = split_train_test(dataset)
    return train_dataset, test_dataset, attributes


def read_chees_data(path="./data/chess/krkopt.csv"):
    dataset, attributes = read_csv(path)
    train_dataset, test_dataset = split_train_test(dataset)
    return train_dataset, test_dataset, attributes

if __name__ == "__main__":
    read_diagnosis_data()
