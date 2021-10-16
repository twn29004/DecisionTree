## 核心代码展示及解释

### 概述

本次实验主要采用python来实现决策树的构建。分别采用三种不纯度度量来进行决策树构建过程中选择属性的标准。除此之外，对于数据集中不同类型的属性，采取不同的处理方式。本次实验中实现了两个版本的决策树。

一种是所有的节点都是两个分支。每次划分利用不纯度度量选择最优的属性以及对应的值。对于离散的属性，实验中通过选择最优划分的值，将等于该值的属性划分为一个分支，不等于该值的属性划分为一个分支。对于连续值，同样选择最优划分的值，小于等于该值的作为一个分支，大于该值的作为一个分支。

另一种是与上一种的区别在于对离散值属性的改进。将二叉树变化为多叉树。如果最优的划分属性为离散属性时，实验中将其含有的所有的值都作为下一个分支。

本次实验所用的决策树没有引入剪枝操作。训练集和测试集按照7:3的比例划分。

本实验完整代码已提交至github仓库。[twn29004/DecisionTree (github.com)](https://github.com/twn29004/DecisionTree)

### 实验中一些特殊情况的处理

1. 划分过程中，某一个子树中含有的对象数目为0.这个时候选择父亲节点中数目最多的类别作为该子树的类别。
2. 在多叉树中，对于离散型属性，如果测试集中出现了训练集中没有的值，即测试数据不知道该忘那个子树继续搜索。本次实验中采取的处理方法为随机的选择一个子树继续判断。

### 对于数据集的划分操作

```python
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
```

上述代码为对数据集的划分操作。其中**split_discrete_dataset()**函数为对于离散值的属性进行划分。本次实验中实现了两个版本，一个是只返回对应属性对应值划分出的数据集，另一个属性是返回等于对应属性对应值的数据集和不等于对应值的数据集。**split_continuous_dataset()**函数是对于连续值属性的划分，其中**less_dataset**为数据集中对应属性小于对应值的数据集，**more_dataset**为大于的部分。

### 熵作为不纯度度量

#### 计算数据集的熵

```python
def cal_entropy(dataset):
    '''计算数据集的熵
    args:
        dataset: 为list类型的数据，是一个二维的list。每一行表示一个对象，每一列表示一个属性，其中最后一列为类别
    return:
        res: 数据集的熵
    
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
```

上述代码为计算数据集熵的代码，其通过统计各个类别出现的频率来计算该数据集对应的熵。其计算公式如下:

​									$$Entrophy(t)=-\sum_{j}p(j|t)*log(p(j|t))$$

其中$p(j|t)$即为代码中的$rate$变量。此外，对于长度为0的数据集，其熵为0

#### 使用熵来选择最优划分的属性

```python
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
```

上述函数为使用熵这一不纯度度量来作为数据集划分的标准。实验中通过属性的值的类型来判断该属性是连续性属性还是离散型属性。如果值的类型为*int*，*float*则说明该属性属于连续性属性，否则使用离散型属性进行划分。

对于连续性属性，我们不仅需要找到一个最优的划分属性，还需要找到一个最优划分的值。因此，我们需要对于一个属性的各个值都需要计算其信息增益。如果当前的划分的信息增益大于之前的划分，则更新对应的变量。

对于离散型的属性，不同的树有不同的构建方式。对于二叉树类型的决策树，其构建方式与连续值基本一致，这里不在赘述。对于多叉树类型的决策树。离散型的属性不再需要找到一个最优的划分的值。只需要计算对于该属性每个值进行划分带来的信息增益，然后进行对比。

### 基尼指数作为不纯度度量

#### 计算数据集的基尼指数

```python
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
```

上述代码为计算数据集的基尼指数，其方法是通过计算每个类别出现的频率，然后对各个类别出现的频率的平方求和，再使用1减去频率平方的和。其计算公式如下:

​														$$gini(t)=1-\sum_j(p(j|t)^2)$$

#### 使用基尼指数来选择最优划分的属性

```python
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
```

上述代码为使用基尼指数这一标准来进行划分。其基本方法与使用熵来划分呢类似，不同的地方在于对于基尼指数的更新。基尼指数越小说明节点越纯净，因此，**当前的划分计算产生的基尼指数小于最优的时**，说明现在的划分更合理，则更新相应的参数。

### 分类错误率作为不纯度度量

#### 计算分类错误率

```python
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
```

上述代码为计算数据集的分类错误率。首先统计各个类别出现的频率，然后选择最大的频率作为正确的类，其他的分类都是错误的，因此分类错误率为

​																$$Error(t)=1-max_{j}p(j|t)$$

#### 使用分类错误率来选择最优划分属性。

使用分类错误率与使用基尼指数划分的代码基本一致，这里不在赘述。

### 构建决策树

构建决策树采用的方法为递归的构建。首先选择一个最优的属性作为根节点，然后根据这个属性将数据集划分为两部分或者几部分。再根据划分出的新的数据集选择最优的属性对新的数据集进行划分。**每个属性只是用一次**。递归中止的条件为数据集中的所有属性都被用来划分了，或者当前节点中只有一个类别了。当所有属性都用完时，选择目前数据集中最多的类别作为当前子树的类别

本次实验选择一个多重的字典来存储决策树。其基本结构如下：*dict\[attributes\]\[value\]*。其中*attributes*为对应的属性，*value*为属性对应的值。

对于二叉树，value是一个元组，由最优划分的值和子树标志组成。离散值子树的标志为*true*和*false*。连续值属性的标志为*less*, *more*。最终构成的*key*值为*tuple([value, flag])*.

对于多叉树，连续值属性和离散值属性具有不同的树结构，其中离散值属性的*key*值为*dict\[attributes\]\[value\]*

*value*就是离散值对应的值。

其具体实现代码如下:

```python
def create_tree(dataset, attributes, criterion="gini"):
    '''根据数据集创建决策树
    args:
        datset: 为list类型的数据，是一个二维的list。每一行表示一个对象，每一列表示一个属性，其中最后一列为类别
        attributes: 属性对应的字符串,每次划分都会从中删除使用的属性
        criterion: 选择属性的不纯度度量方法. 默认为"gini",可以设置为"entrophy","cls error"
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
```

### 使用决策树进行分类

类似于构建决策树的过程，同样是递归的调用分类的程序。首先获取根节点所需要的属性，然后获取对象对应属性的值。根据对象的值确定该获取根节点的哪个子树。然后再重复上述过程，直到到达叶子节点即可获得该对象的类别。具体代码实现如下：

```python
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
```

### 四种分类性能评价指标

对于accuracy. precision, recall,和f1-score这四个分类指标，本次实验采用sklearn这一机器学习库提供的方法进行计算。
