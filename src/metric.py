
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def output_fusion_matrix(predict, ground_truth):
    print("混淆矩阵如下: ")
    labels = set(ground_truth)
    cf_matrix = confusion_matrix(y_pred=predict, y_true=ground_truth, labels=list(labels))
    print("labels = ", labels)
    print(cf_matrix)

def output_classification_report(preict, ground_truth):
    r = classification_report(y_true=ground_truth, y_pred=preict)
    print("分类结果报告: ")
    print(r)

def cal_acc(predict, ground_truth):
    """计算分类精度
    args:
        predict:预测的各个对象的类别数
        ground_truth:基准值
    returns:
        分类精度
    """
    assert len(predict) == len(ground_truth), "预测标签与基准值的长度不一致"
    count = 0
    for index in range(len(predict)):
        if predict[index] == ground_truth[index]:
            count += 1
    return float(count / len(predict))


