from sklearn import tree

from process_data import split_data_label
from process_data import trasfrom_dis_to_int

from metric import output_classification_report as output_cls_report, output_fusion_matrix

# from process_data import read_balance_scale_data as read_data
from process_data import read_bank_data as read_data

train_dataset, test_dataset, attributes = read_data()
print(train_dataset[0])
for idx in [1,2,3,4,6,7,8,10,15]:
    train_dataset = trasfrom_dis_to_int(train_dataset, idx=idx)
    test_dataset = trasfrom_dis_to_int(test_dataset, idx=idx)

train_data, train_label = split_data_label(train_dataset)
test_data, test_label = split_data_label(test_dataset)


decision_tree = tree.DecisionTreeClassifier(criterion="entropy")
decision_tree.fit(X=train_data, y=train_label)
preicit = decision_tree.predict(X=test_data)
output_fusion_matrix(preicit, test_label)
output_cls_report(preicit, test_label)

print("训练集的精度为: ")
preicit_train = decision_tree.predict(X=train_data)
output_fusion_matrix(preicit_train, train_label)
output_cls_report(preicit_train, train_label)