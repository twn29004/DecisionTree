from process_data import read_bank_data

train_dataset, test_dataset, attributes = read_bank_data()


idx = 1

aSet = {object[1] for object in train_dataset}

for value in aSet:
    cnt = 0
    for object in train_dataset:
        if (object[idx] == value):
            cnt += 1
    print(cnt)