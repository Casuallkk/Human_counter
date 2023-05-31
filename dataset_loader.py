"""自己写的，用于将之前下的一个bbox为cvs文件的数据集处理成txt格式，最后并没有用上"""
from sklearn.model_selection import train_test_split
import csv
import os

trainval_percent = 0.9
# (训练集+验证集)和测试集的比例
train_percent = 0.9
# 训练集和验证集的比例

dataset_devkit_path = 'dataset/human'
saveBasePath = os.path.join(dataset_devkit_path, 'labels')
trainval_file = open(os.path.join(saveBasePath, 'trainval.txt'), 'w')
test_file = open(os.path.join(saveBasePath, 'test.txt'), 'w')
train_file = open(os.path.join(saveBasePath, 'train.txt'), 'w')
val_file = open(os.path.join(saveBasePath, 'val.txt'), 'w')
image_folder = 'dataset/human/images/'

if __name__ == '__main__':
    print("Start reading data...")
    with open('dataset/human/train/ground_truth.csv', 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        image_data = list(csv_reader)
    print("Photo nums:", len(image_data))

    trainval_data, test_data = train_test_split(image_data, train_size=trainval_percent)
    train_data, val_data = train_test_split(trainval_data, train_size=train_percent)
    print("Train and val size:", len(trainval_data))
    print("Train size:", len(train_data))

    for row in trainval_data:
        row[0] = image_folder + row[0]
        trainval_file.write(' '.join(row) + '\n')
    for row in test_data:
        row[0] = image_folder + row[0]
        test_file.write(' '.join(row) + '\n')
    for row in train_data:
        train_file.write(' '.join(row) + '\n')
    for row in val_data:
        val_file.write(' '.join(row) + '\n')
    print("Generate txt in csv.")


