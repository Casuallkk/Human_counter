"""获取voc格式的annotation，对代码进行了优化"""
import os
import random
import xml.etree.ElementTree as ET
import numpy as np
from utils.dataloader import YoloDataset

annotation_mode = 0
classes_path = 'model_data/my_classes.txt'
trainval_percent = 0.9
train_percent = 0.9
VOCdevkit_path = 'VOCdevkit'
VOCdevkit_sets = [('2007', 'train'), ('2007', 'val')]
classes, _ = YoloDataset.get_classes(classes_path)
photo_nums = np.zeros(len(VOCdevkit_sets))
nums = np.zeros(len(classes))


def convert_annotation(year, image_id, list_file):
    in_file = open(os.path.join(VOCdevkit_path, 'VOC%s/Annotations/%s.xml' % (year, image_id)), encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0
        if obj.find('difficult') is not None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)),
             int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

        nums[classes.index(cls)] = nums[classes.index(cls)] + 1


if __name__ == "__main__":
    random.seed(0)
    test = os.path.abspath(VOCdevkit_path)
    if " " in os.path.abspath(VOCdevkit_path):
        raise ValueError("Error!")

    if annotation_mode == 0 or annotation_mode == 1:
        print("Generate txt in ImageSets.")
        xmlfilepath = os.path.join(VOCdevkit_path, 'VOC2007/Annotations')
        saveBasePath = os.path.join(VOCdevkit_path, 'VOC2007/ImageSets/Main')
        temp_xml = os.listdir(xmlfilepath)
        total_xml = []
        for xml in temp_xml:
            if xml.endswith(".xml"):
                total_xml.append(xml)

        trainval = random.sample(range(len(total_xml)), int(len(total_xml) * trainval_percent))
        train = random.sample(trainval, int((len(total_xml) * trainval_percent) * train_percent))

        print("train and val size", int(len(total_xml) * trainval_percent))
        print("train size", int(int(len(total_xml) * trainval_percent) * train_percent))
        ftrainval = open(os.path.join(saveBasePath, 'trainval.txt'), 'w')
        ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')
        ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w')
        fval = open(os.path.join(saveBasePath, 'val.txt'), 'w')

        for i in range(len(total_xml)):
            name = total_xml[i][:-4] + '\n'
            if i in trainval:
                ftrainval.write(name)
                if i in train:
                    ftrain.write(name)
                else:
                    fval.write(name)
            else:
                ftest.write(name)

        ftrainval.close()
        ftrain.close()
        fval.close()
        ftest.close()
        print("Generate txt in ImageSets done.")

    if annotation_mode == 0 or annotation_mode == 2:
        print("Generate 2007_train.txt and 2007_val.txt for train.")
        type_index = 0
        for year, image_set in VOCdevkit_sets:
            image_ids = open(os.path.join(VOCdevkit_path, 'VOC%s/ImageSets/Main/%s.txt' % (year, image_set)),
                             encoding='utf-8').read().strip().split()
            list_file = open('%s_%s.txt' % (year, image_set), 'w', encoding='utf-8')
            for image_id in image_ids:
                list_file.write('%s/VOC%s/JPEGImages/%s.jpg' % (os.path.abspath(VOCdevkit_path), year, image_id))

                convert_annotation(year, image_id, list_file)
                list_file.write('\n')
            photo_nums[type_index] = len(image_ids)
            type_index += 1
            list_file.close()
        print("Generate 2007_train.txt and 2007_val.txt for train done.")