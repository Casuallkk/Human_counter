"""单张图片预测及目录遍历检测， 对代码进行了优化。"""
import os
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

from yolo import YOLO

if __name__ == "__main__":
    mode = "predict"
    count = True
    test_interval = 100
    img = "img/street.jpg"
    origin_path = "img/"
    save_path = "img_out/"
    yolo = YOLO()

    if mode == "predict":
        # 用于对单张图片进行预测
        image = Image.open("img/street.jpg")
        result_image = yolo.detect_image(image)
        plt.imshow(result_image)
        plt.axis('off')
        plt.show()

    elif mode == "dir_predict":
        # 用于对origin_path中所有图片进行预测
        print("Start prediction...")
        img_names = os.listdir(origin_path)
        for img_name in tqdm(img_names):
            # tqdm用于生成进度条
            if img_name.lower().endswith('.jpg'):
                image_path = os.path.join(origin_path, img_name)
                image = Image.open(image_path)
                result_image = yolo.detect_image(image)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                result_image.save(os.path.join(
                    save_path, img_name.replace(".jpg")), quality=95, subsampling=0)
    else:
        raise AssertionError(
            "Please specify the correct mode: 'predict', 'dir_predict'.")
