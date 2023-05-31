"""自己写的，图片处理的一些基本操作"""
import numpy as np
import cv2
from PIL import Image


class ImageProcesser:
    def __init__(self, img):
        self.img = img

    @staticmethod
    def resize(img, size, letterbox_image, mode='PIL'):
        """
        使用PTL或opencv对图像resize
        :param img: original image(ndarray)
        :param size: shape of original image(list)
        :param letterbox_image: 是否对图像进行padding以获得不失真的resize(bool)
        :param mode: 使用PTL还是opencv(str)
        :return:new_image: resized image(ndarray)
        """
        w, h = size
        if mode == 'PIL':
            iw, ih = img.size
            if letterbox_image:
                scale = min(w / iw, h / ih)
                nw = int(iw * scale)
                nh = int(ih * scale)
                img = img.resize((nw, nh), Image.BICUBIC)
                new_image = Image.new('RGB', size, (128, 128, 128))
                new_image.paste(img, ((w - nw) // 2, (h - nh) // 2))
            else:
                new_image = img.resize((w, h), Image.BICUBIC)
        else:
            if letterbox_image:
                shape = img.shape
                if isinstance(size, int):
                    size = (size, size)
                r = min(size[0] / shape[0], size[1] / shape[1])
                new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
                dw, dh = size[1] - new_unpad[0], size[0] - new_unpad[1]
                dw /= 2
                dh /= 2

                if shape[::-1] != new_unpad:  # resize
                    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_CUBIC)
                top, bottom = int(dh - 0.1), int(dh + 0.1)
                left, right = int(dw - 0.1), int(dw + 0.1)
                new_image = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                               value=(128, 128, 128))  # add border
            else:
                new_image = cv2.resize(img, (w, h))
        return new_image

    @staticmethod
    def cvtColor(img):
        """
        转RGB格式
        :param img: original image(ndarray)
        :return: img: result image(ndarray)
        """
        if len(np.shape(img)) == 3 and np.shape(img)[2] == 3:
            return img
        else:
            img = img.convert('RGB')
            return img

    @staticmethod
    def preprocess(img):
        """
        Normalization
        :param img: original image(ndarray)
        :return: img: normalized image(ndarray)
        """
        img /= 255.0
        return img
