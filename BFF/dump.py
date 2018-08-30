
import os
import pathlib
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

IMAGE_SIZE = 128
CHANNEL_NUM = 3
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE * CHANNEL_NUM

CLASS_NUM = 101

EPOCH_SIZE = 30
BATCH_SIZE = 300

LEARNING_RATE = 1e-4

DATA_PATH = pathlib.Path('./dataset')
TRAIN_IMAGES_PATH = DATA_PATH / 'meta/train.txt'
LABELS_PATH = DATA_PATH / 'meta/classes.txt'
TEST_IMAGES_PATH = DATA_PATH / 'meta/test.txt'
CHECKPOINT = '../checkpoint/dish_101_cnn.ckpt'

class_df = pd.read_table(LABELS_PATH, header=None)


def image2array(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        return None

    # リサイズ&正規化
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = image.flatten().astype(np.float32) / 255.0

    return image


def load_images(images_path):
    images = []
    labels = []

    count = 0

    # print('\n- load', labels_path.name)

    with images_path.open() as f:
        # 各行のファイル名と正解ラベルを取り出しリスト化する
        for line in tqdm(f):
            dishname, filename = line.rstrip().split('/')
            label = class_df.at[dishname, 'lable']
            image_path = str("./dataset/images/" + dishname + '/' + filename + '.jpg')
            image = image2array(image_path)
            if image is None:
                print('not image:', image_path)
                continue
            images.append(image)
            labels.append(int(label))

            if count % 750 == 749:
                train_list = images, labels
                fname = "dataset/meta/train_dump/train_list" + str(int((count+1)/750)) + '.txt'
                f = open(fname, 'wb')
                pickle.dump(train_list, f)
                f.close()
                images = []
                labels = []

            count += 1

    assert len(images) == len(labels)

    return images, labels


if __name__ == '__main__':
    class_df = class_df.rename(columns={0: 'class'})
    class_df['lable'] = [i for i in range(CLASS_NUM)]
    class_df = class_df.set_index('class')
    test_list = load_images(TRAIN_IMAGES_PATH)