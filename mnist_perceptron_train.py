import os
import pathlib
import cv2
import random
import numpy as np
import tensorflow as tf
from sklearn import metrics
from tqdm import tqdm


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

IMAGE_SIZE = 28
CHANNEL_NUM = 1
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE * CHANNEL_NUM

CLASS_NUM = 10

EPOCH_SIZE = 30
BATCH_SIZE = 100

LEARNING_RATE = 0.5

DATA_PATH = pathlib.Path('./MNIST_data/processed/')
TRAIN_IMAGES_PATH = DATA_PATH / 'images/train/'
TRAIN_LABELS_PATH = DATA_PATH / 'labels/train.csv'
TEST_IMAGES_PATH = DATA_PATH / 'images/test/'
TEST_LABELS_PATH = DATA_PATH / 'labels/test.csv'

CHECKPOINT = './checkpoint/mnist_perceptron.ckpt'


def image2array(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None

    # リサイズ&正規化
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = image.flatten().astype(np.float32) / 255.0

    return image


def load_images(images_path, labels_path):
    images = []
    labels = []

    print('\n- load', labels_path.name)

    with labels_path.open() as f:
        # 各行のファイル名と正解ラベルを取り出しリスト化する
        for line in tqdm(f):
            filename, label = line.rstrip().split(',')
            image_path = str(images_path / filename)
            image = image2array(image_path)
            if image is None:
                print('not image:', image_path)
                continue
            images.append(image)
            labels.append(int(label))

    assert len(images) == len(labels)

    return images, labels


def inference(x):
    # モデルの重み
    W = tf.Variable(tf.zeros([IMAGE_PIXELS, CLASS_NUM]))
    # モデルのバイアス
    b = tf.Variable(tf.zeros([CLASS_NUM]))
    # モデル出力
    y = tf.matmul(x, W) + b

    return y


def loss(onehot_labels, logits):
    # 損失関数はクロスエントロピーとする
    return tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)


def training(loss_value):
    # 勾配降下アルゴリズム(最急降下法)を用いてクロスエントロピーを最小化する
    return tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss_value)


def train(trains, tests):
    # データとラベルに分ける
    train_x, train_y = trains
    test_x, test_y = tests

    with tf.Graph().as_default():
        # 画像データ
        x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS])
        # 出力データ
        y = inference(x)
        # 正解データ
        labels = tf.placeholder(tf.int64, [None])
        y_ = tf.one_hot(labels, depth=CLASS_NUM, dtype=tf.float32)

        # 損失関数
        loss_value = loss(y_, y)
        # 学習
        train_step = training(loss_value)

        # 予測値と正解値を比較してbool値にする
        prediction = tf.argmax(tf.nn.softmax(y), 1)
        correct_prediction = tf.equal(prediction, labels)
        # これを正解率とする
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 保存の準備
        saver = tf.train.Saver()
        # セッションの作成
        sess = tf.Session()
        # セッションの開始及び初期化
        sess.run(tf.global_variables_initializer())

        # 学習
        print('\n- start training')
        for epoch in range(EPOCH_SIZE):
            # ミニバッチ法
            keys = list(range(len(train_x)))
            random.shuffle(keys)
            for i in range(len(keys) // BATCH_SIZE):
                batch_keys = keys[BATCH_SIZE*i:BATCH_SIZE*(i+1)]
                batch_x = np.asarray([train_x[key] for key in batch_keys])
                batch_y = np.asarray([train_y[key] for key in batch_keys])
                # 確率的勾配降下法によりクロスエントロピーを最小化するような重みを更新する
                sess.run(train_step, feed_dict={x: batch_x, labels: batch_y})
            # 1epoch毎に学習データに対して精度を出す
            train_accuracy, train_loss = sess.run([accuracy, loss_value], feed_dict={x: train_x, labels: train_y})
            print(f'[epoch {epoch+1:02d}] acc={train_accuracy:12.10f} loss={train_loss:12.10f}')

        # 学習が終わったら評価データに対して精度を出す
        test_accuracy, test_loss, prediction_y = sess.run([accuracy, loss_value, prediction], feed_dict={x: test_x, labels: test_y})
        print('\n- test accuracy')
        print(f'acc={test_accuracy:12.10f} loss={test_loss:12.10f}')

        print('\n- report')
        print(metrics.classification_report(test_y, prediction_y, target_names=[f'class {c}' for c in range(CLASS_NUM)]))
        print(metrics.confusion_matrix(test_y, prediction_y))

        # 完成したモデルを保存する
        saver.save(sess, CHECKPOINT)


if __name__ == '__main__':
    # 学習データをロードする
    train_lsit = load_images(TRAIN_IMAGES_PATH, TRAIN_LABELS_PATH)
    # 評価データをロードする
    test_list = load_images(TEST_IMAGES_PATH, TEST_LABELS_PATH)
    # 学習開始
    train(train_lsit, test_list)
