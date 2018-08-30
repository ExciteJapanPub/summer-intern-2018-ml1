
import os
import pathlib
import cv2
import random
import numpy as np
import tensorflow as tf
from sklearn import metrics
from tqdm import tqdm
import pandas as pd
import pickle


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

IMAGE_SIZE = 128
CHANNEL_NUM = 3
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE * CHANNEL_NUM

CLASS_NUM = 101

EPOCH_SIZE = 1
BATCH_SIZE = 256

LEARNING_RATE = 1e-4

DATA_PATH = pathlib.Path('./dataset')
TRAIN_IMAGES_PATH = DATA_PATH / 'meta/train.txt'
# TRAIN_LABELS_PATH = DATA_PATH / 'labels/train.csv'
LABELS_PATH = DATA_PATH / 'meta/classes.txt'
# TEST_LABELS_PATH = DATA_PATH / 'labels/test.csv'
TEST_IMAGES_PATH = DATA_PATH / 'meta/test.txt'
# CHECKPOINT = './checkpoint/mnist_cnn.ckpt'
SAVED_CHECKPOINT = '../checkpoint/dish_101_cnn.ckpt'
CHECKPOINT = '../checkpoint/dish_101_cnn_1000.ckpt'

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
    images_for_save = []
    labels_for_save = []

    # print('\n- load', labels_path.name)

    with images_path.open() as f:
        # 各行のファイル名と正解ラベルを取り出しリスト化する
        for line in tqdm(f):
            dishname, filename = line.rstrip().split('/')
            label = class_df.at[dishname, 'lable']
            # print("label_str = " + dishname)
            # print("label = " + str(label))
            image_path = str("./dataset/images/" + dishname + '/' + filename + '.jpg')
            image = image2array(image_path)
            if image is None:
                print('not image:', image_path)
                continue
            images.append(image)
            labels.append(int(label))

    assert len(images) == len(labels)

    return images, labels


def inference(x, keep_prob):
    # 重みを標準偏差0.1の正規分布で初期化する
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    # バイアスを0.1の定数で初期化する
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # 畳み込みを行う
    def conv2d(x, W):
        # 縦横ともにストライドは1でゼロパディングを行う
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    # 畳み込み層を作成する
    def conv_layer(x, filter_size, filter_in, filter_out):
        # 重み
        W = weight_variable([filter_size, filter_size, filter_in, filter_out])
        # バイアス
        b = bias_variable([filter_out])
        # 活性化関数
        return tf.nn.relu(conv2d(x, W) + b)

    # プーリング層を作成する
    def pool_layer(x, image_size):
        # MAXプーリング（カーネルサイズ2px*2px、縦横ともにストライドは2、ゼロパディング）
        h = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # 画像サイズは半分になる
        return h, int(image_size / 2)

    # 全結合層を作成する
    def dense_layer(x, dense_in, dense_out):
        # 重み
        W = weight_variable([dense_in, dense_out])
        # バイアス
        b = bias_variable([dense_out])
        # 結合
        return tf.matmul(x, W) + b

    # 平坦化されたベクトルを画像に戻す
    x_image = tf.reshape(x, [-1, IMAGE_SIZE, IMAGE_SIZE, CHANNEL_NUM])

    # 畳み込み層のフィルターサイズ 畳み込みは3px*3pxのカーネルサイズで行う
    filter_size = 3

    # 第1畳み込み層
    conv1_in = CHANNEL_NUM
    conv1_out = 32
    conv1 = conv_layer(x_image, filter_size, conv1_in, conv1_out)
    # 第1プーリング層
    pool1, out_size = pool_layer(conv1, IMAGE_SIZE)

    # 第2畳み込み層
    conv2_in = conv1_out
    conv2_out = 64
    conv2 = conv_layer(pool1, filter_size, conv2_in, conv2_out)
    # 第2プーリング層
    pool2, out_size = pool_layer(conv2, out_size)

    # 画像を平坦化してベクトルにする
    dimension = out_size * out_size * conv2_out
    x_flatten = tf.reshape(pool2, [-1, dimension])

    # 全結合層
    fc = dense_layer(x_flatten, dimension, conv2_out)
    # 活性化関数
    fc = tf.nn.relu(fc)

    # ドロップアウト
    drop = tf.nn.dropout(fc, keep_prob)

    # モデル出力
    y = dense_layer(drop, conv2_out, CLASS_NUM)

    return y


def loss(onehot_labels, logits):
    # 損失関数はクロスエントロピーとする
    return tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)


def training(loss_value):
    # 勾配降下アルゴリズム(Adam)を用いてクロスエントロピーを最小化する
    return tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss_value)


def train(trains, tests):
    # データとラベルに分ける
    train_x, train_y = trains
    test_x, test_y = tests

    with tf.Graph().as_default():
        # dropout率
        keep_prob = tf.placeholder(tf.float32)
        # 画像データ
        x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS])
        # 出力データ
        y = inference(x, keep_prob)
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

        # モデルの読み込み
        saver.restore(sess, SAVED_CHECKPOINT)

        # 学習
        print('\n- start training')
        for epoch in range(EPOCH_SIZE):
            # ミニバッチ法
            keys = list(range(len(train_x)))
            random.shuffle(keys)
            for i in tqdm(range(len(keys) // BATCH_SIZE)):
                batch_keys = keys[BATCH_SIZE*i:BATCH_SIZE*(i+1)]
                batch_x = np.asarray([train_x[key] for key in batch_keys])
                batch_y = np.asarray([train_y[key] for key in batch_keys])
                # 確率的勾配降下法によりクロスエントロピーを最小化するような重みを更新する
                sess.run(train_step, feed_dict={x: batch_x, labels: batch_y, keep_prob: 0.5})
            # 1epoch毎に学習データに対して精度を出す
            train_accuracy, train_loss = sess.run([accuracy, loss_value], feed_dict={x: train_x, labels: train_y, keep_prob: 1.0})
            print(f'[epoch {epoch+1:02d}] acc={train_accuracy:12.10f} loss={train_loss:12.10f}')

        # 学習が終わったら評価データに対して精度を出す
        test_accuracy, test_loss, prediction_y = sess.run([accuracy, loss_value, prediction], feed_dict={x: test_x, labels: test_y, keep_prob: 1.0})
        print('\n- test accuracy')
        print(f'acc={test_accuracy:12.10f} loss={test_loss:12.10f}')

        print('\n- report')
        print(metrics.classification_report(test_y, prediction_y, target_names=[f'class {c}' for c in range(CLASS_NUM)]))
        print(metrics.confusion_matrix(test_y, prediction_y))

        # 完成したモデルを保存する
        saver.save(sess, CHECKPOINT)


if __name__ == '__main__':
    class_df = class_df.rename(columns={0: 'class'})
    class_df['lable'] = [i for i in range(CLASS_NUM)]
    class_df = class_df.set_index('class')
    # print(class_df)
    # 学習データをロードする
    # train_lsit = load_images(TRAIN_IMAGES_PATH)
    print('start loading data set.')

    images = []
    labels = []
    for i in range(1, CLASS_NUM,5):
        fname = "train_list" + str(i) + ".txt"
        f = open("./dataset/meta/train_dump/" + fname, "rb")
        load_data = pickle.load(f)
        images.extend(load_data[0])
        labels.extend(load_data[1])
    train_list = images, labels
    for i in range(1, CLASS_NUM, 30):
        fname = "test_list" + str(i) + ".txt"
        f = open("./dataset/meta/test_dump/" + fname, "rb")
        load_data = pickle.load(f)
        images.extend(load_data[0])
        labels.extend(load_data[1])
    test_list = images, labels
    # 評価データをロードする
    # test_list = load_images(TEST_IMAGES_PATH)
    # 学習開始
    train(train_list, test_list)

