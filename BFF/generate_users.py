import numpy as np
from numpy.random import *
import argparse
import bff_train as train
import tensorflow as tf
import csv

# define constant
N = 5
NUM_USERS = 10


# define user data
class User:
    def __init__(self, user_id):
        self.feature_food = np.zeros(N)
        self.user_id = user_id


# define picture data
class Picture:
    def __init__(self):
        self.pic_id = None
        self.user_id = None
        self.file_path = None
        self.food_num = None


def search_user_by_userid(users, user_id):
    for user in users:
        if user.user_id == user_id:
            return user

    return None


def update_feature(users, user_id, label):
    user = search_user_by_userid(users, user_id)
    assert user is not None
    user.feature_food[label] += 1
    # for test
    if user_id == 0:
        pass
    else:
        for i in range(0, N):
            user.feature_food[i] += randint(5)


def input_pic(path, user_id):
    picture = Picture()
    picture.file_path = path
    picture.user_id = user_id

    return picture


def predict(picture):
    # 画像の準備
    image = train.image2array(picture.file_path)
    if image is None:
        print('not image:', picture.file_path)
        return None

    with tf.Graph().as_default():
        # 予測
        x = np.asarray([image])
        y = tf.nn.softmax(train.inference(x, 1.0))
        class_label = tf.argmax(y, 1)

        # 保存の準備
        saver = tf.train.Saver()
        # セッションの作成
        sess = tf.Session()
        # セッションの開始及び初期化
        sess.run(tf.global_variables_initializer())

        # モデルの読み込み
        saver.restore(sess, train.CHECKPOINT)

        # 実行
        probas, predicted_label = sess.run([y, class_label])

        # 結果
        label = predicted_label[0]
        probas = [f'{p:5.3f}' for p in probas[0]]
        print(f'prediction={label} probas={probas} image={picture.file_path}')

        picture.food_num = label

        return label


def save_user_data(users):
    f = open('user_data.csv', 'w')
    for i in range(0, NUM_USERS):
        writer = csv.writer(f)

        csvlist = []
        csvlist.append(users[i].user_id)
        csvlist.append(users[i].feature_food)

        writer.writerow(csvlist)

    f.close()


# show user_id and feature_vector in console(for test use)
def show_users():
    f = open('user_data.csv', 'r', errors='', newline='')
    usr_data = csv.reader(f, delimiter=',', doublequote=True, lineterminator='¥r¥n', skipinitialspace=True)
    for row in usr_data:
        print("user_id:"+str(row[0]))
        print("feature_vector:"+str(row[1]))

def generate_users():
    users = []
    path_list = [
       '0_003.jpg',
       '0_003.jpg',
       '0_003.jpg',
       '0_003.jpg',
       '0_003.jpg',
       '0_008.jpg',
       '1_008.jpg',
       '2_008.jpg',
       '3_008.jpg',
       '4_008.jpg']

    for i in range(0, NUM_USERS):
       users.append(User(i))
       path = "/Users/excite1/Work/summer-intern-2018-ml1/DISH_data/raw/images/test/" + path_list[i]
       picture = input_pic(path, i)
       label = predict(picture)
       update_feature(users, i, label)

    save_user_data(users)


if __name__ == '__main__':
    generate_users()
    # for test use
    show_users()