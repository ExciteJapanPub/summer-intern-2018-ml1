
import numpy as np
import tensorflow as tf
import bff_train as train
import pandas as pd
import csv
import argparse
from scipy.spatial.distance import cosine
from random import random

N = 101
COUNTRY_NUM = 5
ING_NUM = 14
CALORIE_NUM = 3
CATEGORY_PATH = "/Users/excite1/Work/summer-intern-2018-ml1/BFF/category.ver2.1.csv"

category_df = pd.read_csv(CATEGORY_PATH)
pictures = []

# FOOD_DICT = {0:'udon', 1:'omurice', 2:'curry rice', 3:'fried rice', 4:'humberg'}

from numpy.random import *

# categoryのデータをpandasで読み込み

class User:
    def __init__(self, user_id):
        self.feature_food = np.zeros(N)
        self.feature_country = np.zeros(COUNTRY_NUM)
        self.feature_ing = np.zeros(ING_NUM)
        self.feature_calorie = np.zeros(CALORIE_NUM)
        self.user_id = user_id

class Picture:
    def __init__(self):
        self.pic_id = None
        self.user_id = None
        self.file_path = None
        self.food_num = None
        self.feature_vector = None


# FOOD_DICT = {0:'udon', 1:'omurice', 2:'curry rice', 3:'fried rice', 4:'humberg'}


def normalize(v):
    n = 0
    for i in v:
        n += i * i
    return v/np.sqrt(n)


def input_pic(path, user_id):
    picture = Picture()
    picture.file_path = path
    picture.user_id = user_id

    pictures.append(picture)

    return picture


def load_category(label):
    user_category = category_df.loc[label, :]
    country = user_category[['中華', '和食', 'アメリカ', 'イタリアン・フレンチ', 'アジア・エスニック']]
    ingredient = user_category[['牛肉', '豚肉', '鶏肉', '魚', '海産物',	'野菜', '卵', '豆', '芋',	'乳製品',
                               'ご飯', '麵類', '揚げ物', '小麦粉']]
    calorie = user_category[['高カロリー',	 '中カロリー',  '低カロリー']]
    return country.values, ingredient.values, calorie.values


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
        # print(f'prediction={label} probas={probas} image={picture.file_path}')

        picture.food_num = label
        picture.feature_vector = load_category(label)
        # feature_vectorを参照して代入
        return label


def search_user_by_userid(users, user_id):
    for user in users:
        if int(user.user_id) == user_id:
            return user

    return None

def serch_picture_by_userid(user_id):

    result = []

    for picture in pictures:
        if picture.user_id == int(user_id):
            result.append(picture)
    return result


def update_feature(users, user_id, label):
    label = int(label)
    user = search_user_by_userid(users, user_id)
    assert user is not None

    # print(user.feature_calorie[0])
    # print(user.feature_food)
    user.feature_food[label] += 1
    # print(user.feature_food)

    country_vector, ing_vector, carolie_vector = load_category(label)
    user.feature_country = user.feature_country + country_vector
    user.feature_ing = user.feature_ing + ing_vector
    user.feature_calorie = user.feature_calorie + carolie_vector
    # print(user.feature_calorie)


def generator():
    print('start generate users data')
    user_lst = []
    for i in range(10):

        user = User(i)
        user_lst.append(user)

        with train.TEST_IMAGES_PATH.open() as f:
            lines = f.readlines()
            for j in range(2):
                k = np.random.randint(25250)
                line = lines[k]
                dish_name, filename = line.rstrip().split('/')
                path = "./static/dataset/images/" + dish_name + '/' + filename + '.jpg'

                picture = input_pic(path, i)
                label = predict(picture)
                # print(label)
                update_feature(user_lst, i, label)

    # user.feature_food = np.random.randint(10, size=(1, N))
    # user.feature_country = np.random.randint(10, size=(1, COUNTRY_NUM))
    # user.feature_ing = np.random.randint(10, size=(1, ING_NUM))
    # user.feature_calorie = np.random.randint(10, size=(1, CALORIE_NUM))

    print(user_lst[1].user_id)
    print(user_lst[1].feature_food)
    print(user_lst[1].feature_country)
    print(user_lst[1].feature_ing)
    print(user_lst[1].feature_calorie)
    return user_lst


# USERS = generator()


def calc_BFF_similarity(users):
    similarity = []
    for user in users:
        similarity.append(calc_BFF_rank(user.user_id, users))

    return similarity


# USERS = generator()


def show_BFF_rank(usr_id, users):
   arr = calc_BFF_rank(usr_id, users)
   friend_list = np.argsort(arr)[::-1]
   print(friend_list[:5])

   return friend_list, np.sort(arr)[::-1]


def load_users():
    # user_list = []
    # f = open('user_data.csv', 'r', errors='', newline='')
    # usr_data = csv.reader(f, delimiter='.', doublequote=True, lineterminator='¥r¥n', skipinitialspace=True)
    # for row in usr_data:
    #     user = User(row[0])
    #     print(row[0])
    #     user.feature_food = int(row[1])
    #     user.feature_country = int(row[2])
    #     user.feature_ing = int(row[3])
    #     user.feature_calorie = int(row[4])
    #     user_list.append(user)

    user_df = pd.read_csv('user_data2.csv', header=None)
    return user_df.values


def calc_BFF_rank(usr_id, users):


   user = search_user_by_userid(users, usr_id)
   sim_arr = []
   my_food = normalize(user.feature_food)
   my_country = normalize(user.feature_country)
   my_ing = normalize(user.feature_ing)
   my_calorie = normalize(user.feature_calorie)

   weight_food = 1
   weight_country = 1
   weight_ingredient = 1
   weight_calorie = 1
   sim_arr_lst = []
   for user in users:
       sim_arr = []
       #print(user.feature_food)
       your_food = normalize(user.feature_food)
       your_country = normalize(user.feature_country)
       your_ing = normalize(user.feature_ing)
       your_calorie = normalize(user.feature_calorie)
       sim_arr.append(1 - cosine(your_food, my_food)*weight_food)
       sim_arr.append(1 - cosine(your_country, my_country)*weight_country)
       sim_arr.append(1 - cosine(your_ing, my_ing)*weight_ingredient)
       sim_arr.append(1 - cosine(your_calorie, my_calorie)*weight_calorie)
       sim_arr_lst.append(np.mean(sim_arr))

   return sim_arr_lst


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Best food friend", add_help=True)
    parser.add_argument('--pathimage', '-t', type=str,
                        default="dataset/images/apple_pie/134.jpg")
    parser.add_argument('--your_id', '-i', type=int, default=0)

    args = parser.parse_args()
    user_id = args.your_id

    # generator()

    '''
    print('path->')
    path = input()
    user_id = input()
    '''

    #
    # path = args.pathimage
    #
    # picture = input_pic(path, user_id)
    #
    # label = predict(picture)
    #
    # update_feature(USERS, user_id, label)

    # print(calc_BFF_similarity(users))
    # show_BFF_rank(user_id, USERS)

    #print(load_category(0))
