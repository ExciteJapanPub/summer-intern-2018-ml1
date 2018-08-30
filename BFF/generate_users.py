import numpy as np
from numpy.random import *
import csv
import pandas as pd
from tqdm import tqdm
from best_friend_forever import User


# define constant
FOOD_NUM = 101
COUNTRY_NUM = 5
ING_NUM = 14
CALORIE_NUM = 3

NUM_USERS = 100

CATEGORY_PATH = "/Users/excite1/Work/summer-intern-2018-ml1/BFF/category.ver2.1.csv"

category_df = pd.read_csv(CATEGORY_PATH)



def search_user_by_userid(users, user_id):
    for user in users:
        if user.user_id == user_id:
            return user

    return None


# load category vectors from csv file
def load_category(label):
    user_category = category_df.loc[label, :]
    country = user_category[['中華', '和食', 'アメリカ', 'イタリアン・フレンチ', 'アジア・エスニック']]
    ingredient = user_category[['牛肉', '豚肉', '鶏肉', '魚', '海産物',	'野菜', '卵', '豆', '芋',	'乳製品',
                               'ご飯', '麵類', '揚げ物', '小麦粉']]
    calorie = user_category[['高カロリー',	 '中カロリー',  '低カロリー']]
    return country.values, ingredient.values, calorie.values


# update user's food vector with random number
def update_feature(users, user_id):
    user = search_user_by_userid(users, user_id)
    assert user is not None
    for i in range(0, FOOD_NUM):
        user.feature_food[i] += randint(10)
        coefficient = user.feature_food[i]
        country_vector, ing_vector, carolie_vector = load_category(i)
        user.feature_country += country_vector.astype('int64') * coefficient
        user.feature_ing += ing_vector.astype('int64') * coefficient
        user.feature_calorie += carolie_vector.astype('int64') * coefficient


# save user_data as a csv file
def save_user_data(users):
    f = open('user_data.csv', 'w')
    for i in range(0, NUM_USERS):
        writer = csv.writer(f)

        csvlist = []
        csvlist.append(users[i].user_id)
        csvlist.append(users[i].feature_food)
        csvlist.append(users[i].feature_country)
        csvlist.append(users[i].feature_ing)
        csvlist.append(users[i].feature_calorie)

        writer.writerow(csvlist)

    f.close()


# show user_id and vectors in console(for test use)
def show_users():
    f = open('user_data.csv', 'r', errors='', newline='')
    usr_data = csv.reader(f, delimiter=',', doublequote=True, lineterminator='¥r¥n', skipinitialspace=True)
    for row in usr_data:
        print("user_id:"+str(row[0]))
        print("feature_vector:"+str(row[1]))
        print("feature_country:"+str(row[2]))
        print("feature_ingredient:"+str(row[3]))
        print("feature_calorie:"+str(row[4]))

def generate_users():
    users = []

    print("Generating user data...")
    for i in tqdm(range(0, NUM_USERS)):
       users.append(User(i))
       update_feature(users, i)

    save_user_data(users)
    print("Generated. user_data file saved")

if __name__ == '__main__':
    generate_users()
    # for test check use
    # show_users()