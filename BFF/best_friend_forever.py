
import numpy as np
import tensorflow as tf
import bff_train as train

N = 5

class User:
    def __init__(self, user_id):
        self.feature_food = np.zeros(N)
        self.user_id = user_id

    def get_feature_food(self):
        return self.feature_food


class Picture:
    def __init__(self):
        self.pic_id = None
        self.user_id = None
        self.file_path = None
        self.food_num = None


FOOD_DICT = {0:'udon', 1:'omurice', 2:'curry rice', 3:'fried rice', 4:'humberg'}


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


def search_user_by_userid(users, user_id):
    for user in users:
        if user.user_id == user_id:
            return user

    return None


def update_feature(users, user_id, label):
    user = search_user_by_userid(users, user_id)
    assert user is not None
    user.feature_food[label] += 1


# ---------


def calc_BFF_rank(usr_id, users):
   user = search_user_by_userid(users, usr_id)
   sim_arr = []
   my_vector = user.get_feature_food()
   i = 0
   for user in users:
       print(i)
       your_vector = user.get_feature_food()
       print(my_vector, your_vector)
       sim_arr.append(np.inner(your_vector, my_vector))
       i += 1

   return sim_arr


def show_BFF_rank(usr_id, users):
   arr = calc_BFF_rank(usr_id, users)
   temp = np.argsort(arr)[::-1]
   print(temp[:5])



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

   for i in range(0, 10):
       users.append(User(i))
       path = "/Users/excite1/Work/summer-intern-2018-ml1/DISH_data/raw/images/test/" + path_list[i]
       picture = input_pic(path, i)
       label = predict(picture)
       update_feature(users, i, label)

   return users


if __name__ == '__main__':

    users = generate_users()

    '''
    print('path->')
    path = input()
    user_id = input()
    '''

    # path = "/Users/excite1/Work/summer-intern-2018-ml1/DISH_data/raw/images/test/2_003.jpg"
    user_id = 0
    #
    # picture = input_pic(path, user_id)
    #
    # label = predict(picture)
    #
    # update_feature(users, user_id, label)

    calc_BFF_rank(user_id, users)

    show_BFF_rank(user_id, users)