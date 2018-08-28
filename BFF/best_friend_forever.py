
import numpy as np
import tensorflow as tf
import bff_train as train


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

def calc_BFF_rank(usr_id, list_usr):
   user = User()
   sim_arr = []
   my_vector = user.get_feature_food(usr_id)
   i = 0
   for user in range(0, len(list_usr)):
       your_vector = user.get_feature_food(usr_id)
       sim_arr[i] = np.inner(your_vector, my_vector)
       i += 1

   return sim_arr


def show_BFF_rank(usr_id, list_usr):
   arr = calc_BFF_rank(usr_id, list_usr)
   temp = np.argsort(arr)
   print(temp[:5])



def generate_users(path):
   users = []
   for i in range(0, 10):
       users[i] = User(i).user_id
       input_pic(path, i)
       predict()
       update_feature()

   return users
