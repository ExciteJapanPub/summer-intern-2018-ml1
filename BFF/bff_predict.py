import numpy as np
import tensorflow as tf
import bff_train as train
from tqdm import tqdm

SAVED_CHECKPOINT = '../checkpoint/dish_101_cnn.ckpt'


def predict(image_path):
    # 画像の準備
    image = train.image2array(image_path)
    if image is None:
        print('not image:', image_path)
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
        saver.restore(sess, SAVED_CHECKPOINT)

        # 実行
        probas, predicted_label = sess.run([y, class_label])

        # 結果
        label = predicted_label[0]
        probas = [f'{p:5.3f}' for p in probas[0]]
        print(f'prediction={label} probas={probas} image={image_path}')

        return label


if __name__ == '__main__':
    # 予測
    path_list = []
    with train.TEST_IMAGES_PATH.open() as f:
        for line in f:
            dishname, filename = line.rstrip().split('/')
            path_list.append(dishname + '/' + filename + '.jpg')
            break
    for path in path_list:
        predict('./dataset/images/' + path)

