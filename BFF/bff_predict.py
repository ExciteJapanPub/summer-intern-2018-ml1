import numpy as np
import tensorflow as tf
import mnist_cnn_train as train


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
        saver.restore(sess, train.CHECKPOINT)

        # 実行
        probas, predicted_label = sess.run([y, class_label])

        # 結果
        label = predicted_label[0]
        probas = [f'{p:5.3f}' for p in probas[0]]
        print(f'prediction={label} probas={probas} image={image_path}')

        return label


if __name__ == '__main__':
    # 予測
    path_list = [
        '0_10.jpg',
        '1_1097.jpg',
        '2_868.jpg',
        '3_8561.jpg',
        '4_8613.jpg',
        '5_955.jpg',
        '6_5441.jpg',
        '7_1100.jpg',
        '8_5360.jpg',
        '9_105.jpg',
        '0_3764.jpg',
        '1_1527.jpg',
        '2_583.jpg',
        '3_9905.jpg',
        '4_3780.jpg',
        '5_3855.jpg',
        '6_1982.jpg',
        '7_4966.jpg',
        '8_3206.jpg',
        '9_6157.jpg',
    ]
    for path in path_list:
        predict(str(train.TEST_IMAGES_PATH / path))
