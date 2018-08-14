import pathlib
import shutil
import random

import cv2
import numpy as np
import pandas as pd


# 平滑化
def apply_smoothing(src):
    # フィルター
    average_square = (5, 5)

    return cv2.blur(src, average_square)


# ガウシアンノイズ
def add_gaussian_noise(src):
    row, col, ch = src.shape
    mean = 0
    sigma = 15
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = src + gauss

    return noisy


# salt&pepperノイズ
def add_salt_pepper_noise(src):
    s_vs_p = 0.5
    amount = 0.004
    out = src.copy()
    # Salt mode
    num_salt = np.ceil(amount * src.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in src.shape]
    out[coords[:-1]] = (255, 255, 255)
    # Pepper mode
    num_pepper = np.ceil(amount * src.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in src.shape]
    out[coords[:-1]] = (0, 0, 0)

    return out


# コントラスト調整
def adjust_contrast(src):
    # ルックアップテーブルの生成
    min_table = 50
    max_table = 205
    diff_table = max_table - min_table

    lut_hc = np.arange(256, dtype='uint8')
    lut_lc = np.arange(256, dtype='uint8')

    # ハイコントラストLUT作成
    for i in range(0, min_table):
        lut_hc[i] = 0
    for i in range(min_table, max_table):
        lut_hc[i] = 255 * (i - min_table) / diff_table
    for i in range(max_table, 255):
        lut_hc[i] = 255

    # ローコントラストLUT作成
    for i in range(256):
        lut_lc[i] = min_table + i * diff_table / 255

    # LUT変換
    distorted = [
        cv2.LUT(src, lut_hc),
        cv2.LUT(src, lut_lc),
    ]

    return distorted


# ガンマ補正
def apply_gamma_correction(src):
    # ルックアップテーブルの生成
    gamma_list = [0.75, 1.5]

    # ガンマ補正LUT作成
    lut_list = []
    for gamma in gamma_list:
        lut_g = np.asarray([255 * pow(float(i) / 255, 1.0 / gamma) for i in range(256)])
        lut_list.append(lut_g)

    distorted = []

    # LUT変換
    for LUT in lut_list:
        distorted.append(cv2.LUT(src, LUT))

    return distorted


# 反転
def flip(src):
    return cv2.flip(src, 1)


# 回転
def rotate(src):
    def calculate_affine(size, angle):
        h, w = size
        angle_rad = angle / 180.0 * np.pi

        # 回転後の画像サイズを計算
        w_rot = int(np.round(h * np.absolute(np.sin(angle_rad)) + w * np.absolute(np.cos(angle_rad))))
        h_rot = int(np.round(h * np.absolute(np.cos(angle_rad)) + w * np.absolute(np.sin(angle_rad))))
        size_rot = (w_rot, h_rot)

        # 元画像の中心を軸に回転する
        center = (w / 2, h / 2)
        scale = 1.0
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

        # 平行移動を加える (rotation + translation)
        affine_matrix = rotation_matrix.copy()
        affine_matrix[0][2] = affine_matrix[0][2] - w / 2 + w_rot / 2
        affine_matrix[1][2] = affine_matrix[1][2] - h / 2 + h_rot / 2

        return affine_matrix, size_rot

    distorted = []
    size = src.shape[:2]
    angle_list = [10, -10]

    for angle in angle_list:
        affine_matrix, size_rot = calculate_affine(size, angle)
        distorted.append(cv2.warpAffine(src, affine_matrix, size_rot, flags=cv2.INTER_CUBIC))

    return distorted


def distort(src):
    # 各種水増し
    distorted = [
        src,
        apply_smoothing(src),
        add_gaussian_noise(src),
        add_salt_pepper_noise(src),
    ]
    distorted.extend(adjust_contrast(src))
    distorted.extend(apply_gamma_correction(src))

    # 反転
    flipped = [flip(image) for image in distorted]
    distorted.extend(flipped)

    # 回転
    result = []
    for distortion in distorted:
        result.append(distortion)
        result.extend(rotate(distortion))

    return result


def load(a_path, i_path, l_path):
    shutil.unpack_archive(str(a_path), a_path.parent)
    images = []
    labels = []
    with open(l_path) as f:
        for line in f:
            filename, label = line.rstrip().split(',')
            images.append(i_path / filename)
            labels.append(int(label))
    return images, labels


def make_images(path, images, labels):
    path.mkdir(parents=True, exist_ok=True)
    new_images = []
    new_labels = []
    for image_path, label in zip(images, labels):
        image_input = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        distorted_images = distort(image_input)
        for i, distorted_image in enumerate(distorted_images):
            new_image_path = path / f'{image_path.stem}_{i:0>3}.jpg'
            cv2.imwrite(str(new_image_path), distorted_image)
            new_images.append(new_image_path)
            new_labels.append(label)
    return new_images, new_labels


def make_labellist(path, kind, images, labels):
    path.mkdir(parents=True, exist_ok=True)
    pairs = [(image.name, label) for image, label in zip(images, labels)]
    random.shuffle(pairs)

    names = [name for name, _ in pairs]
    target = [label for _, label in pairs]
    df = pd.DataFrame({'name': names, 'target': target})
    df.to_csv(path / f'{kind}_distortion.csv', index=False, header=False)


def main():
    def pipeline(kind):

        data_path = pathlib.Path() / 'DISH_data'

        archive_path = data_path / 'raw' / 'images' / f'{kind}.tar.gz'
        images_path = data_path / 'raw' / 'images' / kind
        labels_path = data_path / 'raw' / 'labels' / f'{kind}.csv'

        images, labels = load(archive_path, images_path, labels_path)

        processed_path = data_path / 'processed'
        images, labels = make_images(processed_path / 'images' / kind, images, labels)
        make_labellist(processed_path / 'labels', kind, images, labels)

    print('Processing train data ...')
    pipeline('train')

    print('Processing test data ...')
    pipeline('test')


if __name__ == '__main__':
    main()
