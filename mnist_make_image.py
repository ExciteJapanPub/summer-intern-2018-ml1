"""
参考
https://gist.github.com/xkumiyu/c93222f2dce615f4b264a9e71f6d49e0
"""

import gzip
import pathlib
import struct

import numpy as np
import pandas as pd
from PIL import Image


def load(x_path, y_path):
    with gzip.open(x_path) as fx, gzip.open(y_path) as fy:
        fx.read(4)
        fy.read(4)
        N, = struct.unpack('>i', fy.read(4))
        if N != struct.unpack('>i', fx.read(4))[0]:
            raise RuntimeError('wrong pair of MNIST images and labels')
        fx.read(8)

        images = np.empty((N, 784), dtype=np.uint8)
        labels = np.empty(N, dtype=np.uint8)

        for i in range(N):
            labels[i] = ord(fy.read(1))
            for j in range(784):
                images[i, j] = ord(fx.read(1))
    return images, labels


def make_images(path, images, labels):
    path.mkdir(parents=True, exist_ok=True)
    for (i, image), label in zip(enumerate(images), labels):
        filepath = path / '{}_{}.jpg'.format(label, i)
        Image.fromarray(image.reshape(28, 28)).save(filepath)


def make_labellist(path, kind, labels):
    path.mkdir(parents=True, exist_ok=True)
    filepaths = [
        '{}_{}.jpg'.format(label, i) for i, label in enumerate(labels)
    ]
    df = pd.DataFrame({'name': filepaths, 'target': labels.tolist()})
    df.to_csv(path / '{}.csv'.format(kind), index=False, header=False)


def main():
    def pipeline(kind):
        _kind = kind
        if kind == 'test':
            _kind = 't10k'

        data_path = pathlib.Path() / 'MNIST_data'

        images_path = data_path / 'raw' / f'{_kind}-images-idx3-ubyte.gz'
        labels_path = data_path / 'raw' / f'{_kind}-labels-idx1-ubyte.gz'

        images, labels = load(images_path, labels_path)

        processed_path = data_path / 'processed'
        make_images(processed_path / 'images' / kind, images, labels)
        make_labellist(processed_path / 'labels', kind, labels)

    print('Processing train data ...')
    pipeline('train')

    print('Processing test data ...')
    pipeline('test')


if __name__ == '__main__':
    main()
