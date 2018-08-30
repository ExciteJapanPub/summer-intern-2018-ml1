# Flask などの必要なライブラリをインポートする
import os

from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import numpy as np
import pandas as pd
from best_friend_forever import input_pic, predict

UPLOAD_FILE_PATH = './static/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif'])

# class dic
LABELS_PATH = './dataset/meta/classes.txt'
CLASS_NUM = 101

class_df = pd.read_table(LABELS_PATH, header=None)
class_df = class_df.rename(columns={0: 'class'})
class_df['label'] = [i for i in range(CLASS_NUM)]
class_df = class_df.set_index('label')

# 自身の名称を app という名前でインスタンス化する
app = Flask(__name__)

# ここからウェブアプリケーション用のルーティングを記述
@app.route('/send', methods=['GET', 'POST'])
def send():
    if request.method == 'GET':
        title = "upload"
        return render_template('upload.html', title=title)

    elif request.method == 'POST':
        title = "picture information"
        img_file = request.files['img_file']
        if img_file and allowed_file(img_file.filename):
            img_file.save(os.path.join(UPLOAD_FILE_PATH, img_file.filename))
            img_url = './static/uploads/' + img_file.filename
            dishname = get_picture_info(img_url)
            return render_template('disp.html', title=title, img_url=img_url, dishname=dishname)
        else:
            return ''' <p>許可されていない拡張子です</p> '''

def get_picture_info(img_url):
    user_id = 0
    path = img_url
    picture = input_pic(path, user_id)
    label = predict(picture)
    dishname = class_df.at[label, 'class']

    return dishname

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


if __name__ == '__main__':
    print(app.url_map)
    app.debug = True # デバッグモード有効化
    app.run(host='127.0.0.1') # どこからでもアクセス可能に