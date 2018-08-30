# Flask などの必要なライブラリをインポートする
import os

from flask import Flask, render_template, request, redirect, url_for, send_from_directory
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
@app.route('/')
def index():
    send_path = url_for("send", _external=True)
    ranking_path = url_for("ranking", _external=True)
    return render_template('index.html', send_path=send_path, ranking_path=ranking_path)

@app.route('/ranking', methods=['GET', 'POST'])
def ranking():
    title = "your BFF"
    friend_list = [2, 4, 6, 5, 7, 8, 10, 22, 88]
    friend_similarity = [0.95, 0.94, 0.93, 0.90, 0.88, 0.74, 0.55, 0.43, 0.21]
    user_name = 0

    if request.method == 'GET':
        return render_template('ranking.html', title=title)

    if request.method == 'POST':
        number_bff = request.form['number_BFF']
        if int(number_bff) > len(friend_list):
            return redirect(url_for('ranking'))
        return render_template('ranking.html',
                               title=title, user_name=user_name, friend_list=friend_list,
                               friend_similarity=friend_similarity, number_BFF=int(number_bff))
    else:
        return redirect(url_for('ranking'))

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