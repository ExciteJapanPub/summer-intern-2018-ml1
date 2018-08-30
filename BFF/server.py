# Flask などの必要なライブラリをインポートする
import os

from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import numpy as np

UPLOAD_FILE_PATH = './uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif'])

# 自身の名称を app という名前でインスタンス化する
app = Flask(__name__)

# メッセージをランダムに表示するメソッド
def picked_up():
    messages = [
        "こんにちは、あなたの名前を入力してください",
        "やあ！お名前は何ですか？",
        "あなたの名前を教えてね"
    ]
    # NumPy の random.choice で配列からランダムに取り出し
    return np.random.choice(messages)

# ここからウェブアプリケーション用のルーティングを記述
# index にアクセスしたときの処理
@app.route('/')
def index():
    title = "ようこそ"
    message = picked_up()
    # index.html をレンダリングする
    return render_template('index.html',
                           message=message, title=title)

# /post にアクセスしたときの処理
@app.route('/post', methods=['GET', 'POST'])
def post():
    title = "こんにちは"
    if request.method == 'POST':
        # リクエストフォームから「名前」を取得して
        name = request.form['name']
        # index.html をレンダリングする
        return render_template('index.html',
                               name=name, title=title)
    else:
        # エラーなどでリダイレクトしたい場合はこんな感じで
        return redirect(url_for('index'))

@app.route('/upload', method=['GET', 'POST'])
def uploadpage():
    if request.method == 'GET':
        title = "upload"
        return render_template('upload.html', title=title)

    elif request.method == 'POST':
        title = "picture information"
        img_file = request.files['img_file']
        if img_file and allowed_file(img_file.filename):
            img_file.save(os.path.join(UPLOAD_FILE_PATH, img_file.filename))
            img_url = '/uploads/' + img_file.filename
            dishname, tags = get_picture_info(img_url)
            return render_template('disp.html', title=title, img_url=img_url, tag=tags, dishname=dishname)
        else:
            return ''' <p>許可されていない拡張子です</p> '''


def get_picture_info(img_url):
    user_id = 0
    path = img_url
    picture = input_pic(path, user_id)
    label = predict(picture)
    dishname =

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


if __name__ == '__main__':
    app.debug = True # デバッグモード有効化
    app.run(host='127.0.0.1') # どこからでもアクセス可能に