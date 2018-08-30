# Flask などの必要なライブラリをインポートする
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import best_friend_forever as bff

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




@app.route('/upload')
def disp_uploadpage():
    title = "upload"
    return render_template('upload.html', title=title)


if __name__ == '__main__':
    app.debug = True # デバッグモード有効化
    app.run(host='127.0.0.1') # どこからでもアクセス可能に