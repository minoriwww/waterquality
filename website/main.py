# -*- coding: utf-8 -*-
import os
from flask import Flask, request, url_for, send_from_directory, g
from werkzeug import secure_filename
from flask import render_template
from flask import make_response, redirect
from predict import train_and_predict

ALLOWED_EXTENSIONS = set(['png','PNG', 'jpg', 'JPG','jpeg','JPEG', 'gif', 'pdf'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.getcwd()
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


@app.before_request
def before_request():
    g.file_url = None


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

@app.route('/resetimage', methods=['GET', 'POST'])
def resetimage():
    resp = make_response(redirect(url_for("endpoint", imgurl='')))
    resp.set_cookie('imgurl', '')
    resp.set_cookie('prediction', '')
    # resp.set_cookie('username', 'the username')
    return resp

@app.route("/")
def endpoint(imgurl='', prediction=-1):
    imgurl = request.cookies.get('imgurl')
    prediction = request.cookies.get('prediction')
    return make_response(render_template("index.html", imgurl=imgurl, prediction=prediction))


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file_url = url_for('uploaded_file', filename=filename)
            g.file_url = file_url
            print(g.file_url)

            resp = make_response(render_template("index.html", imgurl=file_url))
            resp.set_cookie('imgurl', file_url)
            # return html + '<br><img src=' + file_url + '>'
            return resp
    return make_response(render_template("index.html", imgurl=''))


@app.route('/calculating', methods=['GET', 'POST'])
def calculating():
    print("in calculating")
    print(g.file_url)
    imgurl = request.cookies.get('imgurl')
    prediction = str(train_and_predict(data_path=imgurl))
    # resp = make_response(render_template("index.html", prediction=1))
    response = make_response(redirect(url_for("endpoint", prediction=1)))

    response.set_cookie('prediction', prediction)
    print(imgurl)

    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True,threaded=True)
