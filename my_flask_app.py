from flask import Flask, render_template, request, send_from_directory
from flask_uploads import UploadSet, configure_uploads, IMAGES
from time import sleep
import os

app = Flask(__name__, static_url_path='')

photos = UploadSet('photos', IMAGES)

app.config['UPLOADED_PHOTOS_DEST'] = 'C:/Users/sebys/Documents/Verified_image'
configure_uploads(app, photos)


path = 'C:/Users/sebys/Documents/python_scripts/python-licence/static/css'
@app.route('/static/css/<path:path>')
def send_css(path):
    return send_from_directory('css', path)


path = 'C:/Users/sebys/Documents/python_scripts/python-licence/static/js'
@app.route('/static/js/<path:path>')
def send_js(path):
    return send_from_directory('js', path)


@app.route('/', methods=['GET', 'POST'])
def upload(name=None, lastItem=None):
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'])
        # print(request.files['photo'])
        # print(filename)
        list_img = []
        path_out = 'C:/Users/sebys/Documents/Verified_image/output'

        if os.listdir(path_out) == []:
            sleep(30)
            for f in os.listdir(path_out):
                print('if', f)
                if f.endswith('.jpg'):
                    list_img.append(path_out + '/' + str(f))
                    print('+++++++++++++++', list_img)

                # item = list_img[len(list_img) - 1]
            return render_template('upload.html', name=list_img, lastItem=list_img[len(list_img) - 1])
        else:
            sleep(10)
            for f in os.listdir(path_out):
                print('else', f)
                if f.endswith('.jpg'):
                    list_img.append(path_out + '/' + str(f))
                    print('+++++++++++++++', list_img)

                # item = list_img[len(list_img) - 1]
            return render_template('upload.html', name=list_img, lastItem=list_img[len(list_img) - 1])

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
