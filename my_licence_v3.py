from models import get_model
from PIL import Image
from tensorflow.keras.optimizers import Adam
from time import sleep, time
from utils import get_acc2, get_stats, get_acc
import numba as nb
import numpy as np
import os
import tensorflow as tf


batch_size = 64
num_classes = 2
no_epoch = 128
init_lr = 0.001


@nb.jit(nopython=True)
def split_image(img, w, h):
    rows, cols, _ = img.shape
    k1 = int(np.floor(rows // w))
    k2 = int(np.floor(cols // h))
    data = np.zeros((k1 * k2, w, h, 3), dtype=np.float32)
    idx = - 1

    for x in range(k1):
        for y in range(k2):
            idx += 1
            data[idx, :, :, :] = img[x * w : (x + 1) * w, y * w : (y + 1) * w, :]

    return data


@nb.jit(nopython=True)
def unsplit_image(img, mat):
    r, c = mat.shape

    for i in range(r):
        for j in range(c):
            if mat[i, j] == 0:
                img[i * 64 : (i + 1) * 64, j * 64 : (j + 1) * 64, 2] = 0.0
    return img


def proc_image(input_path, output_path):
    # get_acc(model, test_folder, w=64, h=64, rescale=False)

    img = np.array(Image.open(input_path), dtype=np.float32)
    w, h, _ = img.shape
    r, c = w // 64, h // 64
    # s1 = time()
    data = split_image(img / 255.0, 64, 64)
    # s2 = time() - s1
    # print('CPU - 1 -> Duration:', s2)
    # s1 = time()
    mat = np.argmax(model.predict(data), axis=1).reshape((r, c))
    # s2 = time() - s1
    # print('GPU -> Duration:', s2)
    # img = np.uint8(img)
    # s1 = time()
    new_image = Image.fromarray(np.uint8(unsplit_image(img, mat)), 'RGB')
    # s2 = time() - s1
    # print('CPU - 2 -> Duration:', s2)
    new_image.save(output_path)


if __name__ == '__main__':
    with tf.device('/gpu:0'):
        path_checkpoint = 'C:/Users/sebys/Documents/python_scripts/python-licence/checkpoint-126.hdf5'
        model = get_model(4, 64, 64, 3, num_classes)
        opt = Adam(lr=init_lr, decay=init_lr / no_epoch)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        model.summary()
        model.load_weights(path_checkpoint)
        list_img = []
        a = 0
        print('Watching folder...')

        while True:
            path = 'C:/Users/sebys/Documents/Verified_image'
            folder_name = '/output'

            if not os.path.isdir(path + folder_name):
                os.mkdir(path + folder_name)

            for f in os.listdir(path):
                if f.endswith(('.png', '.jpg', '.bmp')) and not any(f in i for i in list_img):
                    a += 1
                    list_img.append(f)
                    # print(list_img)
                    input_path = path + '/' + f
                    # output_path = path + folder_name + '/' + f[: - 4] + '_out' + '.png'
                    output_path = path + folder_name + '/' + f'image{a}' + '.jpg'
                    print('==>', input_path, '==>', output_path)
                    start_time = time()
                    proc_image(input_path, output_path)
                    elapsed_time = time() - start_time
                    print('Processed image is ready! ==> Duration:', str(np.floor(elapsed_time)), 'seconds')

            sleep(1)
