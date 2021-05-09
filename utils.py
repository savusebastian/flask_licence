from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.utils import to_categorical
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os


def load_data(filename):
    data = np.load(filename)
    x = data['XT'].astype('float32')
    y = data['XN'].astype('float32')
    return x, y


def gen_data(path, opt=0, filename=''):
    image_categories = [o for o in os.listdir(path) if os.path.isdir(os.path.join(path, o))]
    data = []
    labels = []

    for category in image_categories:
        no_images = 0
        new_path = os.path.join(path, category)
        list_path_images = [os.path.join(new_path, f) for f in os.listdir(new_path)]

        for image_path in list_path_images:
            im = Image.open(image_path)
            #im = im.resize((100, 100), Image.ANTIALIAS)
            img = np.array(im)
            labels.append(image_categories.index(category))
            data.append(img)
            no_images += 1

        print('%i images processed in category %s' % (no_images, category))

    x = np.array(data, dtype=np.float32) / 255.0
    y = np.array(labels)

    if opt == 1:
        idx = np.arange(0, x.shape[0])
        np.random.shuffle(idx)
        x = x[idx, :, :, :]
        y = y[idx]

    y = to_categorical(y, num_classes=len(image_categories))
    np.savez(filename, XT=x, XN=y)
    print('INFO: data saved')

def get_stats(model, path, w=200, h=200, rescale=False):
    image_categories = [o for o in os.listdir(path) if os.path.isdir(os.path.join(path, o))]
    list_imag = []
    list_true_cat = []

    for category in image_categories:
        print('start processing category %s' % category)
        new_path = os.path.join(path, category)
        list_path_images = [os.path.join(new_path, f) for f in os.listdir(new_path)]

        for image_path in list_path_images:
            im = Image.open(image_path)

            if rescale == True:
                im = im.resize((w, h), Image.ANTIALIAS)

            list_imag.append(np.array(im, dtype=np.float32) / 255.0)
            list_true_cat.append(image_categories.index(category))

    list_imag=np.array(list_imag)
    list_true_cat = np.array(list_true_cat)
    print('start prediction')
    list_pred_cat = np.argmax(model.predict(list_imag), axis=1)
    conf_mat = confusion_matrix(list_true_cat, list_pred_cat)
    print(conf_mat)
    print(classification_report(list_true_cat, list_pred_cat, target_names=image_categories))
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(conf_mat, classes=image_categories, title='Confusion matrix, without normalization')
    #normalized
    #np.set_printoptions(precision=2)
    #plot_confusion_matrix(conf_mat, classes=image_categories, normalize = True, title='Confusion matrix, with normalization')

    plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    # This function prints and plots the confusion matrix.
    # Normalization can be applied by setting `normalize=True`.

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def get_acc2(model, path, w=200, h=200, rescale=False):
    image_categories = [o for o in os.listdir(path) if os.path.isdir(os.path.join(path, o))]
    conf_mat = np.zeros((len(image_categories), len(image_categories)), np.int32)
    data = np.zeros((1, w, h, 3), dtype=np.float32)
    list_true_cat =[]
    list_pred_cat =[]

    for category in image_categories:
        print('start processing category %s with index %i' % (category,image_categories.index(category)))
        new_path = os.path.join(path, category)
        list_path_images = [os.path.join(new_path, f) for f in os.listdir(new_path)]

        for image_path in list_path_images:
            im = Image.open(image_path)

            if rescale == True:
                im = im.resize((w, h), Image.ANTIALIAS)

            cat_1 = image_categories.index(category)
            data[0, :, :, :] = np.array(im, dtype=np.float32) / 255.0
            cat_2 = np.argmax(model.predict(data))

            if cat_1 != cat_2:
                print('%d %d' % (cat_1, cat_2))


def get_acc(model, path, w=200, h=200, rescale=False):
    image_categories = [o for o in os.listdir(path) if os.path.isdir(os.path.join(path, o))]
    print(image_categories)
    conf_mat = np.zeros((len(image_categories), len(image_categories)), np.int32)
    data = np.zeros((1, w, h, 3), dtype=np.float32)
    list_true_cat =[]
    list_pred_cat =[]

    for category in image_categories:
        print('start processing category %s with index %i' % (category, image_categories.index(category)))
        new_path = os.path.join(path, category)
        list_path_images = [os.path.join(new_path, f) for f in os.listdir(new_path)]

        for image_path in list_path_images:
            im = Image.open(image_path)

            if rescale == True:
                im = im.resize((w, h), Image.ANTIALIAS)

            data[0, :, :, :] = np.array(im, dtype=np.float32) / 255.0
            # print('%s %i' % (category, image_categories.index(category)))
            conf_mat[image_categories.index(category), np.argmax(model.predict(data))] +=1
            list_true_cat.append(image_categories.index(category))
            list_pred_cat.append(np.argmax(model.predict(data)))
    list_true_cat = np.array(list_true_cat)
    list_pred_cat = np.array(list_pred_cat)

    #conf_mat = confusion_matrix(list_true_cat, list_pred_cat)
    print(conf_mat)
    print(classification_report(list_true_cat, list_pred_cat, target_names=image_categories))
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(conf_mat, classes=image_categories, title='Confusion matrix, without normalization')
    # normalized
    # np.set_printoptions(precision=2)
    # plot_confusion_matrix(conf_mat, classes=image_categories, normalize = True, title='Confusion matrix, with normalization')
    plt.show()


if __name__ == '__main__':
    FULL_PATH = 'E:/PythonProjects/datasets/Dataset_4Set2016'
    #FULL_PATH = 'E:/PythonProjects/datasets/TST2'
    gen_data(FULL_PATH+'/train',1,'./patches/train3.npz')
    gen_data(FULL_PATH+'/test',0,'./patches/test3.npz')
