from models import get_model
from tensorflow.keras.optimizers import Adam
from utils import get_acc2, get_stats, get_acc
import tensorflow as tf

batch_size = 64
num_classes = 2
no_epoch = 20
init_lr = 0.001

if __name__ == '__main__':
    model = get_model(4, 64, 64, 3, num_classes)
    opt = Adam(lr=init_lr, decay=init_lr / no_epoch)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.load_weights('/gdrive/My Drive/Colab Notebooks/checkpoints/checkpoint-118.hdf5')
    model.summary()
    get_acc(model, '/content/datasetlice/test', w=64, h=64, rescale=False)
