from models import get_model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from utils import load_data
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


FULL_PATH = '/content/datasetlice'

batch_size = 64
num_classes = 2
no_epoch = 130
init_lr = 0.0001

if __name__ == '__main__':
    aug1 = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    train_generator = aug1.flow_from_directory(
        directory=FULL_PATH + '/train',
        target_size=(64, 64),
        color_mode='rgb',
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=42)

    valid_generator = aug1.flow_from_directory(
        directory=FULL_PATH + '/test',
        target_size=(64, 64),
        color_mode='rgb',
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=42)

    # initialize the model
    print('[INFO] compiling model...')

    model = get_model(4, 64, 64, 3, num_classes)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
    early_stopper = EarlyStopping(min_delta=0.001, patience=10)

    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])
    model.summary()

    filepath = '/gdrive/My Drive/Colab Notebooks/checkpoints/' + 'checkpoint-{epoch:02d}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, mode='max')
    csv_logger = CSVLogger('/gdrive/My Drive/Colab Notebooks/log-data-drona.csv', append=True, separator=',')
    callbacks_list = [checkpoint, csv_logger, lr_reducer,early_stopper]
    # train the network
    print('[INFO] training network...')

    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
    STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size
    H = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=STEP_SIZE_TRAIN,
        validation_data=valid_generator,
        validation_steps=STEP_SIZE_VALID,
        epochs=no_epoch,
        verbose=1,
        callbacks=callbacks_list)
    # save the model to disk
    print('[INFO] serializing network...')

    # plot the training loss and accuracy
    plt.style.use('ggplot')
    plt.figure()
    N = len(H.history['loss'])

    plt.figure()
    plt.plot(np.arange(0, N), H.history['acc'], label='train_acc')
    plt.plot(np.arange(0, N), H.history['val_acc'], label='val_acc')
    plt.title('Training Loss and Accuracy')
    plt.xlabel('Epoch #')
    plt.ylabel('Loss/Accuracy')
    plt.legend(loc='lower left')
    # plt.savefig('plot')

    plt.figure()
    plt.plot(np.arange(0, N), H.history['loss'], label='train_loss')
    plt.plot(np.arange(0, N), H.history['val_loss'], label='val_loss')
    plt.title('Validation Loss and Accuracy')
    plt.xlabel('Epoch #')
    plt.ylabel('Loss/Accuracy')
    plt.legend(loc='lower left')
    # plt.savefig('plot')
