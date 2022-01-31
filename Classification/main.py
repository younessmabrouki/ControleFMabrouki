import os
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from keras_preprocessing.image import ImageDataGenerator

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def analyse_data(path):
    total = 0
    for subpath, directories, files in os.walk(path):
        if len(files):
            print('Classe {} ; nb d exemples {}'.format(subpath, len(files)))
            total += len(files)
        else:
            print('nb classes: {}'.format(len(directories)))
    print('Nombre total d exemples : {}'.format(total))


def create_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(6, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


def load_generators(path, batch_size, ratio):
    data_generator = ImageDataGenerator(rescale=1. / 255, validation_split=ratio)
    train_generator = data_generator.flow_from_directory(
        directory=path,
        target_size=(200, 200), shuffle=False, class_mode='categorical',
        batch_size=batch_size, subset='training'
    )
    validation_generator = data_generator.flow_from_directory(
        directory=path,
        target_size=(200, 200), shuffle=False, class_mode='categorical',
        batch_size=batch_size, subset='validation'
    )
    return train_generator, validation_generator


def main():
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print(tf.config.list_physical_devices('GPU'))

    path = "./data/dataClassif"

    input_shape = (200, 200, 3)
    model = create_model(input_shape)

    validation_ratio = 0.25
    batch_size = 32
    samples = 2379
    nb_validation_samples = samples * validation_ratio
    nb_train_samples = samples - nb_validation_samples

    train_generator, validation_generator = load_generators(path, batch_size, validation_ratio)
    history = model.fit(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=4,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size
    )

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.legend(['train', 'test'])
    plt.show()


if __name__ == '_main_':
    main()