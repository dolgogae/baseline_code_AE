import tensorflow as tf
from tensorflow import keras

class AE_models():
    def __init__(self, model, input_shape, **kwargs):

        if model == 'basic_ae':
            print("Loading basic AE model...")
            self.input_shape = (28,28,1)
            self.model = self.basic()

        self.model.summary()
        self.model.compile(optimizer='adadelta', loss="binary_crossentropy")

    def basic(self):
        model = keras.models.Sequential()
        # model.add(keras.layers.Flatten(input_shape=self.input_shape))
        # encoding layers
        model.add(keras.layers.Conv2D(128, (3,3), activation='relu', padding='same', input_shape=self.input_shape))
        model.add(keras.layers.MaxPooling2D((2,2), padding='same'))
        model.add(keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'))
        model.add(keras.layers.MaxPooling2D((2,2), padding='same'))
        model.add(keras.layers.Conv2D(16, (3,3), activation='relu', padding='same'))
        model.add(keras.layers.MaxPooling2D((2,2), padding='same'))
        model.add(keras.layers.Conv2D(16, (3,3), activation='relu', padding='same'))
        model.add(keras.layers.MaxPooling2D((2,2), padding='same'))

        # decoding layers
        model.add(keras.layers.Conv2D(16, (3,3), activation='relu', padding='same'))
        model.add(keras.layers.UpSampling2D((2,2)))
        model.add(keras.layers.Conv2D(16, (3,3), activation='relu', padding='same'))
        model.add(keras.layers.UpSampling2D((2,2)))
        model.add(keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'))
        model.add(keras.layers.UpSampling2D((2,2)))
        model.add(keras.layers.Conv2D(128, (3,3), activation='relu'))
        model.add(keras.layers.UpSampling2D((2,2)))
        model.add(keras.layers.Conv2D(1, (3,3), activation='sigmoid', padding='same'))

        return model