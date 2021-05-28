from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
from tensorflow import keras
import numpy as np

input_img = Input(shape=(28, 28, 1))  

# fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# (x_train, _), (x_test, _) = keras.datasets.fashion_mnist

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  

print(x_test.shape)

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(128, (3,3), activation='relu', padding='same', input_shape=(28,28,1)))
model.add(keras.layers.MaxPooling2D((2,2), padding='same'))
model.add(keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D((2,2), padding='same'))
model.add(keras.layers.Conv2D(16, (3,3), activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D((2,2), padding='same'))

        # decoding layers
model.add(keras.layers.Conv2D(16, (3,3), activation='relu', padding='same'))
model.add(keras.layers.UpSampling2D((2,2)))
model.add(keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(keras.layers.UpSampling2D((2,2)))
model.add(keras.layers.Conv2D(128, (3,3), activation='relu'))
model.add(keras.layers.UpSampling2D((2,2)))
model.add(keras.layers.Conv2D(1, (3,3), activation='sigmoid', padding='same'))

# x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
# x = MaxPooling2D((2, 2), padding='same')(x)
# x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
# x = MaxPooling2D((2, 2), padding='same')(x)
# x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
# encoded = MaxPooling2D((2, 2), padding='same')(x)

# # 이 시점에서 표현(representatoin)은 (4,4,8) 즉, 128 차원

# x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
# x = UpSampling2D((2, 2))(x)
# x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
# x = UpSampling2D((2, 2))(x)
# x = Conv2D(16, (3, 3), activation='relu')(x)
# x = UpSampling2D((2, 2))(x)
# decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

# autoencoder = Model(input_img, decoded)
model.compile(optimizer='adadelta', loss='binary_crossentropy')


model.fit(x_train, x_train,
                epochs=50,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test))
                # callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])