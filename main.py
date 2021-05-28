import tensorflow as tf
from tensorflow import keras
import numpy as np
# from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from ae_model import AE_models
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from keras.datasets import mnist


img_size = 28
fashion_mnist = keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()#fashion_mnist.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

# print(y_train.shape)

X_train = np.reshape(X_train, (len(X_train), img_size, img_size, 1))
X_test = np.reshape(X_test, (len(X_test), img_size, img_size, 1))
# X_train = X_train.reshape(-1, img_size, img_size, 1)
X_noise = X_train + np.random.normal(loc=.0, scale=.5, size=X_train.shape)

def train(model, train_model):
    checkpoint_cb = keras.callbacks.ModelCheckpoint('./weights/weights.hdf5', 
                                                    save_best_only=False,
                                                    monitorsss='val_loss')
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, 
                                                    restore_best_weights=False,
                                                    monitor='val_loss')
    tensorboard_cb = TensorBoard(log_dir='./autoencoder')

    print("######################")
    print("# start training...  #")
    print("######################")

    train_model.model.fit(X_train, X_train, 
                        epochs=300, 
                        batch_size=128,
                        shuffle=True,
                        validation_data=(X_test, X_test),
                        callbacks=[tensorboard_cb, checkpoint_cb, early_stopping_cb])

    train_model.model.save_weights("./weights/weights.h5")

def view(train_model):
    train_model.model.load_weights("./weights/weights.h5")
    decoded_imgs = train_model.model.predict(X_test)

    n=10
    plt.figure(figsize=(20,4))
    for i in range(1,n+1):
        ax = plt.subplot(2, n, i)
        plt.imshow(X_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
         # 재구성본 출력
        ax = plt.subplot(2, n, i + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

def main():
    model = "basic_ae"
    image_shape = X_train.shape
    train_model = AE_models(model, image_shape)

    train(model, train_model)

    view(train_model)



if __name__ == '__main__':
    main()