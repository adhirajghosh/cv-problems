import numpy as np
import tensorflow as tf 
from scipy.io import loadmat

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import GlobalAvgPool2D


def resNet_conversion(images, batch_size=512):
    model = Sequential()    
    resNet = ResNet50(weights='imagenet', include_top=False)
    model.add(resNet)
    model.add(GlobalAvgPool2D())
    
    is_bw = False
    if len(images.shape) ==3:
        is_bw = True
    total = images.shape[0]
    features = []
    print('total:', total)
    for i in range(0, total, batch_size):
        print(i, end=', ')
        im = images[i:i+batch_size]
        if is_bw:
            im  = np.repeat(im[:,:,:,None], 3, axis=3)

        im_ = tf.image.resize(im, [224,224])
        im_ = preprocess_input(im_)
        feat = model.predict(im_)
        features.append(np.array(feat))
    features = np.concatenate(features, axis=0)
    return features


def get_svnh():
    train = loadmat('../Data/svnh_numbers/train_32x32.mat')
    x_train = train['X']
    y_train = np.squeeze(train['y']) -1

    test = loadmat('../Data/svnh_numbers/test_32x32.mat')
    x_test = test['X']
    y_test = np.squeeze(test['y']) -1

    x_train = np.einsum('ijkl->lijk', x_train)
    x_test = np.einsum('ijkl->lijk', x_test)

    return x_train, y_train, x_test, y_test

def get_mnist():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    return x_train, y_train, x_test, y_test
