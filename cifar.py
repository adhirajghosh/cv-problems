# %%

from IPython.core.display import display, HTML

display(HTML("<style>.container { width:100% !important; }</style>"))

from dataLoader import importData
import tensorflow as tf
import numpy as np
import wandb
from soft_max_classifier import get_train_val_subsets
import os
from tensorflow.keras import datasets, layers, models
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

def train_model(x_train, y_train, task, num_epochs=20, batch_size=64,
                checkpoint_filepath='checkpoint/', verbose=True):
    x_train_, y_train_, x_val_, y_val_ = get_train_val_subsets(x_train, y_train)
    nClass = len(np.unique(y_train))
    num_dim = x_train.shape[1]

    # model = tf.keras.models.Sequential([
    #
    #     tf.keras.layers.InputLayer(input_shape=(num_dim,)),
    #     tf.keras.layers.Dense(3072, activation='relu'),
    #     tf.keras.layers.Dense(3072, activation='relu'),
    #     # tf.keras.layers.Dense(1000, activation = 'linear'),
    #     tf.keras.layers.Dense(nClass, activation='softmax')
    # ])
    #
    # model.compile(
    #     optimizer='adam',
    #     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    #     metrics=['accuracy'],
    # )

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32,32,3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(nClass))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    if not os.path.exists(checkpoint_filepath):
        os.makedirs(checkpoint_filepath)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        # filepath= checkpoint_filepath + '{epoch:02d}-{val_loss:.4f}.hdf5',
        filepath=checkpoint_filepath + task+ '_'+ 'best_val_acc.hdf5',
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        metric='val_accuracy',
        save_freq='epoch', verbose=verbose)

    model.fit(x_train_, y_train_, epochs=num_epochs, validation_data=(x_val_, y_val_),
              batch_size=batch_size, callbacks=[model_checkpoint_callback])

    model.load_weights(checkpoint_filepath + task+ '_'+ 'best_val_acc.hdf5')
    return model


# %%
def main(target):
    wandb.init(project="problems-with-cv", entity="adhirajghosh")
    x_train, y_train, x_test, y_test, _ = importData(6)
    model = train_model(x_train, y_train, 'cifar10')

    # scores = model.predict(x_test)
    #
    # print('deep learned accuracy')
    # lab = np.argmax(scores, axis=1)
    # deep_acc = np.mean(y_test == lab)
    # print(deep_acc)
    #
    # if target == 'fashion_mnist':
    #     x_train_, y_train_, x_test_, y_test_, _ = importData(0)
    # elif target == 'cifar10':
    #     x_train_, y_train_, x_test_, y_test_, _ = importData(6)
    # elif target == 'cifar100':
    #     x_train_, y_train_, x_test_, y_test_, _ = importData(7)
    #
    # print('deep learned auroc')
    # all_x = np.concatenate([x_test, x_test_], axis=0)
    # all_y = np.zeros(all_x.shape[0], dtype=int)
    # all_y[:x_test.shape[0]] = 1
    # scores = model.predict(all_x)
    # s = np.max(scores, axis=1)
    # deep_roc = roc_auc_score(all_y, s)
    # print(deep_roc)
    #
    # clf = make_pipeline(StandardScaler(),
    #                     SVC(random_state=0, tol=1e-5))
    # clf.fit(x_train, y_train)
    # label = clf.predict(x_test)
    # score = clf.decision_function(x_test)
    # svm_acc = np.mean(label == y_test)
    # print('SVM accuracy')
    #
    # print('accuracy:', svm_acc)
    #
    # print('svm auroc')
    #
    # score = clf.decision_function(all_x)
    # s = np.max(score, axis=1)
    # svm_roc = roc_auc_score(all_y, s)
    # print(svm_roc)
    #
    # wandb.log({
    #     "DL Accuracy": deep_acc,
    #     "DL AUROC": deep_roc,
    #     "SVM Accuracy": svm_acc,
    #     "SVM AUROC": svm_roc
    # })

if __name__ == '__main__':
    main('cifar100')

