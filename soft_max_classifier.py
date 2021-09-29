import numpy as np
import tensorflow as tf
import os 

def train_model(x_train, y_train, num_epochs=20, batch_size=64, 
                checkpoint_filepath = 'checkpoint/', verbose=True):
    x_train_, y_train_, x_val_, y_val_ = get_train_val_subsets(x_train, y_train)
    nClass = len(np.unique(y_train))
    num_dim = x_train.shape[1]

    model = tf.keras.models.Sequential([
    
    tf.keras.layers.InputLayer(input_shape=(num_dim,)),
    #tf.keras.layers.Dense(1000, activation = 'relu'),
    #tf.keras.layers.Dense(1000, activation = 'linear'),
    tf.keras.layers.Dense(nClass, activation = 'softmax')
    ])


    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy'],
    )

    if not os.path.exists(checkpoint_filepath):
        os.makedirs(checkpoint_filepath)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        #filepath= checkpoint_filepath + '{epoch:02d}-{val_loss:.4f}.hdf5',
        filepath = checkpoint_filepath+'best_val_acc.hdf5',
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        metric = 'val_accuracy',
        save_freq = 'epoch', verbose = verbose)

    model.fit(x_train_, y_train_, epochs=num_epochs, validation_data=(x_val_, y_val_), 
        batch_size = batch_size, callbacks=[model_checkpoint_callback])

    model.load_weights(checkpoint_filepath + 'best_val_acc.hdf5')
    return model


def get_num_per_class(trainGt):
    class_labels = np.unique(trainGt)
    #num_each_class = np.zeros((len(class_labels), 1), dtype=int) 
    num_each_class = []
    print("class labels ", class_labels)
    for i in class_labels:
        temp = (trainGt==i)
        num_each_class.append(np.count_nonzero(temp))

    return num_each_class    

def get_train_val_subsets(trainFeat, trainGt):
    num_each_class = get_num_per_class(trainGt)
    class_labels = np.unique(trainGt)

    start_index = 0

    train_set = []
    val_set = []
    label_train = []
    label_val = []

    for i in range(len(class_labels)):
        #print(num_each_class[i])
        num_val = int(0.1*num_each_class[i])
        #print("i ",i, "num_val ",num_val)

        end_index = start_index+num_each_class[i]

        #print("curr num ", num_each_class[i], "start ", start_index, "end ", end_index)

        curr_trainfeat = trainFeat[start_index:end_index-num_val,:]
        curr_valfeat = trainFeat[end_index-num_val:end_index,:]
        curr_trainGt = trainGt[start_index:end_index-num_val]
        curr_valGt = trainGt[end_index-num_val:end_index]

        start_index += num_each_class[i]
        # print("shape curr_train ", curr_trainfeat.shape)
        # print("shape curr_val ", curr_valfeat.shape)
        # print(curr_valGt)

        train_set.append(curr_trainfeat)
        val_set.append(curr_valfeat)
        label_train.append(curr_trainGt)
        label_val.append(curr_valGt)

    x_train = np.concatenate(train_set, axis=0)
    y_train = np.concatenate(label_train)
    x_val = np.concatenate(val_set, axis=0)
    y_val = np.concatenate(label_val)

    return x_train, y_train, x_val, y_val

