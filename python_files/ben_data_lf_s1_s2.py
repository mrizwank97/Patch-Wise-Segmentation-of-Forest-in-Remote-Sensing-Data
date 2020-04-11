import keras
import random 
import numpy as np
import pandas as pd
from glob import glob
from skimage.io import imread
from keras import backend as K
from matplotlib import pyplot as plt
from keras.layers.merge import concatenate
from keras import regularizers, optimizers
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Input, Dense, Activation, Flatten, Dropout, BatchNormalization

def image_generator(files, batch_size=32):
    from skimage.io import imread
    from random import sample, choice
    while True:
        batch_files = sample(files, batch_size)
        batch_Y = []
        batch_s1X = []
        batch_s2X = []
        for idx, input_path in enumerate(batch_files):
            image = np.array(imread(input_path), dtype=float)
            s1 = image[:,:,10:]
            s2 = image[:,:,:10]
            temp = input_path.split('/')[-1]
            Y = list(df.loc[temp])
            batch_Y += [Y]
            batch_s1X += [s1]
            batch_s2X += [s2]
        s1X = np.array(batch_s1X)
        s2X = np.array(batch_s2X)
        Y = np.array(batch_Y)
        yield([s1X, s2X], Y)

def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2*((prec*rec)/(prec+rec+K.epsilon()))

def build_callbacks():
    checkpointer = ModelCheckpoint(filepath="../models/ben_data_lf_s1_s2.h5", monitor='val_f1', verbose=1, save_best_only=True, save_weights_only=False, mode='max')
    reduce = keras.callbacks.ReduceLROnPlateau(monitor='val_f1', factor=0.1, patience=4, mode='max')
    early = keras.callbacks.EarlyStopping(monitor='val_f1', min_delta=1e-4, patience=15, mode='max')
    csv = keras.callbacks.CSVLogger('../logs/ben_data_lf_s1_s2.csv', separator=',')
    callbacks = [checkpointer, reduce, early, csv]
    return callbacks

files = glob('/scratch/mrkhalid/ben_data'+"/**/**/**/*.tif")
print('Total no. of images ' + str(len(files)))
for i in range(100):
    random.shuffle(files)
ne = len(files)
train_files = files[:int(.7*ne)]
val_files = files[int(.7*ne):int(.85*ne)]
test_files = files[int(.85*ne):ne]
print('Training Dataset Size ' + str(len(train_files)))
print('Validation Dataset Size ' + str(len(val_files)))
print('Test Dataset Size ' + str(len(test_files)))

df = pd.read_csv(r'/scratch/mrkhalid/annotated.csv')
df['ID'] = df['ID'] + '.tif'
df.set_index("ID", inplace=True)
bs = 64
train_generator = image_generator(train_files, batch_size=bs)
val_generator = image_generator(val_files, batch_size=bs)
test_generator = image_generator(test_files, batch_size=bs)

s1_i = Input(shape=(120,120,2))
s1 = Conv2D(filters=32, kernel_size=(3,3), padding="same")(s1_i)
s1 = BatchNormalization()(s1)
s1 = Activation('relu')(s1)
s1 = Conv2D(filters=32, kernel_size=(3,3), padding="same")(s1)
s1 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(s1)
s1 = BatchNormalization()(s1)
s1 = Activation('relu')(s1)
s1 = Dropout(0.25)(s1)
s1 = Conv2D(filters=64, kernel_size=(3,3), padding="same")(s1)
s1 = BatchNormalization()(s1)
s1 = Activation('relu')(s1)
s1 = Conv2D(filters=64, kernel_size=(3,3), padding="same")(s1)
s1 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(s1)
s1 = BatchNormalization()(s1)
s1 = Activation('relu')(s1)
s1 = Dropout(0.25)(s1)
s1 = Conv2D(filters=128, kernel_size=(3,3), padding="same")(s1)
s1 = BatchNormalization()(s1)
s1 = Activation('relu')(s1)
s1 = Conv2D(filters=128, kernel_size=(3,3), padding="same")(s1)
s1 = BatchNormalization()(s1)
s1 = Activation('relu')(s1)
s2_i = Input(shape=(120,120,10))
s2 = Conv2D(filters=32, kernel_size=(3,3), padding="same")(s2_i)
s2 = BatchNormalization()(s2)
s2 = Activation('relu')(s2)
s2 = Conv2D(filters=32, kernel_size=(3,3), padding="same")(s2)
s2 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(s2)
s2 = BatchNormalization()(s2)
s2 = Activation('relu')(s2)
s2 = Dropout(0.25)(s2)
s2 = Conv2D(filters=64, kernel_size=(3,3), padding="same")(s2)
s2 = BatchNormalization()(s2)
s2 = Activation('relu')(s2)
s2 = Conv2D(filters=64, kernel_size=(3,3), padding="same")(s2)
s2 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(s2)
s2 = BatchNormalization()(s2)
s2 = Activation('relu')(s2)
s2 = Dropout(0.25)(s2)
s2 = Conv2D(filters=128, kernel_size=(3,3), padding="same")(s2)
s2 = BatchNormalization()(s2)
s2 = Activation('relu')(s2)
s2 = Conv2D(filters=128, kernel_size=(3,3), padding="same")(s2)
s2 = BatchNormalization()(s2)
s2 = Activation('relu')(s2)
x = concatenate([s1,s2])
x = Conv2D(filters=128, kernel_size=(3,3), padding="same")(x)
x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.25)(x)
x = Conv2D(filters=128, kernel_size=(3,3), padding="same")(x)
x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.25)(x)
x = Flatten()(x)
x = Dense(512, activation="relu",)(x)
x = Dense(512, activation="relu",)(x)
out = Dense(19, activation='sigmoid')(x)
model = Model(inputs=[s1_i, s2_i], outputs=out)
model.compile(optimizers.adam(), loss="binary_crossentropy", metrics=['accuracy', recall, precision, f1])

train_steps = len(train_files) // bs
val_steps = len(val_files) // bs
test_steps =len(test_files) // bs
history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=train_steps,
                    validation_data=val_generator,
                    validation_steps=val_steps,
                    epochs=100,
                    callbacks = build_callbacks(),
                    #use_multiprocessing=True,
                    #max_queue_size = 128,
                    #workers=4,
                    verbose=2
                    )

loss, acc, rec, prec, f1 = model.evaluate_generator(test_generator,steps=test_steps)
print(loss)
print(acc)
print(rec)
print(prec)
print(f1)	
