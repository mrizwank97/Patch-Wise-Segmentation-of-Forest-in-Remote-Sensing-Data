import keras
import random 
import numpy as np
import pandas as pd
from glob import glob
from skimage.io import imread
from keras import backend as K
from keras.models import Sequential
from matplotlib import pyplot as plt
from keras import regularizers, optimizers
from keras.layers import Conv2D, MaxPooling2D
from keras_preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization

def image_generator(files, batch_size=32):
    from skimage.io import imread
    from random import sample, choice
    while True:
        batch_files = sample(files, batch_size)
        batch_Y = []
        batch_X = []
        for idx, input_path in enumerate(batch_files):
            image = np.array(imread(input_path), dtype=float)[:,:,:10]
            #image[:,:,0]= (image[:,:,0]-image[:,:,0].min())/(image[:,:,0].max()-image[:,:,0].min())
            #image[:,:,1]= (image[:,:,1]-image[:,:,1].min())/(image[:,:,1].max()-image[:,:,1].min())
            temp = input_path.split('/')[-1]
            Y = list(df.loc[temp])
            batch_Y += [Y]
            batch_X += [image]
        X = np.array(batch_X)
        Y = np.array(batch_Y)
        yield(X, Y)
        
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
    checkpointer = ModelCheckpoint(filepath="../models/ben_data_vgg_s2_moa.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='max')
    reduce = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=4, mode='max')
    early = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=15, mode='max')
    csv = keras.callbacks.CSVLogger('../logs/ben_data_vgg_s2_moa.csv', separator=',')
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

model = Sequential()
model.add(BatchNormalization(input_shape=(120,120,10)))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(units=512,activation="relu"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(units=512,activation="relu"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(units=19, activation="sigmoid"))
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
                    use_multiprocessing=True,
                    max_queue_size = 128,
                    workers=4,
                    verbose=2
                    )

loss, acc, rec, prec, f1 = model.evaluate_generator(test_generator,steps=test_steps)
print(loss)
print(acc)
print(rec)
print(prec)
print(f1)
