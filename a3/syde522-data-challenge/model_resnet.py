from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras.utils.np_utils import to_categorical
from keras.applications.resnet50 import ResNet50
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

X_train = np.load(
    "drive/My Drive/Colab Notebooks/syde 522 Kaggle Challenge/train_x.npy")
Y_train = np.load(
    "drive/My Drive/Colab Notebooks/syde 522 Kaggle Challenge/train_label.npy")
X_test = np.load(
    "drive/My Drive/Colab Notebooks/syde 522 Kaggle Challenge/test_x.npy")

NUM_CLASSES = 20
EPOCHS = 10
BATCH_SIZE = 32


# Normalize
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
Y_train = to_categorical(Y_train, num_classes=NUM_CLASSES)

# Split the train and the validation set for the fitting
X_train, X_val, Y_train, Y_val = train_test_split(
    X_train, Y_train, test_size=0.2, random_state=42)

INPUT_SHAPE = X_train.shape[1:]

# Set the CNN model - Using ResNet
base = ResNet50(input_shape=INPUT_SHAPE, include_top=False, weights='imagenet')
x = base.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
pred = Dense(NUM_CLASSES, activation="softmax")(x)
model = Model(inputs=base.input, output=pred)

# Define the optimizer
opt = RMSprop(lr=0.0001, decay=1e-6)

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


# reduce the learning rate by half when the val_acc plateaus
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)

# early stop when val_loss doesnt decrease past baseline
early_stop = EarlyStopping(monitor='val_loss',
                           verbose=1,
                           mode='min',
                           baseline=0.05,
                           patience=5)

model.fit(X_train, Y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_data=(X_val, Y_val),
          shuffle=True,
          callbacks=[learning_rate_reduction, early_stop])

# Get results and write to csv
results = model.predict(X_test)
results = np.argmax(results, axis=1)
results = pd.Series(results, name='Predicted')

submission = pd.concat([pd.Series(range(0, 200), name='Id'), results], axis=1)

submission.to_csv(
    "drive/My Drive/Colab Notebooks/syde 522 Kaggle Challenge/cnn_predicted.csv", index=False)
