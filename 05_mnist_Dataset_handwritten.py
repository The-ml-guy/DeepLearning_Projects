#mnist dataset using Tensorflow keras and 

import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten

#used to loaddata which is already kept by keras 
(X_train,y_train),(X_test,y_test) = keras.datasets.mnist.load_data()

X_train
#to check the shape of X_train
X_train.shape
#to check the shape of Y_train
y_train.shape

import matplotlib.pyplot as plt
plt.imshow(X_train[0])

X_train = X_train.astype("float32")/255.0
X_test = X_test.astype("float32")/255.0

X_train

# we will start with a model which will be a keras sequestial model
model = Sequential()

model.add(Flatten(input_shape=(28,28)))
model.add(Dense(128,activation='relu'))
model.add(Dense(10,activation='softmax'))

#to get the summeary of the model that what are the thigs sused in the model
model.summary()

#now we will compile the model with out one hot encode for Spase_categorical_crossentropy, 
#or if we use the categorial_crossentropy then we ahve to use the onehot encode

model.compile(loss='sparse_categorical_crossentropy',optimizer='Adam')

model.fit(X_train,y_train, epochs=10, validation_split=0.2)

#poch 1/10
# 1500/1500 ━━━━━━━━━━━━━━━━━━━━ 6s 4ms/step - loss: 0.2902 - val_loss: 0.1637
# Epoch 2/10
# 1500/1500 ━━━━━━━━━━━━━━━━━━━━ 7s 5ms/step - loss: 0.1286 - val_loss: 0.1130
# Epoch 3/10
# 1500/1500 ━━━━━━━━━━━━━━━━━━━━ 6s 4ms/step - loss: 0.0882 - val_loss: 0.1035
# Epoch 4/10
# 1500/1500 ━━━━━━━━━━━━━━━━━━━━ 7s 5ms/step - loss: 0.0650 - val_loss: 0.0903
# Epoch 5/10
# 1500/1500 ━━━━━━━━━━━━━━━━━━━━ 5s 4ms/step - loss: 0.0506 - val_loss: 0.0905
# Epoch 6/10
# 1500/1500 ━━━━━━━━━━━━━━━━━━━━ 7s 4ms/step - loss: 0.0397 - val_loss: 0.0827
# Epoch 7/10
# 1500/1500 ━━━━━━━━━━━━━━━━━━━━ 6s 4ms/step - loss: 0.0306 - val_loss: 0.0874
# Epoch 8/10
# 1500/1500 ━━━━━━━━━━━━━━━━━━━━ 7s 5ms/step - loss: 0.0246 - val_loss: 0.0941
# Epoch 9/10
# 1500/1500 ━━━━━━━━━━━━━━━━━━━━ 9s 4ms/step - loss: 0.0201 - val_loss: 0.0908
# Epoch 10/10
# 1500/1500 ━━━━━━━━━━━━━━━━━━━━ 7s 4ms/step - loss: 0.0172 - val_loss: 0.0939
y_prob = model.predict(X_test)

y_pred = y_prob.argmax(axis=1)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

#And the accuracy is very good around 0.9778 or (97.7 %)
