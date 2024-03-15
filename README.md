# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset

## Neural Network Model

![image](https://github.com/aldrinlijo04/mnist-classification/assets/118544279/eef099d4-ccf0-4148-8d61-3cbe8c06ac37)


## DESIGN STEPS

### Step01 : Import Libraries:

tensorflow as tf (or tensorflow.keras for a higher-level API)

### Step02 : Load and Preprocess Data:

Use tf.keras.datasets.mnist.load_data() to get training and testing data.
Normalize pixel values (e.g., divide by 255) for better training.
Consider one-hot encoding labels for multi-class classification.

### Step03 : Define Model Architecture:

Use a sequential model (tf.keras.Sequential).
Start with a Convolutional layer (e.g., Conv2D) with filters and kernel size.
Add pooling layers (e.g., MaxPooling2D) for dimensionality reduction.
Repeat Conv2D and MaxPooling for feature extraction (optional).
Flatten the output from the convolutional layers.
Add Dense layers (e.g., Dense) with neurons for classification.
Use appropriate activation functions (e.g., ReLU) and output activation (e.g., softmax for 10 classes).

### Step04 : Compile the Model:

Specify optimizer (e.g., Adam), loss function (e.g., categorical_crossentropy), and metrics (e.g., accuracy).

### Step05 : Train the Model:

Use model.fit(X_train, y_train, epochs=...) to train.
Provide validation data (X_test, y_test) for monitoring performance.

### Step06 : Evaluate the Model:

Use model.evaluate(X_test, y_test) to assess accuracy and other metrics.

## PROGRAM

### Name: ALDRIN LIJO J E
### Register Number:212222240007

#### PREPROCESSING
```py
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten
from tensorflow.keras.metrics import CategoricalCrossentropy
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
```
#### DATA LOADING AND PREPROCESSING
```py
(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train.shape
test = x_train[4]
plt.imshow(test,cmap='gray')
x_train.max()
x_train.min()
X_train = x_train/255.0
X_test = x_test/255.0
X_train.max(),X_train.min()
X_test.max(),X_test.min()
X_train.shape
y_train[0]
from tensorflow.keras import utils
y_train = utils.to_categorical(y_train, 10)
y_test_scaled = utils.to_categorical(y_test, 10)
y_train.shape
img = X_train[23]
plt.imshow(img)

X_train = X_train.reshape(-1,28,28,1)
X_test  = X_test.reshape(-1,28,28,1)
type(y_train)
X_train.shape
```
#### MODEL ARCHITECTURE
```py
model = Sequential()
model.add(Conv2D(32,(3,3),activation="relu",input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(16,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(16,activation="relu"))
model.add(Dense(12,activation='relu'))
model.add(Dense(10,activation="softmax"))

model.summary()
model.compile('adam', loss ='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train,y_train,epochs=20,validation_data=(X_test,y_test))
metrics = pd.DataFrame(model.history.history)
metrics.head()
metrics[['loss','val_loss']].plot()
metrics[['accuracy','val_accuracy']].plot()
pred = np.argmax(model.predict(X_test),axis=1)
pred[0:10]
y_test[0:10]
print(type(y_test))
print(type(pred))
y_test = y_test.ravel()
pred=pred.ravel()
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
```
#### PREDICTION
```py
img = image.load_img('/content/7.jpg')
type(img)
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0
x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)
print(x_single_prediction)
plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')
img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0

x_single_prediction = np.argmax(
    model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),
     axis=1)
print(x_single_prediction)
print("ALDRIN LIJO")
print("REG NO:212222240007")
```
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![download](https://github.com/aldrinlijo04/mnist-classification/assets/118544279/0b53ec61-cf83-47af-995c-6b292dc3481e)


### Classification Report

![image](https://github.com/aldrinlijo04/mnist-classification/assets/118544279/64d8fdff-c521-4110-b975-4ea53c88be9f)


### Confusion Matrix

![image](https://github.com/aldrinlijo04/mnist-classification/assets/118544279/6811b825-3397-4dec-b5e6-d706b5d48322)


### New Sample Data Prediction

#### INPUT:
![7](https://github.com/aldrinlijo04/mnist-classification/assets/118544279/b4d5ba59-c4eb-427d-ac85-2db360e52ac3)
#### OUTPUT:

![image](https://github.com/aldrinlijo04/mnist-classification/assets/118544279/78c59b88-7817-44e3-8147-217b6f9395f4)
![image](https://github.com/aldrinlijo04/mnist-classification/assets/118544279/2e63912c-f88c-4306-8eb2-fbd4354cc1a5)
![download](https://github.com/aldrinlijo04/mnist-classification/assets/118544279/61a4943e-0cf4-4509-b4d3-5ad0eeb9a851)

## RESULT
Thus, a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is written and executed successfully.
