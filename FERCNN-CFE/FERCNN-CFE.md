```python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os,cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pylab import rcParams
rcParams['figure.figsize'] = 20, 10

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import keras

from keras.utils import np_utils
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from keras.models import Sequential
from keras.layers import Dense , Activation , Dropout ,Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.metrics import categorical_accuracy
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
from keras.optimizers import *
from keras.layers.normalization import BatchNormalization
import os
print(os.listdir("./input/CFE"))
```

    ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    


```python
data_path = './input/CFE'
data_dir_list = os.listdir(data_path)

img_rows=256
img_cols=256
num_channel=1

num_epoch=10

img_data_list=[]


for dataset in data_dir_list:
    img_list=os.listdir(data_path+'/'+ dataset)
    print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
    for img in img_list:
        input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
        #input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        input_img_resize=cv2.resize(input_img,(48,48))
        img_data_list.append(input_img_resize)
        
img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data = img_data/255
img_data.shape
```

    Loaded the images of dataset-angry
    
    Loaded the images of dataset-disgust
    
    Loaded the images of dataset-fear
    
    Loaded the images of dataset-happy
    
    Loaded the images of dataset-neutral
    
    Loaded the images of dataset-sad
    
    Loaded the images of dataset-surprise
    
    




    (1610, 48, 48, 3)




```python
num_classes = 7

num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')

labels[0:229]=0 #230
labels[230:459]=1 #230
labels[460:689]=2 #230
labels[690:919]=3 #230
labels[920:1149]=4 #230
labels[1150:1379]=5 #230
labels[1380:1610]=6 #230

names = ['anger','disgust','fear','happy','neutral','sadness','surprise']

def getLabel(id):
    return ['anger','disgust','fear','happy','neutral','sadness','surprise'][id]
```


```python
Y = np_utils.to_categorical(labels, num_classes)

#Shuffle the dataset
x,y = shuffle(img_data,Y, random_state=2)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=2)
x_test=X_test
```


```python
input_shape=(48,48,3)

model = Sequential()
model.add(Conv2D(6, (5, 5), input_shape=input_shape, padding='same', activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(16, (5, 5), padding='same', activation = 'relu'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation = 'softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer='adam')
```


```python
model.summary()
model.get_config()
model.layers[0].get_config()
model.layers[0].input_shape
model.layers[0].output_shape
model.layers[0].get_weights()
np.shape(model.layers[0].get_weights()[0])
model.layers[0].trainable
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              (None, 48, 48, 6)         456       
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 24, 24, 6)         0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 24, 24, 16)        2416      
    _________________________________________________________________
    activation (Activation)      (None, 24, 24, 16)        0         
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 12, 12, 16)        0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 10, 10, 64)        9280      
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 5, 5, 64)          0         
    _________________________________________________________________
    flatten (Flatten)            (None, 1600)              0         
    _________________________________________________________________
    dense (Dense)                (None, 128)               204928    
    _________________________________________________________________
    dropout (Dropout)            (None, 128)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 7)                 903       
    =================================================================
    Total params: 217,983
    Trainable params: 217,983
    Non-trainable params: 0
    _________________________________________________________________
    




    True




```python
from keras import callbacks
filename='CFE_model.csv'
filepath="Best-weights-my_model-{epoch:03d}-{loss:.4f}-{acc:.4f}.hdf5"

csv_log=callbacks.CSVLogger(filename, separator=',', append=False)
checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [csv_log,checkpoint]
callbacks_list = [csv_log]
```


```python
hist = model.fit(X_train, y_train, batch_size=7, epochs=50, verbose=1, validation_data=(X_test, y_test),callbacks=callbacks_list)
```

    Epoch 1/50
    196/196 [==============================] - 4s 20ms/step - loss: 1.9490 - accuracy: 0.1250 - val_loss: 1.9471 - val_accuracy: 0.1446
    Epoch 2/50
    196/196 [==============================] - 5s 24ms/step - loss: 1.9466 - accuracy: 0.1323 - val_loss: 1.9483 - val_accuracy: 0.1446
    Epoch 3/50
    196/196 [==============================] - 2s 12ms/step - loss: 1.9467 - accuracy: 0.1389 - val_loss: 1.9478 - val_accuracy: 0.1446
    Epoch 4/50
    196/196 [==============================] - 4s 19ms/step - loss: 1.9465 - accuracy: 0.1389 - val_loss: 1.9490 - val_accuracy: 0.1446
    Epoch 5/50
    196/196 [==============================] - 3s 15ms/step - loss: 1.9464 - accuracy: 0.1316 - val_loss: 1.9490 - val_accuracy: 0.1198
    Epoch 6/50
    196/196 [==============================] - 4s 19ms/step - loss: 1.9461 - accuracy: 0.1301 - val_loss: 1.9489 - val_accuracy: 0.1446
    Epoch 7/50
    196/196 [==============================] - 4s 19ms/step - loss: 1.9461 - accuracy: 0.1477 - val_loss: 1.9493 - val_accuracy: 0.1446
    Epoch 8/50
    196/196 [==============================] - 2s 12ms/step - loss: 1.9462 - accuracy: 0.1425 - val_loss: 1.9494 - val_accuracy: 0.1446
    Epoch 9/50
    196/196 [==============================] - 4s 19ms/step - loss: 1.9464 - accuracy: 0.1265 - val_loss: 1.9490 - val_accuracy: 0.1446
    Epoch 10/50
    196/196 [==============================] - 4s 22ms/step - loss: 1.9461 - accuracy: 0.1411 - val_loss: 1.9490 - val_accuracy: 0.1198
    Epoch 11/50
    196/196 [==============================] - 3s 14ms/step - loss: 1.9466 - accuracy: 0.1345 - val_loss: 1.9491 - val_accuracy: 0.1446
    Epoch 12/50
    196/196 [==============================] - 4s 19ms/step - loss: 1.9460 - accuracy: 0.1374 - val_loss: 1.9496 - val_accuracy: 0.1198
    Epoch 13/50
    196/196 [==============================] - 5s 23ms/step - loss: 1.9462 - accuracy: 0.1308 - val_loss: 1.9495 - val_accuracy: 0.1446
    Epoch 14/50
    196/196 [==============================] - 2s 12ms/step - loss: 1.9459 - accuracy: 0.1382 - val_loss: 1.9498 - val_accuracy: 0.1446
    Epoch 15/50
    196/196 [==============================] - 3s 17ms/step - loss: 1.9460 - accuracy: 0.1382 - val_loss: 1.9497 - val_accuracy: 0.1446
    Epoch 16/50
    196/196 [==============================] - 2s 12ms/step - loss: 1.9459 - accuracy: 0.1440 - val_loss: 1.9498 - val_accuracy: 0.1198
    Epoch 17/50
    196/196 [==============================] - 4s 19ms/step - loss: 1.9460 - accuracy: 0.1308 - val_loss: 1.9497 - val_accuracy: 0.1281
    Epoch 18/50
    196/196 [==============================] - 4s 22ms/step - loss: 1.9460 - accuracy: 0.1257 - val_loss: 1.9499 - val_accuracy: 0.1446
    Epoch 19/50
    196/196 [==============================] - 3s 15ms/step - loss: 1.9459 - accuracy: 0.1294 - val_loss: 1.9500 - val_accuracy: 0.1281
    Epoch 20/50
    196/196 [==============================] - 4s 19ms/step - loss: 1.9459 - accuracy: 0.1330 - val_loss: 1.9497 - val_accuracy: 0.1198
    Epoch 21/50
    196/196 [==============================] - 4s 22ms/step - loss: 1.9460 - accuracy: 0.1404 - val_loss: 1.9499 - val_accuracy: 0.1446
    Epoch 22/50
    196/196 [==============================] - 3s 13ms/step - loss: 1.9460 - accuracy: 0.1418 - val_loss: 1.9498 - val_accuracy: 0.1446
    Epoch 23/50
    196/196 [==============================] - 4s 20ms/step - loss: 1.9460 - accuracy: 0.1491 - val_loss: 1.9499 - val_accuracy: 0.1405
    Epoch 24/50
    196/196 [==============================] - 3s 17ms/step - loss: 1.9460 - accuracy: 0.1338 - val_loss: 1.9499 - val_accuracy: 0.1198
    Epoch 25/50
    196/196 [==============================] - 3s 16ms/step - loss: 1.9460 - accuracy: 0.1396 - val_loss: 1.9498 - val_accuracy: 0.1198
    Epoch 26/50
    196/196 [==============================] - 3s 16ms/step - loss: 1.9459 - accuracy: 0.1323 - val_loss: 1.9499 - val_accuracy: 0.1198
    Epoch 27/50
    196/196 [==============================] - 3s 14ms/step - loss: 1.9460 - accuracy: 0.1462 - val_loss: 1.9500 - val_accuracy: 0.1198
    Epoch 28/50
    196/196 [==============================] - 3s 13ms/step - loss: 1.9459 - accuracy: 0.1462 - val_loss: 1.9501 - val_accuracy: 0.1198
    Epoch 29/50
    196/196 [==============================] - 3s 14ms/step - loss: 1.9460 - accuracy: 0.1382 - val_loss: 1.9503 - val_accuracy: 0.1446
    Epoch 30/50
    196/196 [==============================] - 3s 13ms/step - loss: 1.9460 - accuracy: 0.1367 - val_loss: 1.9504 - val_accuracy: 0.1198
    Epoch 31/50
    196/196 [==============================] - 3s 13ms/step - loss: 1.9459 - accuracy: 0.1316 - val_loss: 1.9504 - val_accuracy: 0.1198
    Epoch 32/50
    196/196 [==============================] - 4s 19ms/step - loss: 1.9460 - accuracy: 0.1330 - val_loss: 1.9501 - val_accuracy: 0.1198
    Epoch 33/50
    196/196 [==============================] - 2s 12ms/step - loss: 1.9459 - accuracy: 0.1360 - val_loss: 1.9501 - val_accuracy: 0.1198
    Epoch 34/50
    196/196 [==============================] - 3s 15ms/step - loss: 1.9479 - accuracy: 0.1360 - val_loss: 1.9500 - val_accuracy: 0.1322
    Epoch 35/50
    196/196 [==============================] - 3s 13ms/step - loss: 1.9462 - accuracy: 0.1396 - val_loss: 1.9499 - val_accuracy: 0.1446
    Epoch 36/50
    196/196 [==============================] - 4s 19ms/step - loss: 1.9460 - accuracy: 0.1206 - val_loss: 1.9499 - val_accuracy: 0.1446
    Epoch 37/50
    196/196 [==============================] - 3s 15ms/step - loss: 1.9460 - accuracy: 0.1330 - val_loss: 1.9501 - val_accuracy: 0.1198
    Epoch 38/50
    196/196 [==============================] - 3s 16ms/step - loss: 1.9459 - accuracy: 0.1345 - val_loss: 1.9501 - val_accuracy: 0.1446
    Epoch 39/50
    196/196 [==============================] - 4s 20ms/step - loss: 1.9462 - accuracy: 0.1367 - val_loss: 1.9503 - val_accuracy: 0.1198
    Epoch 40/50
    196/196 [==============================] - 3s 14ms/step - loss: 1.9462 - accuracy: 0.1338 - val_loss: 1.9501 - val_accuracy: 0.1198
    Epoch 41/50
    196/196 [==============================] - 4s 18ms/step - loss: 1.9460 - accuracy: 0.1272 - val_loss: 1.9501 - val_accuracy: 0.1198
    Epoch 42/50
    196/196 [==============================] - 3s 15ms/step - loss: 1.9457 - accuracy: 0.1433 - val_loss: 1.9500 - val_accuracy: 0.1281
    Epoch 43/50
    196/196 [==============================] - 3s 15ms/step - loss: 1.9462 - accuracy: 0.1396 - val_loss: 1.9503 - val_accuracy: 0.1281
    Epoch 44/50
    196/196 [==============================] - 4s 23ms/step - loss: 1.9460 - accuracy: 0.1535 - val_loss: 1.9499 - val_accuracy: 0.1446
    Epoch 45/50
    196/196 [==============================] - 2s 12ms/step - loss: 1.9459 - accuracy: 0.1404 - val_loss: 1.9500 - val_accuracy: 0.1446
    Epoch 46/50
    196/196 [==============================] - 3s 17ms/step - loss: 1.9462 - accuracy: 0.1411 - val_loss: 1.9503 - val_accuracy: 0.1198
    Epoch 47/50
    196/196 [==============================] - 3s 13ms/step - loss: 1.9459 - accuracy: 0.1396 - val_loss: 1.9501 - val_accuracy: 0.1198
    Epoch 48/50
    196/196 [==============================] - 3s 16ms/step - loss: 1.9461 - accuracy: 0.1308 - val_loss: 1.9505 - val_accuracy: 0.1198
    Epoch 49/50
    196/196 [==============================] - 3s 17ms/step - loss: 1.9460 - accuracy: 0.1352 - val_loss: 1.9502 - val_accuracy: 0.1446
    Epoch 50/50
    196/196 [==============================] - 3s 16ms/step - loss: 1.9459 - accuracy: 0.1411 - val_loss: 1.9501 - val_accuracy: 0.1446
    


```python
keras.utils.plot_model(model, to_file='model.png', show_shapes=True)
```

    ('Failed to import pydot. You must `pip install pydot` and install graphviz (https://graphviz.gitlab.io/download/), ', 'for `pydotprint` to work.')
    


```python
score = model.evaluate(X_test, y_test, verbose=0)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])

test_image = X_test[0:1]
print (test_image.shape)

print(model.predict(test_image))
print(model.predict_classes(test_image))
print(y_test[0:1])

res = model.predict_classes(X_test[9:18])
plt.figure(figsize=(10, 10))

for i in range(0, 9):
    plt.subplot(330 + 1 + i)
    plt.imshow(x_test[i],cmap=plt.get_cmap('gray'))
    plt.gca().get_xaxis().set_ticks([])
    plt.gca().get_yaxis().set_ticks([])
    plt.ylabel('prediction = %s' % getLabel(res[i]), fontsize=14)
# show the plot
plt.show()
```

    Test Loss: 1.950067400932312
    Test accuracy: 0.14462809264659882
    (1, 48, 48, 3)
    [[0.14589688 0.14647803 0.14495741 0.14334056 0.1376446  0.13579555
      0.14588696]]
    WARNING:tensorflow:From C:\Users\Sivajee\AppData\Local\Temp/ipykernel_12384/2922075397.py:9: Sequential.predict_classes (from tensorflow.python.keras.engine.sequential) is deprecated and will be removed after 2021-01-01.
    Instructions for updating:
    Please use instead:* `np.argmax(model.predict(x), axis=-1)`,   if your model does multi-class classification   (e.g. if it uses a `softmax` last-layer activation).* `(model.predict(x) > 0.5).astype("int32")`,   if your model does binary classification   (e.g. if it uses a `sigmoid` last-layer activation).
    [1]
    [[0. 0. 1. 0. 0. 0. 0.]]
    


    
![png](output_9_1.png)
    



```python
# visualizing losses and accuracy
%matplotlib inline

train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['accuracy']
val_acc=hist.history['val_accuracy']

epochs = range(len(train_acc))

plt.plot(epochs,train_loss,'r', label='train_loss')
plt.plot(epochs,val_loss,'b', label='val_loss')
plt.title('train_loss vs val_loss')
plt.legend()
plt.figure()

plt.plot(epochs,train_acc,'r', label='train_acc')
plt.plot(epochs,val_acc,'b', label='val_acc')
plt.title('train_acc vs val_acc')
plt.legend()
plt.figure()
```




    <Figure size 432x288 with 0 Axes>




    
![png](output_10_1.png)
    



    
![png](output_10_2.png)
    



    <Figure size 432x288 with 0 Axes>



```python

```
