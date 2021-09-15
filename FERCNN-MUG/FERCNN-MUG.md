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
print(os.listdir("./input/MUG"))
```

    ['an', 'di', 'fe', 'ha', 'ne', 'sa', 'su']
    


```python
data_path = './input/MUG'
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

    Loaded the images of dataset-an
    
    Loaded the images of dataset-di
    
    Loaded the images of dataset-fe
    
    Loaded the images of dataset-ha
    
    Loaded the images of dataset-ne
    
    Loaded the images of dataset-sa
    
    Loaded the images of dataset-su
    
    




    (401, 48, 48, 3)




```python
num_classes = 7

num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')

labels[0:56]=0 #140
labels[57:127]=1 #140
labels[127:174]=2 #140
labels[175:261]=3 #140
labels[262:286]=4 #140
labels[287:334]=5 #140
labels[335:400]=6 #140

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
filename='mug_model.csv'
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
    49/49 [==============================] - 1s 21ms/step - loss: 1.9186 - accuracy: 0.1735 - val_loss: 1.9348 - val_accuracy: 0.1803
    Epoch 2/50
    49/49 [==============================] - 1s 16ms/step - loss: 1.8959 - accuracy: 0.2059 - val_loss: 1.9421 - val_accuracy: 0.1803
    Epoch 3/50
    49/49 [==============================] - 1s 24ms/step - loss: 1.8888 - accuracy: 0.2118 - val_loss: 1.9405 - val_accuracy: 0.1803
    Epoch 4/50
    49/49 [==============================] - 1s 27ms/step - loss: 1.8722 - accuracy: 0.2294 - val_loss: 1.9347 - val_accuracy: 0.1803
    Epoch 5/50
    49/49 [==============================] - 1s 24ms/step - loss: 1.8714 - accuracy: 0.2265 - val_loss: 1.9346 - val_accuracy: 0.1803
    Epoch 6/50
    49/49 [==============================] - 1s 14ms/step - loss: 1.8625 - accuracy: 0.2441 - val_loss: 1.9111 - val_accuracy: 0.1475
    Epoch 7/50
    49/49 [==============================] - 1s 17ms/step - loss: 1.7899 - accuracy: 0.3265 - val_loss: 1.8328 - val_accuracy: 0.2459
    Epoch 8/50
    49/49 [==============================] - 1s 25ms/step - loss: 1.6233 - accuracy: 0.3559 - val_loss: 1.6658 - val_accuracy: 0.3607
    Epoch 9/50
    49/49 [==============================] - 1s 24ms/step - loss: 1.4526 - accuracy: 0.4882 - val_loss: 1.5913 - val_accuracy: 0.3443
    Epoch 10/50
    49/49 [==============================] - 1s 24ms/step - loss: 1.2752 - accuracy: 0.4971 - val_loss: 1.5428 - val_accuracy: 0.4098
    Epoch 11/50
    49/49 [==============================] - 1s 16ms/step - loss: 1.0961 - accuracy: 0.6088 - val_loss: 1.3786 - val_accuracy: 0.5246
    Epoch 12/50
    49/49 [==============================] - 1s 15ms/step - loss: 0.9635 - accuracy: 0.6529 - val_loss: 1.2746 - val_accuracy: 0.5246
    Epoch 13/50
    49/49 [==============================] - 1s 24ms/step - loss: 0.8262 - accuracy: 0.7029 - val_loss: 1.1944 - val_accuracy: 0.6230
    Epoch 14/50
    49/49 [==============================] - 1s 24ms/step - loss: 0.8030 - accuracy: 0.7000 - val_loss: 1.2682 - val_accuracy: 0.5738
    Epoch 15/50
    49/49 [==============================] - 1s 24ms/step - loss: 0.6765 - accuracy: 0.7353 - val_loss: 1.0647 - val_accuracy: 0.6721
    Epoch 16/50
    49/49 [==============================] - 1s 19ms/step - loss: 0.6088 - accuracy: 0.7735 - val_loss: 1.0644 - val_accuracy: 0.6393
    Epoch 17/50
    49/49 [==============================] - 1s 12ms/step - loss: 0.5110 - accuracy: 0.8176 - val_loss: 1.1173 - val_accuracy: 0.6393
    Epoch 18/50
    49/49 [==============================] - 1s 12ms/step - loss: 0.5050 - accuracy: 0.8059 - val_loss: 1.3430 - val_accuracy: 0.5082
    Epoch 19/50
    49/49 [==============================] - 1s 19ms/step - loss: 0.5094 - accuracy: 0.8441 - val_loss: 1.0988 - val_accuracy: 0.6885
    Epoch 20/50
    49/49 [==============================] - 1s 19ms/step - loss: 0.4927 - accuracy: 0.7853 - val_loss: 1.0359 - val_accuracy: 0.6393
    Epoch 21/50
    49/49 [==============================] - 1s 20ms/step - loss: 0.4281 - accuracy: 0.8324 - val_loss: 1.0947 - val_accuracy: 0.6721
    Epoch 22/50
    49/49 [==============================] - 1s 19ms/step - loss: 0.3772 - accuracy: 0.8529 - val_loss: 1.0156 - val_accuracy: 0.6393
    Epoch 23/50
    49/49 [==============================] - 1s 14ms/step - loss: 0.3520 - accuracy: 0.8765 - val_loss: 1.0534 - val_accuracy: 0.6885
    Epoch 24/50
    49/49 [==============================] - 1s 13ms/step - loss: 0.2810 - accuracy: 0.8971 - val_loss: 1.1172 - val_accuracy: 0.7213
    Epoch 25/50
    49/49 [==============================] - 1s 17ms/step - loss: 0.3104 - accuracy: 0.9029 - val_loss: 1.0570 - val_accuracy: 0.7377
    Epoch 26/50
    49/49 [==============================] - 1s 23ms/step - loss: 0.2485 - accuracy: 0.9147 - val_loss: 0.9903 - val_accuracy: 0.7049
    Epoch 27/50
    49/49 [==============================] - 1s 23ms/step - loss: 0.2826 - accuracy: 0.8853 - val_loss: 1.2973 - val_accuracy: 0.6393
    Epoch 28/50
    49/49 [==============================] - 1s 23ms/step - loss: 0.2563 - accuracy: 0.9000 - val_loss: 1.2992 - val_accuracy: 0.6557
    Epoch 29/50
    49/49 [==============================] - 1s 11ms/step - loss: 0.2102 - accuracy: 0.9000 - val_loss: 0.8771 - val_accuracy: 0.7705
    Epoch 30/50
    49/49 [==============================] - 1s 13ms/step - loss: 0.2287 - accuracy: 0.9147 - val_loss: 1.1556 - val_accuracy: 0.6885
    Epoch 31/50
    49/49 [==============================] - 1s 14ms/step - loss: 0.2001 - accuracy: 0.9353 - val_loss: 0.9824 - val_accuracy: 0.7377
    Epoch 32/50
    49/49 [==============================] - 1s 16ms/step - loss: 0.1966 - accuracy: 0.9206 - val_loss: 1.0361 - val_accuracy: 0.7049
    Epoch 33/50
    49/49 [==============================] - 1s 25ms/step - loss: 0.1607 - accuracy: 0.9382 - val_loss: 1.1901 - val_accuracy: 0.7213
    Epoch 34/50
    49/49 [==============================] - 1s 25ms/step - loss: 0.1611 - accuracy: 0.9353 - val_loss: 1.0434 - val_accuracy: 0.7049
    Epoch 35/50
    49/49 [==============================] - 1s 12ms/step - loss: 0.1429 - accuracy: 0.9559 - val_loss: 1.2068 - val_accuracy: 0.6721
    Epoch 36/50
    49/49 [==============================] - 1s 13ms/step - loss: 0.1640 - accuracy: 0.9382 - val_loss: 1.0769 - val_accuracy: 0.7541
    Epoch 37/50
    49/49 [==============================] - 1s 22ms/step - loss: 0.1502 - accuracy: 0.9471 - val_loss: 0.9600 - val_accuracy: 0.7541
    Epoch 38/50
    49/49 [==============================] - 1s 22ms/step - loss: 0.1123 - accuracy: 0.9735 - val_loss: 1.0700 - val_accuracy: 0.6885
    Epoch 39/50
    49/49 [==============================] - 1s 22ms/step - loss: 0.1838 - accuracy: 0.9412 - val_loss: 0.8191 - val_accuracy: 0.7541
    Epoch 40/50
    49/49 [==============================] - 1s 21ms/step - loss: 0.1549 - accuracy: 0.9471 - val_loss: 1.4036 - val_accuracy: 0.7049
    Epoch 41/50
    49/49 [==============================] - 1s 13ms/step - loss: 0.1243 - accuracy: 0.9529 - val_loss: 1.2575 - val_accuracy: 0.7213
    Epoch 42/50
    49/49 [==============================] - 1s 14ms/step - loss: 0.0701 - accuracy: 0.9882 - val_loss: 1.2575 - val_accuracy: 0.7541
    Epoch 43/50
    49/49 [==============================] - 1s 27ms/step - loss: 0.1432 - accuracy: 0.9500 - val_loss: 1.3263 - val_accuracy: 0.7049
    Epoch 44/50
    49/49 [==============================] - 1s 26ms/step - loss: 0.1324 - accuracy: 0.9588 - val_loss: 1.3332 - val_accuracy: 0.7377
    Epoch 45/50
    49/49 [==============================] - 1s 24ms/step - loss: 0.1390 - accuracy: 0.9559 - val_loss: 1.1336 - val_accuracy: 0.7213
    Epoch 46/50
    49/49 [==============================] - 1s 14ms/step - loss: 0.0980 - accuracy: 0.9706 - val_loss: 1.3445 - val_accuracy: 0.7541
    Epoch 47/50
    49/49 [==============================] - 1s 15ms/step - loss: 0.1143 - accuracy: 0.9647 - val_loss: 1.2182 - val_accuracy: 0.7213
    Epoch 48/50
    49/49 [==============================] - 1s 19ms/step - loss: 0.1119 - accuracy: 0.9647 - val_loss: 1.1771 - val_accuracy: 0.7377
    Epoch 49/50
    49/49 [==============================] - 1s 20ms/step - loss: 0.1638 - accuracy: 0.9441 - val_loss: 1.2080 - val_accuracy: 0.7541
    Epoch 50/50
    49/49 [==============================] - 1s 15ms/step - loss: 0.0927 - accuracy: 0.9647 - val_loss: 1.4342 - val_accuracy: 0.7869
    


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

    Test Loss: 1.4342236518859863
    Test accuracy: 0.7868852615356445
    (1, 48, 48, 3)
    [[1.9568986e-05 1.8867319e-04 9.9219865e-01 5.8780662e-03 7.4400631e-04
      3.4273820e-04 6.2825601e-04]]
    WARNING:tensorflow:From C:\Users\Sivajee\AppData\Local\Temp/ipykernel_15236/2922075397.py:9: Sequential.predict_classes (from tensorflow.python.keras.engine.sequential) is deprecated and will be removed after 2021-01-01.
    Instructions for updating:
    Please use instead:* `np.argmax(model.predict(x), axis=-1)`,   if your model does multi-class classification   (e.g. if it uses a `softmax` last-layer activation).* `(model.predict(x) > 0.5).astype("int32")`,   if your model does binary classification   (e.g. if it uses a `sigmoid` last-layer activation).
    [2]
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
