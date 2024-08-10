
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle

## here we are creating the model by using the sequential
model=Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(100,100,3)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D(pool_size=(2,2))
])

##here we are adding the optimizesr and metrices and loss functions for model training 
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


### now we are displaying the model summary
##Model: "sequential_2"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ conv2d_5 (Conv2D)                    │ (None, 98, 98, 32)          │             896 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_3 (MaxPooling2D)       │ (None, 49, 49, 32)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_6 (Conv2D)                    │ (None, 47, 47, 64)          │          18,496 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_4 (MaxPooling2D)       │ (None, 23, 23, 64)          │               0 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 19,392 (75.75 KB)
 Trainable params: 19,392 (75.75 KB)
 Non-trainable params: 0 (0.00 B)




##displaying the all layers in the models 
for layer in model.layers:
    print(layer)



### display the all layers in the model 
len(model.layers) 

###displaying the all the layers one by one 
layer1=model.layer[0] 
print(f"first layer name is {layer.name}")
print(f" input of the layer is {layer.input}")
print(f"output of the layer is {layer.output}") 

















