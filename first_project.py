!pip install kaggle

## herer the kaggle.json is the api_key for the kaggle 
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

## here we are downloading the dataset in the format of zip file 

!kaggle competitions download -c dogs-vs-cats

## next we are extracting the total zip files in the dataset
from zipfile import ZipFile
data="dogs-vs-cats.zip"
with ZipFile(data,"r") as file: 
  file.extractall()
  print("the dataset is extracted")


## now we are extracting the train.zip files 
dataset="train.zip"
with ZipFile(dataset,"r") as zip: 
  zip.extractall()
  print("alltraining dastasets are extracted ")


## now we are going to extract the files that are in the testdataset 
dataset="test1.zip"
with ZipFile(dataset,"r") as zip: 
  zip.extractall()
  print("alltraining dastasets are extracted ")






import os 
path,dirs,files=next(os.walk("train"))

file_count=len(files)
print('the number of files in the train dataaset',file_count)


file_names=os.listdir("train")
file_names


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from google.colab.patches import cv2_imshow 


##display the one of image 
ima=mpimg.imread("train/cat.10179.jpg")
plt.imshow(ima)
plt.show()


file_names=os.listdir("train")
for i in range(5):
  name=file_names[i]
  print(name.split(".")[0])



##now we are going the count the number of dog files and number of cat files 
file_names=os.listdir("train")
dog_count=0
cat_count=0 
for imag_file in file_names: 
  if imag_file.split(".")[0] == "dog": 
    dog_count +=1
  else: 
    cat_count +=1 
print("the number of cat files are",cat_count)
print('the number of dog file are ',dog_count)


original_folder="train"

from PIL import Image
for filename in os.listdir(original_folder):
  if filename.endswith((".jpg","jpeg",".png")):
    img_path=os.path.join(original_folder,filename)
    img=Image.open(img_path)
    img_resized=img.resize((224,224),Image.ANTIALIAS)
    img_resized.save(img_path)

for filename in os.listdir(original_folder):
  image_path=os.path.join(original_folder,filename)
  img=Image.open(image_path)

print(os.listdir(original_folder)[0:5])




for i in os.listdir(original_folder)[0:5]: 
  img=mpimg.imread(f"train/{i}")
  plt.imshow(img)
  plt.show()


###now create the output for the cat and dog images if cat o else 1
filename=os.listdir("train/")[0:2000]
labels=[]
for i in filename: 
  if i.split(".")[0] =="dog": 
    labels.append(1)
  else: 
    labels.append(0)

values,counts=np.unique(labels,return_counts=True)
values,counts



##now we load 
from PIL import Image
import numpy as np 
image_list=[]
for i in filename: 
  image_path=os.path.join("train",i)
  img=Image.open(image_path)
  image_array=np.array(img)
  img_array=image_array
  image_list.append(img_array)
all_images=np.array(image_list)




all_images.shape
x=all_images 
y=np.asarray(labels)


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
x_train.shape,x_test.shape,y_train.shape,y_test.shape




x_train_scaled=x_train/255
x_test_scaled=x_test/255

import tensorflow as tf 
from tensorflow.keras import datasets, layers, models
model = models.Sequential([
    # First convolutional layer with max pooling
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224,224,3)),
    layers.MaxPooling2D((2, 2)),
    
    # Second convolutional layer with max pooling
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Third convolutional layer with max pooling
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Flatten the feature maps to a 1D vector
    layers.Flatten(),
    
     # Output layer with 10 neurons (one for each class)
    layers.Dense(1,activation="sigmoid")
])

model.summary()


model.compile(optimizer='adam',
              loss='binary_crossentropy', # Change loss function to binary crossentropy
              metrics=['accuracy'])

history=model.fit(x_train_scaled,y_train,epochs=2)
score, acc = model.evaluate(x_test_scaled, y_test)
print('Test Loss =', score)
print('Test Accuracy =', acc)


import cv2

def image_prediction(image_path):
  input_imga_path=image_path

  input_image=cv2.imread(input_imga_path)
  cv2_imshow(input_image)
  input_image=cv2.resize(input_image,(224,224))
  input_image=input_image/255
  input_prediction=model.predict(np.array([input_image]))
  if input_prediction < 0.5: 
    print("the image is a cat")
  else: 
    print("the image is a dog")
  print(input_prediction)
for i in os.listdir("train")[0:5]: 
  path=os.path.join("train",i)
  image_prediction(path)

for i in os.listdir("test1"): 
  path=os.path.join("test1",i)
  image_prediction(path)
  













