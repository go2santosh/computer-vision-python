# import all the required libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Activation, MaxPooling2D,Dense,Flatten,Dropout
import numpy as np

# Initialize a model
catDogImageclassifier = Sequential()

# Add a two-dimensional convolutional layer which will have 32 filters @ 3x3 size, input image shape is 64*64*3
catDogImageclassifier.add(Conv2D(32,(3,3), input_shape=(64,64,3)))

# add the Max Pooling layer to avoid the network to be overly complex computationally
catDogImageclassifier.add(MaxPooling2D(pool_size =(2,2)))

# add three convolutional blocks, each block has a Cov2D, ReLU, and Max Pooling Layer
catDogImageclassifier.add(Conv2D(32,(3,3)))
catDogImageclassifier.add(Activation('relu'))
catDogImageclassifier.add(MaxPooling2D(pool_size =(2,2)))

catDogImageclassifier.add(Conv2D(32,(3,3)))
catDogImageclassifier.add(Activation('relu'))
catDogImageclassifier.add(MaxPooling2D(pool_size =(2,2)))

catDogImageclassifier.add(Conv2D(32,(3,3)))
catDogImageclassifier.add(Activation('relu'))
catDogImageclassifier.add(MaxPooling2D(pool_size=(2,2)))

# flatten the dataset which will transform the pooled feature map matrix into one column
catDogImageclassifier.add(Flatten())

# add the dense function
catDogImageclassifier.add(Dense(64))
catDogImageclassifier.add(Activation('relu'))

# add the Dropout layer to overcome overfitting
catDogImageclassifier.add(Dropout(0.5))

# add one more fully connected layer to get the output in n-dimensional classes
catDogImageclassifier.add(Dense(1))

# add the Sigmoid function to convert to probabilities
catDogImageclassifier.add(Activation('sigmoid'))

# print a summary of the network
catDogImageclassifier.summary()

# compile the network
catDogImageclassifier.compile(optimizer = 'rmsprop', loss ='binary_crossentropy', metrics =['accuracy'])

# data augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255, shear_range =0.25,zoom_range = 0.25, horizontal_flip =True)
test_datagen = ImageDataGenerator(rescale = 1./255)

# load the training data
training_set = train_datagen.flow_from_directory('datasets/dogs-vs-cats/train1', target_size=(64, 64), batch_size= 32, class_mode='binary')

# load the testing data
test_set = test_datagen.flow_from_directory('datasets/dogs-vs-cats/test1', target_size = (64,64), batch_size = 32, class_mode ='binary')

# begin the training
from IPython.display import display
from PIL import Image 

catDogImageclassifier.fit(training_set, steps_per_epoch =625, epochs = 10, validation_data =test_set, validation_steps = 1000)

# save the trained model
catDogImageclassifier.save('catdog_cnn_model.h5')

print("Successfully completed.")