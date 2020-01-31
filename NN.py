#ML script for pose detection

# Importing the libraries
#import mysql.connector as mariadb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pymongo
from sklearn.model_selection import GridSearchCV

#Connect to MongoDB database
myclient = pymongo.MongoClient("mongodb+srv://alex:1234@cluster1-kuer5.mongodb.net/test?retryWrites=true")
mydb = myclient["posture_training_2"]
backward_shoulders_col = mydb["backward_shoulders"]
forward_head_col = mydb["forward_head"]
forward_shoulders_col = mydb["forward_shoulders"]
healthy_posture_col = mydb["healthy_posture"]
left_head_flexion_col = mydb["left_head_flexion"]
left_head_rotation_col = mydb["left_head_rotation"]
left_shoulder_elevation_col = mydb["left_shoulder_elevation"]
left_torso_rotation_col = mydb["left_torso_rotation"]
right_head_flexion_col = mydb["right_head_flexion"]
right_head_rotation_col = mydb["right_head_rotation"]
right_shoulder_elevation_col = mydb["right_shoulder_elevation"]
right_torso_rotation_col = mydb["right_torso_rotation"]
tucked_chin_col = mydb["tucked_chin"]
nothing_col = mydb["nothing"]

#Retrieve all documents from MongoDB collection
all_forward_head = pd.DataFrame(list(forward_head_col.find())).head(53)
all_healthy_posture = pd.DataFrame(list(healthy_posture_col.find())).head(53)
all_left_head_flexion = pd.DataFrame(list(left_head_flexion_col.find()))
all_left_head_rotation = pd.DataFrame(list(left_head_rotation_col.find())).head(53)
all_left_shoulder_elevation = pd.DataFrame(list(left_shoulder_elevation_col.find())).head(53)
#all_left_torso_rotation = pd.DataFrame(list(left_torso_rotation_col.find())).head(53)
all_nothing = pd.DataFrame(list(nothing_col.find()))
all_right_head_flexion = pd.DataFrame(list(right_head_flexion_col.find()))
all_right_head_rotation = pd.DataFrame(list(right_head_rotation_col.find())).head(53)
all_right_shoulder_elevation = pd.DataFrame(list(right_shoulder_elevation_col.find())).head(53)
#all_right_torso_rotation = pd.DataFrame(list(right_torso_rotation_col.find())).head(53)
all_tucked_chin = pd.DataFrame(list(tucked_chin_col.find())).head(53)
all_forward_shoulders = pd.DataFrame(list(forward_shoulders_col.find()))
all_backward_shoulders = pd.DataFrame(list(backward_shoulders_col.find()))

all_poses = pd.concat([all_forward_head, all_healthy_posture, all_left_head_flexion, all_left_head_rotation, 
                       all_left_shoulder_elevation, all_nothing, 
                       all_right_head_flexion, all_right_head_rotation, all_right_shoulder_elevation,
                       all_tucked_chin, all_forward_shoulders, all_backward_shoulders], ignore_index=True)

X = all_poses.loc[:, ['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'x5', 'y5', 
                          'x6', 'y6', 'x7', 'y7', 'x8', 'y8', 'x9', 'y9', 
                          'x10', 'y10', 'x11', 'y11', 'x12', 'y12', 'x13', 'y13', 'x14', 'y14', 
                          'x15', 'y15', 'x16', 'y16', 'x17', 'y17', 'x18', 'y18', 'x19', 'y19']].values
y = all_poses.loc[:, ['pose']].values

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
print(labelencoder_y.classes_)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size = 0.5, random_state = 0)

#Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

import keras
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(output_dim=256, activation='tanh', input_shape = (38, )))
model.add(layers.Dense(output_dim=128, activation='tanh'))
model.add(layers.Dense(output_dim=64, activation='tanh'))
model.add(layers.Dense(output_dim=12, activation='sigmoid'))

#model.add(layers.Dense(output_dim=128, activation='relu'))
#model.add(layers.Dense(output_dim=12, activation='softmax'))

model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=300, batch_size=25)

y_pred = model.predict(X_test)
#y_pred = (y_pred > 0.5)
#y_test = (y_test > 0.5)



results = model.evaluate(X_test, y_test)
print(results)
#X_train[0:1]
#X_train[0:1].shape
#asd = model.predict(X_train[0:1])

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")