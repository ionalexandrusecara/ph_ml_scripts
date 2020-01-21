#ML script for pose detection

# Importing the libraries
#import mysql.connector as mariadb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymongo

#username
username = "thesecara"

#Connect to MongoDB database
myclient = pymongo.MongoClient("mongodb+srv://alex:1234@cluster1-kuer5.mongodb.net/test?retryWrites=true")
mydb = myclient["posture_training"]
leanleft_col = mydb["leanleft"]
leanright_col = mydb["leanright"]
perfect_col = mydb["perfect"]
nothing_col = mydb["nothing"]

#Retrieve all documents from MongoDB collection
all_lefts = pd.DataFrame(list(leanleft_col.find()))
all_rights = pd.DataFrame(list(leanright_col.find()))
all_perfects = pd.DataFrame(list(perfect_col.find()))
all_nothings = pd.DataFrame(list(nothing_col.find()))

all_poses = pd.concat([all_lefts, all_rights, all_perfects, all_nothings], ignore_index=True)

X = all_poses.loc[:, ['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'x5', 'y5', 
                          'x6', 'y6', 'x7', 'y7', 'x8', 'y8', 'x9', 'y9', 
                          'x10', 'y10', 'x11', 'y11', 'x12', 'y12', 'x13', 'y13', 'x14', 'y14', 
                          'x15', 'y15', 'x16', 'y16', 'x17', 'y17', 'x18', 'y18', 'x19', 'y19']].values
y = all_poses.loc[:, ['pose']].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#0 - left, 3 - right, 2 - perfect, 1 - nothing

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X, y)

#Today's date for the collection - username for DB
from datetime import date
today = date.today()
user_db = myclient[str(username)]
user_col = user_db[str(today)]
user_res_col = user_db[str(str(today)+ "_res")]

#Get the pose feed from the user's today's collection
daily_poses = pd.DataFrame(list(user_col.find()))
X_daily = daily_poses.loc[:, ['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'x5', 'y5', 
                          'x6', 'y6', 'x7', 'y7', 'x8', 'y8', 'x9', 'y9', 
                          'x10', 'y10', 'x11', 'y11', 'x12', 'y12', 'x13', 'y13', 'x14', 'y14', 
                          'x15', 'y15', 'x16', 'y16', 'x17', 'y17', 'x18', 'y18', 'x19', 'y19']].values


# Predicting the pose
y_daily_pred = classifier.predict(X_daily)

#test
print("left:", np.count_nonzero(y_daily_pred == 0))
print("nothing:", np.count_nonzero(y_daily_pred == 1))
print("perfect:", np.count_nonzero(y_daily_pred == 2))
print("right:", np.count_nonzero(y_daily_pred == 3))

#dictionary to store in the collection
res_dict = { "left_poses": np.count_nonzero(y_daily_pred == 0), "nothing_poses": np.count_nonzero(y_daily_pred == 1), "perfect_poses": np.count_nonzero(y_daily_pred == 2), "right_poses": np.count_nonzero(y_daily_pred == 3)}

#delete all documents inside user results collection in order to put the new ones in 
x = user_res_col.delete_many({})
x = user_res_col.insert_one(res_dict)

