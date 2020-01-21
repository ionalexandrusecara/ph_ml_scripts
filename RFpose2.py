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
all_left_torso_rotation = pd.DataFrame(list(left_torso_rotation_col.find())).head(53)
all_nothing = pd.DataFrame(list(nothing_col.find()))
all_right_head_flexion = pd.DataFrame(list(right_head_flexion_col.find()))
all_right_head_rotation = pd.DataFrame(list(right_head_rotation_col.find())).head(53)
all_right_shoulder_elevation = pd.DataFrame(list(right_shoulder_elevation_col.find())).head(53)
all_right_torso_rotation = pd.DataFrame(list(right_torso_rotation_col.find())).head(53)
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 500, criterion = 'gini', random_state = 0)
classifier.fit(X_train, y_train)

#Fit XGBoost
#from xgboost import XGBClassifier
#classifier = XGBClassifier()
#classifier.fit(X_train, y_train)

#from sklearn import svm
#classifier = svm.SVC(kernel='poly') # Linear Kernel
#classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):

    

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

plot_confusion_matrix(cm = cm, normalize = False, target_names = ['BS', 'FH', 'FS', 'HP', 'LHF', 
                                                                  'LHR', 'LSE', 'N', 'RHF', 'RHR', 'RSE', 'TC'], title = 'Confusion Matrix')

#SAVE THE MODEL
import pickle

with open('ph_trained_model.pkl', 'wb') as f:
    pickle.dump(classifier, f)
    
with open('ph_trained_model.pkl', 'rb') as f:
    classifier_loaded = pickle.load(f)
    
classifier_loaded


#startups_collection = mydb["startups"]
#Retrieve all documents from MongoDB collection - startups
#startups = pd.DataFrame(list(startups_collection.find()))

#for index, row in startups.iterrows():
#    if(row['predict_reach_series_a'] is None):
#        email = row['email']
#        X_entity = startups.loc[index, ['country', 'category', 'seed_raised', 'seed_participants', 'no_founders', 'no_degree_types', 'no_degree_subjects', 'no_degree_institutions', 'MBA_count', 'time_founded_to_first_funding', 'time_founded_to_first_milestone', 'time_first_funding_to_first_milestone']].values
#        X_entity = X_entity.reshape((1,12))
#        X_entity[:, 0] = labelencoder_X_1.transform(X_entity[:, 0])
#        X_entity[:, 1] = labelencoder_X_2.transform(X_entity[:, 1])
#        X_entity=onehotencoder.transform(X_entity).toarray()
#        prediction = classifier.predict(X_entity)
#        prediction = labelencoder_y.inverse_transform(prediction)
#        prediction_query = { "email": email}
#       new_prediction = { "$set": { "predict_reach_series_a": prediction[0] } }
#        startups_collection.update_one(prediction_query, new_prediction)
#
#print("Predictions made!")