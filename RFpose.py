#ML script for pose detection

# Importing the libraries
#import mysql.connector as mariadb
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import itertools
import pymongo

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

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
print(X_test[0])

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

'''def plot_confusion_matrix(cm,
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

plot_confusion_matrix(cm = cm, normalize = False, target_names = ['L', 'N', 'P', 'R'], title = 'Confusion Matrix')'''

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