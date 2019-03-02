import numpy as np
import pandas as pd
import pickle
import keras
from sklearn.model_selection import train_test_split
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.models import load_model

from keras.models import Sequential, load_model
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score

print("Loading Dataset...")
X = pd.read_csv("data/X.csv", sep = " ", header = None, dtype = float)
X = X.values
print("Done")

people = ["Bush", "Williams"]
y_filenames = ["data/y_bush_vs_others.csv", "data/y_williams_vs_others.csv"]

# def one_hot_label(y):
#     y_labels = []
#     for i in range(len(y)):
#         if y[i] == 1:
#             y_labels += [[1,0]]
#         else:
#             y_labels += [[0,1]]
#     return np.array(y_labels)

# def convert(y_predict):
#     new_y_predict = []
#     for i in range(len(y_predict)):
#         if(y_predict[i] < .5):
#             new_y_predict += [0.0]
#         else:
#             new_y_predict += [1.0]
#     return np.array(new_y_predict)

bush_train = {'f1': 0}
bush_test = {'f1': 0}
williams_train =  {'f1': 0}
williams_test = {'f1': 0}
bush_f1 = [0,0]
williams_f1 = [0,0]

rs = 9472

while(True):
    for i in range(len(people)):
        y_temp = pd.read_csv(y_filenames[i], header = None)
        y = y_temp.values.ravel()
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 1./3,random_state = rs,shuffle=True,stratify=y)
        X_train_reshape = X_train.reshape(-1,64,64,1)
        X_test_reshape = X_test.reshape(-1,64,64,1)
        file = open("finish2.txt", "a")   # for testing
        print("----------------------------------------------------------------------------------------")
        model = load_model("best_test.h5")
        model.fit(x = X_train_reshape, y = y_train, epochs = 20, batch_size = 64)
        y_train_predict = model.predict_classes(X_train_reshape)
        y_test_predict = model.predict_classes(X_test_reshape)
        if(people[i] == "Bush"):  # for testing
            if(bush_train['f1'] <= f1_score(y_train, y_train_predict)):
                bush_train['f1'] = f1_score(y_train, y_train_predict)
                if(bush_test['f1'] <= f1_score(y_test, y_test_predict)):
                    bush_test['f1'] = f1_score(y_test, y_test_predict)
                    model.save("bush.h5")
                    bush_f1[0] = bush_train['f1']
                    bush_f1[1] = bush_test['f1']
                    pickle.dump((bush_f1), open('bush.pkl', 'wb'))
                    file.write("\nBush Train: "+str(bush_train)+"\nBush Test: "+str(bush_test))
        if(people[i] == "Williams"): # for testing
            if(williams_train['f1'] <= f1_score(y_train, y_train_predict)):
                williams_train['f1'] = f1_score(y_train, y_train_predict)
                if(williams_test['f1'] <= f1_score(y_test, y_test_predict)):
                    williams_test['f1'] = f1_score(y_test, y_test_predict)
                    model.save("williams.h5")
                    williams_f1[0] = williams_train['f1']
                    williams_f1[1] = williams_test['f1']
                    pickle.dump((williams_f1), open('williams.pkl', 'wb'))
                    file.write("\nWilliams Train: "+str(williams_train)+"\nWilliams Test: "+str(williams_test))