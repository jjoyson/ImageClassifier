import numpy as np
import pandas as pd
import pickle
import keras
from sklearn.model_selection import train_test_split
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.models import load_model

from keras.models import Sequential
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score

print("Loading Dataset...")
X = pd.read_csv("data/X.csv", sep = " ", header = None, dtype = float)
X = X.values
print("Done")

people = ["Bush", "Williams"]
y_filenames = ["data/y_bush_vs_others.csv", "data/y_williams_vs_others.csv"]
# bush = [0,0]
# williams = [0,0]

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

# bush_train = {'f1': 0, 'summary': 'People: Bush Activates'}
# bush_test = {'f1': 0.9258160237388724, 'summary': 'People: Bush Activates'}

# williams_train =  {'f1': 0, 'summary': 'People: Williams Activates'}
# williams_test = {'f1': 0.56, 'summary': 'People: Williams Activates'}

# # Williams Test: {'f1': 0.56, 'summary': 'People: Williams Activate: tanh Kernel: 4 Drop: 0.4 Filter: 512 Pool: 4'}
# # {'f1': 1.0, 'summary': 'People: Bush Activate1: tanh Activate2: tanh Activate3: relu Activate4: relu Loss: binary_crossentropy'}
# williams_test = {'f1': 0.6666666666666667, 'summary': 'People: Williams Activate: tanh Kernel: 4 Drop: 0.2 Filter: 512 Pool: 4'}
# bush_test = {'f1': 0.9266862170087976, 'summary': 'People: Bush Activate: tanh Kernel: 4 Drop: 0.5 Filter: 512 Pool: 3'}
# # Williams Test: {'f1': 0.6923076923076924, 'summary': 'People: Williams Activate1: relu Activate2: tanh Activate3: relu Activate4: sigmoid Loss: binary_crossentropy'}

bush_train = {'f1': 0}
bush_test = {'f1': 0}
williams_train =  {'f1': 0}
williams_test = {'f1': 0}
bush_f1 = [0,0]
williams_f1 = [0,0]

rs = 0000

# bush_f1 = [1.0,0.9271137026239066]
# williams_f1 = [1.0,0.7142857142857143]

# pickle.dump((bush_f1), open('bush.pkl', 'wb'))
# pickle.dump((williams_f1), open('williams.pkl', 'wb'))

while(True):
    for i in range(len(people)):
        y_temp = pd.read_csv(y_filenames[i], header = None)
        y = y_temp.values.ravel()
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 1./3,random_state = rs,shuffle=True,stratify=y)
        X_train_reshape = X_train.reshape(-1,64,64,1)
        X_test_reshape = X_test.reshape(-1,64,64,1)
        # y_train_reshape = one_hot_label(y_train)
        # y_test_reshape = one_hot_label(y_test)
        file = open("finish.txt", "a")   # for testing
        print("----------------------------------------------------------------------------------------")
        activate1 = "tanh"
        activate2 = "tanh"
        activate3 = "tanh"
        activate4 = "tanh"
        pool = 3
        drop = .5
        model = Sequential()
        model.add(Conv2D(input_shape = (64,64,1), filters = 32, kernel_size = 3, strides = 1, padding = 'same', activation = activate1))
        model.add(MaxPooling2D(pool_size = pool, padding = 'same'))
        model.add(Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = 'same', activation = activate2))
        model.add(MaxPooling2D(pool_size = pool, padding = 'same'))
        model.add(Conv2D(filters = 128, kernel_size = 3, strides = 1, padding = 'same', activation = activate3))
        model.add(MaxPooling2D(pool_size = pool, padding = 'same'))
        model.add(Dropout(drop))
        model.add(Flatten())
        model.add(Dense(128, activation = activate4))
        model.add(Dropout(drop - .1))
        model.add(Dense(1, activation = "sigmoid"))
        model.compile(optimizer = keras.optimizers.Adam(),loss="binary_crossentropy", metrics = ['accuracy'])
        model.fit(x = X_train_reshape, y = y_train, epochs = 30, batch_size = 64)
        y_train_predict = model.predict_classes(X_train_reshape)
        y_test_predict = model.predict_classes(X_test_reshape)
        if(people[i] == "Bush"):  # for testing
            if(bush_train['f1'] <= f1_score(y_train, y_train_predict)):
                bush_train['f1'] = f1_score(y_train, y_train_predict)
                if(bush_test['f1'] <= f1_score(y_test, y_test_predict)):
                    bush_test['f1'] = f1_score(y_test, y_test_predict)+
                    model.save("bush.h5")
                    bush_f1[0] = bush_train['f1']
                    bush_f1[1] = bush_test['f1']
                    pickle.dump((bush_f1), open('bush.pkl', 'wb'))
                    file.write("\nBush Train: "+str(bush_train)+"\nBush Test: "+str(bush_test)+ "act: "+activate1)
        if(people[i] == "Williams"): # for testing
            if(williams_train['f1'] <= f1_score(y_train, y_train_predict)):
                williams_train['f1'] = f1_score(y_train, y_train_predict)
                if(williams_test['f1'] <= f1_score(y_test, y_test_predict)):
                    williams_test['f1'] = f1_score(y_test, y_test_predict)
                    model.save("williams.h5")
                    williams_f1[0] = williams_train['f1']
                    williams_f1[1] = williams_test['f1']
                    pickle.dump((williams_f1), open('williams.pkl', 'wb'))
                    file.write("\nWilliams Train: "+str(williams_train)+"\nWilliams Test: "+str(williams_test)+ "act: "+activate1)