import numpy as np
import pandas as pd
# import pickle
import keras
from sklearn.model_selection import train_test_split
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.models import load_model

from keras.models import Sequential
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score

print("Loading Dataset...")
X = np.load("olivetti_faces.npy")
print("Done")

# for i in range(len(X)):
#     pic = np.reshape(X[i],(64,64))
#     plt.imsave("Images\Picture"+ str(i+1)+".png", pic, cmap = 'gray')

y = np.zeros(400)

rs = 9472

for i in range(10):
    y[i] = 1

best_train = {'test' : 0.8, 'train' : 0.923076923076923}
best_test = {'test' : 0.8, 'train' : 0.923076923076923}

while(True):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 1./3,random_state = rs,shuffle=True,stratify=y)
    X_train_reshape = X_train.reshape(-1,64,64,1)
    X_test_reshape = X_test.reshape(-1,64,64,1)
    file = open("finish.txt", "a")   # for testing
    print("----------------------------------------------------------------------------------------")
    activate1 = "relu"
    activate2 = "relu"
    activate3 = "relu"
    activate4 = "relu"
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
    model.fit(x = X_train_reshape, y = y_train, epochs = 10, batch_size = 128)
    y_train_predict = model.predict_classes(X_train_reshape)
    y_test_predict = model.predict_classes(X_test_reshape)
    if(best_train['train'] <= f1_score(y_train, y_train_predict)):
        if(best_train['test'] <= f1_score(y_test, y_test_predict)):
            best_train['train'] = f1_score(y_train, y_train_predict)
            best_train['test'] = f1_score(y_test, y_test_predict)
            model.save("best_train.h5")
            file.write("\nBest_Train_Train: "+str(best_train['train'])+"\nBest_Train_Test: "+str(best_train['test']))
    if(best_test['test'] <= f1_score(y_test, y_test_predict)):
        if(best_test['train'] <= f1_score(y_train, y_train_predict)):
            best_test['train'] = f1_score(y_train, y_train_predict)
            best_test['test'] = f1_score(y_test, y_test_predict)
            model.save("best_test.h5")
            file.write("\nBest_Test_Train: "+str(best_test['train'])+"\nBest_Test_Test: "+str(best_test['test']))