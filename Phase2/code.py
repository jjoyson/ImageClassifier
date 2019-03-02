import numpy as np
import pandas as pd
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA

# print("Loading Dataset...")
# X = pd.read_csv("data/X.csv", sep = " ", header = None, dtype = float)
# X = X.values
# print("Done")

# people = ["Bush", "Williams"]
# y_filenames = ["data/y_bush_vs_others.csv", "data/y_williams_vs_others.csv"]

# neighbors = [1,3,5]
# kernel = ["linear"]
# degree = [1, 2, 3]
# C = [-1]
# gamma = [-3,-2,-1,0]
# all_components = min(len(X),len(X[0]))
# scale = 50
# # 82, 163, 244 tried
# limit = 4096
# svd_solver = ["auto", "full"]
# white = [False]

# for i in range(len(neighbors)):
#     for j in range(len(people)):
#         y = pd.read_csv(y_filenames[j], header = None)
#         y_person = y.values.ravel()
            
#         print("KNeighboresClassifier: "+str(neighbors[i])+" neighbors")
#         model = KNeighborsClassifier(n_neighbors=neighbors[i])
#         print(people[j]+ " Model")
#         model.fit(X, y_person) 
#         print(people[j]+ " Fit")

#         stratified_cv_results = cross_validate(estimator = model, X = X, y = y_person, cv=StratifiedKFold(n_splits = 3, shuffle=True, random_state = 9472), scoring=('precision', 'recall', 'f1'), return_train_score=False)
#         print(people[j]+ ":\n",stratified_cv_results)

# for i in range(len(kernel)):
#     for k in range(len(C)):
#         for j in range(len(gamma)):
#             for l in range(len(degree)):
#                 for m in range(len(people)):
#                     y = pd.read_csv(y_filenames[m], header = None)
#                     y_person = y.values.ravel()
#                     model = SVC(C = 10**C[k], kernel = kernel[i], gamma = 10**gamma[j], degree = degree[l])
#                     model.fit(X, y_person)
#                     stratified_cv_results = cross_validate(estimator = model, X = X, y = y_person, n_jobs = -1, cv=StratifiedKFold(n_splits = 3, shuffle=True, random_state = 9472), scoring=('precision', 'recall', 'f1'), return_train_score=False)
#                     print(people[m], " C 10^",C[k]," Kernel ",kernel[i], " Gamma 10^",gamma[j], " Degree ",degree[l])
#                     print(people[m], ":\n",stratified_cv_results,"\n")
#                     f.write(str(people[m])+" C "+str(C[k])+" Gamma "+str(gamma[j])+" Degree "+str(degree[l]))
#                     f.write("\n"+str(stratified_cv_results)+"\n")

# bush_f1 = [0.1416768033,0.05406815,0.03204905,0.6404965167]
# williams_f1 = [0.03508772,0,0,0.5239316233]
# pickle.dump((bush_f1), open('bush.pkl', 'wb'))
# pickle.dump((williams_f1), open('williams.pkl', 'wb'))

# bush_neighbor_f1_mean = {'1 n_components': 48, '5 svd_solver': 'full', '3 white': True, '3 svd_solver': 'full', '1 svd_solver': 'full', '5 white': False, '1 Neighbor': 0.17349324496336768, '3 n_components': 48, '1 white': False, '5 Neighbor': 0.050309804550932875, '5 n_components': 23, '3 Neighbor': 0.10920160509201605}
# williams_neighbor_f1_mean = {'1 n_components': 13, '5 svd_solver': 'auto', '3 white': True, '3 svd_solver': 'auto', '1 svd_solver': 'auto', '5 white': True, '1 Neighbor': 0.24524756852343063, '3 n_components': 13, '1 white': False, '5 Neighbor': 0.07017543859649122, '5 n_components': 13, '3 Neighbor': 0.15102974828375287}

# # 82,163,244,1-69
# bush_neighbor_f1_mean = {'1 n_components': 48, '5 svd_solver': 'full', '3 white': True, '3 svd_solver': 'auto', '1 svd_solver': 'full', '5 white': False, '1 Neighbor': 0.17349324496336768, '3 n_components': 58, '1 white': False, '5 Neighbor': 0.050309804550932875, '5 n_components': 23, '3 Neighbor': 0.11594090930374117}
# williams_neighbor_f1_mean = {'1 n_components': 13, '5 svd_solver': 'auto', '3 white': True, '3 svd_solver': 'auto', '1 svd_solver': 'auto', '5 white': True, '1 Neighbor': 0.24524756852343063, '3 n_components': 13, '1 white': False, '5 Neighbor': 0.07017543859649122, '5 n_components': 13, '3 Neighbor': 0.15102974828375287}


bush_f1 = [0.17349324496336768,0.648431703843807]
williams_f1 = [0.24524756852343063,0.5239316239316238]
pickle.dump((bush_f1), open('bush.pkl', 'wb'))
pickle.dump((williams_f1), open('williams.pkl', 'wb'))

# # for i in range(81,limit,300):
# #     print("Component: ",i)
# #     for j in range(len(svd_solver)):
# #         for z in range(len(white)):
# #             pca = PCA(n_components=i, svd_solver=svd_solver[j], whiten= white[z]).fit_transform(X)
# #             for k in range(len(neighbors)):
# #                 for l in range(len(people)):
# #                     y = pd.read_csv(y_filenames[l], header = None)
# #                     y_person = y.values.ravel()
# #                     model = KNeighborsClassifier(n_neighbors=neighbors[k])
# #                     model.fit(pca, y_person) 
# #                     stratified_cv_results = cross_validate(estimator = model, X = pca, y = y_person, n_jobs = -1, cv=StratifiedKFold(n_splits = 3, shuffle=True, random_state = 9472), scoring=('precision', 'recall', 'f1'), return_train_score=False)
# #                     if(neighbors[k] == 1):
# #                         if(people[l] == "Bush"):
# #                             if(np.mean(stratified_cv_results['test_f1']) > bush_neighbor_f1_mean['1 Neighbor']):
# #                                 bush_neighbor_f1_mean['1 Neighbor'] = np.mean(stratified_cv_results['test_f1'])
# #                                 bush_neighbor_f1_mean['1 n_components'] = i
# #                                 bush_neighbor_f1_mean['1 svd_solver'] = svd_solver[j]
# #                                 bush_neighbor_f1_mean['1 white'] = white[z]
# #                         else:
# #                             if(np.mean(stratified_cv_results['test_f1']) > williams_neighbor_f1_mean['1 Neighbor']):
# #                                 williams_neighbor_f1_mean['1 Neighbor'] = np.mean(stratified_cv_results['test_f1'])
# #                                 williams_neighbor_f1_mean['1 n_components'] = i
# #                                 williams_neighbor_f1_mean['1 svd_solver'] = svd_solver[j]
# #                                 williams_neighbor_f1_mean['1 white'] = white[z]
# #                     elif(neighbors[k] == 3):
# #                         if(people[l] == "Bush"):
# #                             if(np.mean(stratified_cv_results['test_f1']) > bush_neighbor_f1_mean['3 Neighbor']):
# #                                 bush_neighbor_f1_mean['3 Neighbor'] = np.mean(stratified_cv_results['test_f1'])
# #                                 bush_neighbor_f1_mean['3 n_components'] = i
# #                                 bush_neighbor_f1_mean['3 svd_solver'] = svd_solver[j]
# #                                 bush_neighbor_f1_mean['3 white'] = white[z]
# #                         else:
# #                             if(np.mean(stratified_cv_results['test_f1']) > williams_neighbor_f1_mean['3 Neighbor']):
# #                                 williams_neighbor_f1_mean['3 Neighbor'] = np.mean(stratified_cv_results['test_f1'])
# #                                 williams_neighbor_f1_mean['3 n_components'] = i
# #                                 williams_neighbor_f1_mean['3 svd_solver'] = svd_solver[j]
# #                                 williams_neighbor_f1_mean['3 white'] = white[z]
# #                     else:
# #                         if(people[l] == "Bush"):
# #                             if(np.mean(stratified_cv_results['test_f1']) > bush_neighbor_f1_mean['5 Neighbor']):
# #                                 bush_neighbor_f1_mean['5 Neighbor'] = np.mean(stratified_cv_results['test_f1'])
# #                                 bush_neighbor_f1_mean['5 n_components'] = i
# #                                 bush_neighbor_f1_mean['5 svd_solver'] = svd_solver[j]
# #                                 bush_neighbor_f1_mean['5 white'] = white[z]
# #                         else:
# #                             if(np.mean(stratified_cv_results['test_f1']) > williams_neighbor_f1_mean['5 Neighbor']):
# #                                 williams_neighbor_f1_mean['5 Neighbor'] = np.mean(stratified_cv_results['test_f1'])
# #                                 williams_neighbor_f1_mean['5 n_components'] = i
# #                                 williams_neighbor_f1_mean['5 svd_solver'] = svd_solver[j]
# #                                 williams_neighbor_f1_mean['5 white'] = white[z]
# #     print("Final Bush ", bush_neighbor_f1_mean, " Final Williams ", williams_neighbor_f1_mean)

# bush_SVC_f1_mean = {'n_components': 0, 'svd_solver': 'temp', 'white': False, 'gamma' : 0, 'C' : 0, 'degree' : 0, 'kernel' : 'temp', 'f1' : 0}
# williams_SVC_f1_mean = {'n_components': 0, 'svd_solver': 'temp', 'white': False, 'gamma' : 0, 'C' : 0, 'degree' : 0, 'kernel' : 'temp', 'f1' : 0}

# bush_SVC_f1_mean = {'kernel': 'temp', 'C': 0, 'n_components': 0, 'degree': 0, 'f1': 0, 'white': False, 'gamma': 0, 'svd_solver': 'temp'}
# williams_SVC_f1_mean = {'kernel': 'linear', 'C': 1, 'n_components': 52, 'degree': 0, 'f1': 0.03508771929824561, 'white': True, 'gamma': 0, 'svd_solver': 'full'}

# bush_SVC_f1_mean = {'kernel': 'linear', 'C': 0.1, 'n_components': 2181, 'degree': 0, 'f1': 0.6484317038438072, 'white': False, 'gamma': 0, 'svd_solver': 'randomized'}
# williams_SVC_f1_mean = {'kernel': 'linear', 'C': 0.1, 'n_components': 2181, 'degree': 0, 'f1': 0.5172839506172839, 'white': False, 'gamma': 0, 'svd_solver': 'full'}

# for i in range(4096,2000,-350):
#     file = open("temp.txt", "a") 
#     print("Component: ",i)
#     for j in range(len(svd_solver)):
#         for z in range(len(white)):
#             pca = PCA(n_components=i, svd_solver = svd_solver[j], whiten=white[z]).fit_transform(X)
#             for x in range(len(kernel)):
#                 for a in range(len(C)):
#                     print(kernel[x], C[a])
#                     if( kernel[x] == "linear"):
#                         for u in range(len(people)):
#                             y = pd.read_csv(y_filenames[u], header = None)
#                             y_person = y.values.ravel()
#                             model = SVC(C = 10**C[a], kernel = kernel[x])
#                             model.fit(pca, y_person)
#                             stratified_cv_results = cross_validate(estimator = model, X = pca, y = y_person, n_jobs = -1, cv=StratifiedKFold(n_splits = 3, shuffle=True, random_state = 9472), scoring=('precision', 'recall', 'f1'), return_train_score=False)
#                             print(people[u])
#                             if(people[u] == "Bush"):
#                                 if(np.mean(stratified_cv_results['test_f1']) > bush_SVC_f1_mean['f1']):
#                                     bush_SVC_f1_mean['f1'] = np.mean(stratified_cv_results['test_f1'])
#                                     bush_SVC_f1_mean['n_components'] = i
#                                     bush_SVC_f1_mean['svd_solver'] = svd_solver[j]
#                                     bush_SVC_f1_mean['white'] = white[z]
#                                     bush_SVC_f1_mean['kernel'] = kernel[x]
#                                     bush_SVC_f1_mean['C'] = 10**C[a]
#                             else:
#                                 if(np.mean(stratified_cv_results['test_f1']) > williams_SVC_f1_mean['f1']):
#                                     williams_SVC_f1_mean['f1'] = np.mean(stratified_cv_results['test_f1'])
#                                     williams_SVC_f1_mean['n_components'] = i
#                                     williams_SVC_f1_mean['svd_solver'] = svd_solver[j]
#                                     williams_SVC_f1_mean['white'] = white[z]
#                                     williams_SVC_f1_mean['kernel'] = kernel[x]
#                                     williams_SVC_f1_mean['C'] = 10**C[a]
#                     else:
#                         for v in range(len(gamma)):
#                             if( kernel[x] == "rbf"):
#                                 for u in range(len(people)):
#                                     y = pd.read_csv(y_filenames[u], header = None)
#                                     y_person = y.values.ravel()
#                                     model = SVC(C = 10**C[a], kernel = kernel[x], gamma = 10**gamma[v])
#                                     model.fit(pca, y_person)
#                                     stratified_cv_results = cross_validate(estimator = model, X = pca, y = y_person, n_jobs = -1, cv=StratifiedKFold(n_splits = 3, shuffle=True, random_state = 9472), scoring=('precision', 'recall', 'f1'), return_train_score=False)
#                                     if(people[u] == "Bush"):
#                                         if(np.mean(stratified_cv_results['test_f1']) > bush_SVC_f1_mean['f1']):
#                                             bush_SVC_f1_mean['f1'] = np.mean(stratified_cv_results['test_f1'])
#                                             bush_SVC_f1_mean['n_components'] = i
#                                             # bush_SVC_f1_mean['svd_solver'] = svd_solver[j]
#                                             # bush_SVC_f1_mean['white'] = white[z]
#                                             bush_SVC_f1_mean['kernel'] = kernel[x]
#                                             bush_SVC_f1_mean['C'] = 10**C[a]
#                                             bush_SVC_f1_mean['gamma'] = 10**gamma[v]
#                                     else:
#                                         if(np.mean(stratified_cv_results['test_f1']) > williams_SVC_f1_mean['f1']):
#                                             williams_SVC_f1_mean['f1'] = np.mean(stratified_cv_results['test_f1'])
#                                             williams_SVC_f1_mean['n_components'] = i
#                                             # williams_SVC_f1_mean['svd_solver'] = svd_solver[j]
#                                             # williams_SVC_f1_mean['white'] = white[z]
#                                             williams_SVC_f1_mean['kernel'] = kernel[x]
#                                             williams_SVC_f1_mean['C'] = 10**C[a]
#                                             williams_SVC_f1_mean['gamma'] = 10**gamma[v]
#                             else:
#                                 for r in range(len(degree)):
#                                     for u in range(len(people)):
#                                         y = pd.read_csv(y_filenames[u], header = None)
#                                         y_person = y.values.ravel()
#                                         model = SVC(C = 10**C[a], kernel = kernel[x], gamma = 10**gamma[v], degree = degree[r])
#                                         model.fit(pca, y_person)
#                                         stratified_cv_results = cross_validate(estimator = model, X = pca, y = y_person, n_jobs = -1, cv=StratifiedKFold(n_splits = 3, shuffle=True, random_state = 9472), scoring=('precision', 'recall', 'f1'), return_train_score=False)
#                                         if(people[u] == "Bush"):
#                                             if(np.mean(stratified_cv_results['test_f1']) > bush_SVC_f1_mean['f1']):
#                                                 bush_SVC_f1_mean['f1'] = np.mean(stratified_cv_results['test_f1'])
#                                                 bush_SVC_f1_mean['n_components'] = i
#                                                 # bush_SVC_f1_mean['svd_solver'] = svd_solver[j]
#                                                 # bush_SVC_f1_mean['white'] = white[z]
#                                                 bush_SVC_f1_mean['kernel'] = kernel[x]
#                                                 bush_SVC_f1_mean['C'] = 10**C[a]
#                                                 bush_SVC_f1_mean['gamma'] = 10**gamma[v]
#                                                 bush_SVC_f1_mean['degree'] = degree[r]
#                                         else:
#                                             if(np.mean(stratified_cv_results['test_f1']) > williams_SVC_f1_mean['f1']):
#                                                 williams_SVC_f1_mean['f1'] = np.mean(stratified_cv_results['test_f1'])
#                                                 williams_SVC_f1_mean['n_components'] = i
#                                                 # williams_SVC_f1_mean['svd_solver'] = svd_solver[j]
#                                                 # williams_SVC_f1_mean['white'] = white[z]
#                                                 williams_SVC_f1_mean['kernel'] = kernel[x]
#                                                 williams_SVC_f1_mean['C'] = 10**C[a]
#                                                 williams_SVC_f1_mean['gamma'] = 10**gamma[v]   
#                                                 williams_SVC_f1_mean['degree'] = degree[r]                                                                
#             print("Final Bush ", bush_SVC_f1_mean, " Final Williams ", williams_SVC_f1_mean)
#     file.write("\nBush: "+str(bush_SVC_f1_mean)+"\nWilliams: "+str(williams_SVC_f1_mean))
#     file.close() 

# # for i in range(29,limit,1):
# #     file = open("temp.txt", "a") 
# #     print("Component: ",i)
# #     for j in range(len(svd_solver)):
# #         for z in range(len(white)):
# #             pca = PCA(n_components=i, svd_solver=svd_solver[j], whiten= white[z]).fit_transform(X)
# #             for u in range(len(people)):
# #                 y = pd.read_csv(y_filenames[u], header = None)
# #                 y_person = y.values.ravel()
# #                 model = SVC(kernel = "poly", C = 1, gamma = .1, degree = .1)
# #                 model.fit(pca, y_person)
# #                 stratified_cv_results = cross_validate(estimator = model, X = pca, y = y_person, n_jobs = -1, cv=StratifiedKFold(n_splits = 3, shuffle=True, random_state = 9472), scoring=('precision', 'recall', 'f1'), return_train_score=False)
# #                 if(people[u] == "Bush"):
# #                     if(np.mean(stratified_cv_results['test_f1']) > bush_SVC_f1_mean['f1']):
# #                         bush_SVC_f1_mean['f1'] = np.mean(stratified_cv_results['test_f1'])
# #                         bush_SVC_f1_mean['n_components'] = i
# #                         bush_SVC_f1_mean['svd_solver'] = svd_solver[j]
# #                         bush_SVC_f1_mean['white'] = white[z]
# #                         bush_SVC_f1_mean['kernel'] = kernel[x]
# #                         bush_SVC_f1_mean['C'] = 10**C[a]
# #                         bush_SVC_f1_mean['gamma'] = 10**gamma[v]
# #                         bush_SVC_f1_mean['degree'] = degree[r]
# #                 else:
# #                     if(np.mean(stratified_cv_results['test_f1']) > williams_SVC_f1_mean['f1']):
# #                         williams_SVC_f1_mean['f1'] = np.mean(stratified_cv_results['test_f1'])
# #                         williams_SVC_f1_mean['n_components'] = i
# #                         williams_SVC_f1_mean['svd_solver'] = svd_solver[j]
# #                         williams_SVC_f1_mean['white'] = white[z]
# #                         williams_SVC_f1_mean['kernel'] = kernel[x]
# #                         williams_SVC_f1_mean['C'] = 10**C[a]
# #                         williams_SVC_f1_mean['gamma'] = 10**gamma[v]   
# #                         williams_SVC_f1_mean['degree'] = degree[r]                                                           
# #     print("Final Bush ", bush_SVC_f1_mean, " Final Williams ", williams_SVC_f1_mean)
# #     file.write("\nBush: "+str(bush_SVC_f1_mean)+"\nWilliams: "+str(williams_SVC_f1_mean))
# #     file.close() 