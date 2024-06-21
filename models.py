import numpy as np
import pandas as pd

# def matrix_multiply(X,Y):
#     if X.shape[1] == Y.shape[0]:
#         result = np.empty((X.shape[0], Y.shape[1]))
        
#         for i in range(X.shape[0]):
#             for j in range(Y.shape[1]):
#                 result[i, j] =  0
#                 for a in X[i]:
#                     for b in Y[:, j]:
#                         result[i, j] += a*b

#         return result
#     else:
#         raise KeyError('check the dimension of matrix')
    
def trans_matrix(X):
    # n, m = X.shape[0], X.shape[1]
    # result = np.empty((m, n))

    # for i in range(n):
    #     for j in range(m):
    #         result[j, i] = X[i, j]
    return np.transpose(X)

def inv_matrix(X):
    return np.linalg.inv(X)

class linear_model:
    def  __init__(self):
        pass
        
    class linear_regression:
        def __init__(self):
            self.beta = None
            self.Xtrain = None
            self.ytrain = None

        def fit(self, X, y):
            if X.shape[0] == y.shape[0]:
                self.Xtrain = X
                self.ytrain = y
                self.beta = inv_matrix(trans_matrix(X)@X)@trans_matrix(X)@y
            else:
                raise ValueError('check dimension')
            
        def metrics(self):
            R_2 = 0
            ssr = 0
            sse = 0
            for i in self.ytrain:
                sse += (self.predict(self.Xtrain) - i)**2
                ssr += (self.predict(self.Xtrain) - np.mean(self.ytrain))**2
            sst = sse + ssr

            mse = sse/ (self.Xtrain.shape[0] - self.Xtrain.shape[1] - 1)

            print(f'mse: {mse}')
            print(f'R_2: {ssr/sst}')

            


        def predict(self, X):
            print(type(self.X))
            print(type(self.beta))
            y_pred = X@self.beta
            return y_pred
        

from sklearn import datasets
from sklearn.model_selection import train_test_split

data = pd.read_csv('boston_data.csv')
X = data.iloc[:,:-1]
y = data.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr_our = linear_model.linear_regression()
lr_our.fit(X_train, y_train)
lr_our.metrics()

