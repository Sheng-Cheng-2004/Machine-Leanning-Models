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
            ssr = 0
            sse = 0
            y_pred = self.predict(self.Xtrain)
            for i in range(len(self.ytrain)):
                sse += (y_pred[i] - self.ytrain[i])**2
                ssr += (y_pred[i] - np.mean(self.ytrain))**2

            sst = sse + ssr
            mse = sse/ (self.Xtrain.shape[0] - self.Xtrain.shape[1] - 1)
            print(f'mse: {mse}')
            print(f'R_2: {ssr/sst}')

        def predict(self, X):
            if isinstance(X, np.ndarray):
                y_pred = X@self.beta
                return y_pred
            else:
                raise ValueError('X is not np.ndarray')
            

class metrics:
    def mse(x1, x2):
        try:
            sse = 0
            for i in range(len(x1)):
                sse += (x1[i] - x2[i])**2
            mse = sse/len(x1)
            return mse
        except:
            pass
            
            
            
if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    data = pd.read_csv('boston_data.csv')
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    lr_our = linear_model.linear_regression()
    lr_our.fit(X_train, y_train)
    lr_our.metrics()

