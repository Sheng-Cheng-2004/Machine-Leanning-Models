import numpy as np
import pandas as pd

def mse(x1, x2):
    try:
        sse = 0
        for i in range(len(x1)):
            sse += (x1[i] - x2[i])**2
        mse = sse/len(x1)
        return mse
    except:
        print('error occur')

