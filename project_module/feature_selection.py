from scipy.stats import spearmanr, pearsonr
import numpy as np

def my_pearsonr(x, y):
    row, col = x.shape
    coef_arr = np.zeros(col)
    p_val_arr = np.zeros(col)
    
    for i in range(col):
        coef_arr[i], p_val_arr[i] = pearsonr(x[:, i], y)
        
    return coef_arr, p_val_arr

def my_spearmanr(x, y):
    coef_arr, p_val_arr = spearmanr(x, y)
    coef_arr = coef_arr[: -1, -1]
    p_val_arr = p_val_arr[: -1, -1]
    
    return coef_arr, p_val_arr
    

class SelectKBestByCoefficient():
    def __init__(self, k: int, method: str = 'pearson'):
        self.k = k
        if method == 'pearson':
            self.func = my_pearsonr
        elif method == 'spearman':
            self.func = my_spearmanr
        
    def fit(self, X, Y):
        x, y = np.array(X), np.array(Y)
        
        corr, _ = self.func(x, y)
        
        self.select_idx = np.argsort(np.abs(corr))[:: -1][: self.k]
        
        return self
        
    def transform(self, X):
        x = np.array(X)
        x_new = x[:, self.select_idx]
        
        return x_new