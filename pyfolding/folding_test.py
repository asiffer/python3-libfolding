import numpy as np
def squared_norm(X): 
    """
    Return the squared norm of a dataset
    
    Parameters:
    -----------
    `X:` ndarray (t,n)
    
    Return:
    -----------
    `X_square_norm =\[ \sqrt{\sum_{j=1}^{t}(X[:][t]^{2})}\]` array (1,n) (||X||² in the article)
    """
    return np.expand_dims((np.sum(np.square(X), axis=0)), axis=0) #shape (1,n)

def cov_norm(X, Y): #X shape (t,n) and Y shape (1,n)
    """
    Parameters:
    -----------
    `X:` ndarray (t,n)
    `Y:` ndarray (1,n)
    
    Return:
    -----------
    `cov_norm =` \[cov_norm =\frac{1}{n} \sum_{i=1}^{n} (X_i- \mu_x) \cdot (Y_i - \mu_{||X||^2})\]
    """
    sum_temp = 0
    for i in range(len(X.T)):
        sum_temp += (X.T[i]-np.mean(X,axis=1)) * (Y[0][i]-np.mean(Y))    
    return sum_temp/len(X.T) #shape (1,n)

def folding_test(X): #See Algorithm 1 and equation (5) in https://hal.archives-ouvertes.fr/hal-01951676/document
    """
    Return a number phi >1 if the dataset is unimodal and <1 if multimodal
    https://hal.archives-ouvertes.fr/hal-01951676/document equation (5)
    
    Parameters:
    -----------
    `X:` ndarray (n,t)
    
    Return:
    -----------
    `folding_score:` float
    """
    X = X.T
    D = np.trace(np.cov(X)) #float
    X_square_norm = squared_norm(X) #shape(1,n)
    s_2 = 0.5 * np.linalg.solve(np.cov(X), cov_norm(X,X_square_norm)) #shape (t,1)
    s_2=np.tile(s_2 ,(len(X.T), 1)).T #Turn it into a zshape (t,n) to subsrtact to each column of X
    X_reduced = np.sqrt(np.sum(np.square(X-s_2), axis=0))
    X_reduced = np.expand_dims(X_reduced,axis=0) #shape(1,n)
    return (np.var(X_reduced)/D) * (1 + len(X))**2 #float
