# %load ./src/linear_models.py
import numpy as np
import pandas as pd

def build_design_matrix(X, PHI):
    """Build the design matrix
    
    Parameter
    ---------
    X: array-like,
        Input vector
    PHI: dictionary,
        basis functions
        
    Returns
    -------
    dm: pandas-DataFrame
        design matrix
    """
    dm = pd.DataFrame(np.array([
        PHI["phi_0"](X),
        PHI["phi_1"](X),
        PHI["phi_2"](X),
    ]).T, columns=list(PHI.keys()))
    return dm

def maximum_likelihood_regression(X, PHI, Y, reg=0):
    """Calculate maximum likelihood parameter
    
    Parameter
    ---------
    X: array-like,
        input vector
    PHI: dictionary,
        basis functions
    Y: array-like,
        target vector
    reg: float, default = 0
        parameter of l2 regularization
    
    Returns
    -------
    ml_params: array-like
        maximum likelihood parameter
    """
    # get the design matrix
    DM = build_design_matrix(X, PHI).values
    
    # get the regularizatioin weights
    L = np.identity(DM.shape[1]) * reg
    
    # convert to matrices to ease multiplication
    DMM = np.matrix(DM)
    LM = np.matrix(L)
    YM = np.matrix(np.atleast_2d(Y).T)
    
    # compute parameter
    WM = np.linalg.inv(DMM.T*DM+LM)*DMM.T*YM
    return np.squeeze(np.asarray(WM))