import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.preprocessing import LabelBinarizer
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


########## REGRESSOR ##########
class LeastSquaresRegressor(BaseEstimator, RegressorMixin):
    
    def __init__(self, phi, reg=0):
        self.phi = phi
        self.reg = reg
        self.w = None
        
    def fit(self, X, y):
        self.w = maximum_likelihood_regression(X, self.phi, y, reg=self.reg)
        
    def predict(self, X):
        return np.dot(self.w, build_design_matrix(X, self.phi).T)


class BayesRegressor(BaseEstimator, RegressorMixin):
    
    def __init__(self, phi, a, b_std, M0=None):
        # misc attributes
        ndim = len(phi)
        self.phi = phi
        self.b = 1. / (b_std * b_std)
        self._seen_samples = 0
        
        if not M0:
            M0 = stats.norm.rvs(size=ndim)
        # update belief
        self._update_belief(
            np.matrix(M0).T,
            np.matrix(1./a*np.identity(ndim)),
        )
    
    def predict(self, X):
        DM = np.matrix(build_design_matrix(X, self.phi))
        mu = np.squeeze(np.asarray(self.M_prior.T*DM.T))
        sig = np.squeeze(np.asarray(np.sqrt(np.diag(DM*self.S_prior*DM.T))))
        return mu, sig
    
    def fit(self, X, Y):
        S_post = self._update_S(X)
        M_post = self._update_M(X, Y)
        self.b = self._update_beta(X, Y)
        self._update_belief(M_post, S_post)
    
    def _update_belief(self, M_post, S_post):
        self.M_prior = M_post
        self.S_prior = S_post
        self.prior = stats.multivariate_normal(
            np.squeeze(np.asarray(self.M_prior)), 
            self.S_prior,
        )
        return True
    
    def _update_beta(self, X, Y):
        DM = np.matrix(build_design_matrix(X, self.phi))
        self._seen_samples += 1
        return self._seen_samples / np.linalg.norm(np.matrix(Y).T - DM*self.M_prior)
    
    def _update_S(self, X):
        DM = np.matrix(build_design_matrix(X, self.phi))
        return np.linalg.inv(np.linalg.inv(self.S_prior) + self.b*DM.T*DM)
    
    def _update_M(self, X, Y):
        # get design matrix
        DM = np.matrix(build_design_matrix(X, self.phi))
        # reshape Y
        Y_ = np.matrix(np.atleast_2d(Y).reshape(-1, 1))
        # return M_post
        return self._update_S(X) * (np.linalg.inv(self.S_prior)*self.M_prior + self.b*DM.T*Y_)
    
########## CLASSIFIER ##########
class LeastSquaresClassifier(BaseEstimator, ClassifierMixin):
    """Linear Classifer trained on Least Squares"""
    
    def __init__(self):
        self.lb = LabelBinarizer()
        self.W_opt = None
        self.binary = False
    
    def fit(self, X, y):
        """Fit the Classifier to the data set
        
        Parameter
        ---------
        X: array-like
            input vectors
            
        y: array-like
            target vector
        """
        X_ = self._add_dummy(X)
        # inspect if two or more classes
        if len(np.unique(y)) > 2:
            self.binary = False
            T_ = np.matrix(self.lb.fit_transform(y))
        else:
            self.binary = True
            T_ = np.matrix(y)
        self.W_opt = np.linalg.pinv(X_)*T_
        
    def predict(self, X):
        """Predict the class of unseen data
        
        Parameter
        ---------
        X: array-like
            unsee input vectors
            
        Returns
        -------
        preds: array-like
            classes of unseen data
        """
        X_ = self._add_dummy(X)
        preds = (self.W_opt.T * X_.T).T
        if self.binary:
            return np.sign(preds)
        else:
            return self.lb.inverse_transform(preds)
    
    def _add_dummy(self, X):
        """Add a dummy input for bias weight
        
        Parameter
        ---------
        X: array-like
            input vectors
            
        Returns
        -------
        X_: array-like
            input vectors including
            ones for bias weight
        """
        return np.matrix(np.hstack([np.ones((X.shape[0],1)), X]))
    
    def discriminant_functions(self):
        """Print the discriminant function"""
        if self.W_opt is None:
            raise ValueError("You first have to call fit")
        # build the template
        template = "f_{}(x) = {:.03f}"
        for i in range(1, self.W_opt.shape[0]):
            template += " {:+.03f}" + "x_{}".format(i)

        # fill the template
        for idx, row in enumerate(self.W_opt.T, start=1):
            params = [idx]
            params += list(np.squeeze(np.asarray(row)))
            print(template.format(*params))
    
########## UTILS ##########
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
    results = []
    for k in PHI.keys():
        results.append(PHI[k](X))
    dm = pd.DataFrame(
        np.array(results).T,
        columns=PHI.keys()
    )
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

def plot_posterior(br, x_min, x_max):
    x = np.linspace(x_min, x_max, 100)
    mu, sig = br.predict(x)
    plt.plot(x, mu)
    plt.fill_between(
        x,
        mu - 3*sig,
        mu + 3*sig,
        alpha=.25,
    );
    
    
def plot_belief(br, lower, upper, resolution=100):
    xx, yy = np.meshgrid(
        np.linspace(lower, upper, resolution),
        np.linspace(lower, upper, resolution)
    )
    zz = br.prior.pdf(np.array([xx.ravel(), yy.ravel()]).T).reshape(xx.shape)
    plt.imshow(zz, cmap="inferno")
    plt.xticks(np.linspace(0, resolution,len(np.arange(lower, upper+.5, .5))), 
               map(str, np.arange(lower, upper+.5, .5)), 
               rotation=45,
               ha="right",
              )
    plt.yticks(np.linspace(0, resolution,len(np.arange(lower, upper+.5, .5))), 
               map(str, np.arange(lower, upper+.5, .5)), 
               rotation=45,
               ha="right",
              )