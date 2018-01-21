import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.preprocessing import LabelBinarizer
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


########## REGRESSOR ##########
class LeastSquaresRegressor(BaseEstimator, RegressorMixin):
    """Least Squares Regression
    
    Parameter
    ---------
    phi: dict,
        dict of design matrix transformations    
    reg: float
        factor of l2 regularization
    """
    def __init__(self, phi, reg=0):
        self.phi = phi
        self.reg = reg
        self.w = None
        
    def fit(self, X, y):
        """Fit the Regressor to the data set
        
        Parameter
        ---------
        X: array-like
            input vectors
            
        y: array-like
            target vector
        """
        self.w = maximum_likelihood_regression(X, self.phi, y, reg=self.reg)
        
    def predict(self, X):
        """Predict the value of unseen data
        
        Parameter
        ---------
        X: array-like
            unsee input vectors
            
        Returns
        -------
        preds: array-like
            estimated value of unseen data
        """
        return np.dot(self.w, build_design_matrix(X, self.phi).T)


class BayesRegressor(BaseEstimator, RegressorMixin):
    """Bayesian Regression
    
    Parameter
    ---------
    phi: dict,
        dict of design matrix transformations 
    a: float,
        used for covariance matrix initialization 
    b_std: float,
        used as prior of std 
    M0: array-like,
        prior means of the multivariate gauss
    """
    
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
        """Predict the value of unseen data
        
        Parameter
        ---------
        X: array-like
            unsee input vectors
            
        Returns
        -------
        mu: array-like,
            estimated means of unseen data
        sig: array-like,
            estimated std of unseen data
        """
        DM = np.matrix(build_design_matrix(X, self.phi))
        mu = np.squeeze(np.asarray(self.M_prior.T*DM.T))
        sig = np.squeeze(np.asarray(np.sqrt(np.diag(DM*self.S_prior*DM.T))))
        return mu, sig
    
    def fit(self, X, Y):
        """Fit the Regressor to the data set
        
        Parameter
        ---------
        X: array-like
            input vectors
        Y: array-like
            target vector
        """
        S_post = self._update_S(X)
        M_post = self._update_M(X, Y)
        self.b = self._update_beta(X, Y)
        self._update_belief(M_post, S_post)
    
    def _update_belief(self, M_post, S_post):
        """Update internal priors
        
        Parameter
        ---------
        M_post: array-like,
            posterior mus
        S_post: array-like,
            posterior stds
            
        Returns
        -------
        success: boolean,
            update was successful
        """
        self.M_prior = M_post
        self.S_prior = S_post
        self.prior = stats.multivariate_normal(
            np.squeeze(np.asarray(self.M_prior)), 
            self.S_prior,
        )
        return True
    
    def _update_beta(self, X, Y):
        """Update confidence
        
        Parameter
        ---------
        X: array-like,
            input vectors
        Y: array-like,
            target vectors
            
        Returns
        -------
        beta: float,
            std of beliefs
        """
        DM = np.matrix(build_design_matrix(X, self.phi))
        self._seen_samples += 1
        return self._seen_samples / np.linalg.norm(np.matrix(Y).T - DM*self.M_prior)
    
    def _update_S(self, X):
        """Update stds
        
        Parameter
        ---------
        X: array-like,
            input vectors
        
        Returns
        -------
        s_updated: array-like,
            updated std of beliefs
        """
        DM = np.matrix(build_design_matrix(X, self.phi))
        return np.linalg.inv(np.linalg.inv(self.S_prior) + self.b*DM.T*DM)
    
    def _update_M(self, X, Y):
        """Update mus
        
        Parameter
        ---------
        X: array-like,
            input vectors
        Y: array-like,
            target vectors
        
        Returns
        -------
        s_updated: array-like,
            updated mus of beliefs
        """
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
    
    
class FisherClassifier(BaseEstimator, ClassifierMixin):
    """Classifier based on Fisher's discriminant function"""
    
    def __init__(self):
        self.w = None
        self.th = None
        self.cls_greater_th = None
    
    def fit(self, X, y):
        """Fit the Classifier to the data set
        
        Parameter
        ---------
        X: array-like
            input vectors
        y: array-like
            target vector
        """
        # check for non binary classification
        cls = np.unique(y)
        if len(cls) != 2:
            raise NotImplementedError("Currently only two class separation implemented")

        # compute means and variances per class
        mu1 = np.mat(X[y==cls[0]]).mean(axis=0).T
        mu2 = np.mat(X[y==cls[1]]).mean(axis=0).T
        s1 = (X[y==cls[0]].T - mu1) * (X[y==cls[0]].T - mu1).T
        s2 = (X[y==cls[1]].T - mu2) * (X[y==cls[1]].T - mu2).T
        self.w = np.linalg.inv(s1+s2) * (mu2 - mu1)
        
        # compare mean of disctributions
        kde1 = self._kde(X[y==cls[0]] * self.w)
        kde2 = self._kde(X[y==cls[1]] * self.w)
        lower, upper = (X*self.w).min(), (X*self.w).max()
        peak1 = self._peak_kde(kde1, lower, upper)
        peak2 = self._peak_kde(kde2, lower, upper)
        if peak1 > peak2:
            self.cls_greater_th = {True: cls[0], False: cls[1]}
        else:
            self.cls_greater_th = {True: cls[1], False: cls[0]}
        
        # get optimal threshold
        self.th = self._optimal_theta(X, y, lower, upper)
        
    def predict(self, X, th=None):
        """Predict the class of unseen data
        
        Parameter
        ---------
        X: array-like
            unseen input vectors
        th: float, default None
            threshold in transformed space
            
        Returns
        -------
        pred: array-like,
            predicitons of unseen data
        """
        if not th:
            th = self.th
        x_1d = np.squeeze(np.asarray(np.mat(X) * self.w))
        pred = list(map(lambda p: self.cls_greater_th[p], x_1d >= th))
        return np.array(pred)
    
    def _kde(self, X):
        """Estimate a gaussian kernel
        
        Parameter
        ---------
        X: array-like,
            input vectors
        
        Returns
        -------
        kde: scipy.stats.gaussian_kde,
            kernel density estimate of input data
        """
        X = np.squeeze(np.asarray(X))
        return stats.gaussian_kde(X)
    
    def _optimal_theta(self, X, y, lower, upper, res=100):
        """Optimal threshold for classification
        
        Parameter
        ---------
        X: array-like,
            input vectors
        y: array-like,
            target vectors
        lower: float,
            lower value for threshold
        upper: float,
            upper value for threshold
        res: int, default=100,
            stepsize of threshold calculation
        
        Returns
        -------
        th: float,
            optimal threshold
        """
        errors = []
        for t in np.linspace(lower, upper, res):
            preds = self.predict(X, th=t)
            errors.append(np.sum(preds != y))
        errors = np.array(errors)
        best_th_idx = int(np.median(np.argwhere(errors == np.min(errors))))
        return np.linspace(lower, upper, res)[best_th_idx]
        
    def _peak_kde(self, kde, lower, upper, res=100):
        """Peak of estimated probability density function
        
        Parameter
        ---------
        kde: scipy.stats.gaussian_kde,
            estimated kernel density
        lower: float,
            lower value for peak
        upper: float,
            upper value for peak
        res: int, default=100,
            stepsize of peak calculation
        
        Returns
        -------
        peak: float,
            argmax of probability density function
        """
        peak_idx = np.argmax(kde.pdf(np.linspace(lower, upper, res)))
        return np.linspace(lower, upper, res)[peak_idx]


class Perceptron(BaseEstimator, ClassifierMixin):
    """Perceptron for binary classification
    
    Parameter
    ---------
    phi: dict,
        basis functions
    w: array-like, default=None,
        initial weights 
    t: float, default=1
        laerning step size
    max_iter: int, default=100,
        maximal number of iterations    
    """
    
    def __init__(self, phi, w=None, t=1, max_iter=100):
        self.t = t
        self.phi = phi
        if w and (len(w) != len(phi)):
            raise ValueError("Number of weights ({}) must equal number of features ({})".format(len(w), len(phi)))
        if not w:
            self.w = np.random.normal(size=len(phi))
        else:
            self.w = np.array(w)
        self.max_iter = max_iter
        self.updates = 0
        self.w_history = [self.w.copy()]
            
    def fit(self, X, y):
        """Fit the Classifier to the data set
        
        Parameter
        ---------
        X: array-like
            input vectors
        y: array-like
            target vector
        """
        # check for binary classification
        cls = np.unique(y)
        if len(cls) != 2:
            raise NotImplementedError("Only implemented for n=2 classes")
            
        # build feature vector
        X_ = build_design_matrix(X, self.phi).values
        
        # reset updates
        self.updates = 0
        
        # iterate over epochs
        for epoch in range(1, self.max_iter+1):
            err = 0
            # iterate over complete dataset
            for row, label in zip(X_, y):
                pred = np.dot(self.w, row)

                # if missclassificaiton, update
                if np.sign(pred) != np.sign(label):
                    err += 1
                    self._update_w(row, label)
                    
            # early convergence
            if err == 0:
                print("Early success in epoch {} ({} updates)".format(epoch, self.updates))
                break
    
    def predict(self, X):
        """Predict the class of unseen data
        
        Parameter
        ---------
        X: array-like
            unseen input vectors
            
        Returns
        -------
        pred: array-like,
            predicitons of unseen data
        """
        X_ = build_design_matrix(np.asarray(X), self.phi).values
        return np.sign(np.dot(self.w, X_.T))
    
    def _update_w(self, row, label):
        """Update internal weights
        
        Parameter
        ---------
        row: array-like,
            single input vector
        label: int,
            target label
        """
        self.w += row * label
        self.updates +=1
        self.w_history.append(self.w.copy()) # otherwise just reference


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
    """Plot posterior distribution
    
    Parameter
    ---------
    br: BayesRegression,
        fitted Bayes Regressor
    xmin: float,
        minimal x for plotting
    xmax: float,
        maximal x for plotting
    """
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
    """Plot prior belief
    
    Parameter
    ---------
    br: BayesRegression,
        fitted Bayes Regressor
    lower: float,
        minimal x, y for plotting
    upper: float,
        maximal x, y for plotting
    """
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