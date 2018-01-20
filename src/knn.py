import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from scipy.spatial import KDTree
from sklearn.metrics.regression import mean_squared_error as mse

########## CLASSIFIER ##########
class KNNC(BaseEstimator, ClassifierMixin):
    """K-nearest neighbour classifier
    
    Parameter
    ---------
    k: int, default=4,
        number of nearest neighbours
    search: "brute" or "kdtree",
        implementation of the neighbour search
    """
        
    def __init__(self, k=4, search="brute"):
        self.k = k
        self.search = search
        
    def fit(self, X, y):
        """Fit the Classifier to the data set
        
        Parameter
        ---------
        X: array-like
            input vectors
            
        y: array-like
            target vector
        """
        self._points = X
        self._label_list = y
        self._labels = np.unique(y)
        self.kdt = KDTree(X)
    
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
        # get nearest neighbors
        knn = np.apply_along_axis(
            lambda x: self._calc_distance(x),
            axis=1,
            arr = np.atleast_2d(X)
        )
        # map labels
        lbls = np.apply_along_axis(
            lambda x: self._decode_labels(x),
            axis=1,
            arr = knn
        )
        # get most common label
        cls = np.apply_along_axis(
            lambda x: self._get_most_common_label(x),
            axis=1,
            arr = lbls
        )
        return cls
    
    def predict_proba(self, X):
        """Predict the class-probabilities of unseen data
        
        Parameter
        ---------
        X: array-like
            unsee input vectors
            
        Returns
        -------
        probas: array-like
            class-probabilities of unseen data
        """
        # get nearest neighbors
        knn = np.apply_along_axis(
            lambda x: self._calc_distance(x),
            axis=1,
            arr = np.atleast_2d(X)
        )
        # map labels
        lbls = np.apply_along_axis(
            lambda x: self._decode_labels(x),
            axis=1,
            arr = knn
        )
        # get probas per label
        probas = np.apply_along_axis(
            lambda x: self._get_count_per_label(x),
            axis=1,
            arr=lbls
        )
        
        return probas        
        
    def score(self, X, y):
        """Calculate the accuracy
        
        Parameter
        ---------
        X: array-like,
            input vector
        y: array-like
            target-vector
        
        Returns
        -------
        score: float,
            accuracy score
        """
        pred = self.predict(X)
        return np.sum(np.equal(pred, y))/len(y)
    
    def _calc_distance(self, x):
        """Calculate distances
        
        Parameter
        ---------
        x: array-like,
            single input-vector
        
        Returns
        -------
        result:
            k nearest-neighbors
        """
        if self.search == "brute":
            dists = np.linalg.norm(np.ones_like(self._points)*x - self._points, axis=1)
            sorted_dists = np.argsort(dists)
            result = sorted_dists[:self.k]
        elif self.search == "kdtree":
            result = np.array([self.kdt.query(x, self.k)[1]])
        return result
    
    def _decode_labels(self, idx):
        """Decode class labels
        
        Parameter
        ---------
        idx: int,
            index of class label
        
        Returns
        -------
        label: string/int,
            decoded class label
        """
        return self._label_list[idx]
    
    def _get_most_common_label(self, lbls):
        """Calculate most common label
        
        Parameter
        ---------
        lbls: array-like
            list of labels
        
        Returns
        -------
        lbl: string/int,
            most common label
        """
        (vals, cnts) = np.unique(lbls, return_counts=True)
        return vals[np.argmax(cnts)]
    
    def _get_count_per_label(self, lbls):
        """Calculate count per label
        
        Parameter
        ---------
        lbls: array-like,
            list of labels
            
        Returns
        -------
        cnts: float,
            count per label
        """
        _, c = np.unique(
            np.concatenate((lbls, self._labels)),
            return_counts=True)
        return (c-1)/np.sum((c-1))
    
########## REGRESSOR ##########
class KNNR(BaseEstimator, RegressorMixin):
    """K-nearest neighbour regressor
    
    Parameter
    ---------
    k: int, default=4,
        number of nearest neighbours
    search: "brute" or "kdtree",
        implementation of the neighbour search
    combiner: "average" or "median",
        implementation of neighbour weighting
    """

    def __init__(self, k=4, search="brute", combiner="average"):
        self.k = k
        self.search = search
        self.combiner = combiner
        
    def fit(self, X, y):
        """Fit the Regressor to the data set
        
        Parameter
        ---------
        X: array-like
            input vectors
            
        y: array-like
            target vector
        """
        self._points = X
        self._targets = y
        self.kdt = KDTree(X)
    
    def predict(self, X):
        """Predict the value of unseen data
        
        Parameter
        ---------
        X: array-like
            unsee input vectors
            
        Returns
        -------
        scores: array-like
            values of unseen data
        """
        # get nearest neighbors
        knn = np.apply_along_axis(
            lambda x: self._calc_distance(x),
            axis=1,
            arr = np.atleast_2d(X)
        )
        # score the nearest neighbors
        scores = np.apply_along_axis(
            lambda x: self._combine(self._targets[x]),
            axis=1,
            arr = np.atleast_2d(knn)
        )
        return scores
        
    def _combine(self, nn):
        """Combine values of nearest neighbourds
        
        Parameter
        ---------
        nn: array-like,
            values of nearest neighbours
        
        Returns
        -------
        result: float,
            combined values
        """
        if self.combiner == "average":
            result = np.mean(nn)
        elif self.combiner == "median":
            result = np.median(nn)
        else:
            raise NotImplementedError("{} is not implemented".format(self.combiner))
        return result
        
    def score(self, X, y):
        """Calculate the mean squared error
        
        Parameter
        ---------
        X: array-like,
            input vector
        y: array-like
            target-vector
        
        Returns
        -------
        score: float,
            mean squared error
        """
        pred = self.predict(X)
        return mse(y, pred)
        
    def _calc_distance(self, x):
        """Calculate distances
        
        Parameter
        ---------
        x: array-like,
            single input-vector
        
        Returns
        -------
        result:
            k nearest-neighbors
        """
        if self.search == "brute":
            dists = np.linalg.norm(np.ones_like(self._points)*x - self._points, axis=1)
            sorted_dists = np.argsort(dists)
            result = sorted_dists[:self.k]
        elif self.search == "kdtree":
            result = np.array([self.kdt.query(x, self.k)[1]])
        else:
            raise NotImplementedError("{} is not implemented".format(self.search))
        return result