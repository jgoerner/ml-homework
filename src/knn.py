import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from scipy.spatial import KDTree
from sklearn.metrics.regression import mean_squared_error as mse

########## CLASSIFIER ##########
class KNNC(BaseEstimator, ClassifierMixin):
    
    def __init__(self, k=4, search="brute"):
        self.k = k
        self.search = search
        
    def fit(self, X, y):
        self._points = X
        self._label_list = y
        self._labels = np.unique(y)
        self.kdt = KDTree(X)
    
    def predict(self, X):
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
        pred = self.predict(X)
        return np.sum(np.equal(pred, y))/len(y)
    
    def _calc_distance(self, x):
        if self.search == "brute":
            dists = np.linalg.norm(np.ones_like(self._points)*x - self._points, axis=1)
            sorted_dists = np.argsort(dists)
            result = sorted_dists[:self.k]
        elif self.search == "kdtree":
            result = np.array([self.kdt.query(x, self.k)[1]])
        return result
    
    def _decode_labels(self, idx):
        return self._label_list[idx]
    
    def _get_most_common_label(self, lbls):
        (vals, cnts) = np.unique(lbls, return_counts=True)
        return vals[np.argmax(cnts)]
    
    def _get_count_per_label(self, lbls):
        _, c = np.unique(
            np.concatenate((lbls, self._labels)),
            return_counts=True)
        return (c-1)/np.sum((c-1))
    
########## REGRESSOR ##########
class KNNR(BaseEstimator, RegressorMixin):

    def __init__(self, k=4, search="brute", combiner="average"):
        self.k = k
        self.search = search
        self.combiner = combiner
        
    def fit(self, X, y):
        self._points = X
        self._targets = y
        self.kdt = KDTree(X)
    
    def predict(self, X):
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
        if self.combiner == "average":
            result = np.mean(nn)
        elif self.combiner == "median":
            result = np.median(nn)
        else:
            raise NotImplementedError("{} is not implemented".format(self.combiner))
        return result
        
    def score(self, X, y):
        pred = self.predict(X)
        return mse(y, pred)
        
    def _calc_distance(self, x):
        if self.search == "brute":
            dists = np.linalg.norm(np.ones_like(self._points)*x - self._points, axis=1)
            sorted_dists = np.argsort(dists)
            result = sorted_dists[:self.k]
        elif self.search == "kdtree":
            result = np.array([self.kdt.query(x, self.k)[1]])
        else:
            raise NotImplementedError("{} is not implemented".format(self.search))
        return result