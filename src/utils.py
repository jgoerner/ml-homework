from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from sklearn.preprocessing import LabelEncoder

# TODO refactor to capsulated method
def multiclass_confusion_matrix(y, y_pred, name="", only_wrong=False, figsize=(5, 5), normed=False):
    """Plot a confusion multiclass confusion matrix
    
    Parameters
    -----------
    y: array-like, shape = [n_samples, 1]
        true class labels
    y_pred: array-like, shape = [n_samples, 1]
        predicted class labels
    
    Returns
    -----------
    fig, ax: atplotlib.pyplot subplot objects
        Figure and axis elements of the subplot.
        
    """
    le = LabelEncoder()
    y_e = le.fit_transform(y)
    y_pred_e = le.transform(y_pred)
    n_labels = len(np.unique(y))
    cm = np.zeros(shape=(n_labels, n_labels), dtype=int)
    subtitle = ""
    acc = np.around(100*np.sum(y == y_pred)/len(y), decimals=3)
    label_dict = dict(enumerate(le.classes_))
    
    # TODO: vectorize
    for _y, _y_pred in zip(y_e, y_pred_e):
        cm[_y, _y_pred] += 1
    
    # check if normed
    if normed:
        cm = np.round((cm / np.sum(cm, axis=1)[:, np.newaxis])*100, decimals=2)
    
    # check if true positives should be ignored
    if only_wrong:
        np.fill_diagonal(cm, 0)
        subtitle = "\n - only misclassification -"
        
    # generate the confusion matrix plot
    fig, ax =  plot_confusion_matrix(cm, figsize=figsize)
    
    # cosmetics
    labels = le.inverse_transform(range(0,n_labels))
    ax.set_xticks(range(n_labels))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(n_labels))
    ax.set_yticklabels(labels)
    plt.xticks(rotation=90, ha="right")
    ax.set_xlabel("predicted label", fontsize=14)
    ax.set_ylabel("true label", fontsize=14)
    ax.set_title("Confusion Matrix {} (acc: {}%){}".format(name,acc,subtitle), fontsize=20)
    plt.tight_layout()

    return cm, label_dict

def plot_decision_planes(lsc, xmin, xmax, ymin, ymax, resolution=100):
    """Plot decision planes of a Least-Squares-Classifier
    
    Parameters
    ----------
    lsc: LeastSquaresClassifier
        classifier to be investigated
    xmin: float,
        minimal x to plot
    xmax: float,
        maximal x to plot
    ymin: float,
        minimal y to plot
    ymax: float,
        maximal y to plot
    resolution: int, default=100
        resolution of the plot
    """
    w = lsc.W_opt
    x = np.linspace(xmin, xmax, resolution)
    for f1, f2 in combinations(w.T, 2):
        x2 = get_x2(f1, f2, x)
        plt.plot(x, x2, c="k", linestyle="--", alpha=0.3)
    plt.ylim(ymin, ymax);  
    
def get_x2(f1, f2, x):
    """Helper to extract x2 from two discriminant functions
    
    Parameters
    ----------
    f1: 1-d matrix,
        parameter of first discriminant function
    f2: 1-d matrix,
        parameter of second discriminant function
    x: array-like,
        array of x input values
    
    Returns
    -------
    x2: array-like,
        other x coordinate
    """
    num, denom = np.split((f1-f2).A.flatten(), [-1])
    x2 = - np.dot(num, np.vstack([np.ones_like(x), x])) / denom
    return x2

def plot_decision_regions(lsc, xmin, xmax, ymin, ymax, resolution=100):
    """Plot decision regions for a classifier
    
    Parameters
    ----------
    lsc: LeastSquaresClassifier
        classifier to be investigated
    xmin: float,
        minimal x to plot
    xmax: float,
        maximal x to plot
    ymin: float,
        minimal y to plot
    ymax: float,
        maximal y to plot
    resolution: int, default=100
        resolution of the plot
    """
    xx1, xx2 = np.meshgrid(
        np.linspace(xmin, xmax, resolution),
        np.linspace(ymin, ymax, resolution),
    )
    cls = lsc.predict(np.matrix(np.vstack([xx1.ravel(), xx2.ravel()]).T))
    plt.contourf(
        xx1,
        xx2,
        cls.reshape(xx1.shape),
        alpha=0.25,
        cmap="rainbow"
    )