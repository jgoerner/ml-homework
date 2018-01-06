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