from typing import List, Text
import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

def plot_confusion_matrix(cm: np.ndarray, target_names: List[Text], title: Text = 'Confusion Matrix', cmap: matplotlib.colors.LinearSegmentedColormap = None):
    '''
    Return a plt with the confusion matrix given the cm array

    Args:
        cm: confusion matrix from sklearn.metrics.confusion_matrix
        target_names: given classification classes such as [0, 1, 2] the class names, for example: ['high', 'medium', 'low']
        title: the text to display at the top of the matrix
        cmap: the gradient of the values displayed from matplotlib.pyplot.cm
                see http://matplotlib.org/examples/color/colormaps_reference.html
                plt.get_cmap('jet') or plt.cm.Blues
    '''
    if cmap is None:
        cmap = plt.get_cmap('Blues')

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    
    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_title(title)
    disp.plot(ax=ax)
    
    return fig

def print_classification_report(y_true, y_hat):
    '''
    Print classification report
    '''
    report = classification_report(y_true, y_hat)
    print('Classification report:')
    print(report)