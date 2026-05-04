import numpy as np

def standardize_labels(labels, method_type):
    """
    Converts various outputs to: 1 = Outlier, 0 = Inlier
    """
    if method_type == 'sklearn': # IsolationForest, LocalOutlierFactor, EllipticEnvelope
        # Sklearn uses 1 (inlier) and -1 (outlier)
        return np.where(labels == -1, 1, 0)
    
    elif method_type == 'cluster': # DBSCAN, OPTICS
        # They use -1 for noise (outlier) and 0, 1, 2... for clusters
        return np.where(labels == -1, 1, 0)
    
    elif method_type == 'pyod_or_manual': # PyOD o il tuo thresholding di isotree
        # They're already 0 (inlier) and 1 (outlier)
        return labels
    
    return labels