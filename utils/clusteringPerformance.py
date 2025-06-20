'''
   This program is to evaluate clustering performance
   Code author: Shide Du
   Email: shidedums@163.com
   Date: Dec 4, 2019.
'''
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score
import warnings
from scipy import sparse as sp
from scipy.special import comb

warnings.filterwarnings("ignore")


def bestMap(y_pred, y_true):
    from scipy.optimize import linear_sum_assignment
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    np.asarray(ind)
    ind = np.transpose(ind)
    label = np.zeros(y_pred.size)
    for i in range(y_pred.size):
        label[i] = ind[y_pred[i]][1]
    return label.astype(np.int64)


def similarity_function(points):
    """

    :param points:
    :return:
    """
    res = rbf_kernel(points)
    for i in range(len(res)):
        res[i, i] = 0
    return res


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    indx_list = []
    for i in range(len(ind[0])):
        indx_list.append((ind[0][i], ind[1][i]))
    return sum([w[i1, j1] for (i1, j1) in indx_list]) * 1.0 / y_pred.size


def cluster_f(y_true, y_pred):
    N = len(y_true)
    numT = 0
    numH = 0
    numI = 0
    for n in range(0, N):
        C1 = [y_true[n] for x in range(1, N - n)]
        C1 = np.array(C1)
        C2 = y_true[n + 1:]
        C2 = np.array(C2)
        Tn = (C1 == C2) * 1

        C3 = [y_pred[n] for x in range(1, N - n)]
        C3 = np.array(C3)
        C4 = y_pred[n + 1:]
        C4 = np.array(C4)
        Hn = (C3 == C4) * 1

        numT = numT + np.sum(Tn)
        numH = numH + np.sum(Hn)
        numI = numI + np.sum(np.multiply(Tn, Hn))
    if numH > 0:
        p = numI / numH
    if numT > 0:
        r = numI / numT
    if (p + r) == 0:
        f = 0
    else:
        f = 2 * p * r / (p + r)
    return f, p, r


def clustering_purity(labels_true, labels_pred):
    """
    :param y_true:
        data type: numpy.ndarray
        shape: (n_samples,)
        sample: [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20]
    :param y_pred:
        data type: numpy.ndarray
        shape: (n_samples,)
        sample: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
    :return: Purity
    """
    y_true = labels_true.copy()
    y_pred = labels_pred.copy()
    if y_true.shape[1] != 1:
        y_true = y_true.T
    if y_pred.shape[1] != 1:
        y_pred = y_pred.T

    n_samples = len(y_true)

    u_y_true = np.unique(y_true)
    n_true_classes = len(u_y_true)
    y_true_temp = np.zeros((n_samples, 1))
    if n_true_classes != max(y_true):
        for i in range(n_true_classes):
            y_true_temp[np.where(y_true == u_y_true[i])] = i + 1
        y_true = y_true_temp

    u_y_pred = np.unique(y_pred)
    n_pred_classes = len(u_y_pred)
    y_pred_temp = np.zeros((n_samples, 1))
    if n_pred_classes != max(y_pred):
        for i in range(n_pred_classes):
            y_pred_temp[np.where(y_pred == u_y_pred[i])] = i + 1
        y_pred = y_pred_temp

    u_y_true = np.unique(y_true)
    n_true_classes = len(u_y_true)
    u_y_pred = np.unique(y_pred)
    n_pred_classes = len(u_y_pred)

    n_correct = 0
    for i in range(n_pred_classes):
        incluster = y_true[np.where(y_pred == u_y_pred[i])]

        inclunub = np.histogram(incluster, bins=range(1, int(max(incluster)) + 1))[0]
        if len(inclunub) != 0:
            n_correct = n_correct + max(inclunub)

    Purity = n_correct / len(y_pred)

    return Purity


def contingency_matrix(labels_true, labels_pred, eps=None, sparse=False):
    """Build a contingency matrix describing the relationship between labels.

    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        Ground truth class labels to be used as a reference

    labels_pred : array, shape = [n_samples]
        Cluster labels to evaluate

    eps : None or float, optional.
        If a float, that value is added to all values in the contingency
        matrix. This helps to stop NaN propagation.
        If ``None``, nothing is adjusted.

    sparse : boolean, optional.
        If True, return a sparse CSR continency matrix. If ``eps is not None``,
        and ``sparse is True``, will throw ValueError.

        .. versionadded:: 0.18

    Returns
    -------
    contingency : {array-like, sparse}, shape=[n_classes_true, n_classes_pred]
        Matrix :math:`C` such that :math:`C_{i, j}` is the number of samples in
        true class :math:`i` and in predicted class :math:`j`. If
        ``eps is None``, the dtype of this array will be integer. If ``eps`` is
        given, the dtype will be float.
        Will be a ``scipy.sparse.csr_matrix`` if ``sparse=True``.
    """

    if eps is not None and sparse:
        raise ValueError("Cannot set 'eps' when sparse=True")

    classes, class_idx = np.unique(labels_true, return_inverse=True)
    clusters, cluster_idx = np.unique(labels_pred, return_inverse=True)
    n_classes = classes.shape[0]
    n_clusters = clusters.shape[0]
    # Using coo_matrix to accelerate simple histogram calculation,
    # i.e. bins are consecutive integers
    # Currently, coo_matrix is faster than histogram2d for simple cases
    contingency = sp.coo_matrix((np.ones(class_idx.shape[0]),
                                 (class_idx, cluster_idx)),
                                shape=(n_classes, n_clusters),
                                dtype=np.int)
    if sparse:
        contingency = contingency.tocsr()
        contingency.sum_duplicates()
    else:
        contingency = contingency.toarray()
        if eps is not None:
            # don't use += as contingency is integer
            contingency = contingency + eps
    return contingency


def _comb2(n):
    # the exact version is faster for k == 2: use it by default globally in
    # this module instead of the float approximate variant
    return comb(n, 2, exact=1)


def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment as linear_assignment
    ind = linear_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def b3_precision_recall_fscore(labels_true, labels_pred):
    """Compute the B^3 variant of precision, recall and F-score.
    Parameters
    ----------
    :param labels_true: 1d array containing the ground truth cluster labels.
    :param labels_pred: 1d array containing the predicted cluster labels.
    Returns
    -------
    :return float precision: calculated precision
    :return float recall: calculated recall
    :return float f_score: calculated f_score
    Reference
    ---------
    Amigo, Enrique, et al. "A comparison of extrinsic clustering evaluation
    metrics based on formal constraints." Information retrieval 12.4
    (2009): 461-486.
    """
    # Check that labels_* are 1d arrays and have the same size

    labels_pred = bestMap(labels_pred, labels_true)

    # Check that input given is not the empty set
    if labels_true.shape == (0,):
        raise ValueError(
            "input labels must not be empty.")

    # Compute P/R/F scores
    n_samples = len(labels_true)
    true_clusters = {}  # true cluster_id => set of sample indices
    pred_clusters = {}  # pred cluster_id => set of sample indices

    for i in range(n_samples):
        true_cluster_id = labels_true[i]
        pred_cluster_id = labels_pred[i]

        if true_cluster_id not in true_clusters:
            true_clusters[true_cluster_id] = set()
        if pred_cluster_id not in pred_clusters:
            pred_clusters[pred_cluster_id] = set()

        true_clusters[true_cluster_id].add(i)
        pred_clusters[pred_cluster_id].add(i)

    for cluster_id, cluster in true_clusters.items():
        true_clusters[cluster_id] = frozenset(cluster)
    for cluster_id, cluster in pred_clusters.items():
        pred_clusters[cluster_id] = frozenset(cluster)

    precision = 0.0
    recall = 0.0

    intersections = {}

    for i in range(n_samples):
        pred_cluster_i = pred_clusters[labels_pred[i]]
        true_cluster_i = true_clusters[labels_true[i]]

        if (pred_cluster_i, true_cluster_i) in intersections:
            intersection = intersections[(pred_cluster_i, true_cluster_i)]
        else:
            intersection = pred_cluster_i.intersection(true_cluster_i)
            intersections[(pred_cluster_i, true_cluster_i)] = intersection

        precision += len(intersection) / len(pred_cluster_i)
        recall += len(intersection) / len(true_cluster_i)

    precision /= n_samples
    recall /= n_samples

    f_score = 2 * precision * recall / (precision + recall)

    return f_score, precision, recall


# Evaluation metrics of clustering performance
def clusteringMetrics(trueLabel, predictiveLabel):
    y_pred = bestMap(predictiveLabel, trueLabel)
    # Clustering accuracy
    ACC = cluster_acc(trueLabel, y_pred)
    # NMI
    NMI = normalized_mutual_info_score(trueLabel, y_pred)
    # Purity
    Purity = clustering_purity(trueLabel.reshape((-1, 1)), y_pred.reshape(-1, 1))
    # Adjusted rand index
    ARI = metrics.adjusted_rand_score(trueLabel, y_pred)
    # Fscore, Precision, Recall = cluster_f(trueLabel, y_pred)
    Fscore, Precision, Recall = b3_precision_recall_fscore(trueLabel, y_pred)

    return ACC, NMI, Purity, ARI, Fscore, Precision, Recall
