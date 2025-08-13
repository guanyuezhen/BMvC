import numpy as np
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


class ZScoreOutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=3):
        self.threshold = threshold
        self.std = None
        self.mean = None

    def fit(self, x, y=None):
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)
        return self

    def transform(self, x):
        z_scores = np.abs((x - self.mean) / self.std)
        x_cleaned = np.where(z_scores > self.threshold, self.mean, x)
        return x_cleaned

def get_data(data_name):
    # load the datasets
    data_path = "./datasets/" + data_name + ".mat"
    data = sio.loadmat(data_path)
    if data['fea'].shape[1] < data['fea'].shape[0]:
        data_x = data['fea'][0]
    else:
        data_x = data['fea'][:][0]
    data_y = data['gt'].flatten()
    # get the basic information of the datasets
    view_num = data_x.shape[0]
    sample_num = data_x[0].shape[0]
    cluster_num = len(np.unique(data_y))
    input_dims = [data_x[v].shape[1] for v in range(view_num)]
    # random permutation the sample orders
    random_sequence = np.random.permutation(sample_num)
    for v in range(view_num):
        data_x[v] = data_x[v][random_sequence]
    data_y = data_y[random_sequence]
    # denoise the normalize the datasets
    for v in range(view_num):
        # pipeline = Pipeline(steps=[
        #     ('outlier_remover', ZScoreOutlierRemover()),
        #     ('scaler', MinMaxScaler())
        # ])
        pipeline = MinMaxScaler()
        data_x[v] = pipeline.fit_transform(data_x[v])
    print(
        f"Data: {data_name},"
        f" number of data: {sample_num},"
        f" views: {view_num},"
        f" clusters: {cluster_num},"
        f" dims of each view: {input_dims}")

    return data_x, data_y, view_num, sample_num, cluster_num, input_dims
