import torch
import numpy as np
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

My_DATA_PATH = "D://DeepClustering/mvc_datasets/Processed_DATA/DATA/"


def get_data(data_name):
    # load the datasets
    data_path = My_DATA_PATH + data_name + ".mat"
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

    # normalize the datasets
    for v in range(view_num):
        pipeline = MinMaxScaler()
        data_x[v] = pipeline.fit_transform(data_x[v])

    print(
        f"Data: {data_name},"
        f" number of data: {sample_num},"
        f" views: {view_num},"
        f" clusters: {cluster_num},"
        f" dims of each view: {input_dims}")

    data_set = MultiViewDataset(data_name, data_x, data_y)

    return data_set, view_num, sample_num, cluster_num, input_dims


class MultiViewDataset(Dataset):
    def __init__(self, data_name, data_x, data_y):
        super(MultiViewDataset, self).__init__()
        self.data_name = data_name
        self.x = data_x
        self.view_num = data_x.shape[0]
        self.y = data_y.astype(dtype=np.int64)

    def __getitem__(self, index):
        features = dict()
        for v in range(self.x.shape[0]):
            features[v] = torch.from_numpy((self.x[v][index]).astype(np.float32))
        labels = torch.from_numpy(np.array(self.y[index]))
        return features, labels, index

    def __len__(self):
        return self.x[0].shape[0]