import random
import torch
import os
import numpy as np
import torch.nn.functional as F
from utils.dataloader import get_data
from models.models import MvAEModel
from utils.Logger import BaseLogger
from sklearn.cluster import KMeans
from utils.clusteringPerformance import clusteringMetrics
from argparse import ArgumentParser
from models.losses import knn_graph, contrastive_loss

def set_seed():
    seed = 42  # (3407, 4079, 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_name', default="MSRCV1", help='Data directory | MSRCV1 ')
    parser.add_argument('--train_epochs', type=int, default=3000, help='Max. number of epochs')
    parser.add_argument('--alpha', type=int, default=-3, help='Parameter: alpha')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--log_save_dir', default='./logs/', help='Directory to save the results')
    args = parser.parse_args()
    print('Called with args:')
    print(args)
    set_seed()
    random_numbers_for_kmeans = random.sample(range(10000), 20)
    print(random_numbers_for_kmeans)
    os.makedirs(args.log_save_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    latent_dim = 64
    hid_dims = [192, 128]

    data_x, data_y, view_num, sample_num, cluster_num, input_dims = get_data(args.data_name)
    mask_x = []
    for v in range(view_num):
        data_x[v] = torch.from_numpy(data_x[v]).to(dtype=torch.float32, device=device)
        mask_x.append(knn_graph(data_x[v]))
    data_y = torch.from_numpy(data_y).to(device=device)

    logger = BaseLogger(log_save_dir=args.log_save_dir, log_name=args.data_name + '.csv')
    print("====================== start training ======================")
    print("data_name:", args.data_name, "alpha:", args.alpha)
    logger.write_parameters(args.alpha)

    model = MvAEModel(input_dims,
                      view_num,
                      latent_dim=latent_dim,
                      hid_dims=hid_dims,
                      cluster_num=cluster_num
                      )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    losses = []
    mask_z = 0
    for epoch in range(args.train_epochs):
        # Train
        model.train()
        recs, joint_z, specific_y = model(data_x, is_training=True)
        loss_rec = 0
        loss_con = 0
        mask_z = knn_graph(joint_z)
        for v in range(view_num):
            loss_rec += F.mse_loss(recs[v], data_x[v])
            mask = 0.5 * mask_x[v] + 0.5 * mask_z
            loss_con += contrastive_loss(specific_y[v], mask)
        optimizer.zero_grad()
        loss = loss_rec + 10 ** args.alpha * loss_con
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            print("dataset: %s, epoch: %d, loss_rec: %.6f, loss_con: %.6f" % (
                args.data_name, epoch, loss_rec.data.item(), loss_con.data.item()))

        # Test
        if (epoch + 1) % (args.train_epochs // 1) == 0:
            model.eval()
            joint_z = model(data_x)
            with torch.no_grad():
                ACCo, NMIo, Purityo, ARIo, Fscoreo, Precisiono, Recallo = [], [], [], [], [], [], []
                for p in range(len(random_numbers_for_kmeans)):
                    datas = joint_z.clone().cpu().numpy()
                    kmeans = KMeans(n_clusters=cluster_num, n_init=10, random_state=random_numbers_for_kmeans[p])
                    y_pred = kmeans.fit_predict(datas)
                    label = data_y.clone().cpu().numpy()
                    ACCp, NMIp, Purityp, ARIp, Fscorep, Precisionp, Recallp = clusteringMetrics(label, y_pred)
                    ACCo.append(ACCp)
                    NMIo.append(NMIp)
                    Purityo.append(Purityp)
                    ARIo.append(ARIp)
                    Fscoreo.append(Fscorep)
                    Precisiono.append(Precisionp)
                    Recallo.append(Recallp)

                ACCo = sum(ACCo) / len(ACCo) * 100
                NMIo = sum(NMIo) / len(NMIo) * 100
                Purityo = sum(Purityo) / len(Purityo) * 100
                ARIo = sum(ARIo) / len(ARIo) * 100
                Fscoreo = sum(Fscoreo) / len(Fscoreo) * 100
                Precisiono = sum(Precisiono) / len(Precisiono) * 100
                Recallo = sum(Recallo) / len(Recallo) * 100

                scores = [ACCo, NMIo, Purityo, ARIo, Fscoreo, Precisiono, Recallo]
                logger.write_val(epoch, loss, scores)

    logger.close_logger()


