from __future__ import division
from __future__ import print_function

from utils import features_construct_graph, spatial_construct_graph
import os
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from config import Config


def normalize(adata, highly_genes=3000):
    print("start select HVGs")
    sc.pp.filter_genes(adata, min_cells=100)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=highly_genes)
    adata = adata[:, adata.var['highly_variable']].copy()
    adata.X = adata.X / np.sum(adata.X, axis=1).reshape(-1, 1) * 10000
    sc.pp.scale(adata, zero_center=False, max_value=10)
    return adata


def load_ST_file(dataset, highly_genes, k, k1):
    path = "D://DeepClustering/srt_datasets/" + dataset + "/"
    labels_path = path + "metadata.tsv"

    labels = pd.read_table(labels_path, sep='\t')
    labels = labels["ground_truth"].copy()

    ground = labels
    ground = ground.replace('AOB::Gl', '0')
    ground = ground.replace('AOB::Gr', '1')
    ground = ground.replace('AOB::Ml', '2')
    ground = ground.replace('AOE', '3')

    ground = ground.replace('AON::L1_1', '4')
    ground = ground.replace('AON::L1_2', '5')

    ground = ground.replace('AON::L2', '6')
    ground = ground.replace('AcbC', '7')

    ground = ground.replace('AcbSh', '8')
    ground = ground.replace('CC', '9')
    ground = ground.replace('CPu', '10')
    ground = ground.replace('Cl', '11')
    ground = ground.replace('En', '12')
    ground = ground.replace('FRP::L1', '13')

    ground = ground.replace('FRP::L2/3', '14')
    ground = ground.replace('Fim', '15')
    ground = ground.replace('Ft', '16')
    ground = ground.replace('HY::LPO', '17')
    ground = ground.replace('Io', '18')
    ground = ground.replace('LV', '19')
    ground = ground.replace('MO::L1', '20')
    ground = ground.replace('MO::L2/3', '21')
    ground = ground.replace('MO::L5', '22')

    ground = ground.replace('MO::L6', '23')
    ground = ground.replace('MOB::Gl_1', '24')

    ground = ground.replace('MOB::Gl_2', '25')
    ground = ground.replace('MOB::Gr', '26')

    ground = ground.replace('MOB::MI', '27')
    ground = ground.replace('MOB::Opl', '28')
    ground = ground.replace('MOB::lpl', '29')
    ground = ground.replace('Not_annotated', '30')
    ground = ground.replace('ORB::L1', '31')
    ground = ground.replace('ORB::L2/3', '32')

    ground = ground.replace('ORB::L5', '33')
    ground = ground.replace('ORB::L6', '34')
    ground = ground.replace('OT::Ml', '35')
    ground = ground.replace('OT::Pl', '36')
    ground = ground.replace('OT::PoL', '37')
    ground = ground.replace('Or', '38')
    ground = ground.replace('PIR', '39')
    ground = ground.replace('Pal::GPi', '40')
    ground = ground.replace('Pal::MA', '41')
    ground = ground.replace('Pal::NDB', '42')
    ground = ground.replace('Pal::Sl', '43')
    ground = ground.replace('Py', '44')
    ground = ground.replace('SLu', '45')
    ground = ground.replace('SS::L1', '46')
    ground = ground.replace('SS::L2/3', '47')
    ground = ground.replace('SS::L5', '48')
    ground = ground.replace('SS::L6', '49')
    ground = ground.replace('St', '50')
    ground = ground.replace('TH::RT', '51')
    adata = sc.read_visium(path, count_file='filtered_feature_bc_matrix.h5', load_images=True)
    adata.var_names_make_unique()

    adata.obs['ground_truth'] = labels.values
    adata.obs['ground'] = ground.values.astype(int)

    adata.var_names_make_unique()

    adata.X = np.array(sp.csr_matrix(adata.X, dtype=np.float32).todense())
    print(adata)
    adata = normalize(adata, highly_genes=highly_genes)

    fadj = features_construct_graph(adata.X, k=k)
    sadj, graph_nei, graph_neg = spatial_construct_graph(adata.obsm['spatial'], k=k1)

    adata.obsm["fadj"] = fadj
    adata.obsm["sadj"] = sadj
    adata.obsm["graph_nei"] = graph_nei.numpy()
    adata.obsm["graph_neg"] = graph_neg.numpy()
    adata.var_names_make_unique()
    return adata


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    datasets = ['Mouse_Brain_Anterior']
    for i in range(len(datasets)):
        dataset = datasets[i]
        print(dataset)
        if not os.path.exists("../generate_data/"):
            os.mkdir("../generate_data/")
        savepath = "../generate_data/" + dataset + "/"
        config_file = './config/' + dataset + '.ini'
        if not os.path.exists(savepath):
            os.mkdir(savepath)

        config = Config(config_file)
        adata = load_ST_file(dataset, config.fdim, config.k, config.radius)
        print("saving")
        adata.write(savepath + 'BMVC.h5ad')
        print("done")
