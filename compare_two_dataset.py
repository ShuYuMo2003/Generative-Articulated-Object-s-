import yaml
import argparse
import open3d as o3d

from rich import print

from onet_v2.dataset import PartnetMobilityDataset as PartnetMobilityDatasetv2
from onet_v2.onet import ONet

from transformer.utils import str2hash

parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', dest='config',
                    help=('config file.'), required=True)
config = yaml.safe_load(open(parser.parse_args().config).read())



# if _.use_v1_dataset:
from onet.dataset import PartnetMobilityDataset as PartnetMobilityDatasetv1
train_dataset_v1 = PartnetMobilityDatasetv1('dataset/2_onet_dataset', train_ratio=config['train_ratio'], train=True)


train_dataset_v2 = PartnetMobilityDatasetv2('dataset/2_onet_v2_dataset', train_ratio=config['train_ratio'],
                                        selected_categories=config['selected_categories'], train=True)

print('v1 len = ', len(train_dataset_v1))
print('v2 len = ', len(train_dataset_v2))

for i in range(len(train_dataset_v1)):
    data_v1 = train_dataset_v1[i]
    data_v2 = train_dataset_v2[i]

    enc_sp_v1, enc_occ_v1, dec_sp_v1, dec_occ_v1 = data_v1
    enc_sp_v2, enc_occ_v2, dec_sp_v2, dec_occ_v2 = data_v2

    # print('i ====== ', i)
    # print('enc_sp_v1:', enc_sp_v1.shape)
    # print('enc_occ_v1:', enc_occ_v1.shape)
    # print('dec_sp_v1:', dec_sp_v1.shape)
    # print('dec_occ_v1:', dec_occ_v1.shape)

    # print('enc_sp_v2:', enc_sp_v2.shape)
    # print('enc_occ_v2:', enc_occ_v2.shape)
    # print('dec_sp_v2:', dec_sp_v2.shape)
    # print('dec_occ_v2:', dec_occ_v2.shape)
    # print('')

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(enc_sp_v1)
    o3d.io.write_point_cloud('enc_sp_v1.ply', pcd)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(enc_sp_v1[enc_occ_v1 == 1])
    o3d.io.write_point_cloud('enc_sp_v1_occ.ply', pcd)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(enc_sp_v2)
    o3d.io.write_point_cloud('enc_sp_v2.ply', pcd)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(enc_sp_v2[enc_occ_v2 == 1])
    o3d.io.write_point_cloud('enc_sp_v2_occ.ply', pcd)

    print('max enc_sp_v1', enc_sp_v1[enc_occ_v1 == 1].max(dim=0))
    print('min enc_sp_v1', enc_sp_v1[enc_occ_v1 == 1].min(dim=0))

    print('max enc_sp_v2', enc_sp_v2[enc_occ_v2 == 1].max(dim=0))
    print('min enc_sp_v2', enc_sp_v2[enc_occ_v2 == 1].min(dim=0))

    print('point count v1 = ', enc_sp_v1[enc_occ_v1 == 1].shape)
    print('point count v2 = ', enc_sp_v2[enc_occ_v2 == 1].shape)


    break