import os
import pickle
import logging
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from ..data_util import crop_pc, voxelize
from ..build import DATASETS


@DATASETS.register_module()
class S3DIS(Dataset):
    # 定义S3DIS数据集类
    classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'chair', 'table', 'bookcase', 'sofa',
               'board', 'clutter']
    num_classes = 13
    num_per_class = np.array(
        [3370714, 2856755, 4919229, 318158, 375640, 478001, 974733, 650464, 791496, 88727, 1284130, 229758, 2272837],
        dtype=np.int32)
    class2color = {'ceiling': [0, 255, 0], 'floor': [0, 0, 255], 'wall': [0, 255, 255], 'beam': [255, 255, 0],
                   'column': [255, 0, 255], 'window': [100, 100, 255], 'door': [200, 200, 100],
                   'table': [170, 120, 200], 'chair': [255, 0, 0], 'sofa': [200, 100, 100], 'bookcase': [10, 200, 100],
                   'board': [200, 200, 200], 'clutter': [50, 50, 50]}
    cmap = [*class2color.values()]
    gravity_dim = 1  # 重力维度为1

    """
    S3DIS数据集，加载未经块/球体下采样的子采样整个房间作为输入。
    Args:
        data_root (str, optional): 数据根目录，默认为'data/S3DIS/s3disfull'。
        test_area (int, optional): 默认为5。
        voxel_size (float, optional): 下采样的体素大小，默认为0.04。
        voxel_max (NoneType, optional): 每个点云的最大点数。设置为None以使用所有点。默认为None。
        split (str, optional): 默认为'train'。
        transform (NoneType, optional): 默认为None。
        loop (int, optional): 每个epoch的分割循环次数，默认为1。
        presample (bool, optional): 是否在训练之前对每个点云进行下采样。设置为False以进行即时下采样。默认为False。
        variable (bool, optional): 是否使用原始点数。每个点云的点数是可变的。默认为False。
    """

    def __init__(self,
                 data_root: str = 'data/s3dis',
                 test_area: int = 5,
                 voxel_size: float = 0.04,
                 voxel_max=None,
                 split: str = 'train',
                 transform=None,
                 loop: int = 1,
                 presample: bool = False,
                 variable: bool = False,
                 shuffle: bool = True,
                 ):

        super().__init__()
        # 设置数据集属性
        self.split, self.voxel_size, self.transform, self.voxel_max, self.loop = split, voxel_size, transform, voxel_max, loop
        self.presample = presample
        self.variable = variable
        self.shuffle = shuffle

        raw_root = os.path.join(data_root, 'raw')
        self.raw_root = raw_root
        # 加载数据文件列表
        data_list = sorted(os.listdir(raw_root))
        data_list = [item[:-4] for item in data_list if 'Area_' in item]

        # 根据数据集分割加载数据列表
        if split == 'train':
            self.data_list = [item for item in data_list if not 'Area_{}'.format(test_area) in item]
        else:
            self.data_list = [item for item in data_list if 'Area_{}'.format(test_area) in item]

        # 处理数据文件路径
        processed_root = os.path.join(data_root, 'processed')
        filename = os.path.join(processed_root, f's3dis_{split}_area{test_area}_{voxel_size:.3f}_{str(voxel_max)}.pkl')

        # 对数据进行预采样
        if presample and not os.path.exists(filename):
            np.random.seed(0)
            self.data = []
            # 读取每个数据文件并进行处理
            for item in tqdm(self.data_list, desc=f'Loading S3DISFull {split} split on Test Area {test_area}'):
                data_path = os.path.join(raw_root, item + '.npy')
                cdata = np.load(data_path).astype(np.float32)
                # 坐标偏移
                cdata[:, :3] -= np.min(cdata[:, :3], 0)
                # 进行体素化处理
                if voxel_size:
                    coord, feat, label = cdata[:, 0:3], cdata[:, 3:6], cdata[:, 6:7]
                    uniq_idx = voxelize(coord, voxel_size)
                    coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
                    cdata = np.hstack((coord, feat, label))

                self.data.append(cdata)
            npoints = np.array([len(data) for data in self.data])
            logging.info('split: %s, median npoints %.1f, avg num points %.1f, std %.1f' % (
                self.split, np.median(npoints), np.average(npoints), np.std(npoints)))
            os.makedirs(processed_root, exist_ok=True)
            with open(filename, 'wb') as f:
                pickle.dump(self.data, f)
                print(f"{filename} saved successfully")
        elif presample:
            with open(filename, 'rb') as f:
                self.data = pickle.load(f)
                print(f"{filename} load successfully")

        self.data_idx = np.arange(len(self.data_list))
        assert len(self.data_idx) > 0
        logging.info(f"\nTotally {len(self.data_idx)} samples in {split} set")

    def __getitem__(self, idx):
        # 获取要处理的数据索引
        data_idx = self.data_idx[idx % len(self.data_idx)]

        # 如果进行了预采样
        if self.presample:
            # 对数据进行拆分，获取坐标、特征和标签
            coord, feat, label = np.split(self.data[data_idx], [3, 6], axis=1)
        else:
            # 读取原始数据文件路径
            data_path = os.path.join(self.raw_root, self.data_list[data_idx] + '.npy')
            # 加载数据并将其转换为浮点型
            cdata = np.load(data_path).astype(np.float32)
            # 对数据进行坐标调整
            cdata[:, :3] -= np.min(cdata[:, :3], 0)
            # 分别获取坐标、特征和标签数据
            coord, feat, label = cdata[:, :3], cdata[:, 3:6], cdata[:, 6:7]
            # 对数据进行裁剪，利用网格体素采样，每个文件中取固定数量的点
            coord, feat, label = crop_pc(
                coord, feat, label, self.split, self.voxel_size, self.voxel_max,
                downsample=not self.presample, variable=self.variable, shuffle=self.shuffle)

        # 将标签转换为长整型并压缩维度
        label = label.squeeze(-1).astype(np.long)
        # 构建数据字典包含坐标、特征（rgb）和标签
        data = {'pos': coord, 'x': feat, 'y': label}

        # 如果存在数据变换方法，则对数据进行预处理
        if self.transform is not None:
            data = self.transform(data)

        # 如果数据字典中不存在 'heights' 键，则添加 'heights' 键并赋值为坐标中的重力维度的数据
        if 'heights' not in data.keys():
            data['heights'] = torch.from_numpy(coord[:, self.gravity_dim:self.gravity_dim + 1].astype(np.float32))

        # 返回处理后的数据
        return data

    def __len__(self):
        return len(self.data_idx) * self.loop

        # return 1   # debug


"""debug 
from openpoints.dataset import vis_multi_points
import copy
old_data = copy.deepcopy(data)
if self.transform is not None:
    data = self.transform(data)
vis_multi_points([old_data['pos'][:, :3], data['pos'][:, :3].numpy()], colors=[old_data['x'][:, :3]/255.,data['x'][:, :3].numpy()])
"""
if __name__ == "__main__":
    dataset = S3DIS()
    data = dataset.__getitem__(0)
    len = dataset.__len__()
    print(data, len)
