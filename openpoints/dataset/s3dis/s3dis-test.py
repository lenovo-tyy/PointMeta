import os
import pickle
import logging
import sys
sys.path.append("..")
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

from data_util2 import crop_pc, voxelize
# from ..build import DATASETS
from curses import keyname
import numpy as np
import torch
import os
import os.path as osp
import ssl
import sys
import urllib
import h5py
from typing import Optional


class IO:
    @classmethod
    def get(cls, file_path):
        _, file_extension = os.path.splitext(file_path)

        if file_extension in ['.npy']:
            return cls._read_npy(file_path)
        elif file_extension in ['.pcd']:
            return cls._read_pcd(file_path)
        elif file_extension in ['.h5']:
            return cls._read_h5(file_path)
        elif file_extension in ['.txt']:
            return cls._read_txt(file_path)
        else:
            raise Exception('Unsupported file extension: %s' % file_extension)

    # References: https://github.com/numpy/numpy/blob/master/numpy/lib/format.py
    @classmethod
    def _read_npy(cls, file_path):
        return np.load(file_path)
    # # References: https://github.com/dimatura/pypcd/blob/master/pypcd/pypcd.py#L275
    # # Support PCD files without compression ONLY!
    # @classmethod
    # def _read_pcd(cls, file_path):
    #     pc = open3d.io.read_point_cloud(file_path)
    #     ptcloud = np.array(pc.points)
    #     return ptcloud

    @classmethod
    def _read_txt(cls, file_path):
        return np.loadtxt(file_path)

    @classmethod
    def _read_h5(cls, file_path):
        f = h5py.File(file_path, 'r')
        return f['data'][()]


# download
def download_url(url: str, folder: str, log: bool = True,
                 filename: Optional[str] = None):
    r"""Downloads the content of an URL to a specific folder.
    Borrowed from https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/data/download.py
    Args:
        url (string): The url.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """

    if filename is None:
        filename = url.rpartition('/')[2]
        filename = filename if filename[0] == '?' else filename.split('?')[0]

    path = osp.join(folder, filename)

    if osp.exists(path):  # pragma: no cover
        if log:
            print(f'Using existing file {filename}', file=sys.stderr)
        return path

    if log:
        print(f'Downloading {url}', file=sys.stderr)

    os.makedirs(folder, exist_ok=True)
    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)

    with open(path, 'wb') as f:
        # workaround for https://bugs.python.org/issue42853
        while True:
            chunk = data.read(10 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    return path


def fnv_hash_vec(arr):
    """
    FNV64-1A
    """
    assert arr.ndim == 2
    # Floor first for negative coordinates
    arr = arr.copy()
    arr = arr.astype(np.uint64, copy=False)
    hashed_arr = np.uint64(14695981039346656037) * \
        np.ones(arr.shape[0], dtype=np.uint64)
    # [1542355, 1]
    for j in range(arr.shape[1]):
        hashed_arr *= np.uint64(1099511628211)
        hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
    return hashed_arr


def ravel_hash_vec(arr):
    """
    Ravel the coordinates after subtracting the min coordinates.
    """
    assert arr.ndim == 2
    arr = arr.copy()
    arr -= arr.min(0)
    arr = arr.astype(np.uint64, copy=False)
    arr_max = arr.max(0).astype(np.uint64) + 1

    keys = np.zeros(arr.shape[0], dtype=np.uint64)
    # Fortran style indexing
    for j in range(arr.shape[1] - 1):
        keys += arr[:, j]
        keys *= arr_max[j + 1]
    keys += arr[:, -1]
    return keys

def voxelize(coord, voxel_size=0.05, hash_type='fnv', mode=0):
    # 将坐标离散化为体素坐标
    discrete_coord = np.floor(coord / np.array(voxel_size))

    # 根据哈希类型选择哈希方法（默认使用 FNV 哈希）
    if hash_type == 'ravel':
        key = ravel_hash_vec(discrete_coord)  # 使用 ravel_hash_vec 进行哈希处理
    else:
        key = fnv_hash_vec(discrete_coord)  # 使用 fnv_hash_vec 进行哈希处理

    # 对哈希键进行排序
    idx_sort = np.argsort(key)
    key_sort = key[idx_sort]

    # 计算哈希键的唯一值（未重叠？）和出现次数
    _, voxel_idx, count = np.unique(key_sort, return_counts=True, return_inverse=True)
    # 这行代码的目的是在每个体素中随机选择一个点作为训练点。
    # 通过计算累积和和随机生成的整数数组，确保了选择的点在每个体素内，并且在不同的训练迭代中，可以在每个体素中选择不同的点进行训练，增加了训练的多样性。

    if mode == 0:  # 训练模式
        # 随机选择一个体素中的点进行训练
        idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + np.random.randint(0, count.max(), count.size) % count
        idx_unique = idx_sort[idx_select]  # 根据随机选择的索引获取最终选择的索引
        return idx_unique
    else:  # 验证模式
        return idx_sort, voxel_idx, count  # 返回排序后的索引、体素索引和计数


def crop_pc(coord, feat, label, split='train', voxel_size=0.04, voxel_max=None, downsample=True, variable=True, shuffle=True):
    if voxel_size and downsample:
        # Is this shifting a must? I borrow it from Stratified Transformer and Point Transformer.
        coord -= coord.min(0)
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx] if feat is not None else None, label[uniq_idx] if label is not None else None
    if voxel_max is not None:  # 如果 voxel_max 不为 None
        crop_idx = None  # 初始化裁剪索引为 None
        N = len(label)  # 点的数量
        if N >= voxel_max:  # 如果点的数量大于或等于 voxel_max
            init_idx = np.random.randint(N) if 'train' in split else N // 2  # 如果在训练集中，则随机选择一个初始索引；否则选择中间索引
            # 根据点之间的距离对它们进行排序，并选取前 voxel_max 个点
            crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
        elif not variable:  # 如果不是可变的情况（批处理数据）
            cur_num_points = N  # 当前点的数量
            query_inds = np.arange(cur_num_points)  # 创建一个索引数组
            padding_choice = np.random.choice(
                cur_num_points, voxel_max - cur_num_points)  # 随机选择一些点进行填充，使得总点数达到 voxel_max
            crop_idx = np.hstack([query_inds, query_inds[padding_choice]])  # 组合原始索引和填充的索引
        crop_idx = np.arange(coord.shape[0]) if crop_idx is None else crop_idx  # 如果裁剪索引为 None，则将其设为全部索引，否则保持不变

        if shuffle:
            shuffle_choice = np.random.permutation(np.arange(len(crop_idx)))
            crop_idx = crop_idx[shuffle_choice]
        coord, feat, label = coord[crop_idx], feat[crop_idx] if feat is not None else None, label[crop_idx] if label is not None else None
    coord -= coord.min(0)
    return coord.astype(np.float32), feat.astype(np.float32) if feat is not None else None , label.astype(np.long) if label is not None else None


def get_features_by_keys(data, keys='pos,x'):
    key_list = keys.split(',')
    if len(key_list) == 1:
        return data[keys].transpose(1,2).contiguous()
    else:
        return torch.cat([data[key] for key in keys.split(',')], -1).transpose(1,2).contiguous()


def get_class_weights(num_per_class, normalize=False):
    weight = num_per_class / float(sum(num_per_class))
    ce_label_weight = 1 / (weight + 0.02)

    if normalize:
        ce_label_weight = (ce_label_weight *
                           len(ce_label_weight)) / ce_label_weight.sum()
    return torch.from_numpy(ce_label_weight.astype(np.float32))

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
            # 对数据进行裁剪
            coord, feat, label = crop_pc(
                coord, feat, label, self.split, self.voxel_size, self.voxel_max,
                downsample=not self.presample, variable=self.variable, shuffle=self.shuffle)

        # 将标签转换为长整型并压缩维度
        label = label.squeeze(-1).astype(np.long)
        # 构建数据字典包含坐标、特征和标签
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
    dataset = S3DIS(voxel_max=None, presample=False, voxel_size=0.08, data_root='/home/tyy/TYY/Pycharm_Project/PointMetaBase-main (1)/data/s3dis', loop=30)
    data = dataset.__getitem__(0)
    p0 = data['pos']
    for i in range(len(p0)):
        with open("points-nan-0.08-2.txt", "a+") as f:
            f.write(
                "{} {} {}\n".format(p0[i, 0], p0[i, 1], p0[i, 2]))
    f.close()
    len = dataset.__len__()
    print(len)
