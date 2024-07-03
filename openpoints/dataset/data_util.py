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
    # 确保输入的数组是二维的
    arr = arr.copy()
    # 创建数组的副本，以避免修改原数组
    arr = arr.astype(np.uint64, copy=False)
    # 将数组的元素类型转换为无符号64位整数（uint64）
    hashed_arr = np.uint64(14695981039346656037) * \
                 np.ones(arr.shape[0], dtype=np.uint64)
    # 初始化哈希数组，使用FNV-1a哈希算法的初始值（offset basis）
    # [1542355, 1]
    for j in range(arr.shape[1]):
        # 遍历数组的每一列
        hashed_arr *= np.uint64(1099511628211)
        # 乘以FNV-1a哈希算法的质数
        hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
        # 对哈希数组和当前列的元素进行按位异或操作
    return hashed_arr
    # 返回计算得到的哈希数组


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
    # 将坐标离散化为体素坐标(立方体)
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
    # 标签相同的点即在同一个网格中，出现次数即为该网格中的点数
    _, voxel_idx, count = np.unique(key_sort, return_counts=True, return_inverse=True)
    # 这行代码的目的是在每个体素中随机选择一个点作为训练点。
    # 通过计算累积和和随机生成的整数数组，确保了选择的点在每个体素内，并且在不同的训练迭代中，可以在每个体素中选择不同的点进行训练，增加了训练的多样性。

    if mode == 0:  # 训练模式
        # 每个体素网格中选取一个点进行下采样
        # np.cumsum(np.insert(count, 0, 0)[0:-1]) 得到每个网格单元的起始索引
        # np.random.randint(0, count.max(), count.size) % count 对这些随机整数取模，然后取余数确保其范围在 [0, count[i]) 之间
        idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + np.random.randint(0, count.max(), count.size) % count
        idx_unique = idx_sort[idx_select]  # 根据随机选择的索引获取最终选择的索引
        return idx_unique
    else:  # 验证模式
        return idx_sort, voxel_idx, count  # 返回排序后的索引、体素索引和计数


def crop_pc(coord, feat, label, split='train', voxel_size=0.04, voxel_max=None, downsample=True, variable=True,
            shuffle=True):
    if voxel_size and downsample:
        # 对坐标进行偏移，使得所有点的坐标最小值为0
        coord -= coord.min(0)
        # 对点云进行体素化处理，下采样得到唯一索引
        uniq_idx = voxelize(coord, voxel_size)
        # 根据唯一索引对坐标、特征和标签进行筛选
        coord, feat, label = coord[uniq_idx], feat[uniq_idx] if feat is not None else None, label[
            uniq_idx] if label is not None else None

    if voxel_max is not None:
        crop_idx = None
        N = len(label)  # 获取点的数量

        if N >= voxel_max:
            # 如果点的数量大于或等于最大体素数
            # 在训练时随机选择一个初始点，测试时选择中点
            init_idx = np.random.randint(N) if 'train' in split else N // 2
            # 计算所有点到初始点的距离，并取最近的voxel_max个点
            crop_idx = np.argsort(
                np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
        elif not variable:
            # 如果点的数量小于最大体素数，并且非可变
            cur_num_points = N  # 当前点数
            query_inds = np.arange(cur_num_points)  # 当前点的索引
            # 随机选择一些点进行填充，使得点的数量达到最大体素数
            padding_choice = np.random.choice(
                cur_num_points, voxel_max - cur_num_points)
            crop_idx = np.hstack([query_inds, query_inds[padding_choice]])

        # 如果没有指定裁剪索引，默认使用所有点
        crop_idx = np.arange(coord.shape[0]) if crop_idx is None else crop_idx

        if shuffle:
            # 如果需要打乱点的顺序，随机排列索引
            shuffle_choice = np.random.permutation(np.arange(len(crop_idx)))
            crop_idx = crop_idx[shuffle_choice]

        # 根据裁剪索引筛选坐标、特征和标签
        coord, feat, label = coord[crop_idx], feat[crop_idx] if feat is not None else None, label[
            crop_idx] if label is not None else None

    # 再次对坐标进行偏移，使得所有点的坐标最小值为0
    coord -= coord.min(0)

    return coord.astype(np.float32), feat.astype(np.float32) if feat is not None else None, label.astype(
        np.long) if label is not None else None


def get_features_by_keys(data, keys='pos,x'):
    key_list = keys.split(',')
    if len(key_list) == 1:
        return data[keys].transpose(1,2).contiguous()
    else:
        # data['hei']
        if len(data[('heights')].shape) < 3 :
            data[('heights')] = data[('heights')].unsqueeze(0)
        return torch.cat([data[key] for key in keys.split(',')], -1).transpose(1,2).contiguous()


def get_class_weights(num_per_class, normalize=False):
    weight = num_per_class / float(sum(num_per_class))
    ce_label_weight = 1 / (weight + 0.02)

    if normalize:
        ce_label_weight = (ce_label_weight *
                           len(ce_label_weight)) / ce_label_weight.sum()
    return torch.from_numpy(ce_label_weight.astype(np.float32))
