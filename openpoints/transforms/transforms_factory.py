import torch
from ..utils.registry import Registry

DataTransforms = Registry('datatransforms')


def concat_collate_fn(datas):
    """collate fn for point transformer
    """
    pts, feats, labels, offset, count = [], [], [], [], 0
    for data in datas:
        count += len(data['pos'])
        offset.append(count)
        pts.append(data['pos'])
        feats.append(data['x'])
        labels.append(data['y'])
    data = {'pos': torch.cat(pts), 'x': torch.cat(feats), 'y': torch.cat(labels),
            'o': torch.IntTensor(offset)}
    return data


class Compose(object):
    """Composes several transforms together."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, args):
        for t in self.transforms:
            args = t(args)
        return args


class ListCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, coord, feat, label):
        for t in self.transforms:
            coord, feat, label = t(coord, feat, label)
        return coord, feat, label


def build_transforms_from_cfg(split, datatransforms_cfg):
    """
    为某个特定数据集分割构建数据集转换，由 `datatransforms_cfg` 定义。
    """
    # 获取数据集转换列表、转换参数和组合函数
    transform_list = datatransforms_cfg.get(split, None)
    transform_args = datatransforms_cfg.get('kwargs', None)
    compose_fn = eval(datatransforms_cfg.get('compose_fn', 'Compose'))

    # 如果转换列表为空或长度为 0，则返回 None
    if transform_list is None or len(transform_list) == 0:
        return None

    point_transforms = []

    # 如果转换列表长度大于 1，则构建多个数据集转换
    if len(transform_list) > 1:
        for t in transform_list:
            point_transforms.append(DataTransforms.build({'NAME': t}, default_args=transform_args))
        return compose_fn(point_transforms)
    else:
        # 否则，构建单个数据集转换
        return DataTransforms.build({'NAME': transform_list[0]}, default_args=transform_args)

