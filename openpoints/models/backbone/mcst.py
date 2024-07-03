"""
Mcst
"""
from typing import List, Type
import logging
import torch
import torch.nn as nn
from ..build import MODELS
from ..layers import create_convblock1d, create_convblock2d, create_act, CHANNEL_MAP, \
    create_grouper, furthest_point_sample, random_sample, three_interpolation
import torch.nn.functional as F
import gc
def get_reduction_fn(reduction):
    reduction = 'mean' if reduction.lower() == 'avg' else reduction
    assert reduction in ['sum', 'max', 'mean']
    if reduction == 'max':
        pool = lambda x: torch.max(x, dim=-1, keepdim=False)[0]
    elif reduction == 'mean':
        pool = lambda x: torch.mean(x, dim=-1, keepdim=False)
    elif reduction == 'sum':
        pool = lambda x: torch.sum(x, dim=-1, keepdim=False)
    return pool


def get_aggregation_feautres(p, dp, f, fj, feature_type='dp_fj'):
    if feature_type == 'dp_fj':
        fj = torch.cat([dp, fj], 1)
    elif feature_type == 'dp_fj_df':
        df = fj - f.unsqueeze(-1)
        fj = torch.cat([dp, fj, df], 1)
    elif feature_type == 'pi_dp_fj_df':
        df = fj - f.unsqueeze(-1)
        fj = torch.cat([p.transpose(1, 2).unsqueeze(-1).expand(-1, -1, -1, df.shape[-1]), dp, fj, df], 1)
    elif feature_type == 'dp_df':
        df = fj - f.unsqueeze(-1)
        fj = torch.cat([dp, df], 1)
    return fj


class LocalAggregation(nn.Module):
    """Local aggregation layer for a set
    Set abstraction layer abstracts features from a larger set to a smaller set
    Local aggregation layer aggregates features from the same set
    """

    def __init__(self,
                 channels: List[int],
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'},
                 group_args={'NAME': 'ballquery', 'radius': 0.1, 'nsample': 16},
                 conv_args=None,
                 feature_type='dp_fj',
                 reduction='max',
                 last_act=True,
                 **kwargs
                 ):
        super().__init__()
        if kwargs:
            logging.warning(f"kwargs: {kwargs} are not used in {__class__.__name__}")
        channels1 = channels
        convs1 = []
        for i in range(len(channels1) - 1):  # #layers in each blocks
            convs1.append(create_convblock1d(channels1[i], channels1[i + 1],
                                             norm_args=norm_args,
                                            act_args=None if i == (
                                                    len(channels1) - 2) and not last_act else act_args,
                                             **conv_args)
                          )
        self.convs1 = nn.Sequential(*convs1)
        self.grouper = create_grouper(group_args)
        self.reduction = reduction.lower()
        self.pool = get_reduction_fn(self.reduction)
        self.feature_type = feature_type

    def forward(self, pf, pe) -> torch.Tensor:
        # p: position, f: feature
        p, f = pf
        # preconv
        f = self.convs1(f)
        # grouping
        dp, fj = self.grouper(p, p, f)
        # pe + fj
        f = pe + fj
        f = self.pool(f)
        """ DEBUG neighbor numbers. 
        if f.shape[-1] != 1:
            query_xyz, support_xyz = p, p
            radius = self.grouper.radius
            dist = torch.cdist(query_xyz.cpu(), support_xyz.cpu())
            points = len(dist[dist < radius]) / (dist.shape[0] * dist.shape[1])
            logging.info(
                f'query size: {query_xyz.shape}, support size: {support_xyz.shape}, radius: {radius}, num_neighbors: {points}')
        DEBUG end """
        return f

class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, kernel_size=1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, kernel_size=1, bias=False)
        self.q_conv.weight = nn.Parameter(self.k_conv.weight.clone())
        self.v_conv = nn.Conv1d(channels, channels, kernel_size=1)
        self.trans_conv = nn.Conv1d(channels, channels, kernel_size=1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        gc.collect()
        torch.cuda.empty_cache()
        x_q = self.q_conv(x).permute(0, 2, 1) # b, n, c
        x_k = self.k_conv(x)  # b, c, n
        x_v = self.v_conv(x)
        energy = torch.bmm(x_q, x_k) # b, n, n
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        x_r = torch.bmm(x_v, attention) # b, c, n
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x

class PointTransformer(nn.Module):
    def __init__(self):
        super(PointTransformer, self).__init__()


        self.sa1 = SA_Layer(128)
        self.sa2 = SA_Layer(128)
        self.sa3 = SA_Layer(128)
        self.sa4 = SA_Layer(128)

        self.conv_fuse = nn.Sequential(
                    nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                    nn.BatchNorm1d(1024),
                    nn.LeakyReLU(negative_slope=0.2)
                )
        self.fc1 = nn.Sequential(
            nn.Conv1d(3, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = x[:, 0:3, :]

        batch_size, _, N = x.size()
        x = self.fc1(x)
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x_point_feature = self.conv_fuse(x)
        x_max = torch.max(x_point_feature, 2)[0]
        x_avg = torch.mean(x_point_feature, 2)
        x_max_feature = x_max.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)
        x_avg_feature = x_avg.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)
        x_global_feature = torch.cat((x_max_feature, x_avg_feature), 1)

        return x_point_feature, x_global_feature

class SetAbstraction(nn.Module):
    """The modified set abstraction module in PointNet++ with residual connection support
    """

    def __init__(self,
                 in_channels, out_channels,
                 layers=1,
                 stride=1,
                 group_args={'NAME': 'ballquery',
                             'radius': 0.1, 'nsample': 16},
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'},
                 conv_args=None,
                 sampler='fps',
                 feature_type='dp_fj',
                 use_res=False,
                 is_head=False,
                 **kwargs,
                 ):
        super().__init__()
        self.stride = stride
        self.is_head = is_head
        self.all_aggr = not is_head and stride == 1
        self.use_res = use_res and not self.all_aggr and not self.is_head
        self.feature_type = feature_type

        mid_channel = out_channels // 2 if stride > 1 else out_channels
        channels = [in_channels] + [mid_channel] * \
                   (layers - 1) + [out_channels]
        channels[0] = in_channels #if is_head else CHANNEL_MAP[feature_type](channels[0])
        channels1 = channels
        # channels2 = copy.copy(channels)
        channels2 = [in_channels] + [32,32] * (min(layers, 2) - 1) + [out_channels] # 16
        channels2[0] = 3
        convs1 = []
        convs2 = []

        if self.use_res:
            self.skipconv = create_convblock1d(
                in_channels, channels[-1], norm_args=None, act_args=None) if in_channels != channels[
                -1] else nn.Identity()
            self.act = create_act(act_args)

        # actually, one can use local aggregation layer to replace the following
        for i in range(len(channels1) - 1):  # #layers in each blocks
            convs1.append(create_convblock1d(channels1[i], channels1[i + 1],
                                             norm_args=norm_args if not is_head else None,
                                             act_args=None if i == len(channels) - 2
                                                            and (self.use_res or is_head) else act_args,
                                             **conv_args)
                          )
        self.convs1 = nn.Sequential(*convs1)

        if not is_head:
            for i in range(len(channels2) - 1):  # #layers in each blocks
                convs2.append(create_convblock2d(channels2[i], channels2[i + 1],
                                                 norm_args=norm_args if not is_head else None,
                                                #  act_args=None if i == len(channels) - 2
                                                #                 and (self.use_res or is_head) else act_args,
                                                 act_args=act_args,
                                                **conv_args)
                            )
            self.convs2 = nn.Sequential(*convs2)
            if self.all_aggr:
                group_args.nsample = None
                group_args.radius = None
            self.grouper = create_grouper(group_args)
            self.pool = lambda x: torch.max(x, dim=-1, keepdim=False)[0]
            if sampler.lower() == 'fps':
                self.sample_fn = furthest_point_sample
            elif sampler.lower() == 'random':
                self.sample_fn = random_sample

    def forward(self, pf_pe):
        p, f, pe = pf_pe
        if self.is_head:
            f = self.convs1(f)  # (n, c)
        else:
            if not self.all_aggr:
                idx = self.sample_fn(p, p.shape[1] // self.stride).long()
                new_p = torch.gather(p, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
            else:
                new_p = p
            """ DEBUG neighbor numbers. 
            query_xyz, support_xyz = new_p, p
            radius = self.grouper.radius
            dist = torch.cdist(query_xyz.cpu(), support_xyz.cpu())
            points = len(dist[dist < radius]) / (dist.shape[0] * dist.shape[1])
            logging.info(f'query size: {query_xyz.shape}, support size: {support_xyz.shape}, radius: {radius}, num_neighbors: {points}')
            DEBUG end """
            if self.use_res or 'df' in self.feature_type:
                fi = torch.gather(
                    f, -1, idx.unsqueeze(1).expand(-1, f.shape[1], -1))
                if self.use_res:
                    identity = self.skipconv(fi)
            else:
                fi = None
            # preconv
            f = self.convs1(f)
            # grouping
            dp, fj = self.grouper(new_p, p, f)
            # conv on neighborhood_dp
            pe = self.convs2(dp)
            # pe + fj
            f = pe + fj
            f = self.pool(f)
            if self.use_res:
                f = self.act(f + identity)
            p = new_p
        return p, f, pe

class PointSetAbstractionMsg(nn.Module):
    def __init__(self, k):
        super(PointSetAbstractionMsg, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv1d(9, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Conv1d(256, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        # self.SA4 = PointNetSetAbstractionMsg(16, [0.4, 0.8], [16, 32], 256 + 256, [[256, 256, 512], [256, 384, 512]])
        self.k = k
    def forward(self, p, f):
        # [B, N, C]
        centroid_xyz = p[-1]
        centroid_features = f[-1].permute(0, 2, 1)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        xyz = p[0].to(device)
        centroid_xyz = centroid_xyz.to(device)
        centroid_features = centroid_features.to(device)
        # 计算每个点到质心点的距离
        # 可以使用欧氏距离或其他距离度量方法
        distances = torch.cdist(xyz, centroid_xyz, p=2)  # Compute Euclidean distances
        _, nearest_centroid_indices = torch.topk(distances, k=self.k, largest=False, sorted=True, dim=2)  # Find nearest centroid indices

        epsilon = 1e-8  # Avoid division by zero
        weights = 1.0 / (distances.gather(2, nearest_centroid_indices).clamp_min(
            epsilon))  # Calculate weights based on distance reciprocal

        # 对权重进行归一化处理
        sum_weights = torch.sum(weights, dim=2, keepdim=True)  # 求和得到每个点的总权重
        normalized_weights_flattened = weights / sum_weights  # 对权重进行归一化处理
        # normalized_weights_flattened = self.bns3(weights.permute(0, 2, 1)).permute(0, 2, 1)
        # 为每个点寻找最近质心点，并将质心点坐标作为其坐标
        nearest_centroid_features = torch.gather(centroid_features.unsqueeze(2).expand(-1, -1, 3, -1), 1,
                                    nearest_centroid_indices.unsqueeze(-1).expand(-1, -1, -1, centroid_features.shape[-1])
                                    )
        # Expand normalized weights for element-wise multiplication
        normalized_weights_expanded = normalized_weights_flattened.unsqueeze(-1).expand_as(nearest_centroid_features)

        # Element-wise multiplication for weighted features
        weighted_features = normalized_weights_expanded * nearest_centroid_features

        # Sum the weighted features
        sum_weighted_features = torch.sum(weighted_features, dim=2)
        return sum_weighted_features

class FeaturePropogation(nn.Module):
    """The Feature Propogation module in PointNet++
    """

    def __init__(self, mlp,
                 upsample=True,
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'}
                 ):
        """
        Args:
            mlp: [current_channels, next_channels, next_channels]
            out_channels:
            norm_args:
            act_args:
        """
        super().__init__()
        if not upsample:
            self.linear2 = nn.Sequential(
                nn.Linear(mlp[0], mlp[1]), nn.ReLU(inplace=True))
            mlp[1] *= 2
            linear1 = []
            for i in range(1, len(mlp) - 1):
                linear1.append(create_convblock1d(mlp[i], mlp[i + 1],
                                                  norm_args=norm_args, act_args=act_args
                                                  ))
            self.linear1 = nn.Sequential(*linear1)
        else:
            convs = []
            for i in range(len(mlp) - 1):
                convs.append(create_convblock1d(mlp[i], mlp[i + 1],
                                                norm_args=norm_args, act_args=act_args
                                                ))
            self.convs = nn.Sequential(*convs)

        self.pool = lambda x: torch.mean(x, dim=-1, keepdim=False)

    def forward(self, pf1, pf2=None):
        # pfb1 is with the same size of upsampled points
        if pf2 is None:
            _, f = pf1  # (B, N, 3), (B, C, N)
            f_global = self.pool(f)
            f = torch.cat(
                (f, self.linear2(f_global).unsqueeze(-1).expand(-1, -1, f.shape[-1])), dim=1)
            f = self.linear1(f)
        else:
            p1, f1 = pf1
            p2, f2 = pf2
            if f1 is not None:
                f = self.convs(
                    torch.cat((f1, three_interpolation(p1, p2, f2)), dim=1))
            else:
                f = self.convs(three_interpolation(p1, p2, f2))
        return f

class InvResMLP(nn.Module):
    def __init__(self,
                 in_channels,
                 norm_args=None,
                 act_args=None,
                 aggr_args={'feature_type': 'dp_fj', "reduction": 'max'},
                 group_args={'NAME': 'ballquery'},
                 conv_args=None,
                 expansion=1,
                 use_res=True,
                 num_posconvs=2,#2,
                 less_act=False,
                 **kwargs
                 ):
        super().__init__()
        self.use_res = use_res
        mid_channels = int(in_channels * expansion)
        self.convs = LocalAggregation([in_channels, in_channels],
                                      norm_args=norm_args, act_args=act_args ,#if num_posconvs > 0 else None,
                                      group_args=group_args, conv_args=conv_args,
                                      **aggr_args, **kwargs)
        if num_posconvs < 1:
            channels = []
        elif num_posconvs == 1:
            channels = [in_channels, in_channels]
        elif num_posconvs == 4:
            channels = [in_channels, in_channels, in_channels, in_channels, in_channels]
        elif num_posconvs == 3:
            channels = [in_channels, in_channels, in_channels, in_channels]
        else:
            channels = [in_channels, mid_channels, in_channels]
        pwconv = []
        # point wise after depth wise conv (without last layer)
        for i in range(len(channels) - 1):
            pwconv.append(create_convblock1d(channels[i], channels[i + 1],
                                             norm_args=norm_args,
                                             act_args=act_args if
                                             (i != len(channels) - 2) and not less_act else None,
                                             **conv_args)
                          )
        self.pwconv = nn.Sequential(*pwconv)
        self.act = create_act(act_args)

    def forward(self, pf_pe):
        p, f, pe = pf_pe
        identity = f
        f = self.convs([p, f], pe)
        f = self.pwconv(f)
        if f.shape[-1] == identity.shape[-1] and self.use_res:
            f += identity
        f = self.act(f)
        return [p, f, pe]


@MODELS.register_module()
class McstDecoder(nn.Module):
    def __init__(self,
                 encoder_channel_list: List[int],
                 decoder_layers: int = 2,
                 decoder_stages: int = 4,
                 **kwargs
                 ):
        super().__init__()
        self.k = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transformer = PointTransformer()
        self.setabs = PointSetAbstractionMsg(self.k)
        self.conv_fuse = nn.Sequential(
                    nn.Conv1d(1024 * 3 + 512, 512, kernel_size=1, bias=False),
                    nn.BatchNorm1d(512),
                    nn.LeakyReLU(negative_slope=0.2)
                )
        # self.lbrd = nn.Sequential(
        #     nn.Conv1d(1024 * 3 + 512, 512, 1),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(),
        #     nn.Dropout(0.5)
        # )
        # self.lbr = nn.Sequential(
        #     nn.Conv1d(512, 512, 1),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU()
        # )


    def forward(self, p, f):
        torch.cuda.empty_cache()
        point_feature, global_feature = self.transformer(p[0])
        local_feature = self.setabs(p, f)
        local_feature = local_feature.permute(0, 2, 1).to(self.device)
        point_feature = point_feature.to(self.device)
        global_feature = global_feature.to(self.device)
        x = torch.cat([point_feature, global_feature, local_feature], dim=1)
        x = self.conv_fuse(x)
        # concatenated_features 的形状为 [B, N, 2*C]，即每个点的特征和最近质心点的特征进行了拼接

        # x = self.lbrd(x)
        # x = self.lbr(x)

        return x
