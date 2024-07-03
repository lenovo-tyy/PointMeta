"""
(Distributed) training script for scene segmentation
This file currently supports training and testing on S3DIS
If more than 1 GPU is provided, will launch multi processing distributed training by default
if you only wana use 1 GPU, set `CUDA_VISIBLE_DEVICES` accordingly
Author: Guocheng Qian @ 2022, guocheng.qian@kaust.edu.sa
"""
import __init__
import argparse, yaml, os, logging, numpy as np, csv, wandb, glob
from tqdm import tqdm
import torch, torch.nn as nn
from torch import distributed as dist, multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch_scatter import scatter
from openpoints.utils import set_random_seed, save_checkpoint, load_checkpoint, resume_checkpoint, setup_logger_dist, \
    cal_model_parm_nums, Wandb, generate_exp_directory, resume_exp_directory, EasyConfig, dist_utils, find_free_port
from openpoints.utils import AverageMeter, ConfusionMatrix, get_mious
from openpoints.dataset import build_dataloader_from_cfg, get_features_by_keys, get_class_weights
from openpoints.dataset.data_util import voxelize
from openpoints.dataset.semantic_kitti.semantickitti import load_label_kitti, load_pc_kitti, remap_lut_read, remap_lut_write, get_semantickitti_file_list
from openpoints.transforms import build_transforms_from_cfg
from openpoints.optim import build_optimizer_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg
from openpoints.loss import build_criterion_from_cfg
from openpoints.models import build_model_from_cfg
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def write_to_csv(oa, macc, miou, ious, best_epoch, cfg, write_header=True, area=5):
    ious_table = [f'{item:.2f}' for item in ious]
    header = ['method', 'Area', 'OA', 'mACC', 'mIoU'] + cfg.classes + ['best_epoch', 'log_path', 'wandb link']
    data = [cfg.cfg_basename, str(area), f'{oa:.2f}', f'{macc:.2f}',
            f'{miou:.2f}'] + ious_table + [str(best_epoch), cfg.run_dir,
                                           wandb.run.get_url() if cfg.wandb.use_wandb else '-']
    with open(cfg.csv_path, 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(data)
        f.close()


def generate_data_list(cfg):
    if 's3dis' in cfg.dataset.common.NAME.lower():
        raw_root = os.path.join(cfg.dataset.common.data_root, 'raw')
        data_list = sorted(os.listdir(raw_root))
        data_list = [os.path.join(raw_root, item) for item in data_list if
                     'Area_{}'.format(cfg.dataset.common.test_area) in item]
    elif 'scannet' in cfg.dataset.common.NAME.lower():
        data_list = glob.glob(os.path.join(cfg.dataset.common.data_root, cfg.dataset.test.split, "*.pth"))
    elif 'semantickitti' in cfg.dataset.common.NAME.lower():
        if cfg.dataset.test.split == 'val':
            split_no = 1
        else:
            split_no = 2
        data_list = get_semantickitti_file_list(os.path.join(cfg.dataset.common.data_root, 'sequences'),
                                                str(cfg.dataset.test.test_id + 11))[split_no]
    elif 'line_seg' == cfg.dataset.common.NAME.lower():
        raw_root = os.path.join(cfg.dataset.common.data_root, 'raw')
        data_list = sorted(os.listdir(raw_root))
        data_list = [os.path.join(raw_root, item) for item in data_list if
                     'Area_{}'.format(cfg.dataset.common.test_area) in item]
    else:
        raise Exception('dataset not supported yet'.format(args.data_name))
    return data_list


def load_data(data_path, cfg):
    label, feat = None, None
    if 's3dis' in cfg.dataset.common.NAME.lower():
        data = np.load(data_path)  # xyzrgbl, N*7
        coord, feat, label = data[:, :3], data[:, 3:6], data[:, 6]
        feat = np.clip(feat / 255., 0, 1).astype(np.float32)
    elif 'line_seg' in cfg.dataset.common.NAME.lower():
        data = np.load(data_path)  # xyzrgbl, N*7
        coord, feat, label = data[:, :3], data[:, 3:6], data[:, 6]
        feat = np.clip(feat / 255., 0, 1).astype(np.float32)
    elif 'scannet' in cfg.dataset.common.NAME.lower():
        data = torch.load(data_path)  # xyzrgbl, N*7
        coord, feat = data[0], data[1]
        if cfg.dataset.test.split != 'test':
           label = data[2]
        else:
            label = None
        feat = np.clip((feat + 1) / 2., 0, 1).astype(np.float32)
    elif 'semantickitti' in cfg.dataset.common.NAME.lower():
        coord = load_pc_kitti(data_path[0])
        if cfg.dataset.test.split != 'test':
            label = load_label_kitti(data_path[1], remap_lut_read)
    coord -= coord.min(0)

    idx_points = []
    voxel_idx, reverse_idx_part,reverse_idx_sort = None, None, None
    voxel_size = cfg.dataset.common.get('voxel_size', None)

    if voxel_size is not None:
        # idx_sort: original point indicies sorted by voxel NO.
        # voxel_idx: Voxel NO. for the sorted points
        idx_sort, voxel_idx, count = voxelize(coord, voxel_size, mode=1)
        if cfg.get('test_mode', 'multi_voxel') == 'nearest_neighbor':
            idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + np.random.randint(0, count.max(), count.size) % count
            idx_part = idx_sort[idx_select]
            npoints_subcloud = voxel_idx.max()+1
            idx_shuffle = np.random.permutation(npoints_subcloud)
            idx_part = idx_part[idx_shuffle] # idx_part: randomly sampled points of a voxel
            reverse_idx_part = np.argsort(idx_shuffle, axis=0) # revevers idx_part to sorted
            idx_points.append(idx_part)
            reverse_idx_sort = np.argsort(idx_sort, axis=0)
        else:
            for i in range(count.max()):
                idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
                idx_part = idx_sort[idx_select]
                np.random.shuffle(idx_part)
                idx_points.append(idx_part)
    else:
        idx_points.append(np.arange(label.shape[0]))
    return coord, feat, label, idx_points, voxel_idx, reverse_idx_part, reverse_idx_sort


# 定义一个名为 main 的函数，接收两个参数 gpu 和 cfg
def main(gpu, cfg):
    # 如果配置 cfg 中设置了分布式训练
    if cfg.distributed:
        # 如果 cfg 设置了多进程
        if cfg.mp:
            # 将当前 GPU ID 分配给 cfg.rank
            cfg.rank = gpu
        # 使用 PyTorch 的分布式初始化方法初始化进程组
        dist.init_process_group(
            backend=cfg.dist_backend,  # 分布式后端（通常选择 NCCL 或 Gloo）
            init_method=cfg.dist_url,  # 分布式初始化方法
            world_size=cfg.world_size,  # 进程组中的进程数
            rank=cfg.rank  # 当前进程的排名（编号）
        )
        # 对所有进程进行同步，等待所有进程都完成这个点之前的工作
        dist.barrier()

    # 设置日志记录器
    setup_logger_dist(cfg.log_path, cfg.rank, name=cfg.dataset.common.NAME)
    # 如果当前进程的排名为0（即主进程）
    if cfg.rank == 0:
        # 启动 W&B（Wandb）的记录功能
        Wandb.launch(cfg, cfg.wandb.use_wandb)
        # 如果是训练过程，则创建一个 TensorBoard 的 SummaryWriter 对象来记录日志，否则为 None
        writer = SummaryWriter(log_dir=cfg.run_dir) if cfg.is_training else None
    else:
        # 如果不是主进程，则 writer 设为 None
        writer = None
    # 设置随机种子
    set_random_seed(cfg.seed + cfg.rank, deterministic=cfg.deterministic)
    # 启用 Torch 的 cuDNN 加速
    torch.backends.cudnn.enabled = True
    # 打印配置信息
    logging.info(cfg)

    # 如果模型中没有指定输入通道数，则使用 encoder_args 中的通道数
    if cfg.model.get('in_channels', None) is None:
        cfg.model.in_channels = cfg.model.encoder_args.in_channels
    # 根据配置文件构建模型，并将其放置到指定的设备上
    model = build_model_from_cfg(cfg.model).to(cfg.rank)
    model_size = cal_model_parm_nums(model)
    # 打印模型信息和参数数量
    logging.info(model)
    logging.info('Number of params: %.4f M' % (model_size / 1e6))

    # 如果需要同步 BN 层
    if cfg.sync_bn:
        # 将模型转换为支持同步 BN 的模型
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # 打印信息，表示正在使用同步的 BatchNorm
        logging.info('Using Synchronized BatchNorm ...')
    # 如果是分布式训练
    if cfg.distributed:
        # 设置当前设备的 GPU ID
        torch.cuda.set_device(gpu)
        # 使用 PyTorch 的分布式并行训练（DistributedDataParallel），将模型放置到指定设备上
        model = nn.parallel.DistributedDataParallel(
            model.cuda(),  # 模型移至 GPU 上
            device_ids=[cfg.rank],  # 当前进程所在设备的 GPU ID
            output_device=cfg.rank  # 输出设备的 GPU ID
        )
        # 打印信息，表示正在使用分布式数据并行训练
        logging.info('Using Distributed Data parallel ...')

    # optimizer & scheduler
    optimizer = build_optimizer_from_cfg(model, lr=cfg.lr, **cfg.optimizer)
    scheduler = build_scheduler_from_cfg(cfg, optimizer)

    # build dataset
    val_loader = build_dataloader_from_cfg(
        cfg.get('val_batch_size', cfg.batch_size),  # 获取验证集的批次大小
        cfg.dataset,  # 数据集配置
        cfg.dataloader,  # 数据加载器配置
        datatransforms_cfg=cfg.datatransforms,  # 数据预处理配置
        split='val',  # 数据集拆分为验证集
        distributed=cfg.distributed  # 是否分布式
    )
    # 打印验证集数据集长度
    logging.info(f"length of validation dataset: {len(val_loader.dataset)}")
    num_classes = val_loader.dataset.num_classes if hasattr(val_loader.dataset, 'num_classes') else None
    if num_classes is not None:
        assert cfg.num_classes == num_classes
    # 打印数据集的类别数信息
    logging.info(f"number of classes of the dataset: {num_classes}")
    cfg.classes = val_loader.dataset.classes if hasattr(val_loader.dataset, 'classes') else np.arange(num_classes)
    cfg.cmap = np.array(val_loader.dataset.cmap) if hasattr(val_loader.dataset, 'cmap') else None
    validate_fn = validate if 'sphere' not in cfg.dataset.common.NAME.lower() else validate_sphere

    # optionally resume from a checkpoint
    model_module = model.module if hasattr(model, 'module') else model
    if cfg.pretrained_path is not None:
        if cfg.mode == 'resume':
            resume_checkpoint(cfg, model, optimizer, scheduler, pretrained_path=cfg.pretrained_path)
        else:
            if cfg.mode == 'val':
                best_epoch, best_val = load_checkpoint(model, pretrained_path=cfg.pretrained_path)
                val_miou, val_macc, val_oa, val_ious, val_accs = validate_fn(model, val_loader, cfg, num_votes=1)
                with np.printoptions(precision=2, suppress=True):
                    logging.info(
                        f'Best ckpt @E{best_epoch},  val_oa , val_macc, val_miou: {val_oa:.2f} {val_macc:.2f} {val_miou:.2f}, '
                        f'\niou per cls is: {val_ious}')
                return val_miou
            elif cfg.mode == 'test':
                best_epoch, best_val = load_checkpoint(model, pretrained_path=cfg.pretrained_path)
                data_list = generate_data_list(cfg)
                logging.info(f"length of test dataset: {len(data_list)}")
                test_miou, test_macc, test_oa, test_ious, test_accs, _ = test(model, data_list, cfg)

                if test_miou is not None:
                    with np.printoptions(precision=2, suppress=True):
                        logging.info(
                            f'Best ckpt @E{best_epoch},  test_oa , test_macc, test_miou: {test_oa:.2f} {test_macc:.2f} {test_miou:.2f}, '
                            f'\niou per cls is: {test_ious}')
                    cfg.csv_path = os.path.join(cfg.run_dir, cfg.run_name + '_test.csv')
                    write_to_csv(test_oa, test_macc, test_miou, test_ious, best_epoch, cfg)
                return test_miou

            elif 'encoder' in cfg.mode:
                logging.info(f'Finetuning from {cfg.pretrained_path}')
                load_checkpoint(model_module.encoder, cfg.pretrained_path, cfg.get('pretrained_module', None))
            else:
                logging.info(f'Finetuning from {cfg.pretrained_path}')
                load_checkpoint(model, cfg.pretrained_path, cfg.get('pretrained_module', None))
    else:
        logging.info('Training from scratch')

    if 'freeze_blocks' in cfg.mode:
        for p in model_module.encoder.blocks.parameters():
            p.requires_grad = False

    train_loader = build_dataloader_from_cfg(
        cfg.batch_size,  # 训练集的批次大小
        cfg.dataset,  # 数据集配置
        cfg.dataloader,  # 数据加载器配置
        datatransforms_cfg=cfg.datatransforms,  # 数据预处理配置
        split='train',  # 数据集拆分为训练集
        distributed=cfg.distributed,  # 是否分布式
    )
    # 打印训练集数据集长度
    logging.info(f"length of training dataset: {len(train_loader.dataset)}")

    cfg.criterion_args.weight = None
    if cfg.get('cls_weighed_loss', False):
        if hasattr(train_loader.dataset, 'num_per_class'):
            cfg.criterion_args.weight = get_class_weights(train_loader.dataset.num_per_class, normalize=True)
        else:
            logging.info('`num_per_class` attribute is not founded in dataset')
    criterion = build_criterion_from_cfg(cfg.criterion_args).cuda()


    # 开始训练
    if cfg.use_amp:  # 若使用自动混合精度（Automatic Mixed Precision, AMP）
        scaler = torch.cuda.amp.GradScaler()  # 创建梯度缩放器对象
    else:
        scaler = None  # 否则梯度缩放器为空

    # 初始化变量以存储验证指标及最佳模型信息
    val_miou, val_macc, val_oa, val_ious, val_accs = 0., 0., 0., [], []
    best_val, macc_when_best, oa_when_best, ious_when_best, best_epoch = 0., 0., 0., [], 0

    # 开始训练循环，迭代每个 epoch
    for epoch in range(cfg.start_epoch, cfg.epochs + 1):
        if cfg.distributed:
            train_loader.sampler.set_epoch(epoch)  # 设置分布式训练的 epoch
        if hasattr(train_loader.dataset, 'epoch'):  # 某些数据集将数据集长度设置为固定的步数。
            train_loader.dataset.epoch = epoch - 1  # 设置数据集的 epoch

        # 在一个 epoch 内执行训练，返回训练损失和指标
        train_loss, train_miou, train_macc, train_oa, _, _ = \
            train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, epoch, cfg)

        is_best = False  # 初始化为 False
        # 每隔一定周期进行验证
        if epoch % cfg.val_freq == 0:
            # 执行验证，并获取验证指标
            val_miou, val_macc, val_oa, val_ious, val_accs = validate_fn(model, val_loader, cfg)
            # 如果当前验证得到的 miou 比之前的最佳 miou 更好
            if val_miou > best_val:
                is_best = True  # 设置为 True，表示找到更好的模型
                best_val = val_miou  # 更新最佳 miou
                macc_when_best = val_macc  # 更新最佳 macc
                oa_when_best = val_oa  # 更新最佳 oa
                ious_when_best = val_ious  # 更新最佳 ious
                best_epoch = epoch  # 更新最佳 epoch
                with np.printoptions(precision=2, suppress=True):
                    logging.info(
                        f'Find a better ckpt @E{epoch}, val_miou {val_miou:.2f} val_macc {macc_when_best:.2f}, val_oa {oa_when_best:.2f}'
                        f'\nmious: {val_ious}')

        lr = optimizer.param_groups[0]['lr']  # 获取当前学习率
        # 打印当前 epoch 的训练和验证信息
        logging.info(f'Epoch {epoch} LR {lr:.6f} '
                     f'train_miou {train_miou:.2f}, val_miou {val_miou:.2f}, best val miou {best_val:.2f}')
        # 如果 writer 不为空，则将训练和验证指标写入 TensorBoard
        if writer is not None:
            writer.add_scalar('best_val', best_val, epoch)
            writer.add_scalar('val_miou', val_miou, epoch)
            writer.add_scalar('macc_when_best', macc_when_best, epoch)
            writer.add_scalar('oa_when_best', oa_when_best, epoch)
            writer.add_scalar('val_macc', val_macc, epoch)
            writer.add_scalar('val_oa', val_oa, epoch)
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('train_miou', train_miou, epoch)
            writer.add_scalar('train_macc', train_macc, epoch)
            writer.add_scalar('lr', lr, epoch)

        # 如果设置为每个 epoch 更新 scheduler
        if cfg.sched_on_epoch:
            scheduler.step(epoch)  # 更新 scheduler 的学习率

        # 如果当前进程为 rank 为 0，则保存检查点
        if cfg.rank == 0:
            save_checkpoint(cfg, model, epoch, optimizer, scheduler,
                            additioanl_dict={'best_val': best_val},
                            is_best=is_best
                            )
            is_best = False  # 重置为 False
    # 不保存文件到 Wandb 以节省空间
    # if writer is not None:
    #     Wandb.add_file(os.path.join(cfg.ckpt_dir, f'{cfg.run_name}_ckpt_best.pth'))
    # Wandb.add_file(os.path.join(cfg.ckpt_dir, f'{cfg.logname}_ckpt_latest.pth'))


    # 验证模型性能
    with np.printoptions(precision=2, suppress=True):
        logging.info(
            f'Best ckpt @E{best_epoch},  val_oa {oa_when_best:.2f}, val_macc {macc_when_best:.2f}, val_miou {best_val:.2f}, '
            f'\niou per cls is: {ious_when_best}')

    if cfg.world_size < 2:  # 不支持多 GPU 测试
        # 测试
        load_checkpoint(model, pretrained_path=os.path.join(cfg.ckpt_dir, f'{cfg.run_name}_ckpt_best.pth'))
        cfg.csv_path = os.path.join(cfg.run_dir, cfg.run_name + f'.csv')
        if 'sphere' in cfg.dataset.common.NAME.lower():
            test_miou, test_macc, test_oa, test_ious, test_accs = validate_sphere(model, val_loader, cfg)
        else:
            data_list = generate_data_list(cfg)
            test_miou, test_macc, test_oa, test_ious, test_accs, _ = test(model, data_list, cfg)
        with np.printoptions(precision=2, suppress=True):
            logging.info(
                f'Best ckpt @E{best_epoch},  test_oa {test_oa:.2f}, test_macc {test_macc:.2f}, test_miou {test_miou:.2f}, '
                f'\niou per cls is: {test_ious}')
        if writer is not None:
            writer.add_scalar('test_miou', test_miou, epoch)
            writer.add_scalar('test_macc', test_macc, epoch)
            writer.add_scalar('test_oa', test_oa, epoch)
        write_to_csv(test_oa, test_macc, test_miou, test_ious, best_epoch, cfg, write_header=True)
        logging.info(f'save results in {cfg.csv_path}')
        if cfg.use_voting:
            load_checkpoint(model, pretrained_path=os.path.join(cfg.ckpt_dir, f'{cfg.run_name}_ckpt_best.pth'))
            set_random_seed(cfg.seed)
            val_miou, val_macc, val_oa, val_ious, val_accs = validate_fn(model, val_loader, cfg, num_votes=20,
                                                                         data_transform=data_transform)
            if writer is not None:
                writer.add_scalar('val_miou20', val_miou, cfg.epochs + 50)

            ious_table = [f'{item:.2f}' for item in val_ious]
            data = [cfg.cfg_basename, 'True', f'{val_oa:.2f}', f'{val_macc:.2f}', f'{val_miou:.2f}'] + ious_table + [
                str(best_epoch), cfg.run_dir]
            with open(cfg.csv_path, 'w', encoding='UT8') as f:
                writer = csv.writer(f)
                writer.writerow(data)
    else:
        logging.warning('暂不支持多 GPU 测试。需要在此训练之后进行测试。')
    if writer is not None:
        writer.close()
    dist.destroy_process_group()
    wandb.finish(exit_code=True)



def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, epoch, cfg):
    loss_meter = AverageMeter()
    cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
    model.train()  # set model to training mode
    pbar = tqdm(enumerate(train_loader), total=train_loader.__len__(), ascii=True)
    num_iter = 0
    for idx, data in pbar:
        keys = data.keys() if callable(data.keys) else data.keys
        for key in keys:
            data[key] = data[key].cuda(non_blocking=True)
        num_iter += 1
        target = data['y'].squeeze(-1)
        """ debug
        from openpoints.dataset import vis_points
        vis_points(data['pos'].cpu().numpy()[0], labels=data['y'].cpu().numpy()[0])
        vis_points(data['pos'].cpu().numpy()[0], data['x'][0, :3, :].transpose(1, 0))
        end of debug """
        data['x'] = get_features_by_keys(data, cfg.feature_keys)
        with torch.cuda.amp.autocast(enabled=cfg.use_amp):
            logits = model(data)
            loss = criterion(logits, target) if 'mask' not in cfg.criterion_args.NAME.lower() \
                else criterion(logits, target, data['mask'])

        if cfg.use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        # optimize
        if num_iter == cfg.step_per_update:
            if cfg.get('grad_norm_clip') is not None and cfg.grad_norm_clip > 0.:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_norm_clip, norm_type=2)
            num_iter = 0

            if cfg.use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()
            if not cfg.sched_on_epoch:
                scheduler.step(epoch)

        # update confusion matrix
        cm.update(logits.argmax(dim=1), target)
        loss_meter.update(loss.item())

        if idx % cfg.print_freq:
            pbar.set_description(f"Train Epoch [{epoch}/{cfg.epochs}] "
                                 f"Loss {loss_meter.val:.3f} Acc {cm.overall_accuray:.2f}")
    miou, macc, oa, ious, accs = cm.all_metrics()
    return loss_meter.avg, miou, macc, oa, ious, accs


@torch.no_grad()
def validate(model, val_loader, cfg, num_votes=1, data_transform=None):
    model.eval()  # set model to eval mode
    cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
    pbar = tqdm(enumerate(val_loader), total=val_loader.__len__(), desc='Val', ascii=True)
    for idx, data in pbar:
        keys = data.keys() if callable(data.keys) else data.keys
        for key in keys:
            data[key] = data[key].cuda(non_blocking=True)
        target = data['y'].squeeze(-1)
        data['x'] = get_features_by_keys(data, cfg.feature_keys)
        logits = model(data)
        if 'mask' not in cfg.criterion_args.NAME or cfg.get('use_maks', False):
            cm.update(logits.argmax(dim=1), target)
        else:
            mask = data['mask'].bool()
            cm.update(logits.argmax(dim=1)[mask], target[mask])

        """visualization in debug mode
        from openpoints.dataset.vis3d import vis_points, vis_multi_points
        coord = data['pos'].cpu().numpy()[0]
        pred = logits.argmax(dim=1)[0].cpu().numpy()
        label = target[0].cpu().numpy()
        if cfg.ignore_index is not None:
            if (label == cfg.ignore_index).sum() > 0:
                pred[label == cfg.ignore_index] = cfg.num_classes
                label[label == cfg.ignore_index] = cfg.num_classes
        vis_multi_points([coord, coord], labels=[label, pred])
        """
        # tp, union, count = cm.tp, cm.union, cm.count
        # if cfg.distributed:
        #     dist.all_reduce(tp), dist.all_reduce(union), dist.all_reduce(count)
        # miou, macc, oa, ious, accs = get_mious(tp, union, count)
        # with np.printoptions(precision=2, suppress=True):
        #     logging.info(f'{idx}-th cloud,  test_oa , test_macc, test_miou: {oa:.2f} {macc:.2f} {miou:.2f}, '
        #                 f'\niou per cls is: {ious}')

    tp, union, count = cm.tp, cm.union, cm.count
    if cfg.distributed:
        dist.all_reduce(tp), dist.all_reduce(union), dist.all_reduce(count)
    miou, macc, oa, ious, accs = get_mious(tp, union, count)
    return miou, macc, oa, ious, accs


@torch.no_grad()
def validate_sphere(model, val_loader, cfg, num_votes=1, data_transform=None):
    """
    validation for sphere sampled input points with mask.
    in this case, between different batches, there are overlapped points.
    thus, one point can be evaluated multiple times.
    In this validate_mask, we will avg the logits.
    """
    model.eval()  # set model to eval mode
    cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
    if cfg.get('visualize', False):
        from openpoints.dataset.vis3d import write_obj
        cfg.vis_dir = os.path.join(cfg.run_dir, 'visualization')
        os.makedirs(cfg.vis_dir, exist_ok=True)
        cfg.cmap = cfg.cmap.astype(np.float32) / 255.

    pbar = tqdm(enumerate(val_loader), total=val_loader.__len__(), ascii=True)
    all_logits, idx_points = [], []
    for idx, data in pbar:
        for key in data.keys():
            data[key] = data[key].cuda(non_blocking=True)
        data['x'] = get_features_by_keys(data, cfg.feature_keys)
        logits = model(data)
        all_logits.append(logits)
        idx_points.append(data['input_inds'])
    all_logits = torch.cat(all_logits, dim=0).transpose(1, 2).reshape(-1, cfg.num_classes)
    idx_points = torch.cat(idx_points, dim=0).flatten()

    if cfg.distributed:
        dist.all_reduce(all_logits), dist.all_reduce(idx_points)

    # average overlapped predictions to subsampled points
    all_logits = scatter(all_logits, idx_points, dim=0, reduce='mean')

    # now, project the original points to the subsampled points
    # these two targets would be very similar but not the same
    # val_points_targets = all_targets[val_points_projections]
    # torch.allclose(val_points_labels, val_points_targets)
    all_logits = all_logits.argmax(dim=1)
    val_points_labels = torch.from_numpy(val_loader.dataset.clouds_points_labels[0]).squeeze(-1).to(all_logits.device)
    val_points_projections = torch.from_numpy(val_loader.dataset.projections[0]).to(all_logits.device).long()
    val_points_preds = all_logits[val_points_projections]

    del all_logits, idx_points
    torch.cuda.empty_cache()

    cm.update(val_points_preds, val_points_labels)
    miou, macc, oa, ious, accs = cm.all_metrics()

    if cfg.get('visualize', False):
        dataset_name = cfg.dataset.common.NAME.lower()
        coord = val_loader.dataset.clouds_points[0]
        colors = val_loader.dataset.clouds_points_colors[0].astype(np.float32)
        gt = val_points_labels.cpu().numpy().squeeze()
        pred = val_points_preds.cpu().numpy().squeeze()
        gt = cfg.cmap[gt, :]
        pred = cfg.cmap[pred, :]
        # output pred labels
        if 's3dis' in dataset_name:
            file_name = f'{dataset_name}'
        else:
            file_name = f'{dataset_name}'

        # save per room
        rooms = val_loader.dataset.clouds_rooms[0]

        for idx in tqdm(range(len(rooms)-1), desc='save visualization', ascii=True):
            start_idx, end_idx = rooms[idx], rooms[idx+1]
            # write_obj(coord[start_idx:end_idx], colors[start_idx:end_idx],
            #             os.path.join(cfg.vis_dir, f'input-{file_name}-{idx}.obj'))
            # # output ground truth labels
            # write_obj(coord[start_idx:end_idx], gt[start_idx:end_idx],
            #             os.path.join(cfg.vis_dir, f'gt-{file_name}-{idx}.obj'))
            # # output pred labels
            write_obj(coord[start_idx:end_idx], pred[start_idx:end_idx],
                        os.path.join(cfg.vis_dir, f'pix4point-{file_name}-{idx}.obj'))
    return miou, macc, oa, ious, accs


# TODO: multi gpu support. Warp to a dataloader.
@torch.no_grad()
def test(model, data_list, cfg, num_votes=1):
    """using a part of original point cloud as input to save memory.
    Args:
        model (_type_): _description_
        test_loader (_type_): _description_
        cfg (_type_): _description_
        num_votes (int, optional): _description_. Defaults to 1.
    Returns:
        _type_: _description_
    """
    model.eval()  # set model to eval mode
    all_cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
    set_random_seed(0)
    cfg.visualize = cfg.get('visualize', False)
    if cfg.visualize:
        from openpoints.dataset.vis3d import write_obj
        cfg.vis_dir = os.path.join(cfg.run_dir, 'visualization')
        os.makedirs(cfg.vis_dir, exist_ok=True)
        cfg.cmap = cfg.cmap.astype(np.float32) / 255.

    # data
    trans_split = 'val' if cfg.datatransforms.get('test', None) is None else 'test'
    pipe_transform = build_transforms_from_cfg(trans_split, cfg.datatransforms)

    dataset_name = cfg.dataset.common.NAME.lower()
    len_data = len(data_list)

    cfg.save_path = cfg.get('save_path', f'results/{cfg.task_name}/{cfg.dataset.test.split}/{cfg.cfg_basename}')
    if 'semantickitti' in cfg.dataset.common.NAME.lower():
        cfg.save_path = os.path.join(cfg.save_path, str(cfg.dataset.test.test_id + 11), 'predictions')
    os.makedirs(cfg.save_path, exist_ok=True)

    gravity_dim = cfg.datatransforms.kwargs.gravity_dim
    nearest_neighbor = cfg.get('test_mode', 'multi_voxel') == 'nearest_neighbor'
    for cloud_idx, data_path in enumerate(data_list):
        logging.info(f'Test [{cloud_idx}]/[{len_data}] cloud')
        cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
        all_logits = []
        coord, feat, label, idx_points, voxel_idx, reverse_idx_part, reverse_idx  = load_data(data_path, cfg)
        if label is not None:
            label = torch.from_numpy(label.astype(np.int).squeeze()).cuda(non_blocking=True)

        len_part = len(idx_points)
        nearest_neighbor = len_part == 1
        pbar = tqdm(range(len(idx_points)), ascii=True)
        for idx_subcloud in pbar:
            pbar.set_description(f"Test on {cloud_idx}-th cloud [{idx_subcloud}]/[{len_part}]]")
            if not (nearest_neighbor and idx_subcloud>0):
                idx_part = idx_points[idx_subcloud]
                coord_part = coord[idx_part]
                coord_part -= coord_part.min(0)

                feat_part =  feat[idx_part] if feat is not None else None
                data = {'pos': coord_part}
                if feat_part is not None:
                    data['x'] = feat_part
                if pipe_transform is not None:
                    data = pipe_transform(data)
                if 'heights' in cfg.feature_keys and 'heights' not in data.keys():
                    if 'semantickitti' in cfg.dataset.common.NAME.lower():
                        data['heights'] = torch.from_numpy((coord_part[:, gravity_dim:gravity_dim + 1] - coord_part[:, gravity_dim:gravity_dim + 1].min()).astype(np.float32)).unsqueeze(0)
                    else:
                        data['heights'] = torch.from_numpy(coord_part[:, gravity_dim:gravity_dim + 1].astype(np.float32)).unsqueeze(0)
                if not cfg.dataset.common.get('variable', False):
                    if 'x' in data.keys():
                        data['x'] = data['x'].unsqueeze(0)
                    data['pos'] = data['pos'].unsqueeze(0)
                else:
                    data['o'] = torch.IntTensor([len(coord)])
                    data['batch'] = torch.LongTensor([0] * len(coord))

                for key in data.keys():
                    data[key] = data[key].cuda(non_blocking=True)
                data['x'] = get_features_by_keys(data, cfg.feature_keys)
                logits = model(data)
                """visualization in debug mode. !!! visulization is not correct, should remove ignored idx.
                from openpoints.dataset.vis3d import vis_points, vis_multi_points
                vis_multi_points([coord, coord_part], labels=[label.cpu().numpy(), logits.argmax(dim=1).squeeze().cpu().numpy()])
                """

            all_logits.append(logits)
        all_logits = torch.cat(all_logits, dim=0)
        if not cfg.dataset.common.get('variable', False):
            all_logits = all_logits.transpose(1, 2).reshape(-1, cfg.num_classes)

        if not nearest_neighbor:
            # average merge overlapped multi voxels logits to original point set
            idx_points = torch.from_numpy(np.hstack(idx_points)).cuda(non_blocking=True)
            all_logits = scatter(all_logits, idx_points, dim=0, reduce='mean')
        else:
            # interpolate logits by nearest neighbor
            all_logits = all_logits[reverse_idx_part][voxel_idx][reverse_idx]
        pred = all_logits.argmax(dim=1)
        if label is not None:
            cm.update(pred, label)
        """visualization in debug mode
        from openpoints.dataset.vis3d import vis_points, vis_multi_points
        vis_multi_points([coord, coord], labels=[label.cpu().numpy(), all_logits.argmax(dim=1).squeeze().cpu().numpy()])
        """
        if cfg.visualize:
            gt = label.cpu().numpy().squeeze() if label is not None else None
            pred = pred.cpu().numpy().squeeze()
            gt = cfg.cmap[gt, :] if gt is not None else None
            pred = cfg.cmap[pred, :]
            # output pred labels
            if 's3dis' in dataset_name:
                file_name = f'{dataset_name}-Area{cfg.dataset.common.test_area}-{cloud_idx}'
            else:
                file_name = f'{dataset_name}-{cloud_idx}'

            write_obj(coord, feat,
                    #   os.path.join(cfg.vis_dir, f'input-{file_name}.obj'))
                      os.path.join(cfg.vis_dir, f'input-{file_name}.txt'))
            # output ground truth labels
            if gt is not None:
                write_obj(coord, gt,
                        # os.path.join(cfg.vis_dir, f'gt-{file_name}.obj'))
                        os.path.join(cfg.vis_dir, f'gt-{file_name}.txt'))
            # output pred labels
            write_obj(coord, pred,
                    #   os.path.join(cfg.vis_dir, f'pred-{file_name}.obj'))
                      os.path.join(cfg.vis_dir, f'pred-{file_name}.txt'))

        if cfg.get('save_pred', False):
            if 'semantickitti' in cfg.dataset.common.NAME.lower():
                pred = pred + 1
                pred = pred.cpu().numpy().squeeze()
                pred = pred.astype(np.uint32)
                upper_half = pred >> 16  # get upper half for instances
                lower_half = pred & 0xFFFF  # get lower half for semantics (lower_half.shape) (100k+, )
                lower_half = remap_lut_write[lower_half]  # do the remapping of semantics
                pred = (upper_half << 16) + lower_half  # reconstruct full label
                pred = pred.astype(np.uint32)
                frame_id = data_path[0].split('/')[-1][:-4]
                store_path = os.path.join(cfg.save_path, frame_id + '.label')
                pred.tofile(store_path)
            elif 'scannet' in cfg.dataset.common.NAME.lower():
                pred = pred.cpu().numpy().squeeze()
                label_int_mapping={0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 14, 13: 16, 14: 24, 15: 28, 16: 33, 17: 34, 18: 36, 19: 39}
                pred=np.vectorize(label_int_mapping.get)(pred)
                save_file_name=data_path.split('/')[-1].split('_')
                save_file_name=save_file_name[0]+'_'+save_file_name[1]+'.txt'
                save_file_name=os.path.join(cfg.save_path,save_file_name)
                np.savetxt(save_file_name, pred, fmt="%d")

        if label is not None:
            tp, union, count = cm.tp, cm.union, cm.count
            miou, macc, oa, ious, accs = get_mious(tp, union, count)
            with np.printoptions(precision=2, suppress=True):
                logging.info(
                    f'[{cloud_idx}]/[{len_data}] cloud,  test_oa , test_macc, test_miou: {oa:.2f} {macc:.2f} {miou:.2f}, '
                    f'\niou per cls is: {ious}')
            all_cm.value += cm.value

    if 'scannet' in cfg.dataset.common.NAME.lower():
        logging.info(f" Please select and zip all the files (DON'T INCLUDE THE FOLDER) in {cfg.save_path} and submit it to"
                     f" Scannet Benchmark https://kaldir.vc.in.tum.de/scannet_benchmark/. ")

    if label is not None:
        tp, union, count = all_cm.tp, all_cm.union, all_cm.count
        if cfg.distributed:
            dist.all_reduce(tp), dist.all_reduce(union), dist.all_reduce(count)
        miou, macc, oa, ious, accs = get_mious(tp, union, count)
        return miou, macc, oa, ious, accs, all_cm#cm
    else:
        return None, None, None, None, None, None



if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser('场景分割训练/测试')
    parser.add_argument('--cfg', type=str, required=True, help='配置文件')
    parser.add_argument('--profile', action='store_true', default=False, help='设置为True以进行速度分析')
    args, opts = parser.parse_known_args()
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.update(opts)  # 用yml中的默认参数进行更新

    if cfg.seed is None:
        cfg.seed = np.random.randint(1, 10000)

    # 先初始化分布式环境，因为日志记录依赖于分布式信息。
    cfg.rank, cfg.world_size, cfg.distributed, cfg.mp = dist_utils.get_dist_info(cfg)
    cfg.sync_bn = cfg.world_size > 1

    # 初始化日志目录
    cfg.task_name = args.cfg.split('.')[-2].split('/')[-2]  # 任务/数据集名称，例如s3dis，modelnet40_cls
    cfg.cfg_basename = args.cfg.split('.')[-2].split('/')[-1]  # 配置文件名，例如pointnext-xl
    tags = [
        cfg.task_name,  # 任务名称（在./cfgs下的名称文件夹）
        cfg.mode,
        cfg.cfg_basename,  # 配置文件名
        f'ngpus{cfg.world_size}',
        f'seed{cfg.seed}',
    ]
    for i, opt in enumerate(opts):
        if 'rank' not in opt and 'dir' not in opt and 'root' not in opt and 'pretrain' not in opt and 'path' not in opt and 'wandb' not in opt and '/' not in opt:
            tags.append(opt)
    cfg.root_dir = os.path.join(cfg.root_dir, cfg.task_name)
    # 根据不同的模式设置训练/测试相关的配置
    cfg.is_training = cfg.mode not in ['test', 'testing', 'val', 'eval', 'evaluation']
    if cfg.mode in ['resume', 'val', 'test']:
        # 如果是恢复、验证或测试模式，调用相应的函数
        resume_exp_directory(cfg, pretrained_path=cfg.pretrained_path)  # 恢复实验目录
        cfg.wandb.tags = [cfg.mode]  # 设置Wandb标签为当前模式
    else:
        # 如果不是上述模式，生成实验目录
        generate_exp_directory(cfg, tags, additional_id=os.environ.get('MASTER_PORT', None))  # 生成实验目录
        cfg.wandb.tags = tags  # 设置Wandb标签为特定的标签集合
    os.environ["JOB_LOG_DIR"] = cfg.log_dir  # 设置环境变量中的作业日志目录
    cfg_path = os.path.join(cfg.run_dir, "cfg.yaml")  # 配置文件路径

    with open(cfg_path, 'w') as f:
        yaml.dump(cfg, f, indent=2)  # 将配置信息写入YAML文件
        os.system('cp %s %s' % (args.cfg, cfg.run_dir))  # 复制配置文件到运行目录
    cfg.cfg_path = cfg_path  # 设置配置文件路径

    # wandb配置
    cfg.wandb.name = cfg.run_name

    # 多进程处理
    if cfg.mp:
        port = find_free_port()
        cfg.dist_url = f"tcp://localhost:{port}"
        print('使用多进程进行分布式训练')
        mp.spawn(main, nprocs=cfg.world_size, args=(cfg,))
    else:
        main(0, cfg)
