# coding: utf-8

import logging
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP

from caspr.data.load import init_loaders
from caspr.data.common_dataset import CommonDataset
from caspr.models.factory import CASPRFactory
from caspr.models.model_wrapper import AutoencoderTeacherTraining, LSTMAutoencoder, TransformerAutoEncoder
from caspr.utils.early_stopping import DistributedEarlyStopping, EarlyStopping
from caspr.utils.metrics import get_metrics
from caspr.utils.onnx import ONNXWrapper
from caspr.utils.score import get_architecture

from caspr.utils.log import XLogger

DDP_BACKEND = "nccl"
DDP_MASTER_ADDR = "localhost"
DDP_MASTER_PORT = "12355"
DDP_LOAD_WORKERS = 1
STD_LOAD_WORKERS = 0

Log = XLogger("caspr_train_churn.log", level='info')
logger = Log.logger


def run_autoencoder(autoenc, optimizer, dataloader_train, criterion, device):
    count = 0
    epoch_start_time = time.time()
    running_loss = 0.0

    for _, y, seq_cat_data, seq_cont_data, non_seq_cat_data, non_seq_cont_data in dataloader_train:
        y, seq_cat_data, seq_cont_data, non_seq_cat_data, non_seq_cont_data = y.to(device), seq_cat_data.to(
            device), seq_cont_data.to(device), non_seq_cat_data.to(device), non_seq_cont_data.to(device)

        _, loss = autoenc.run(y, seq_cat_data, seq_cont_data, non_seq_cat_data, non_seq_cont_data, criterion)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss = (running_loss * count + loss.item()) / (count + 1)

        count = count + 1

        if count % 64 == 0:
            logger.info(loss, count*seq_cat_data.shape[0])
            print(loss, count*seq_cat_data.shape[0])
            time_so_far = time.time() - epoch_start_time
            logger.info("Time taken since start:" + str(time_so_far))
            print("Time taken since start:" + str(time_so_far))

    epoch_end_time = time.time()
    logger.info(epoch_end_time - epoch_start_time)
    print(epoch_end_time - epoch_start_time)

    return running_loss, epoch_end_time - epoch_start_time


def run_autoencoder_val(autoenc, dataloader_val, criterion, device):
    count = 0
    running_loss = 0.0

    for _, y, seq_cat_data, seq_cont_data, non_seq_cat_data, non_seq_cont_data in dataloader_val:
        y, seq_cat_data, seq_cont_data, non_seq_cat_data, non_seq_cont_data = y.to(device), seq_cat_data.to(
            device), seq_cont_data.to(device), non_seq_cat_data.to(device), non_seq_cont_data.to(device)

        _, loss = autoenc.run(y, seq_cat_data, seq_cont_data, non_seq_cat_data, non_seq_cont_data, criterion)

        running_loss = (running_loss * count + loss.item()) / (count + 1)
        count = count + 1

        if count % 64 == 0:
            logger.info(loss, count*seq_cat_data.shape[0])
            print(loss, count*seq_cat_data.shape[0])

    return running_loss


def run_epoch(model, epoch, dataloader, criterion, device, optimizer=None, is_train=True, get_outputs=False):
    model.to(device)
    losses = []
    y_labels = []
    y_preds = []

    if isinstance(model, DDP):
        model = model.module

    for _, y, seq_cat_x, seq_cont_x, non_seq_cat_x, non_seq_cont_x in dataloader:
        if is_train:
            optimizer.zero_grad()

        seq_cat_x = seq_cat_x.to(device)
        seq_cont_x = seq_cont_x.to(device)
        non_seq_cat_x = non_seq_cat_x.to(device)
        non_seq_cont_x = non_seq_cont_x.to(device)
        y = y.to(device)
        # print('seq_cat_x.shape, seq_cont_x.shape, non_seq_cat_x.shape, non_seq_cont_x.shape, y.shape: ', seq_cat_x.shape, seq_cont_x.shape, non_seq_cat_x.shape, non_seq_cont_x.shape, y.shape)
        key_pad_mask = ~torch.isnan(seq_cont_x[:, :, 0])
        # print('key_pad_mask.shape 1: ', key_pad_mask.shape)
        key_pad_mask = key_pad_mask.reshape(key_pad_mask.shape[0], 1, 1, key_pad_mask.shape[1]).byte()
        # print('key_pad_mask.shape 2: ', key_pad_mask.shape)
        # Forward Pass
        y_pred, loss = model.run(y, seq_cat_x, seq_cont_x, non_seq_cat_x, non_seq_cont_x, criterion=criterion, 
                                 pad_mask=key_pad_mask)
        losses.append(loss.detach().cpu().numpy())

        if get_outputs:
            y_labels.append(y)
            y_preds.append(y_pred)

        # Backward Pass and Optimization
        if is_train:
            loss.backward()
            optimizer.step()

    if get_outputs:
        # print("y_labels.type, y_preds.type", type(y_labels), type(y_preds))
        # print("y_labels.len, y_preds.len", len(y_labels), len(y_preds))
        # print("y_labels.type.0, y_preds.type.0", type(y_labels[0]), type(y_preds[0]))
        # print("y_preds.len.0", len(y_preds[0]))
        # print("y_labels.shape, y_preds.shape", y_labels.shape, y_preds.shape)
        y_labels = torch.cat(y_labels, 0).detach().cpu().numpy()
        y_preds = torch.cat(y_preds, 0).detach().cpu().numpy()

    mean_loss = np.mean(np.asarray(losses))
    mode = 'training' if is_train else 'validation'
    logger.info("Average {} loss in epoch {} is {}".format(mode, epoch, mean_loss))
    print("Average {} loss in epoch {} is {}".format(mode, epoch, mean_loss))
    return y_labels, y_preds, mean_loss


def init_lr_schedulers(optimizer, warmup_epochs, reduce_mode='min', reduce_factor=0.1, reduce_patience=4, verbose=True):
    """
    Training batch size grows proportionally with training distribution, mandating upscaling of the learning rate, which in turn reduces the probability of finding the global optimum.
    This function initializes learning rate schedulers for a given optimizer to facilitate dynamic adjustment (reduction) of learning rate during training.
    """
    
    warm_up = lambda epoch: epoch / warmup_epochs if warmup_epochs > 0 & epoch <= warmup_epochs else 1
    scheduler_wu = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=warm_up)
    scheduler_re = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode=reduce_mode, factor=reduce_factor, patience=reduce_patience, verbose=verbose)

    return scheduler_wu, scheduler_re


def train_model(model, criterion, num_epochs, dataloader_train, dataloader_val, device, save_path, lr=1e-3, fix_module_names=None,
                should_decrease=True, patience=10, verbose=True, evaluate_downstream=False, rank=0, world_size=1, warmup_epochs=5, save_onnx=False):

    if isinstance(model, (LSTMAutoencoder, AutoencoderTeacherTraining, TransformerAutoEncoder)) and evaluate_downstream:
        raise ValueError('evaluate_downstream should be set to False when training autoencoder')

    if fix_module_names:
        fix_modules = [module for name, module in model.named_modules() if name in fix_module_names]
        for module in fix_modules:
            for param in module.parameters():
                param.requires_grad = False
            module.eval()

    # print('type(lr), lr: ', type(lr), lr)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    scheduler_wu, scheduler_re = init_lr_schedulers(optimizer, warmup_epochs, reduce_patience=int(patience/2), verbose=verbose)

    if world_size > 1:
        early_stopping = DistributedEarlyStopping(logger, should_decrease, patience, verbose, rank=rank, save_onnx=save_onnx)
    else:
        early_stopping = EarlyStopping(logger, should_decrease, patience, verbose, save_onnx=save_onnx)

    for epoch in range(num_epochs):
        print('epoch: ', epoch)
        start = time.time()

        model.train()
        if fix_module_names:
            for module in fix_modules:
                module.eval()

        run_epoch(model, epoch, dataloader_train, criterion, device, optimizer)

        model.eval()
        with torch.no_grad():
            y_labels, y_preds, mean_val_loss = run_epoch(model, epoch, dataloader_val, criterion, device,
                                                         is_train=False, get_outputs=evaluate_downstream)
            if evaluate_downstream:
                get_metrics(y_labels, y_preds)

            end = time.time()
            logger.info("Time for epoch {0} is {1}\n".format(epoch, (end - start)))
            logger.info("Mean validation loss for epoch {0} is {1}\n".format(epoch, mean_val_loss))
            print("Time for epoch {0} is {1}\n".format(epoch, (end - start)))
            print("Mean validation loss for epoch {0} is {1}\n".format(epoch, mean_val_loss))

            if epoch <= warmup_epochs:
                scheduler_wu.step()
            scheduler_re.step(mean_val_loss)

            early_stopping(mean_val_loss, model, save_path)
            if early_stopping.early_stop:
                logger.info('early stopping at epoch {}'.format(epoch))
                print('early stopping at epoch {}'.format(epoch))
                break

    if rank == 0:
        if save_onnx:
            model_type = get_architecture(model)
            model = ONNXWrapper(save_path, model_type)
        elif isinstance(model, DDP):
            model.module.load_state_dict(torch.load(save_path))
        else:
            model.load_state_dict(torch.load(save_path))
        return model


def __setup_ddp(rank, world_size):

    os.environ['MASTER_ADDR'] = DDP_MASTER_ADDR
    os.environ['MASTER_PORT'] = DDP_MASTER_PORT

    # initialize the process group
    dist.init_process_group(DDP_BACKEND, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def __do_train_ddp(rank, args):

    __setup_ddp(rank, args['world_size'])

    caspr_factory = args['caspr_factory']

    model = caspr_factory.create(args['caspr_arch'], **args['hyper_params'])

    model = DDP(model.cuda(), device_ids=[rank])

    train_loader, val_loader = init_loaders(args['ds_train'], args['ds_val'], args['batch_size'],
                                            num_workers=DDP_LOAD_WORKERS, world_size=args['world_size'], rank=rank)

    train_model(model, args['criterion'], args['num_epochs'], train_loader, val_loader, rank, args['save_path'],
                lr=args['lr'] * args['world_size'], rank=rank, world_size=args['world_size'], **args['kwargs'])

    dist.destroy_process_group()


def train_model_ddp(caspr_factory : CASPRFactory, caspr_arch : str, hyper_params : dict, ds_train, ds_val, criterion, num_epochs, batch_size, save_path, lr=1e-3, **kwargs):
    """
    Distributed Data Parallel implementation of CASPR training. Will use all GPUs available on the current machine.

    Arguments:
    ----------

    caspr_factory:  CASPR model factory for the specified dataset

    caspr_arch: CASPR architecture e.g. TransformerAutoEncoder

    hyper_params:  parameters for instantiating a new CASPR model with the above method

    ds_train:  CommonDataset for training

    ds_val: CommonDataset for validation

    criterion, num_epochs, batch_size, save_path, lr: self explanatory

    **kwargs: any other parameters to be passed to the train_model function by the DDP worker (e.g. evaluate, verbose or patience)

    Returns: Trained model

    """
    logger.info("Setting up model training using torch DDP")
    print("Setting up model training using torch DDP")

    for arg in [caspr_factory, caspr_arch, ds_train, ds_val, criterion, num_epochs, batch_size, save_path, lr]:
        if not arg:
            raise ValueError("Illegal null argument. Check for None values and try again.")

    world_size = torch.cuda.device_count()

    if not torch.cuda.is_available() or world_size < 2:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.warn("DDP mode disabled. Training on %s..." % device)
        model = caspr_factory.create(caspr_arch, device=device, **hyper_params)
        
        init_model_encoder_param_file = f"/home/{jhub_user}/shared/MI_ZHOU/transformer/CASPR/raw_model/caspr_transformer_nopadmask_test_r0_encoder_params"
        model.unified_encoder.load_state_dict(torch.load(init_model_encoder_param_file))
        logger.info('init transformer encoder with param: %s' % init_model_encoder_param_file)
        
        # init_model_param_file = f"/home/{jhub_user}/shared/MI_ZHOU/transformer/CASPR/raw_model/caspr_transformer_paper_v2_1m_unmask_churn"
        # model.load_state_dict(torch.load(init_model_param_file))
        
        train_loader, val_loader = init_loaders(ds_train, ds_val, batch_size, num_workers=STD_LOAD_WORKERS)
        return train_model(model, criterion, num_epochs, train_loader, val_loader, device, save_path, lr, **kwargs)

    logger.info("DDP mode enabled, will train on %d GPUs" % world_size)
    print("DDP mode enabled, will train on %d GPUs" % world_size)

    arguments = locals()

    mp.spawn(__do_train_ddp,
             args=(arguments,),
             nprocs=world_size,
             join=True)

    model = caspr_factory.create(caspr_arch, **hyper_params)
    model.load_state_dict(torch.load(save_path))
    return model


def test_model(model, dataloader_test, criterion, device):
    model.eval()
    with torch.no_grad():
        y_labels, y_preds, _ = run_epoch(
            model, 0, dataloader_test, criterion, device, is_train=False, get_outputs=True)
    return y_labels, y_preds


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    import os
    env_dic = os.environ
    jhub_user = env_dic.get('JUPYTERHUB_USER')
    
    # categorical columns
    # cat_cols_ = ['city', 'gender', 'registered_via']
    # num_activities: key, categorical columns in seq_cols_ + non_seq_cols_; value: unique categorical num
    # embedding层需检查张量内部具体值的大小，并确保它们的值在有效范围内[0, num_embeddings-1]
    num_activities = {
        'city': 22,
        'gender': 3,
        'registered_via': 6
    }
    seq_cols_ = ['is_auto_renew', 'is_cancel_cnt', 'active_days',
           'num_25_sum', 'num_25_stddev', 'num_25_avg', 'num_25_max', 'num_25_min',
           'num_50_sum', 'num_50_stddev', 'num_50_avg', 'num_50_max', 'num_50_min',
           'num_75_sum', 'num_75_stddev', 'num_75_avg', 'num_75_max', 'num_75_min',
           'num_985_sum', 'num_985_stddev', 'num_985_avg', 'num_985_max',
           'num_985_min', 'num_100_sum', 'num_100_stddev', 'num_100_avg',
           'num_100_max', 'num_100_min', 'num_unq_sum', 'num_unq_stddev',
           'num_unq_avg', 'num_unq_max', 'num_unq_min', 'gap_days']
    non_seq_cols_ = ['city', 'bd', 'gender', 'registered_via']
    output_col = 'is_churn'
    cat_cols_ = ['city', 'gender', 'registered_via']
    cont_cols_ = ['is_auto_renew', 'is_cancel_cnt', 'active_days',
           'num_25_sum', 'num_25_stddev', 'num_25_avg', 'num_25_max', 'num_25_min',
           'num_50_sum', 'num_50_stddev', 'num_50_avg', 'num_50_max', 'num_50_min',
           'num_75_sum', 'num_75_stddev', 'num_75_avg', 'num_75_max', 'num_75_min',
           'num_985_sum', 'num_985_stddev', 'num_985_avg', 'num_985_max',
           'num_985_min', 'num_100_sum', 'num_100_stddev', 'num_100_avg',
           'num_100_max', 'num_100_min', 'num_unq_sum', 'num_unq_stddev',
           'num_unq_avg', 'num_unq_max', 'num_unq_min', 'bd', 'gap_days']
    date_cols = []
    output_col = "is_churn"
    
    caspr_factory = CASPRFactory(cat_cols_, num_activities, cont_cols_, seq_cols_, non_seq_cols_, date_cols)

    from caspr.data.load import init_datasets
    caspr_arch = "TransformerChurnModel"
    hyper_params = dict()
    
    seq_len = 15
    df_train = pd.read_csv(f"/home/{jhub_user}/shared/MI_ZHOU/transformer/data/df_train_v2_fix_201610_201702_nomask.csv")
    df_val = pd.read_csv(f"/home/{jhub_user}/shared/MI_ZHOU/transformer/data/df_val_v2_fix_201610_201702_nomask.csv")
    
    ds_val = CommonDataset(df_val, seq_cols_, non_seq_cols_, output_col, cat_cols_, cont_cols_, seq_len)
    ds_train = CommonDataset(df_train, seq_cols_, non_seq_cols_, output_col, cat_cols_, cont_cols_, seq_len)
    
    criterion = nn.CrossEntropyLoss()
    num_epochs = 100
    batch_size = 10240
    save_path = "./raw_model/Churn_freeze_encoder_param_r0"
    
    init_model_encoder_param_file = f"/home/{jhub_user}/shared/MI_ZHOU/transformer/CASPR/raw_model/caspr_transformer_nopadmask_test_r0_encoder_params"
    
    logger.info('train model %s with init transformer encoder with param: %s' % (save_path, init_model_encoder_param_file))
    
    fix_module_names = [
        'unified_encoder', 
        'unified_encoder.emb_non_seq',
        'unified_encoder.emb_non_seq.emb_layers',
        'unified_encoder.emb_non_seq.emb_layers.0',
        'unified_encoder.emb_non_seq.emb_layers.1',
        'unified_encoder.emb_non_seq.emb_layers.2',
        'unified_encoder.emb_non_seq.emb_dropout_layer',
        'unified_encoder.emb_seq',
        'unified_encoder.emb_seq.emb_layers',
        'unified_encoder.emb_seq.emb_dropout_layer',
        'unified_encoder.linear_seq',
        'unified_encoder.linear_non_seq',
        'unified_encoder.transformer_encoder',
        'unified_encoder.transformer_encoder.pos_embedding',
        'unified_encoder.transformer_encoder.layers',
        'unified_encoder.transformer_encoder.layers.0',
        'unified_encoder.transformer_encoder.layers.0.self_attn_layer_norm',
        'unified_encoder.transformer_encoder.layers.0.ff_layer_norm',
        'unified_encoder.transformer_encoder.layers.0.self_attention',
        'unified_encoder.transformer_encoder.layers.0.self_attention.fc_q',
        'unified_encoder.transformer_encoder.layers.0.self_attention.fc_k',
        'unified_encoder.transformer_encoder.layers.0.self_attention.fc_v',
        'unified_encoder.transformer_encoder.layers.0.self_attention.fc_o',
        'unified_encoder.transformer_encoder.layers.0.self_attention.dropout',
        'unified_encoder.transformer_encoder.layers.0.positionwise_feedforward',
        'unified_encoder.transformer_encoder.layers.0.positionwise_feedforward.fc_1',
        'unified_encoder.transformer_encoder.layers.0.positionwise_feedforward.fc_2',
        'unified_encoder.transformer_encoder.layers.0.positionwise_feedforward.dropout',
        'unified_encoder.transformer_encoder.layers.0.dropout',
        'unified_encoder.transformer_encoder.layers.1',
        'unified_encoder.transformer_encoder.layers.1.self_attn_layer_norm',
        'unified_encoder.transformer_encoder.layers.1.ff_layer_norm',
        'unified_encoder.transformer_encoder.layers.1.self_attention',
        'unified_encoder.transformer_encoder.layers.1.self_attention.fc_q',
        'unified_encoder.transformer_encoder.layers.1.self_attention.fc_k',
        'unified_encoder.transformer_encoder.layers.1.self_attention.fc_v',
        'unified_encoder.transformer_encoder.layers.1.self_attention.fc_o',
        'unified_encoder.transformer_encoder.layers.1.self_attention.dropout',
        'unified_encoder.transformer_encoder.layers.1.positionwise_feedforward',
        'unified_encoder.transformer_encoder.layers.1.positionwise_feedforward.fc_1',
        'unified_encoder.transformer_encoder.layers.1.positionwise_feedforward.fc_2',
        'unified_encoder.transformer_encoder.layers.1.positionwise_feedforward.dropout',
        'unified_encoder.transformer_encoder.layers.1.dropout',
        'unified_encoder.transformer_encoder.layers.2',
        'unified_encoder.transformer_encoder.layers.2.self_attn_layer_norm',
        'unified_encoder.transformer_encoder.layers.2.ff_layer_norm',
        'unified_encoder.transformer_encoder.layers.2.self_attention',
        'unified_encoder.transformer_encoder.layers.2.self_attention.fc_q',
        'unified_encoder.transformer_encoder.layers.2.self_attention.fc_k',
        'unified_encoder.transformer_encoder.layers.2.self_attention.fc_v',
        'unified_encoder.transformer_encoder.layers.2.self_attention.fc_o',
        'unified_encoder.transformer_encoder.layers.2.self_attention.dropout',
        'unified_encoder.transformer_encoder.layers.2.positionwise_feedforward',
        'unified_encoder.transformer_encoder.layers.2.positionwise_feedforward.fc_1',
        'unified_encoder.transformer_encoder.layers.2.positionwise_feedforward.fc_2',
        'unified_encoder.transformer_encoder.layers.2.positionwise_feedforward.dropout',
        'unified_encoder.transformer_encoder.layers.2.dropout',
        'unified_encoder.transformer_encoder.layers.3',
        'unified_encoder.transformer_encoder.layers.3.self_attn_layer_norm',
        'unified_encoder.transformer_encoder.layers.3.ff_layer_norm',
        'unified_encoder.transformer_encoder.layers.3.self_attention',
        'unified_encoder.transformer_encoder.layers.3.self_attention.fc_q',
        'unified_encoder.transformer_encoder.layers.3.self_attention.fc_k',
        'unified_encoder.transformer_encoder.layers.3.self_attention.fc_v',
        'unified_encoder.transformer_encoder.layers.3.self_attention.fc_o',
        'unified_encoder.transformer_encoder.layers.3.self_attention.dropout',
        'unified_encoder.transformer_encoder.layers.3.positionwise_feedforward',
        'unified_encoder.transformer_encoder.layers.3.positionwise_feedforward.fc_1',
        'unified_encoder.transformer_encoder.layers.3.positionwise_feedforward.fc_2',
        'unified_encoder.transformer_encoder.layers.3.positionwise_feedforward.dropout',
        'unified_encoder.transformer_encoder.layers.3.dropout',
        'unified_encoder.transformer_encoder.layers.4',
        'unified_encoder.transformer_encoder.layers.4.self_attn_layer_norm',
        'unified_encoder.transformer_encoder.layers.4.ff_layer_norm',
        'unified_encoder.transformer_encoder.layers.4.self_attention',
        'unified_encoder.transformer_encoder.layers.4.self_attention.fc_q',
        'unified_encoder.transformer_encoder.layers.4.self_attention.fc_k',
        'unified_encoder.transformer_encoder.layers.4.self_attention.fc_v',
        'unified_encoder.transformer_encoder.layers.4.self_attention.fc_o',
        'unified_encoder.transformer_encoder.layers.4.self_attention.dropout',
        'unified_encoder.transformer_encoder.layers.4.positionwise_feedforward',
        'unified_encoder.transformer_encoder.layers.4.positionwise_feedforward.fc_1',
        'unified_encoder.transformer_encoder.layers.4.positionwise_feedforward.fc_2',
        'unified_encoder.transformer_encoder.layers.4.positionwise_feedforward.dropout',
        'unified_encoder.transformer_encoder.layers.4.dropout',
        'unified_encoder.transformer_encoder.layers.5',
        'unified_encoder.transformer_encoder.layers.5.self_attn_layer_norm',
        'unified_encoder.transformer_encoder.layers.5.ff_layer_norm',
        'unified_encoder.transformer_encoder.layers.5.self_attention',
        'unified_encoder.transformer_encoder.layers.5.self_attention.fc_q',
        'unified_encoder.transformer_encoder.layers.5.self_attention.fc_k',
        'unified_encoder.transformer_encoder.layers.5.self_attention.fc_v',
        'unified_encoder.transformer_encoder.layers.5.self_attention.fc_o',
        'unified_encoder.transformer_encoder.layers.5.self_attention.dropout',
        'unified_encoder.transformer_encoder.layers.5.positionwise_feedforward',
        'unified_encoder.transformer_encoder.layers.5.positionwise_feedforward.fc_1',
        'unified_encoder.transformer_encoder.layers.5.positionwise_feedforward.fc_2',
        'unified_encoder.transformer_encoder.layers.5.positionwise_feedforward.dropout',
        'unified_encoder.transformer_encoder.layers.5.dropout',
        'unified_encoder.transformer_encoder.dropout'
       ]

    train_model_ddp(caspr_factory, caspr_arch, hyper_params, ds_train, ds_val, criterion, num_epochs, batch_size,
                    save_path, fix_module_names=fix_module_names)