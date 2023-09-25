"""
This file includes the main libraries in the network training module.
"""

import json
import os
import signal
import sys
import time
from functools import partial
from os import path as osp

import numpy as np
import torch
#from dataloader.dataset_fb import FbSequenceDataset
from dataloader.tlio_data import TlioData
from network.losses import get_loss, loss_class
from network.model_factory import get_model
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.logging import logging
from utils.utils import to_device

from importlib import import_module
import json
from argparse import Namespace
from torch import nn
from network.parallel import DataParallel, DataParallelModel, DataParallelCriterion
import vgtk.pc as pctk
import vgtk.point3d as p3dtk

def torch_to_numpy(torch_arr):
    return torch_arr.cpu().detach().numpy()


def get_inference(network, data_loader, device, epoch, transforms=[]):
    """
    Obtain attributes from a data loader given a network state
    Outputs all targets, predicts, predicted covariance params, and losses in numpy arrays
    Enumerates the whole data loader
    """
    targets_all, preds_all, preds_cov_all, losses_all = [], [], [], []
    network.eval()

    for bid, sample in enumerate(data_loader):
        sample = to_device(sample, device)
        for transform in transforms:
            sample = transform(sample)
        feat = sample["feats"]["imu0"]
        pred, pred_cov = network(feat)
        
        if len(pred.shape) == 2:
            targ = sample["targ_dt_World"][:,-1,:]
        else:
            # Leave off zeroth element since it's 0's. Ex: Net predicts 199 if there's 200 GT
            targ = sample["targ_dt_World"][:,1:,:].permute(0,2,1)

        loss = get_loss(pred, pred_cov, targ, epoch)

        targets_all.append(torch_to_numpy(targ))
        preds_all.append(torch_to_numpy(pred))
        preds_cov_all.append(torch_to_numpy(pred_cov))
        losses_all.append(torch_to_numpy(loss))

    targets_all = np.concatenate(targets_all, axis=0)
    preds_all = np.concatenate(preds_all, axis=0)
    preds_cov_all = np.concatenate(preds_cov_all, axis=0)
    losses_all = np.concatenate(losses_all, axis=0)
    attr_dict = {
        "targets": targets_all,
        "preds": preds_all,
        "preds_cov": preds_cov_all,
        "losses": losses_all,
    }
    return attr_dict


def do_train(network, train_loader, device, epoch, optimizer, transforms=[]):
    """
    Train network for one epoch using a specified data loader
    Outputs all targets, predicts, predicted covariance params, and losses in numpy arrays
    """
    train_targets, train_preds, train_preds_cov, train_losses = [], [], [], []
    network.train()
    print(network)
    #for bid, (feat, targ, _, _) in enumerate(train_loader):
    for bid, sample in enumerate(train_loader):
        sample = to_device(sample, device)
        for transform in transforms:
            sample = transform(sample)
        feat = sample["feats"]["imu0"]
        optimizer.zero_grad()
        
        ### >>> check input feature shape ###
        # print("feature shape : ", feat.shape)   # shape => torch.Size([1024, 6, 200]) (batch : 1024, pos : 6, sliding window : 200)
        ### <<< check input feature shape ###
        
        ### >>> Data Parallel ###
        # network = DataParallelModel(network, device_ids=[0, 1])
        # get_loss = DataParallelCriterion(loss_class)
        ### <<< Data Parallel ###
        
        pred, pred_cov = network(feat)
        # print('pred : ', pred, pred_cov, len(pred), len(pred_cov))
        # pred = torch.cat(pred, dim=0)
        # pred_cov = torch.cat(pred_cov, dim=0)

        if len(pred.shape) == 2:
            targ = sample["targ_dt_World"][:,-1,:]
        else:
            # Leave off zeroth element since it's 0's. Ex: Net predicts 199 if there's 200 GT
            targ = sample["targ_dt_World"][:,1:,:].permute(0,2,1)
        
        print('size check : ', pred.shape, pred_cov.shape, targ.shape, epoch)   # torch.Size([1024, 3]) torch.Size([1024, 3]) torch.Size([1024, 3])
        
        loss = get_loss(pred, pred_cov, targ, epoch)

        train_targets.append(torch_to_numpy(targ))
        train_preds.append(torch_to_numpy(pred))
        train_preds_cov.append(torch_to_numpy(pred_cov))
        train_losses.append(torch_to_numpy(loss))
            
        #print("Loss full: ", loss)

        loss = loss.mean()
        loss.backward()

        #print("Loss mean: ", loss.item())
        
        #print("Gradients:")
        #for name, param in network.named_parameters():
        #    if param.requires_grad:
        #        print(name, ": ", param.grad)

        torch.nn.utils.clip_grad_norm_(network.parameters(), 0.1, error_if_nonfinite=True)
        optimizer.step()

    train_targets = np.concatenate(train_targets, axis=0)
    train_preds = np.concatenate(train_preds, axis=0)
    train_preds_cov = np.concatenate(train_preds_cov, axis=0)
    train_losses = np.concatenate(train_losses, axis=0)
    train_attr_dict = {
        "targets": train_targets,
        "preds": train_preds,
        "preds_cov": train_preds_cov,
        "losses": train_losses,
    }
    return train_attr_dict

def do_train_e2pn(network, train_loader, device, epoch, optimizer, transforms=[]):
    """
    Train network for one epoch using a specified data loader
    Outputs all targets, predicts, predicted covariance params, and losses in numpy arrays
    """
    
    # print(network)
    
    train_targets, train_preds, train_preds_cov, train_losses = [], [], [], []
    class LinearReshape(nn.Module):
        def __init__(self,input_dim, output_dim, reshape_shape):
            super(LinearReshape, self).__init__()
            self.linear = nn.Linear(input_dim, output_dim)
            self.reshape_shape = reshape_shape
            
        def forward(self, x):
            # print('x.shape : ', x.shape)
            x = self.linear(x)
            x = x.transpose(0,1)
            x = x.view(-1,20,3)
            return x
    
    input_dim = 204800
    output_dim = 8 * 2 * 20
    reshape_shape = (16, 20, 3)
    
    # print('feat dim : ', feat.shape, 'feat_tmp3 : ', feat_tmp3.shape)
    
    linear_reshape_module = LinearReshape(input_dim, output_dim, reshape_shape).to(device)
    reshape_network = nn.Sequential(linear_reshape_module, network)
    
    # reshape_network = DataParallel(reshape_network)
    reshape_network = reshape_network.cuda()

    reshape_network.train()
        
    #for bid, (feat, targ, _, _) in enumerate(train_loader):
    for bid, sample in enumerate(train_loader):
        sample = to_device(sample, device)
        for transform in transforms:
            sample = transform(sample)
        feat = sample["feats"]["imu0"]
        optimizer.zero_grad()
        
        ### >>> check input feature shape ###
        # print("feature shape : ", feat.shape)   # shape => torch.Size([1024, 6, 200]) (batch : 1024, pos : 6, sliding window : 200) / TLIO
        ### <<< check input feature shape ###
        
        #  (1024, 6, 200) -> (8, 2, 1024, 3)
        
        feat_permute = feat.cpu().transpose(1,2)   # 1024,6,200 -> 1024,200,6
        feat_tmp = feat_permute.reshape(-1, 6)  # 1024,200,6 -> 1024 * 200 , 6
        feat_tmp2 = feat_tmp.view(-1, 3, 2).sum(dim=2)  # 1024,200,6 -> 1024 * 200 * 2, 3
        feat_tmp3 = to_device(feat_tmp2.transpose(0,1),device)  #  204800 , 3 -> 3 , 204800
        
        # feat_tmp2 = to_device(feat_tmp2, device)
        # print(feat_permute.device.type, feat_tmp.device.type, feat_tmp2.device.type)
        check_feat = feat_tmp3.cpu().numpy()    
        
        
        # reshape_network = nn.DataParallel(reshape_network, device_ids = [0,1])
        
        pred, pred_cov = reshape_network(feat_tmp3)
        # print('after outblock shape : ', pred_cov.shape)
        if len(pred.shape) == 2:
            targ = sample["targ_dt_World"][:,-1,:]
        else:
            # Leave off zeroth element since it's 0's. Ex: Net predicts 199 if there's 200 GT
            targ = sample["targ_dt_World"][:,1:,:].permute(0,2,1)
        # print('targ size : ', targ.shape)

        print('size check : ', pred.shape, pred_cov.shape, targ.shape, epoch)
        loss = get_loss(pred, pred_cov, targ, epoch)
        # print('loss executed')

        train_targets.append(torch_to_numpy(targ))
        train_preds.append(torch_to_numpy(pred))
        train_preds_cov.append(torch_to_numpy(pred_cov))
        train_losses.append(torch_to_numpy(loss))
            
        #print("Loss full: ", loss)

        loss = loss.mean()
        loss.backward()

        #print("Loss mean: ", loss.item())
        
        #print("Gradients:")
        #for name, param in network.named_parameters():
        #    if param.requires_grad:
        #        print(name, ": ", param.grad)

        torch.nn.utils.clip_grad_norm_(network.parameters(), 0.1, error_if_nonfinite=True)
        optimizer.step()

    train_targets = np.concatenate(train_targets, axis=0)
    train_preds = np.concatenate(train_preds, axis=0)
    train_preds_cov = np.concatenate(train_preds_cov, axis=0)
    train_losses = np.concatenate(train_losses, axis=0)
    train_attr_dict = {
        "targets": train_targets,
        "preds": train_preds,
        "preds_cov": train_preds_cov,
        "losses": train_losses,
    }
    return train_attr_dict

def do_train_imu_e2pn(network, train_loader, device, epoch, optimizer, transforms=[]):
    """
    Train network for one epoch using a specified data loader
    Outputs all targets, predicts, predicted covariance params, and losses in numpy arrays
    """
    train_targets, train_preds, train_preds_cov, train_losses = [], [], [], []
    network.train()

    #for bid, (feat, targ, _, _) in enumerate(train_loader):
    for bid, sample in enumerate(train_loader):
        sample = to_device(sample, device)
        for transform in transforms:
            sample = transform(sample)
        feat = sample["feats"]["imu0"]
        
                
        optimizer.zero_grad()
        
        ### >>> check input feature shape ###
        # print("feature shape : ", feat.shape)   # shape => torch.Size([1024, 6, 200]) (batch : 1024, pos : 6, sliding window : 200)
        ### <<< check input feature shape ###
        
        #stack rotated imu
        # _, pc = pctk.uniform_resample_np(data['pc'], self.opt.model.input_num)  # don't need to resample pc in imu case
            # normalization 
        pc = feat.cpu().numpy() # 1024,6,200
        pc_tgt = pc.transpose(0,2,1) # 1024,200,6
        pc_tgt = pc_tgt[:,:,:3]+pc_tgt[:,:,3:] #add acc. and ang-vel.
        pc_tgt = torch.from_numpy(pc_tgt).to(torch.device('cuda'))
        
        pc = p3dtk.normalize_np(pc, batch=True)
        pc = pc.transpose(0,2,1)   # 1024,200,6
        pc = torch.from_numpy(pc).to('cuda')
        pc_src, _ = pctk.batch_rotate_point_cloud(pc)  

        # pc_tensor = np.stack([pc_src, pc_tgt], axis=1)  #pc_tensor shape : (1024, 2, 200, 6)
        # pc_tensor = torch.from_numpy(pc_tensor)
        # print("pc_tensor shape : ", pc_tensor.shape)   # shape => torch.Size([1024, 6, 200]) (batch : 1024, pos : 6, sliding window : 200)
        # print(network)
        
        pc_ori, _ = pctk.batch_rotate_point_cloud(pc) 
        
        # pc = torch.from_numpy(pc)
        # pc_ori = torch.from_numpy(pc_ori)
        
        # feat,feat_cov = network(pc)
        # feat_ori, feat_cov_ori = network(pc_ori)
        # print("cos sim of feat : ", cos_sim(feat,feat_ori))
        # print("cos sim of feat_cov : ", cos_sim(feat_cov,feat_cov_ori))
        # assert False

        pred, pred_cov = network(pc_tgt)
        # pred, pred_cov = network(feat)
        
        # print('feat info : ', feat.shape, pc_tensor.shape, type(feat), type(pc_tensor))
        # print('pred : ', pred, pred_cov, len(pred), len(pred_cov))
        # pred = torch.cat(pred, dim=0)
        # pred_cov = torch.cat(pred_cov, dim=0)

        if len(pred.shape) == 2:
            # print("running!")
            targ = sample["targ_dt_World"][:,-1,:]
        else:
            # Leave off zeroth element since it's 0's. Ex: Net predicts 199 if there's 200 GT
            targ = sample["targ_dt_World"][:,1:,:].permute(0,2,1)
        
        # print("sample displacment size check : ",sample["targ_dt_World"][:,1:,:].shape)
        # print('size check : ', pred.shape, pred_cov.shape, targ.shape, epoch) # torch.Size([1024, 64]) torch.Size([1024, 12]) torch.Size([1024, 3])
        
        loss = get_loss(pred, pred_cov, targ, epoch)

        train_targets.append(torch_to_numpy(targ))
        train_preds.append(torch_to_numpy(pred))
        train_preds_cov.append(torch_to_numpy(pred_cov))
        train_losses.append(torch_to_numpy(loss))
            
        #print("Loss full: ", loss)

        loss = loss.mean()
        loss.backward()

        #print("Loss mean: ", loss.item())
        
        #print("Gradients:")
        #for name, param in network.named_parameters():
        #    if param.requires_grad:
        #        print(name, ": ", param.grad)

        torch.nn.utils.clip_grad_norm_(network.parameters(), 0.1, error_if_nonfinite=True)
        optimizer.step()

    train_targets = np.concatenate(train_targets, axis=0)
    train_preds = np.concatenate(train_preds, axis=0)
    train_preds_cov = np.concatenate(train_preds_cov, axis=0)
    train_losses = np.concatenate(train_losses, axis=0)
    train_attr_dict = {
        "targets": train_targets,
        "preds": train_preds,
        "preds_cov": train_preds_cov,
        "losses": train_losses,
    }
    return train_attr_dict

def do_train_e2pn2(network, train_loader, device, epoch, optimizer, transforms=[]):
    """
    Train network for one epoch using a specified data loader
    Outputs all targets, predicts, predicted covariance params, and losses in numpy arrays
    """
    
    # print(network)
    
    train_targets, train_preds, train_preds_cov, train_losses = [], [], [], []
    network.train()

    #for bid, (feat, targ, _, _) in enumerate(train_loader):
    for bid, sample in enumerate(train_loader):
        sample = to_device(sample, device)
        for transform in transforms:
            sample = transform(sample)
        feat = sample["feats"]["imu0"]
        optimizer.zero_grad()
        
        ### >>> check input feature shape ###
        # print("feature shape : ", feat.shape)   # shape => torch.Size([1024, 6, 200]) (batch : 1024, pos : 6, sliding window : 200) / TLIO
        ### <<< check input feature shape ###
        
        #  (1024, 6, 200) -> (8, 2, 1024, 3)
        
        feat_permute = feat.cpu().transpose(0,1)   # 1024,6,200 -> 6,1024,200

        # feat_tmp2 = to_device(feat_tmp2, device)
        # print(feat_permute.device.type, feat_tmp.device.type, feat_tmp2.device.type)
        check_feat = feat_tmp3.cpu().numpy()    
        class conv1dreshape(nn.Module):
            def __init__(self,input_dim, output_dim, reshape_shape):
                super(conv1dreshape, self).__init__()
                self.conv = nn.Conv1d(input_dim, output_dim, 3, 1, 2)
                
            def forward(self, x):
                # print('x.shape : ', x.shape)
                x = self.conv(x)    # 6,1024,200 -> 6,8,200
                x = x.view(3,2,8,200)  #6,8,200 -> 3,2,8,200
                x = x.sum(dim,1)   #3,2,8,200 -> 3,8,200
                x = x.sum(dim,2) #3,8
                x = x.unsqueeze(2).repeat(1,1,20) #3,8,1024
                x = x.transpose(0,1) #8,3,1024
                x = x.transpose(2,1) #8,1024,3
                x = x.unsqueeze(1).repeat(1,2,1,1) #8,2,1024,3
                print('x shape should be 8,2,20,3',x.shape)
                return x
        
        input_dim = 20
        output_dim = 6
        
        # print('feat dim : ', feat.shape, 'feat_tmp3 : ', feat_tmp3.shape)
        
        linear_reshape_module = conv1dreshape(input_dim, output_dim, reshape_shape).to(device)
        reshape_network = nn.Sequential(linear_reshape_module, network)
        
        # reshape_network = DataParallel(reshape_network)
        reshape_network = reshape_network.cuda()
        
        # reshape_network = nn.DataParallel(reshape_network, device_ids = [0,1])
        
        pred, pred_cov = reshape_network(feat_tmp3)
        # print('after outblock shape : ', pred_cov.shape)
        if len(pred.shape) == 2:
            targ = sample["targ_dt_World"][:,-1,:]
        else:
            # Leave off zeroth element since it's 0's. Ex: Net predicts 199 if there's 200 GT
            targ = sample["targ_dt_World"][:,1:,:].permute(0,2,1)
        # print('targ size : ', targ.shape)

        loss = get_loss(pred, pred_cov, targ, epoch)
        # print('loss executed')

        train_targets.append(torch_to_numpy(targ))
        train_preds.append(torch_to_numpy(pred))
        train_preds_cov.append(torch_to_numpy(pred_cov))
        train_losses.append(torch_to_numpy(loss))
            
        #print("Loss full: ", loss)

        loss = loss.mean()
        loss.backward()

        #print("Loss mean: ", loss.item())
        
        #print("Gradients:")
        #for name, param in network.named_parameters():
        #    if param.requires_grad:
        #        print(name, ": ", param.grad)

        torch.nn.utils.clip_grad_norm_(network.parameters(), 0.1, error_if_nonfinite=True)
        optimizer.step()

    train_targets = np.concatenate(train_targets, axis=0)
    train_preds = np.concatenate(train_preds, axis=0)
    train_preds_cov = np.concatenate(train_preds_cov, axis=0)
    train_losses = np.concatenate(train_losses, axis=0)
    train_attr_dict = {
        "targets": train_targets,
        "preds": train_preds,
        "preds_cov": train_preds_cov,
        "losses": train_losses,
    }
    return train_attr_dict


def write_summary(summary_writer, attr_dict, epoch, optimizer, mode):
    """ Given the attr_dict write summary and log the losses """

    mse_loss = np.mean((attr_dict["targets"] - attr_dict["preds"]) ** 2, axis=0)
    ml_loss = np.average(attr_dict["losses"])
    sigmas = np.exp(attr_dict["preds_cov"])
    # If it's sequential, take the last one
    if len(mse_loss.shape) == 2:
        assert mse_loss.shape[0] == 3
        mse_loss = mse_loss[:, -1]
        assert sigmas.shape[1] == 3
        sigmas = sigmas[:,:,-1]
    summary_writer.add_scalar(f"{mode}_loss/loss_x", mse_loss[0], epoch)
    summary_writer.add_scalar(f"{mode}_loss/loss_y", mse_loss[1], epoch)
    summary_writer.add_scalar(f"{mode}_loss/loss_z", mse_loss[2], epoch)
    summary_writer.add_scalar(f"{mode}_loss/avg", np.mean(mse_loss), epoch)
    summary_writer.add_scalar(f"{mode}_dist/loss_full", ml_loss, epoch)
    summary_writer.add_histogram(f"{mode}_hist/sigma_x", sigmas[:, 0], epoch)
    summary_writer.add_histogram(f"{mode}_hist/sigma_y", sigmas[:, 1], epoch)
    summary_writer.add_histogram(f"{mode}_hist/sigma_z", sigmas[:, 2], epoch)
    if epoch > 0:
        summary_writer.add_scalar(
            "optimizer/lr", optimizer.param_groups[0]["lr"], epoch - 1
        )
    logging.info(
        f"{mode}: average ml loss: {ml_loss}, average mse loss: {mse_loss}/{np.mean(mse_loss)}"
    )


def save_model(args, epoch, network, optimizer, best, interrupt=False):
    if interrupt:
        model_path = osp.join(args.out_dir, "checkpoints", "checkpoint_latest.pt")
    if best:
        model_path = osp.join(args.out_dir, "checkpoint_best.pt")        
    else:
        model_path = osp.join(args.out_dir, "checkpoints", "checkpoint_%d.pt" % epoch)
    state_dict = {
        "model_state_dict": network.state_dict(),
        "epoch": epoch,
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
    }
    torch.save(state_dict, model_path)
    logging.info(f"Model saved to {model_path}")


def arg_conversion(args):
    """ Conversions from time arguments to data size """

    if not (args.past_time * args.imu_freq).is_integer():
        raise ValueError(
            "past_time cannot be represented by integer number of IMU data."
        )
    if not (args.window_time * args.imu_freq).is_integer():
        raise ValueError(
            "window_time cannot be represented by integer number of IMU data."
        )
    if not (args.future_time * args.imu_freq).is_integer():
        raise ValueError(
            "future_time cannot be represented by integer number of IMU data."
        )
    if not (args.imu_freq / args.sample_freq).is_integer():
        raise ValueError("sample_freq must be divisible by imu_freq.")

    data_window_config = dict(
        [
            ("past_data_size", int(args.past_time * args.imu_freq)),
            ("window_size", int(args.window_time * args.imu_freq)),
            ("future_data_size", int(args.future_time * args.imu_freq)),
            ("step_size", int(args.imu_freq / args.sample_freq)),
        ]
    )
    net_config = {
        "in_dim": (
            data_window_config["past_data_size"]
            + data_window_config["window_size"]
            + data_window_config["future_data_size"]
        )
        // 32
        + 1
    }

    return data_window_config, net_config


def net_train(args):
    """
    Main function for network training
    """

    try:
        if args.root_dir is None:
            raise ValueError("root_dir must be specified.")
        #if args.train_list is None:
        #    raise ValueError("train_list must be specified.")
        if args.out_dir is not None:
            if not osp.isdir(args.out_dir):
                os.makedirs(args.out_dir)
            if not osp.isdir(osp.join(args.out_dir, "checkpoints")):
                os.makedirs(osp.join(args.out_dir, "checkpoints"))
            if not osp.isdir(osp.join(args.out_dir, "logs")):
                os.makedirs(osp.join(args.out_dir, "logs"))
            with open(
                os.path.join(args.out_dir, "parameters.json"), "w"
            ) as parameters_file:
                parameters_file.write(json.dumps(vars(args), sort_keys=True, indent=4))
            logging.info(f"Training output writes to {args.out_dir}")
        else:
            raise ValueError("out_dir must be specified.")
        #if args.val_list is None:
        #    logging.warning("val_list is not specified.")
        if args.continue_from is not None:
            if osp.exists(args.continue_from):
                logging.info(
                    f"Continue training from existing model {args.continue_from}"
                )
            else:
                raise ValueError(
                    f"continue_from model file path {args.continue_from} does not exist"
                )
        data_window_config, net_config = arg_conversion(args)
    except ValueError as e:
        logging.error(e)
        return

    # Display
    np.set_printoptions(formatter={"all": "{:.6f}".format})
    logging.info(f"Training/testing with {args.imu_freq} Hz IMU data")
    logging.info(
        "Size: "
        + str(data_window_config["past_data_size"])
        + "+"
        + str(data_window_config["window_size"])
        + "+"
        + str(data_window_config["future_data_size"])
        + ", "
        + "Time: "
        + str(args.past_time)
        + "+"
        + str(args.window_time)
        + "+"
        + str(args.future_time)
    )
    logging.info("Perturb on bias: %s" % args.do_bias_shift)
    logging.info("Perturb on gravity: %s" % args.perturb_gravity)
    logging.info("Sample frequency: %s" % args.sample_freq)

    train_loader, val_loader = None, None
    start_t = time.time()
    
    data = TlioData(
        args.root_dir, 
        batch_size=args.batch_size, 
        dataset_style=args.dataset_style, 
        num_workers=args.workers,
        persistent_workers=args.persistent_workers,
    )
    data.prepare_data()
    
    train_list = data.get_datalist("train")

    """
    try:
        train_dataset = FbSequenceDataset(
            args.root_dir, train_list, args, data_window_config, mode="train"
        )
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        )
    except OSError as e:
        logging.error(e)
        return
    """
    train_loader = data.train_dataloader()
    # print("train_loader is : ", train_loader, type(train_loader))
    train_transforms = data.get_train_transforms()

    end_t = time.time()
    logging.info(f"Training set loaded. Loading time: {end_t - start_t:.3f}s")
    logging.info(f"Number of train samples: {len(data.train_dataset)}")

    #if args.val_list is not None:
    if data.val_dataset is not None:
        val_list = data.get_datalist("val")
        """
        try:
            val_dataset = FbSequenceDataset(
                args.root_dir, val_list, args, data_window_config, mode="val"
            )
            val_loader = DataLoader(val_dataset, batch_size=512, shuffle=True)
        except OSError as e:
            logging.error(e)
            return
        """
        val_loader = data.val_dataloader()
        logging.info("Validation set loaded.")
        logging.info(f"Number of val samples: {len(data.val_dataset)}")

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    
    ### >>> modify network to e2pn arch ###
    
    # Load the arguments from the JSON file
    
    def convert_dict_to_namespace(d):
        for key, value in d.items():
            if isinstance(value, dict):
                d[key] = convert_dict_to_namespace(value)
        return Namespace(**d)
    
    # with open('/workspace/equivTLIO/src/SPConvNets/opt.json', 'r') as args_file:
    #     opt_e2pn = json.load(args_file)
    with open('/workspace/equivTLIO/src/SPConvNets/opt-inv.json', 'r') as args_file:
        opt_e2pn = json.load(args_file)
        
    opt_e2pn = convert_dict_to_namespace(opt_e2pn)
    module = import_module('SPConvNets.models')

    network = get_model(args.arch, net_config, args.input_dim, args.output_dim)
    ##e2pn network
    # network = getattr(module, 'reg_so3net').build_model_from(opt_e2pn, None)
    network = getattr(module, 'inv_so3net_pn').build_model_from(opt_e2pn, None)
    
    
    ### <<< modify network to e2pn arch ###
    
    network.to(device)
    # e2pn_model.to(device)
    
    
    ### >>> print model info ###
    # print("input dim : ",args.input_dim, " output dim : ", args.output_dim)
    # print(" >>> network info <<< ")
    # print(network)
    # print(" >>> imported info <<< ")
    # print(e2pn_model)
    # print("batch size : ", args.batch_size)
    ### <<< print model info ###
    
    # total_params = network.get_num_params()
    # logging.info(f'Network "{args.arch}" loaded to device {device}')
    # logging.info(f"Total number of parameters: {total_params}")

    optimizer = torch.optim.Adam(network.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10, verbose=True, eps=1e-12
    )
    logging.info(f"Optimizer: {optimizer}, Scheduler: {scheduler}")

    start_epoch = 0
    if args.continue_from is not None:
        checkpoints = torch.load(args.continue_from)
        start_epoch = checkpoints.get("epoch", 0)
        network.load_state_dict(checkpoints.get("model_state_dict"))
        optimizer.load_state_dict(checkpoints.get("optimizer_state_dict"))
        logging.info(f"Continue from epoch {start_epoch}")
    else:
        # default starting from latest checkpoint from interruption
        latest_pt = os.path.join(args.out_dir, "checkpoints", "checkpoint_latest.pt")
        if os.path.isfile(latest_pt):
            checkpoints = torch.load(latest_pt)
            start_epoch = checkpoints.get("epoch", 0)
            network.load_state_dict(checkpoints.get("model_state_dict"))
            optimizer.load_state_dict(checkpoints.get("optimizer_state_dict"))
            logging.info(
                f"Detected saved checkpoint, starting from epoch {start_epoch}"
            )

    summary_writer = SummaryWriter(osp.join(args.out_dir, "logs"))
    # summary_writer.add_text("info", f"total_param: {total_params}")

    logging.info(f"-------------- Init, Epoch {start_epoch} --------------")
    #attr_dict = get_inference(network, train_loader, device, start_epoch, train_transforms)
    #write_summary(summary_writer, attr_dict, start_epoch, optimizer, "train")
    #if val_loader is not None:
    #    attr_dict = get_inference(network, val_loader, device, start_epoch)
    #    write_summary(summary_writer, attr_dict, start_epoch, optimizer, "val")

    def stop_signal_handler(args, epoch, network, optimizer, signal, frame):
        logging.info("-" * 30)
        logging.info("Early terminate")
        save_model(args, epoch, network, optimizer, best=False, interrupt=True)
        sys.exit()

    best_val_loss = np.inf
    for epoch in range(start_epoch + 1, args.epochs):
        signal.signal(
            signal.SIGINT, partial(stop_signal_handler, args, epoch, network, optimizer)
        )
        signal.signal(
            signal.SIGTERM,
            partial(stop_signal_handler, args, epoch, network, optimizer),
        )

        logging.info(f"-------------- Training, Epoch {epoch} ---------------")
        start_t = time.time()
        # train_attr_dict = do_train(network, train_loader, device, epoch, optimizer, train_transforms)
        train_attr_dict = do_train_imu_e2pn(network, train_loader, device, epoch, optimizer, train_transforms)
        # train_attr_dict = do_train_e2pn(e2pn_model, train_loader, device, epoch, optimizer, train_transforms)
        write_summary(summary_writer, train_attr_dict, epoch, optimizer, "train")
        end_t = time.time()
        logging.info(f"time usage: {end_t - start_t:.3f}s")

        if val_loader is not None:
            val_attr_dict = get_inference(network, val_loader, device, epoch)
            write_summary(summary_writer, val_attr_dict, epoch, optimizer, "val")
            if np.mean(val_attr_dict["losses"]) < best_val_loss:
                best_val_loss = np.mean(val_attr_dict["losses"])
                save_model(args, epoch, network, optimizer, best=True)
        else:
            save_model(args, epoch, network, optimizer, best=False)

    logging.info("Training complete.")

    return

def check_equivariance(self, data):
        in_tensors = data['pc'].to(self.opt.device)
        in_label = data['label'].to(self.opt.device).reshape(-1)
        in_Rlabel = data['R_label'].to(self.opt.device) #if self.opt.debug_mode == 'knownatt' else None #!!!!
        in_R = data['R'].to(self.opt.device)

        feat_conv, x = self.model(in_tensors, in_Rlabel)
        pred, feat, x_feat = x
        n_anchors = feat.shape[-1]
        x_feat = x_feat.reshape(x_feat.shape[0], -1, n_anchors)

        in_tensors_ori = torch.matmul(in_tensors, in_R) # B*n*3, B*3*3
        feat_conv_ori, x_ori = self.model(in_tensors_ori, in_Rlabel)  # bn, bra, b[ca]
        pred_ori, feat_ori, x_feat_ori = x_ori
        n_anchors = feat_ori.shape[-1]
        x_feat_ori = x_feat_ori.reshape(x_feat_ori.shape[0], -1, n_anchors)

        trace_idx_ori = self.trace_idx_ori[in_Rlabel.flatten()] # ba
        trace_idx_ori_p = trace_idx_ori[:,None,None].expand_as(feat_conv_ori) #bcpa
        feat_conv_align = torch.gather(feat_conv, -1, trace_idx_ori_p)

        trace_idx_ori_global = trace_idx_ori[:,None].expand_as(x_feat_ori) #bca
        x_feat_align = torch.gather(x_feat, -1, trace_idx_ori_global)

        # self.logger.log('TestEquiv', f'feat_ori: {feat_ori.shape}, x_feat_ori: {x_feat_ori.shape}')
        # self.logger.log('TestEquiv', f'x_feat: {x_feat.shape}, x_feat_from_ori: {x_feat_from_ori.shape}')
        # self.logger.log('TestEquiv', f'in_Rlabel: {in_Rlabel}, in_R: {in_R}')

        cos_sim_before = self.cos_sim(feat_conv, feat_conv_ori)
        cos_sim_after = self.cos_sim(feat_conv_align, feat_conv_ori)

        self.logger.log('TestEquiv', f'per point cos before: {cos_sim_before}, after: {cos_sim_after}')

        cos_sim_before = self.cos_sim(x_feat, x_feat_ori)
        cos_sim_after = self.cos_sim(x_feat_align, x_feat_ori)
        self.logger.log('TestEquiv', f'global cos before: {cos_sim_before}, after: {cos_sim_after}')
        
def cos_sim(f1, f2):
        ### both bc(p)a
        f1_norm = torch.norm(f1, dim=1)
        f2_norm = torch.norm(f2, dim=1)
        cos_similarity = (f1 * f2).sum(1) / (f1_norm * f2_norm)
        return cos_similarity