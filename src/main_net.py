"""
IMU network training/testing/evaluation for displacement and covariance
Input: Nx6 IMU data
Output: 3x1 displacement, 3x1 covariance parameters
"""
import sys
import os
from os import path as osp

sys.path.insert(0,'/workspace/equivTLIO/src/vgtk')
# sys.path.insert(0, os.path.join(os.path.dirname(__file__),'vgtk') )
# print('sys path : ',sys.path)
from SPConvNets.trainer_tlio import Trainer
# from SPConvNets.options import opt

import network
import network.parallel as parallel

from utils.argparse_utils import add_bool_arg
import json
from argparse import Namespace



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # ------------------ directories -----------------
    # NOTE now they are assumed to be under root_dir with new format
    #parser.add_argument("--train_list", type=str, default=None)
    #parser.add_argument("--val_list", type=str, default=None)
    #parser.add_argument("--test_list", type=str, default=None)
    parser.add_argument(
        "--root_dir", type=str, 
        default="local_data/tlio_golden", help="Path to data directory"
    )
    parser.add_argument("--out_dir", type=str, default="outputs/resnet_seq")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--continue_from", type=str, default=None)
    parser.add_argument("--out_name", type=str, default=None)

    # ------------------ architecture and training -----------------
    parser.add_argument("--lr", type=float, default=1e-04)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=10000, help="max num epochs")
    parser.add_argument("--arch", type=str, default="resnet")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--input_dim", type=int, default=6)
    parser.add_argument("--output_dim", type=int, default=3)
    parser.add_argument("-j", "--workers", type=int, default=4)
    parser.add_argument("--dataset_style", type=str, default="mmap", 
            help="'ram', 'mmap', or 'iter'. See dataloader/tlio_data.py for more details")
    add_bool_arg(parser, "persistent_workers", default=True)

    # ------------------ commons -----------------
    parser.add_argument(
        "--mode", type=str, default="train", choices=["train", "test", "eval"]
    )
    parser.add_argument(
        "--imu_freq", type=float, default=200.0, help="imu_base_freq is a multiple"
    )
    parser.add_argument("--imu_base_freq", type=float, default=1000.0)

    # ----- perturbation -----
    add_bool_arg(parser, "do_bias_shift", default=True)
    parser.add_argument("--accel_bias_range", type=float, default=0.2)  # 5e-2
    parser.add_argument("--gyro_bias_range", type=float, default=0.05)  # 1e-3

    add_bool_arg(parser, "perturb_gravity", default=True)
    parser.add_argument(
        "--perturb_gravity_theta_range", type=float, default=5.0
    )  # degrees

    # ----- window size and inference freq -----
    parser.add_argument("--past_time", type=float, default=0.0)  # s
    parser.add_argument("--window_time", type=float, default=1.0)  # s
    parser.add_argument("--future_time", type=float, default=0.0)  # s

    # ----- for sampling in training / stepping in testing -----
    parser.add_argument("--sample_freq", type=float, default=20.0)  # hz

    # ----- plotting and evaluation -----
    add_bool_arg(parser, "save_plot", default=True)
    parser.add_argument("--rpe_window", type=float, default="2.0")  # s

    args = parser.parse_args()

    ###########################################################
    # Main
    ###########################################################
    # if args.mode == "train":
    #     opt.batch_size = 12
    #     opt.test_batch_size = 24
    #     opt.train_lr.decay_rate = 0.5
    #     opt.train_lr.decay_step = 20000
    #     opt.train_loss.attention_loss_type = 'default'
    #     opt.num_iterations = 80000
    
    def convert_dict_to_namespace(d):
        for key, value in d.items():
            if isinstance(value, dict):
                d[key] = convert_dict_to_namespace(value)
        return Namespace(**d)
    
    # print("args info")
    # for arg_name in vars(args):
    #     arg_value = getattr(args, arg_name)
    #     print(f"{arg_name}: {arg_value}")
    
    
    with open('/workspace/equivTLIO/src/SPConvNets/opt-cls.json', 'r') as args_file:
        opt_e2pn = json.load(args_file)    
    opt_e2pn = convert_dict_to_namespace(opt_e2pn)
    trainer = Trainer(opt_e2pn, args) 
        
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
        
    if args.mode == "train":
        # network.net_train(args)
        trainer.train()
        print('training done!')
    elif args.mode == "test":
        # network.net_test(args)
        trainer.eval_tlio() 
        # trainer.eval_imu() 
        # trainer.eval() 
        
        
    elif args.mode == "eval":
        network.net_eval(args)
    else:
        raise ValueError("Undefined mode")
