from importlib import import_module
from SPConvNets import Dataloader_ModelNet40
from tqdm import tqdm
import torch
import vgtk
import vgtk.pc as pctk
import numpy as np
import os
from os import path as osp
import torch.nn.functional as F
from sklearn.neighbors import KDTree
from SPConvNets.datasets.evaluation.retrieval import modelnet_retrieval_mAP
import vgtk.so3conv.functional as L
from thop import profile
from fvcore.nn import FlopCountAnalysis
from dataloader.tlio_data import TlioData
from dataloader.memmapped_sequences_dataset import MemMappedSequencesDataset
from torch.utils.data import DataLoader

from network.losses import get_loss, loss_class
import vgtk.point3d as p3dtk
from network.test import torch_to_numpy, get_inference, get_datalist, arg_conversion

def val(dataset_test, model, metric, best_acc, test_accs, device, logger, info,
        debug_mode, attention_loss, attention_loss_type, att_permute_loss):
    accs = []
    # lmc = np.zeros([40,60], dtype=np.int32)

    all_labels = []
    all_feats = []
    dataset_test.dataset.set_seed(0)
    
    self.dataset_iter = iter(self.dataset_test)
    data = next(self.dataset_iter)
    
    for transform in self.train_transforms:
        sample = transform(sample)
    feat = sample["feats"]["imu0"]
    
    pc = feat.cpu().numpy() # 1024,6,200
    pc = p3dtk.normalize_np(pc, batch=True)
    
    pc_tgt = pc.transpose(0,2,1) # 1024,200,6
    pc_tgt = pc_tgt[:,:,:3]+pc_tgt[:,:,3:] #add acc. and ang-vel.

    pc_tgt = torch.from_numpy(pc_tgt).to(torch.device('cuda'))
    pred, pred_cov= self.model(pc_tgt)
    
    self.model(torch.rand_like(pc_ori))
    if len(pred.shape) == 2:
        targ = sample["targ_dt_World"][:,-1,:]
    else:
        targ = sample["targ_dt_World"][:,1:,:].permute(0,2,1)
    
    if isinstance(targ, np.ndarray):
        targ = torch.tensor(targ).to(torch.device('cuda'))
    elif targ.device.type == 'cpu':
        targ = targ.to(torch.device('cuda'))
    self.loss = get_loss(pred, pred_cov, targ, epoch)
        
        
    
    for it, data in enumerate(tqdm(dataset_test, miniters=100, maxinterval=600)):
        in_tensors = data['pc'].to(device)
        bdim = in_tensors.shape[0]
        in_label = data['label'].to(device).reshape(-1)
        in_Rlabel = data['R_label'].to(device) if debug_mode == 'knownatt' else None

        pred, feat, x_feat = model(in_tensors, in_Rlabel)

        if attention_loss:
            in_rot_label = data['R_label'].to(device).reshape(bdim)
            loss, cls_loss, r_loss, acc, r_acc = metric(pred, in_label, feat, in_rot_label, 2000)
            attention = F.softmax(feat,1)

            if attention_loss_type == 'no_cls':
                acc = r_acc
                loss = r_loss

            # max_id = attention.max(-1)[1].detach().cpu().numpy()
            # labels = data['label'].cpu().numpy().reshape(-1)
            # for i in range(max_id.shape[0]):
            #     lmc[labels[i], max_id[i]] += 1
        elif att_permute_loss:
            in_rot_label = data['R_label'].to(device).reshape(bdim)
            in_anchor_label = data['anchor_label'].to(device)
            loss, cls_loss, r_loss, acc, r_acc = metric(pred, in_label, feat, in_rot_label, 2000, in_anchor_label)
        else:
            cls_loss, acc = metric(pred, in_label)
            loss = cls_loss

        all_labels.append(in_label.cpu().numpy())
        all_feats.append(x_feat.cpu().numpy())  # feat

        accs.append(acc.cpu())
        # ### comment out if do not need per batch print
        # logger.log("Testing", "Accuracy: %.1f, Loss: %.2f!"%(100*acc.item(), loss.item()))
        # if attention_loss or att_permute_loss:
        #     logger.log("Testing", "Rot Acc: %.1f, Rot Loss: %.2f!"%(100*r_acc.item(), r_loss.item()))

    accs = np.array(accs, dtype=np.float32)
    mean_acc = 100*accs.mean()
    logger.log('Testing', 'Average accuracy is %.2f!!!!'%(mean_acc))
    test_accs.append(mean_acc)

    new_best = best_acc is None or mean_acc > best_acc
    if new_best:
        best_acc = mean_acc
    info_print = info if info == '' else info+': '
    logger.log('Testing', info_print+'Best accuracy so far is %.2f!!!!'%(best_acc))

    mem_used_max_GB = torch.cuda.max_memory_allocated() / (1024*1024*1024)
    torch.cuda.reset_peak_memory_stats()
    logger.log('Testing', f'Mem: {mem_used_max_GB:.3f}GB')

    # self.logger.log("Testing", 'Here to peek at the lmc')
    # self.logger.log("Testing", str(lmc))
    # import ipdb; ipdb.set_trace()
    n = 1
    all_feats = np.concatenate(all_feats, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    logger.log("Testing", "all_feats.shape, {}, all_labels.shape, {}".format(all_feats.shape, all_labels.shape))
    mAP = modelnet_retrieval_mAP(all_feats,all_labels,n)
    logger.log('Testing', 'Mean average precision at %d is %f!!!!'%(n, mAP))

    return mean_acc, best_acc, new_best

def save_model(args, epoch, network, optimizer, best=True, interrupt=False):
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
                        
class Trainer(vgtk.Trainer):
    def __init__(self, opt, args):
        """Trainer for tlio regression. """
        self.attention_model = opt.model.flag.startswith('attention') and opt.debug_mode != 'knownatt'
        self.attention_loss = self.attention_model and opt.train_loss.cls_attention_loss
        self.att_permute_loss = opt.model.flag == 'permutation'
        self.args = args
        if opt.group_test:
            self.rot_set = [None, 'so3'] # 'ico' 'z', 
            if opt.train_rot is None:
                ### Test the icosahedral equivariance when not using rotation augmentation in training
                self.rot_set.append('ico')
        super(Trainer, self).__init__(opt,args)

        if self.attention_loss or self.att_permute_loss:
            self.summary.register(['Loss', 'Acc', 'R_Loss', 'R_Acc'])
        else:
            self.summary.register(['Loss', 'Acc'])
        self.epoch_counter = 0
        self.iter_counter = 0
        if self.opt.group_test:
            # self.best_accs_ori = {None: 0, 'z': 0, 'so3': 0}
            # self.best_accs_aug = {None: 0, 'z': 0, 'so3': 0}
            # self.test_accs_ori = {None: [], 'z': [], 'so3': []}
            # self.test_accs_aug = {None: [], 'z': [], 'so3': []}
            self.best_accs_ori = dict()
            self.best_accs_aug = dict()
            self.test_accs_ori = dict()
            self.test_accs_aug = dict()
            for rot in self.rot_set:
                self.best_accs_ori[rot] = 0
                self.best_accs_aug[rot] = 0
                self.test_accs_ori[rot] = []
                self.test_accs_aug[rot] = []
        else:
            self.test_accs = []
            self.best_acc = None

        
        if self.opt.model.kanchor == 12:
            self.anchors = L.get_anchorsV()
            self.trace_idx_ori, self.trace_idx_rot = L.get_relativeV_index()
            self.trace_idx_ori = torch.tensor(self.trace_idx_ori).to(self.opt.device)
            self.trace_idx_rot = torch.tensor(self.trace_idx_rot).to(self.opt.device)
        else:
            self.anchors = L.get_anchors(self.opt.model.kanchor)

    def _setup_datasets(self,args):
        if self.opt.mode == 'train':
            
            data = TlioData(
                args.root_dir, 
                batch_size=args.batch_size, 
                dataset_style=args.dataset_style, 
                num_workers=args.workers,
                persistent_workers=args.persistent_workers,
            )
            data.prepare_data()
            
            train_list = data.get_datalist("train")   #283
            self.dataset = data.train_dataloader()
            self.train_transforms = data.get_train_transforms()
            self.dataset_iter = iter(self.dataset)
            
            ###
            # dataset = Dataloader_ModelNet40(self.opt, rot=self.opt.train_rot)
            # self.dataset = torch.utils.data.DataLoader(dataset, \
            #                                             batch_size=self.opt.batch_size, \
            #                                             shuffle=True, \
            #                                             num_workers=self.opt.num_thread)
            # self.dataset_iter = iter(self.dataset)

        if self.opt.mode =='train':
            test_batch_size = self.opt.test_batch_size
        else:
            test_batch_size = self.opt.batch_size
        if self.opt.group_test:
            self.datasets_test_ori = dict()
            # for rot_mode in [None, 'z', 'so3']:
            for rot_mode in self.rot_set:
                dataset_test = Dataloader_ModelNet40(self.opt, 'testR', test_aug=False, rot=rot_mode)
                self.datasets_test_ori[rot_mode] = torch.utils.data.DataLoader(dataset_test, \
                                                        batch_size=test_batch_size, \
                                                        shuffle=False, \
                                                        num_workers=self.opt.num_thread)
            self.datasets_test_aug = dict()
            # for rot_mode in [None, 'z', 'so3']:
            for rot_mode in self.rot_set:
                dataset_test = Dataloader_ModelNet40(self.opt, 'testR', test_aug=True, rot=rot_mode)
                self.datasets_test_aug[rot_mode] = torch.utils.data.DataLoader(dataset_test, \
                                                        batch_size=test_batch_size, \
                                                        shuffle=False, \
                                                        num_workers=self.opt.num_thread)
        else:
            dataset_test = Dataloader_ModelNet40(self.opt, 'testR')
            self.dataset_test = torch.utils.data.DataLoader(dataset_test, \
                                                            batch_size=test_batch_size, \
                                                            shuffle=False, \
                                                            num_workers=self.opt.num_thread)


    def _setup_model(self):
        if self.opt.mode == 'train':
            param_outfile = os.path.join(self.root_dir, "params.json")
        else:
            param_outfile = None

        module = import_module('SPConvNets.models')
        self.model = getattr(module, self.opt.model.model).build_model_from(self.opt, param_outfile)

        pytorch_total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.log("Training", "Total number of parameters: {}".format(pytorch_total_params))

            
        input = torch.randn(8, 2, 3).to(self.opt.device)
        # print(self.model)
        macs, params = profile(self.model, inputs=(input, ))
        print(
            "Batch size: 8 | params(M): %.2f | FLOPs(G) %.5f" % (params / (1000 ** 2), macs / (1000 ** 3))
        )
        input = torch.randn(8, 2, 3).to(self.opt.device)
        macs, params = profile(self.model, inputs=(input, ))
        print(
            "Batch size: 8 | params(M): %.2f | FLOPs(G) %.5f" % (params / (1000 ** 2), macs / (1000 ** 3))
        )
        self.profiled = 1 # 0

    # def _setup_metric(self):
    #     if self.attention_loss:
    #         ### loss on category and rotation classification
    #         self.metric = vgtk.AttentionCrossEntropyLoss(self.opt.train_loss.attention_loss_type, self.opt.train_loss.attention_margin)
    #         # self.r_metric = AnchorMatchingLoss()
    #     elif self.att_permute_loss:
    #         ### loss on category classification and anchor alignment
    #         self.metric = vgtk.AttPermuteCrossEntropyLoss(self.opt.train_loss.attention_loss_type, self.opt.train_loss.attention_margin, self.opt.device, \
    #                                                         self.opt.train_loss.anchor_ab_loss, self.opt.train_loss.cross_ab, self.opt.train_loss.cross_ab_T)
    #     else:
    #         ### loss on category classification only
    #         self.metric = vgtk.CrossEntropyLoss()

    # For epoch-based training
    def epoch_step(self, epoch):
        for it, data in tqdm(enumerate(self.dataset)):
            if self.opt.debug_mode == 'check_equiv':
                self._check_equivariance(data)
            else:
                self._optimize(data, epoch)

    # For iter-based training
    def step(self):
        try:
            # print('len of traindata set : ', len(self.dataset_iter))   #113850
            data = next(self.dataset_iter)
            # print("len(data['seq_id']) : ", len(data['seq_id']), 'batch_size : ', self.opt.batch_size )
            if len(data['seq_id']) < self.opt.batch_size:
                print('all data is train is loaded!')
                raise StopIteration
        except StopIteration:
            # New epoch
            self.epoch_counter += 1
            print("[DataLoader]: At Epoch %d!"%self.epoch_counter)
            self.dataset_iter = iter(self.dataset)
            data = next(self.dataset_iter)

        else:
            self._optimize(data, self.epoch_counter)
        # print("train_loader and length of batched dataset is : ", len(self.dataset), len(self.dataset_iter)) # 113850
        
        self.iter_counter += 1

    def cos_sim(self, f1, f2):
        ### both bc(p)a
        f1_norm = torch.norm(f1, dim=1)
        f2_norm = torch.norm(f2, dim=1)
        cos_similarity = (f1 * f2).sum(1) / (f1_norm * f2_norm)
        return cos_similarity

    def _check_equivariance(self, data):
        self.model.eval()
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

    def _optimize(self, sample, epoch):
        self.model = self.model.to(torch.device("cuda"))
        for transform in self.train_transforms:
            sample = transform(sample)
        feat = sample["feats"]["imu0"]
        
        #for gravity compensation 
        gravity = np.array([0,0,-9.8066, 0, 0, 0])
        feat = feat - gravity[np.newaxis, :, np.newaxis]
        
        # print()
        # print('feats info : ')
        # print(feat[0,:,0])
        # print()
        
        self.optimizer.zero_grad()
        pc = feat.cpu().numpy() # 1024,6,200
        pc = p3dtk.normalize_np(pc, batch=True)
        
        pc_tgt = pc.transpose(0,2,1) # 1024,200,6
        pc_tgt = pc_tgt[:,:,:3]+pc_tgt[:,:,3:] #add acc. and ang-vel.

        pc_tgt = torch.from_numpy(pc_tgt).to(torch.device('cuda'))
        
        # pc_ori, _ = pctk.batch_rotate_point_cloud(pc_tgt) 
        from scipy.spatial.transform import Rotation as sciR
        
        rotation_matrix = sciR.random().as_matrix()
        rotation_matrix = torch.from_numpy(rotation_matrix).to('cuda')
        rotation_matrix = rotation_matrix[None].repeat(pc_tgt.shape[0],1,1)
        rotation_matrix = rotation_matrix.float()
        pc_tgt = pc_tgt.float()
        pc_ori = torch.matmul(pc_tgt, rotation_matrix) # B*n*3, B*3*3
        
        # pc_ori = torch.from_numpy(pc_ori).to(torch.device('cuda'))
        
        # print('pred')
        pred, pred_cov = self.model(pc_tgt)
        # print('pred_ori')
        # pred_ori,pred_cov_ori = self.model(pc_ori)
        # print('pred_any')
        # pc_any = torch.rand_like(pc_ori)
        # pc_any2 = torch.rand_like(pc_ori)
        # pred_any, _  = self.model(pc_any)
        # print('pred_any2')
        # pred_any2, _ = self.model(pc_any2)
        # print()
        # print('value of input')
        # print(pc_tgt[:2,:2,:2], pc_ori[:2,:2,:2], pc_any[:2,:2,:2], pc_any2[:2,:2,:2])
        
        # print('model info ', self.model)
        # print()
        # print('in_tensor value : ', pc_tgt)
        # print('in_tensor_ori value : ', pc_ori)
        # print('in_tensor_any value : ', pc_any)
        # print('>> << ')
        # print('value of pred : ', pred.shape)
        # print(pred[:2,:2])
        # print('value of pred_ori : ',pred_ori.shape)
        # print(pred_ori[:2,:2])
        # print('value of pred_any : ', pred_any.shape)
        # print(pred_any[:2,:2])
        # assert False
        
        # assert False
        
        if len(pred.shape) == 2:
            targ = sample["targ_dt_World"][:,-1,:]
        else:
            targ = sample["targ_dt_World"][:,1:,:].permute(0,2,1)
        
        if isinstance(targ, np.ndarray):
            targ = torch.tensor(targ).to(torch.device('cuda'))
        elif targ.device.type == 'cpu':
            targ = targ.to(torch.device('cuda'))
        self.loss = get_loss(pred, pred_cov, targ, epoch)
        
        self.loss = self.loss.mean()
        self.loss.backward()

        # torch.nn.utils.clip_grad_norm_(network.parameters(), 0.1, error_if_nonfinite=True)
        self.optimizer.step()

        log_info = {
            'Loss': self.loss,
        }
        # in_tensors = data['pc'].to(self.opt.device)
        # bdim = in_tensors.shape[0]
        # in_label = data['label'].to(self.opt.device).reshape(-1)
        # in_Rlabel = data['R_label'].to(self.opt.device) #if self.opt.debug_mode == 'knownatt' else None
        # # import ipdb; ipdb.set_trace()

        # ###################### ----------- debug only ---------------------
        # # in_tensorsR = data['pcR'].to(self.opt.device)
        # # import ipdb; ipdb.set_trace()
        # ##################### --------------------------------------------
        # if self.profiled < 1:
        #     self.logger.log('Profile', f'in_tensors: {in_tensors.shape}, in_Rlabel: {in_Rlabel.shape}')
        #     flops = FlopCountAnalysis(self.model, (in_tensors, in_Rlabel))
        #     self.logger.log('Profile', f'flops: {flops.total()/ (1000**3)}')
        #     self.logger.log('Profile', f'flops.by_module(): {flops.by_module()}')
        #     self.profiled +=1

        # pred, feat, x_feat = self.model(in_tensors, in_Rlabel)
        # # x_feat not used in training, but used in eval() for retrieval mAP

        # ##############################################
        # # predR, featR = self.model(in_tensorsR, in_Rlabel)
        # # print(torch.sort(featR[0,0])[0])
        # # print(torch.sort(feat[0,0])[0])
        # # import ipdb; ipdb.set_trace()
        # ##############################################

        # self.optimizer.zero_grad()

        # if self.attention_loss:
        #     in_rot_label = data['R_label'].to(self.opt.device).reshape(bdim)
        #     self.loss, cls_loss, r_loss, acc, r_acc = self.metric(pred, in_label, feat, in_rot_label, 2000)
        # elif self.att_permute_loss:
        #     in_rot_label = data['R_label'].to(self.opt.device).reshape(bdim)
        #     in_anchor_label = data['anchor_label'].to(self.opt.device)
        #     self.loss, cls_loss, r_loss, acc, r_acc = self.metric(pred, in_label, feat, in_rot_label, 2000, in_anchor_label)
        # else:
        #     cls_loss, acc = self.metric(pred, in_label)
        #     self.loss = cls_loss

        # self.loss.backward()
        # self.optimizer.step()

        # Log training stats
        # if self.attention_loss or self.att_permute_loss:
        #     log_info = {
        #         'Loss': cls_loss.item(),
        #         'Acc': 100 * acc.item(),
        #         'R_Loss': r_loss.item(),
        #         'R_Acc': 100 * r_acc.item(),
        #     }
        # else:
        #     log_info = {
        #         'Loss': cls_loss.item(),
        #         'Acc': 100 * acc.item(),
        #     }

        self.summary.update(log_info)


    def _print_running_stats(self, step):
        stats = self.summary.get()
        
        mem_used_max_GB = torch.cuda.max_memory_allocated() / (1024*1024*1024)
        torch.cuda.reset_peak_memory_stats()
        mem_str = f', Mem: {mem_used_max_GB:.3f}GB'

        self.logger.log('Training', f'{step}: {stats}'+mem_str)
        # self.summary.reset(['Loss', 'Pos', 'Neg', 'Acc', 'InvAcc'])

    def test(self, dataset=None, best_acc=None, test_accs=None, info='', epoch=50):
        # new_best, best_acc = self.eval(dataset, best_acc, test_accs, info)
        new_best, best_acc = self.eval_imu(dataset, best_acc, test_accs, info, epoch)
        return new_best, best_acc

    def eval_imu(self, dataset=None, best_acc=None, test_accs=None, info='',epoch=50):
        """
        Main function for network evaluation
        """
        args= self.args
        data_window_config, net_config = arg_conversion(args)
        self.logger.log('Testing','Evaluating test set!'+info)
        torch.cuda.reset_peak_memory_stats()

        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        network = self.model.to(torch.device("cuda"))
        network.eval()

        all_targets, all_errors, all_sigmas = [], [], []
        all_norm_targets, all_angle_targets, all_norm_errors, all_angle_errors = (
            [],
            [],
            [],
            [],
        )
        mse_losses, likelihood_losses, avg_mse_losses, avg_likelihood_losses = (
            [],
            [],
            [],
            [],
        )
        all_mahalanobis = []

        test_list = get_datalist(os.path.join(args.root_dir, "test_list.txt"))

        blacklist = ["loop_hidacori058_20180519_1525"]
        # blacklist = []
        
        with torch.no_grad():
            for data in test_list:
                try:
                    #seq_dataset = FbSequenceDataset(
                    #    args.root_dir, [data], args, data_window_config, mode="eval"
                    #)

                    seq_dataset = MemMappedSequencesDataset(
                        args.root_dir,
                        "test",
                        data_window_config,
                        sequence_subset=[data],
                        store_in_ram=True,
                    )

                    seq_loader = DataLoader(seq_dataset, batch_size=1024, shuffle=False)
                except OSError as e:
                    logging.error(e)
                    continue

                attr_dict = get_inference(network, seq_loader, device, epoch)
                # print("inference is done!")
                norm_targets = np.linalg.norm(attr_dict["targets"][:, :2], axis=1)
                angle_targets = np.arctan2(
                    attr_dict["targets"][:, 0], attr_dict["targets"][:, 1]
                )
                norm_preds = np.linalg.norm(attr_dict["preds"][:, :2], axis=1)
                angle_preds = np.arctan2(attr_dict["preds"][:, 0], attr_dict["preds"][:, 1])
                norm_errors = norm_preds - norm_targets
                angle_errors = angle_preds - angle_targets
                sigmas = np.exp(attr_dict["preds_cov"])
                errors = attr_dict["preds"] - attr_dict["targets"]
                a1 = np.expand_dims(errors, axis=1)
                a2 = np.expand_dims(np.multiply(np.reciprocal(sigmas), errors), axis=-1)
                mahalanobis_dists = np.einsum("tip,tpi->t", a1, a2)

                all_targets.append(attr_dict["targets"])
                all_errors.append(errors)
                all_sigmas.append(sigmas)
                mse_losses.append(errors ** 2)
                avg_mse_losses.append(np.mean(errors ** 2, axis=1).reshape(-1, 1))
                likelihood_losses.append(attr_dict["losses"])
                avg_likelihood_losses.append(
                    np.mean(attr_dict["losses"], axis=1).reshape(-1, 1)
                )
                all_norm_targets.append(norm_targets.reshape(-1, 1))
                all_angle_targets.append(angle_targets.reshape(-1, 1))
                all_norm_errors.append(norm_errors.reshape(-1, 1))
                all_angle_errors.append(angle_errors.reshape(-1, 1))
                all_mahalanobis.append(mahalanobis_dists.reshape(-1, 1))

            arr_targets = np.concatenate(all_targets, axis=0)
            arr_errors = np.concatenate(all_errors, axis=0)
            arr_sigmas = np.concatenate(all_sigmas, axis=0)
            arr_mse_losses = np.concatenate(mse_losses, axis=0)
            arr_avg_mse_losses = np.concatenate(avg_mse_losses, axis=0)
            arr_likelihood_losses = np.concatenate(likelihood_losses, axis=0)
            arr_avg_likelihood_losses = np.concatenate(avg_likelihood_losses, axis=0)
            arr_norm_targets = np.concatenate(all_norm_targets, axis=0)
            arr_norm_errors = np.concatenate(all_norm_errors, axis=0)
            arr_angle_targets = np.concatenate(all_angle_targets, axis=0)
            arr_angle_errors = np.concatenate(all_angle_errors, axis=0)
            arr_mahalanobis = np.concatenate(all_mahalanobis, axis=0)

            arr_data = np.concatenate(
                (
                    arr_targets,
                    arr_errors,
                    arr_sigmas,
                    arr_mse_losses,
                    arr_avg_mse_losses,
                    arr_likelihood_losses,
                    arr_avg_likelihood_losses,
                    arr_norm_targets,
                    arr_norm_errors,
                    arr_angle_targets,
                    arr_angle_errors,
                    arr_mahalanobis,
                ),
                axis=1,
            )

            dataset = pd.DataFrame(
                arr_data,
                index=range(arr_mahalanobis.shape[0]),
                columns=[
                    "targets_x",
                    "targets_y",
                    "targets_z",
                    "errors_x",
                    "errors_y",
                    "errors_z",
                    "sigmas_x",
                    "sigmas_y",
                    "sigmas_z",
                    "mse_losses_x",
                    "mse_losses_y",
                    "mse_losses_z",
                    "avg_mse_losses",
                    "likelihood_losses_x",
                    "likelihood_losses_y",
                    "likelihood_losses_z",
                    "avg_likelihood_losses",
                    "norm_targets",
                    "norm_errors",
                    "angle_targets",
                    "angle_errors",
                    "mahalanobis",
                ],
            )

            dstr = "d"
            if args.do_bias_shift:
                dstr = f"{dstr}-bias-{args.accel_bias_range}-{args.gyro_bias_range}"
            if args.perturb_gravity:
                dstr = f"{dstr}-grav-{args.perturb_gravity_theta_range}"
            dstr = f"{dstr}.pkl"
            if args.out_name is not None:
                dstr = args.out_name
            outfile = os.path.join(args.out_dir, dstr)
            dataset.to_pickle(outfile)

            network.train()

        new_best = 0
        best_acc = 0
        
        return new_best, best_acc

    
    def eval_tlio(args):
        self.logger.log('Testing','Evaluating test set!'+info)
        self.model.eval()
        # self.metric.eval()
        torch.cuda.reset_peak_memory_stats()
        
            
            # write_summary(summary_writer, val_attr_dict, epoch, optimizer, "val")
            
            # mean_acc, best_acc, new_best = val(dataset, self.model, self.metric, 
            #     best_acc, test_accs, self.opt.device, self.logger, info,
            #     self.opt.debug_mode, self.attention_loss, self.opt.train_loss.attention_loss_type, 
            #     self.att_permute_loss)

        
        
        test_list_path = osp.join(args.root_dir, "test_list.txt")
        test_list = get_datalist(test_list_path)

        device = torch.device(
            "cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu"
        )

        # initialize containers
        all_metrics = {}

        with torch.no_grad():
            for data in test_list:
                try:
                    seq_dataset = MemMappedSequencesDataset(
                        args.root_dir,
                        "test",
                        data_window_config,
                        sequence_subset=[data],
                        store_in_ram=True,
                    )
                    seq_loader = DataLoader(seq_dataset, batch_size=1024, shuffle=False)
                except OSError as e:
                    print(e)
                    continue

                # Obtain trajectory
                val_attr_dict = get_inference(self.model, seq_loader, device)
                # val_attr_dict = get_inference(network, seq_loader, device, epoch=50)
                traj_attr_dict = pose_integrate(args, seq_dataset, net_attr_dict["preds"])
                outdir = osp.join(args.out_dir, data)
                if osp.exists(outdir) is False:
                    os.mkdir(outdir)
                outfile = osp.join(outdir, "trajectory.txt")
                trajectory_data = np.concatenate(
                    [
                        traj_attr_dict["ts"].reshape(-1, 1),
                        traj_attr_dict["pos_pred"],
                        traj_attr_dict["pos_gt"],
                    ],
                    axis=1,
                )
                np.savetxt(outfile, trajectory_data, delimiter=",")

                # obtain metrics
                metrics, plot_dict = compute_metrics_and_plotting(
                    args, net_attr_dict, traj_attr_dict
                )
                logging.info(metrics)
                all_metrics[data] = metrics

                outfile_net = osp.join(outdir, "net_outputs.txt")
                net_outputs_data = np.concatenate(
                    [
                        plot_dict["pred_ts"].reshape(-1, 1),
                        plot_dict["preds"],
                        plot_dict["targets"],
                        plot_dict["pred_sigmas"],
                    ],
                    axis=1,
                )
                np.savetxt(outfile_net, net_outputs_data, delimiter=",")

                if args.save_plot:
                    make_plots(args, plot_dict, outdir)

                try:
                    with open(args.out_dir + "/metrics.json", "w") as f:
                        json.dump(all_metrics, f, indent=1)
                except ValueError as e:
                    raise e
                except OSError as e:
                    print(e)
                    continue
                except Exception as e:
                    raise e

        self.model.train()
        return
    
    
    
    
    def eval(self, dataset=None, best_acc=None, test_accs=None, info=''):
        self.logger.log('Testing','Evaluating test set!'+info)
        self.model.eval()
        # self.metric.eval()
        torch.cuda.reset_peak_memory_stats()

        ################## DEBUG ###############################
        # for module in self.model.modules():
        #     if isinstance(module, torch.nn.modules.BatchNorm1d):
        #         module.train()
        #     if isinstance(module, torch.nn.modules.BatchNorm2d):
        #         module.train()
        #     if isinstance(module, torch.nn.modules.BatchNorm3d):
        #         module.train()
            # if isinstance(module, torch.nn.Dropout):
            #     module.train()
        #####################################################

        with torch.no_grad():
            if dataset is None:
                dataset = self.dataset_test
            if best_acc is None:
                best_acc = self.best_acc
            if test_accs is None:
                test_accs = self.test_accs

            val_attr_dict = get_inference(self.model, seq_loader, device)
            # write_summary(summary_writer, val_attr_dict, epoch, optimizer, "val")
            
            # mean_acc, best_acc, new_best = val(dataset, self.model, self.metric, 
            #     best_acc, test_accs, self.opt.device, self.logger, info,
            #     self.opt.debug_mode, self.attention_loss, self.opt.train_loss.attention_loss_type, 
            #     self.att_permute_loss)

        self.model.train()
        # self.metric.train()

        return new_best, best_acc
    
    def train_epoch(self):
        for i in range(self.opt.num_epochs):
            self.lr_schedule.step()
            self.epoch_step(i)

            if i % self.opt.log_freq == 0:
                self._print_running_stats(f'Epoch {i}')

            if i > 0 and i % self.opt.save_freq == 0:
                # self._save_network(f'Epoch{i}')
                save_model(self.args, i, self.model, self.optimizer, best=True, interrupt=True)
    
    def train_iter(self):
        for i in range(self.opt.num_iterations+1):
            # if i == 5:
            #     break
            self.timer.set_point('train_iter')
            self.lr_schedule.step()
            self.step()
            # print({'Time': self.timer.reset_point('train_iter')})
            self.summary.update({'Time': self.timer.reset_point('train_iter')})

            if i % self.opt.log_freq == 0:
                if hasattr(self, 'epoch_counter'):
                    step = f'Epoch {self.epoch_counter}, Iter {i}'
                else:
                    step = f'Iter {i}'
                self._print_running_stats(step)

            if i > 0 and i % self.opt.eval_freq == 0:
                if self.opt.group_test:
                    for key, dataset_test in self.datasets_test_ori.items():
                        info = 'ori_' + str(key)
                        new_best, self.best_accs_ori[key] = self.test(
                            dataset_test, self.best_accs_ori[key], self.test_accs_ori[key], info)
                        if new_best:
                            self.logger.log('Testing', 'New best! Saving this model. '+info)
                            # self._save_network('best_'+info)
                    for key, dataset_test in self.datasets_test_aug.items():
                        info = 'aug_' + str(key)
                        new_best, self.best_accs_aug[key] = self.test(
                            dataset_test, self.best_accs_aug[key], self.test_accs_aug[key], info)
                        if new_best:
                            self.logger.log('Testing', 'New best! Saving this model. '+info)
                            # self._save_network('best_'+info)
                else:
                    # new_best, self.best_acc = self.test(epoch = self.epoch_counter)
                    
                        
                    save_model(self.args, self.epoch_counter, self.model, self.optimizer, best=True, interrupt=True)
                    
                    self.logger.log('Testing', 'New best! Saving this model. ')
                    # self._save_network('best')

            if i > 0 and i % self.opt.save_freq == 0:
                tmp = 0
                # self._save_network(f'Iter{i}')
                