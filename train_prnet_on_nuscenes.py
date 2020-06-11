#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import gc
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from data import ModelNet40
from data import Bunny
import numpy as np
from torch.utils.data import DataLoader
from model import PRNet
from types import SimpleNamespace
import seaborn as sns
from tqdm import tqdm
import gc
from util import npmat2euler
from sklearn.metrics import r2_score

from sacred import Experiment
from sacred import SETTINGS
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.observers import MongoObserver

from nuscenesHelper.utils import scale_to_range, filter_pointcloud
from nuscenesHelper import LidarDataset
from nuscenesHelper import NuScenesHelper
from nuscenes.nuscenes import NuScenes, NuScenesExplorer

# sacred config
SETTINGS.CAPTURE_MODE = 'sys'  # for tqdm
ex = Experiment("train_prnet_on_nuscenes")
observer = MongoObserver.create(url='10.3.54.105:27017', db_name='qcao_scene_flow') 
ex.observers.append(observer)
ex.captured_out_filter = apply_backspaces_and_linefeeds  # for tqdm

CP_PATH = '/root/no_backup/checkpoints/prnet_train_nuscenes/models'
BEST_CP_PATH = '/root/workspace/checkpoints/prnet_train_nuscenes/models'

os.makedirs(CP_PATH, exist_ok=True)
os.makedirs(BEST_CP_PATH, exist_ok=True)


@ex.config
def config():
    NB_EPOCHS = 5
    BATCH_SIZE = 4
    N_FF_DIMS = 1024
    N_EMB_DIMS = 512
    N_POINTS = 1024
    N_SUBSAMPLED_POINTS = 512
    N_KEYPOINTS = 512
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0
    MODEL_PATH = ""
    


@ex.automain
def main(NB_EPOCHS, BATCH_SIZE, 
         N_FF_DIMS, N_EMB_DIMS, 
         N_POINTS, N_SUBSAMPLED_POINTS, 
         N_KEYPOINTS, _run, 
         LEARNING_RATE, 
         WEIGHT_DECAY, MODEL_PATH):
    
    # Get the job id of the cluster in case
    # we want to redo the experiment.
    _run.meta_info['container_name'] = os.environ.get('CONTAINER_NAME', 'no name')
    
    args = SimpleNamespace(
        emb_nn='dgcnn', 
        attention='transformer', 
        head='svd',
        svd_on_gpu=True,
        n_emb_dims=N_EMB_DIMS,
        n_blocks=1,
        n_heads=4,
        n_iters=3,
        discount_factor=0.9,
        n_ff_dims=N_FF_DIMS,
        n_keypoints=N_KEYPOINTS,
        temp_factor=100,
        cat_sampler='gumbel_softmax',
        dropout=0,
        batch_size=BATCH_SIZE,
        test_batch_size=BATCH_SIZE,
        epochs=NB_EPOCHS,
        cycle_consistency_loss=0.1,
        feature_alignment_loss=0.1,
        n_points=N_POINTS,
        rot_factor=4,
        exp_name="prnet_train_nuscenes",
        n_subsampled_points=N_SUBSAMPLED_POINTS,
        model_path=MODEL_PATH,
        num_workers=1,
        seed=2212
    )
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    
    net = PRNet(args).cuda()
    opt = optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Load nuScenes dataset
    nusc = NuScenes(version='v1.0-trainval', dataroot='/datasets_master/nuscenes', verbose=True)
    nuscenesHelper = NuScenesHelper(nusc)
    
    test_scenes = NuScenesHelper.get_small_test_set()
    train_scenes, _ = NuScenesHelper.split_train_val_scenes()
    train_scenes = np.random.permutation(train_scenes)
    num_scenes = len(train_scenes)
    val_scenes = train_scenes[:20]
    train_scenes = train_scenes[20:]

    train_dataset = LidarDataset(nusc, train_scenes, skip=(1, 1), n_rounds=1, get_colors=False, num_points=1024)
    val_dataset = LidarDataset(nusc, val_scenes, skip=(1, 1), n_rounds=1, get_colors=False, num_points=1024)
    test_dataset = LidarDataset(nusc, test_scenes, skip=(1, 1), n_rounds=1, get_colors=False, num_points=1024)

    train_loader = DataLoader(
        train_dataset, 
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = args.num_workers
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size = args.test_batch_size,
        shuffle = False,
        num_workers = args.num_workers
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size = args.test_batch_size,
        shuffle = False,
        num_workers = args.num_workers
    )
    
    # Train
    eval_every = 500
    epoch_factor = args.epochs / 100.0        
    
    if args.model_path is not '':
        assert os.path.exists(args.model_path), "Trying to resume, but model given doesn't exists."

        state = torch.load(args.model_path)
        net.set_state(state['model_state_dict'])
        opt.load_state_dict(state['optimizer_state_dict'])
        start_epoch = state['epoch']
        info_test_best = state['info_test_best']
        it = state['iteration']
        print("Resuming from previous state: %s" % args.model_path)
        print("Previous best: ")
        net.logger.write(info_test_best, write=False) 
    else:
        start_epoch = 0
        info_test_best = None
        it = 0

    
    scheduler = MultiStepLR(opt,
                            milestones=[int(30*epoch_factor), int(60*epoch_factor), int(80*epoch_factor)],
                            gamma=0.1)
    
    
    for epoch in range(start_epoch, args.epochs):
        net.train()
        total_loss = 0
        rotations_ab = []
        translations_ab = []
        rotations_ab_pred = []
        translations_ab_pred = []
        eulers_ab = []
        num_examples = 0
        total_feature_alignment_loss = 0.0
        total_cycle_consistency_loss = 0.0
        total_scale_consensus_loss = 0.0
        
        # Train for 1 epoch
        for data in tqdm(train_loader, position=0, leave=True):
            it += 1
            src = data['source_points'].float().cuda()
            tgt = data['target_points'].float().cuda()
            translation_ab = data['translation'].float().cuda()
            rotation_ab = data['rotation'].float().cuda()

            rotation_ba = data['inv_rotation'].float().cuda()
            translation_ba = data['inv_translation'].float().cuda()

            euler_ab = data['euler'].float().cuda()
            euler_ba = data['inv_euler'].float().cuda()
            
            loss, feature_alignment_loss, cycle_consistency_loss, scale_consensus_loss,\
            rotation_ab_pred, translation_ab_pred = net._train_one_batch(src, tgt, rotation_ab, translation_ab, opt)
            
            batch_size = src.size(0)
            num_examples += batch_size
            total_loss = total_loss + loss * batch_size
            total_feature_alignment_loss = total_feature_alignment_loss + feature_alignment_loss * batch_size
            total_cycle_consistency_loss = total_cycle_consistency_loss + cycle_consistency_loss * batch_size
            total_scale_consensus_loss = total_scale_consensus_loss + scale_consensus_loss * batch_size
            
            rotations_ab.append(rotation_ab.detach().cpu().numpy())
            translations_ab.append(translation_ab.detach().cpu().numpy())
            rotations_ab_pred.append(rotation_ab_pred.detach().cpu().numpy())
            translations_ab_pred.append(translation_ab_pred.detach().cpu().numpy())
            eulers_ab.append(euler_ab.cpu().numpy())
            
            if it % eval_every == 0:
                avg_loss = total_loss / num_examples
                avg_feature_alignment_loss = total_feature_alignment_loss / num_examples
                avg_cycle_consistency_loss = total_cycle_consistency_loss / num_examples
                avg_scale_consensus_loss = total_scale_consensus_loss / num_examples

                rotations_ab = np.concatenate(rotations_ab, axis=0)
                translations_ab = np.concatenate(translations_ab, axis=0)
                rotations_ab_pred = np.concatenate(rotations_ab_pred, axis=0)
                translations_ab_pred = np.concatenate(translations_ab_pred, axis=0)
                eulers_ab = np.degrees(np.concatenate(eulers_ab, axis=0))
                eulers_ab_pred = npmat2euler(rotations_ab_pred)

                r_ab_mse = np.mean((eulers_ab-eulers_ab_pred)**2)
                r_ab_rmse = np.sqrt(r_ab_mse)
                r_ab_mae = np.mean(np.abs(eulers_ab-eulers_ab_pred))

                t_ab_mse = np.mean((translations_ab-translations_ab_pred)**2)
                t_ab_rmse = np.sqrt(t_ab_mse)
                t_ab_mae = np.mean(np.abs(translations_ab-translations_ab_pred))

                r_ab_r2_score = r2_score(eulers_ab, eulers_ab_pred)
                t_ab_r2_score = r2_score(translations_ab, translations_ab_pred)

                # save to sacred
                ex.log_scalar("training.avg_loss", avg_loss, it)
                ex.log_scalar("training.avg_feature_alignment_loss", avg_feature_alignment_loss, it)
                ex.log_scalar("training.avg_cycle_consistency_loss", avg_cycle_consistency_loss, it)
                ex.log_scalar("training.avg_scale_consensus_loss", avg_scale_consensus_loss, it)

                ex.log_scalar("training.r_ab_mse", r_ab_mse, it)
                ex.log_scalar("training.r_ab_rmse", r_ab_rmse, it)            
                ex.log_scalar("training.r_ab_mae", r_ab_mae, it)

                ex.log_scalar("training.t_ab_mse", r_ab_mae, it)
                ex.log_scalar("training.t_ab_rmse", r_ab_mae, it)            
                ex.log_scalar("training.t_ab_mae", t_ab_mae, it)

                ex.log_scalar("training.r_ab_r2_score", r_ab_r2_score, it)
                ex.log_scalar("training.t_ab_r2_score", t_ab_r2_score, it)            


                total_loss = 0
                rotations_ab = []
                translations_ab = []
                rotations_ab_pred = []
                translations_ab_pred = []
                eulers_ab = []
                num_examples = 0
                total_feature_alignment_loss = 0.0
                total_cycle_consistency_loss = 0.0
                total_scale_consensus_loss = 0.0
                with torch.no_grad():
                    info_test = net._test_one_epoch(epoch=epoch, test_loader=val_loader, dataset="nuscenes")

                    # save to sacred
                    ex.log_scalar("eval.avg_loss", info_test['loss'], it)
                    ex.log_scalar("eval.avg_feature_alignment_loss", info_test['feature_alignment_loss'], it)
                    ex.log_scalar("eval.avg_cycle_consistency_loss", info_test['cycle_consistency_loss'], it)
                    ex.log_scalar("eval.avg_scale_consensus_loss", info_test['scale_consensus_loss'], it)

                    ex.log_scalar("eval.r_ab_mse", info_test['r_ab_mse'], it)
                    ex.log_scalar("eval.r_ab_rmse", info_test['r_ab_rmse'], it)            
                    ex.log_scalar("eval.r_ab_mae", info_test['r_ab_mae'], it)

                    ex.log_scalar("eval.t_ab_mse", info_test['r_ab_mae'], it)
                    ex.log_scalar("eval.t_ab_rmse", info_test['r_ab_mae'], it)            
                    ex.log_scalar("eval.t_ab_mae", info_test['t_ab_mae'], it)

                    ex.log_scalar("eval.r_ab_r2_score", info_test['r_ab_r2_score'], it)
                    ex.log_scalar("eval.t_ab_r2_score", info_test['t_ab_r2_score'], it)    

                    savedict = {
                        'epoch': epoch,
                        "iteration": it,
                        'model_state_dict': net.get_state(),
                        'optimizer_state_dict': opt.state_dict(),
                        'info_test_best': info_test_best
                    }

                    path = os.path.join(CP_PATH, 'model.%d.%d.t7' % (epoch, it))
                    torch.save(savedict, path)

                    if info_test_best is None or info_test_best['loss'] > info_test['loss']:
                        info_test_best = info_test
                        info_test_best['stage'] = 'best_test'
                        savedict['info_test_best'] = info_test_best

                        path = os.path.join(BEST_CP_PATH, 'model.best.t7')
                        torch.save(savedict, path)
              
                    
        scheduler.step()
        gc.collect()
if __name__ == '__main__':
    main()