# -*- coding: utf-8 -*-
# Copyright (c) 2021 Decision Transformer (Decision Transformer: Reinforcement Learning via Sequence Modeling) Authors (https://arxiv.org/abs/2106.01345)
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
@File    :   active_rl_main.py
@Time    :   2024/04/12 10:41:15
@Author  :   Mingjiang Liu 
@Version :   1.0
@Desc    :   copied from Decision Transformer, modified for Badminton project: ActiveRL
'''


import gym
import numpy as np
import torch
import torch.nn as nn
import wandb

import argparse
import pickle
import random
import sys

from decision_transformer.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.mlp_bc import MLPBCModel
from decision_transformer.training.act_trainer import ActTrainer
from decision_transformer.training.seq_trainer import SequenceTrainer


def discount_cumsum(x, gamma):
    # return discount return for every timestep
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum


def experiment(
        exp_prefix,
        variant,
):
    device = variant.get('device', 'cuda')
    log_to_wandb = variant.get('log_to_wandb', False)

    dataset = variant['dataset']
    model_type = variant['model_type']
    group_name = f'{exp_prefix}-{dataset}'
    exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'

    # if env_name == 'hopper':
    #     env = gym.make('Hopper-v3')
    #     max_ep_len = 1000
    #     env_targets = [3600, 1800]  # evaluation conditioning targets
    #     scale = 1000.  # normalization for rewards/returns
    # elif env_name == 'halfcheetah':
    #     # for evaluation
    #     env = gym.make('HalfCheetah-v3')
    #     max_ep_len = 1000
    #     env_targets = [12000, 6000]
    #     scale = 1000.
    # elif env_name == 'walker2d':
    #     env = gym.make('Walker2d-v3')
    #     max_ep_len = 1000
    #     env_targets = [5000, 2500]
    #     scale = 1000.
    # elif env_name == 'reacher2d':
    #     from decision_transformer.envs.reacher_2d import Reacher2dEnv
    #     env = Reacher2dEnv()
    #     max_ep_len = 100
    #     env_targets = [76, 40]
    #     scale = 10.
    # else:
        # raise NotImplementedError

    max_ep_len = 100
    scale = 1

    # if model_type == 'bc':
    #     env_targets = env_targets[:1]  # since BC ignores target, no need for different evaluations

    # load dataset
    dataset_path = f'data/{dataset}.pkl'
    with open(dataset_path, 'rb') as f:
        # trajectories: [{key:array}, {key:array}, ...]
        trajectories = pickle.load(f)
    
    state_dim = trajectories[0]["observations"].shape[1]
    act_dim = trajectories[0]["actions"].shape[1]

    # save all path information into separate lists
    mode = variant.get('mode', 'normal')
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns) # M-dims, M = episode_num

    # used for input normalization
    states = np.concatenate(states, axis=0) # states: NxD-array, N timesteps
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(traj_lens) # total-timesteps = N

    print('=' * 50)
    print(f'Starting new experiment: {dataset}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)

    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']
    pct_traj = variant.get('pct_traj', 1.)

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj*num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    def get_batch(batch_size=256):
        # get_batch for Badminton trajectories
        # batch_inds: sample 256 indexs for trajectory selection
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        max_len = 0
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]

            # get sequences from dataset
            s.append(traj['observations'].reshape(1, -1, state_dim))
            a.append(traj['actions'].reshape(1, -1, act_dim))
            r.append(traj['rewards'].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'].reshape(1, -1))
            else:
                d.append(traj['dones'].reshape(1, -1))
            timesteps.append(np.arange(traj['actions'].shape[0]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff
            rtg.append(discount_cumsum(traj['rewards'], gamma=1.).reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                # why add item to rtg? 
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            max_len = max(max_len, s[-1].shape[1])

        for i in range(batch_size):
            # padding and state + reward normalization
            tlen = s[i].shape[1]
            s[i] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[i]], axis=1)
            s[i] = (s[i] - state_mean) / state_std
            a[i] = np.concatenate([np.zeros((1, max_len - tlen, act_dim)), a[i]], axis=1)
            r[i] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[i]], axis=1)
            d[i] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[i]], axis=1)
            #TODO: need scale?
            rtg[i] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[i]], axis=1) / scale
            timesteps[i] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[i]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)
        # return: [batch_size, max_len, xxx]

        return s, a, r, d, rtg, timesteps, mask

    # def eval_episodes(target_rew):
        # def fn(model):
        #     returns, lengths = [], []
        #     for _ in range(num_eval_episodes):
        #         with torch.no_grad():
        #             if model_type == 'dt':
        #                 ret, length = evaluate_episode_rtg(
        #                     env,
        #                     state_dim,
        #                     act_dim,
        #                     model,
        #                     max_ep_len=max_ep_len,
        #                     scale=scale,
        #                     target_return=target_rew/scale,
        #                     mode=mode,
        #                     state_mean=state_mean,
        #                     state_std=state_std,
        #                     device=device,
        #                 )
        #             else:
        #                 ret, length = evaluate_episode(
        #                     env,
        #                     state_dim,
        #                     act_dim,
        #                     model,
        #                     max_ep_len=max_ep_len,
        #                     target_return=target_rew/scale,
        #                     mode=mode,
        #                     state_mean=state_mean,
        #                     state_std=state_std,
        #                     device=device,
        #                 )
        #         returns.append(ret)
        #         lengths.append(length)
        #     return {
        #         f'target_{target_rew}_return_mean': np.mean(returns),
        #         f'target_{target_rew}_return_std': np.std(returns),
        #         f'target_{target_rew}_length_mean': np.mean(lengths),
        #         f'target_{target_rew}_length_std': np.std(lengths),
        #     }
        # return fn

    if model_type == 'dt':
        model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            max_ep_len=max_ep_len,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*variant['embed_dim'],
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
            action_tanh=False,
        )
    elif model_type == 'bc':
        model = MLPBCModel(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
        )
    else:
        raise NotImplementedError

    model = model.to(device=device)

    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )

    if model_type == 'dt':
        loss = nn.CrossEntropyLoss()
        trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            # loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2), # MSE loss
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: loss(a_hat, a), # CrossEntropy loss
        )
    elif model_type == 'bc':
        trainer = ActTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )

    if log_to_wandb:
        # initial wandb
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project='decision-transformer',
            config=variant
        )
        # wandb.watch(model)  # wandb has some bug

    for iter in range(variant['max_iters']):
        outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iter+1, print_logs=True)
        if log_to_wandb:
            wandb.log(outputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='badminton_test') 
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse reward
    parser.add_argument('--pct_traj', type=float, default=1.) # use top x% data to train model 
    parser.add_argument('--batch_size', type=int, default=32) # batch_size samples， 1 sample = K sequences
    parser.add_argument('--model_type', type=str, default='dt')  # dt for decision transformer, bc for behavior cloning
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--num_eval_episodes', type=int, default=100)
    parser.add_argument('--max_iters', type=int, default=100)
    parser.add_argument('--num_steps_per_iter', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)
    
    args = parser.parse_args()

    experiment('BadmintonActiveRL', variant=vars(args))
