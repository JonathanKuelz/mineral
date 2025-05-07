# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import collections
import itertools
import json
import os
import re
from copy import deepcopy
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ... import nets
from ...common import normalizers
from ...common.reward_shaper import RewardShaper
from ...common.timer import Timer
from ...common.tracker import Tracker
from ..agent import Agent
from . import models
from .utils import CriticDataset, adaptive_scheduler, grad_norm, policy_kl, soft_update

from docbrown.templates.warp_env import RolloutTrajectory, WarpEnv

class SHAC(Agent):
    r"""Short-Horizon Actor-Critic."""

    def __init__(self, full_cfg, **kwargs):
        self._env = None
        self.network_config = full_cfg.network
        self.shac_config = full_cfg.shac
        self.max_agent_steps = int(self.shac_config.max_agent_steps)
        super().__init__(full_cfg, **kwargs)

        # --- SAPO Parameters ---
        self.with_logprobs = self.shac_config.get('with_logprobs', False)
        self.with_autoent = self.shac_config.get('with_autoent', False)
        self.entropy_coef = self.shac_config.get('entropy_coef', None)
        self.unscale_entropy_alpha = self.shac_config.get('unscale_entropy_alpha', False)
        self.use_distr_ent = self.shac_config.get('use_distr_ent', False)
        self.entropy_in_return = self.shac_config.get('entropy_in_return', False)
        self.entropy_in_targets = self.shac_config.get('entropy_in_targets', False)
        self.no_actor_entropy = self.shac_config.get('no_actor_entropy', False)

        # --- SHAC Parameters ---
        self.tanh_clamp = self.network_config.get('tanh_clamp', False)  # on actions, if not done in actor dist
        self.actor_loss_avgcritics = self.shac_config.get('actor_loss_avgcritics', False)
        self.critic_lrschedule = self.shac_config.get('critic_lrschedule', True)
        self.actor_detach_z = self.shac_config.get('actor_detach_z', False)

        self.gamma = self.shac_config.get('gamma', 0.99)
        self.critic_method = self.shac_config.get('critic_method', 'one-step')  # ['one-step', 'td-lambda']
        if self.critic_method == 'td-lambda':
            self.lam = self.shac_config.get('lambda', 0.95)
        self.critic_iterations = self.shac_config.get('critic_iterations', 16)  # Called mini-epoch in their paper
        self.target_critic_alpha = self.shac_config.get('target_critic_alpha', 0.4)

        self.max_epochs = self.shac_config.get('max_epochs', 0)  # set to 0 to disable and track by max_agent_steps instead
        self.num_critic_batches = self.shac_config.get('num_critic_batches', 4)

        # --- Encoder ---
        if self.network_config.get("encoder", None) is not None:
            EncoderCls = getattr(nets, self.network_config.encoder)
            encoder_kwargs = self.network_config.get("encoder_kwargs", {})
            self.encoder = EncoderCls(self.obs_space, encoder_kwargs, weight_init_fn=models.weight_init_)
        else:
            f = lambda x: x
            self.encoder = nets.Lambda(f)
        self.encoder.to(self.device)
        # print('Encoder:', self.encoder)

        self.share_encoder = self.shac_config.get("share_encoder", True)
        if self.share_encoder:
            self.actor_encoder = self.encoder
            # print('Actor Encoder: (shared)')
        else:
            self.actor_encoder = deepcopy(self.encoder)
            # print('Actor Encoder:', self.actor_encoder)

        # --- Buffer ---
        self.target_values = None

    @property
    def action_dim(self):
        return self.env.num_act

    @property
    def env(self) -> WarpEnv:
        if self._env is None:
            raise ValueError("Environment not yet set up.")
        return self._env

    @env.setter
    def env(self, env):
        first_setup = self._env is None
        self._env = env
        if first_setup:
            self.__init_post_set_env()

    @property
    def num_actors(self):
        return self.env.num_envs

    @property
    def horizon_len(self):
        return self.env.num_frames

    def __init_post_set_env(self):
        """Perform all the setup steps that were in __init__ previously but can only be done once the environment is set up."""
        self.num_envs = self.env.num_envs
        self.num_obs = self.env.num_obs

        self.observation_space = self.env.observation_space
        try:
            obs_space = {k: v.shape for k, v in self.observation_space.spaces.items()}
        except AttributeError:
            obs_space = {'obs': self.observation_space.shape}
        self.obs_space = obs_space

        self.metrics = self._create_metrics(self.tracker_len, self.metrics_kwargs)

        # --- Normalizers ---
        if self.tanh_clamp:  # legacy
            # unbiased=False -> correction=0
            # https://github.com/NVlabs/DiffRL/blob/a4c0dd1696d3c3b885ce85a3cb64370b580cb913/utils/running_mean_std.py#L34
            rms_config = dict(eps=1e-5, correction=0, initial_count=1e-4, dtype=torch.float32)
        else:
            rms_config = dict(eps=1e-5, initial_count=1, dtype=torch.float64)

        if self.normalize_input:
            self.obs_rms = {}
            for k, v in self.obs_space.items():
                if re.match(self.obs_rms_keys, k):
                    self.obs_rms[k] = normalizers.RunningMeanStd(v, **rms_config)
                else:
                    self.obs_rms[k] = normalizers.Identity()
            self.obs_rms = nn.ModuleDict(self.obs_rms).to(self.device)
        else:
            self.obs_rms = None

        # --- Model ---
        if self.network_config.get("encoder", None) is not None:
            obs_dim = self.encoder.out_dim
        else:
            obs_dim = self.obs_space['obs']
            obs_dim = obs_dim[0] if isinstance(obs_dim, tuple) else obs_dim
            assert obs_dim == self.env.num_obs

        ActorCls = getattr(models, self.network_config.actor)
        CriticCls = getattr(models, self.network_config.critic)
        if self.action_dim == 0:
            self.actor = ActorCls(0, 0, mlp_kwargs={'units': [0]})
        else:
            self.actor = ActorCls(obs_dim, self.action_dim, **self.network_config.get("actor_kwargs", {}))
        self.critic = CriticCls(obs_dim, self.action_dim, **self.network_config.get("critic_kwargs", {}))
        self.actor.to(self.device)
        self.critic.to(self.device)
        # print('Actor:', self.actor)
        # print('Critic:', self.critic, '\n')

        # --- Optim ---
        OptimCls = getattr(torch.optim, self.shac_config.optim_type)

        if self.shac_config.get("actor_detach_encoder", False):
            actor_optim_params = self.actor.parameters()
        else:
            actor_optim_params = itertools.chain(self.actor_encoder.parameters(), self.actor.parameters())
        self.actor_optim = OptimCls(
            actor_optim_params,
            **self.shac_config.get("actor_optim_kwargs", {}),
        )

        if self.shac_config.get("critic_detach_encoder", False):
            critic_optim_params = self.critic.parameters()
        else:
            critic_optim_params = itertools.chain(self.encoder.parameters(), self.critic.parameters())
        self.critic_optim = OptimCls(
            critic_optim_params,
            **self.shac_config.get("critic_optim_kwargs", {}),
        )

        # TODO: encoder_lr? currently overridden by actor_lr
        self.actor_lr = self.actor_optim.defaults["lr"]
        self.critic_lr = self.critic_optim.defaults["lr"]
        self.min_lr, self.max_lr = self.shac_config.get('min_lr', 1e-5), self.shac_config.get('max_lr', self.actor_lr)
        # kl scheduler
        self.last_lr = self.actor_lr
        scheduler_kwargs = self.shac_config.get('scheduler_kwargs', {})
        self.scheduler_kwargs = {**scheduler_kwargs, **dict(min_lr=self.min_lr, max_lr=self.max_lr)}
        self.avg_kl = self.scheduler_kwargs.get('kl_threshold', None)

        # --- Target Networks ---
        self.encoder_target = deepcopy(self.encoder) if not self.shac_config.no_target_critic else self.encoder
        self.critic_target = deepcopy(self.critic) if not self.shac_config.no_target_critic else self.critic

        self.reward_shaper = RewardShaper(**self.shac_config.reward_shaper)

        # --- Entropy ---
        if self.with_autoent:
            if self.shac_config.get("alpha", None) is None:
                if self.shac_config.get("alpha_optim_type", False):
                    OptimCls = getattr(torch.optim, self.shac_config.alpha_optim_type)

                init_alpha = np.log(self.shac_config.init_alpha)
                self.log_alpha = nn.Parameter(torch.tensor(init_alpha, device=self.device, dtype=torch.float32))
                self.alpha_optim = OptimCls([self.log_alpha], **self.shac_config.get("alpha_optim_kwargs", {}))
        target_entropy_scalar = self.shac_config.get("target_entropy_scalar", 1.0)
        self.target_entropy = -self.action_dim * target_entropy_scalar
        # RLPD divides by 2, https://github.com/ikostrikov/rlpd/blob/c90fd4baf28c9c9ef40a81460a2e395092844f88/rlpd/agents/sac/sac_learner.py#L78-L79
        if self.with_autoent or self.entropy_coef is not None:
            print('Target Entropy Scalar:', target_entropy_scalar, 'Target Entropy:', self.target_entropy)

        # --- Episode Metrics ---
        self.episode_rewards = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.episode_lengths = torch.zeros(self.num_envs, dtype=int, device=self.device)
        self.episode_discounted_rewards = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.episode_gamma = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)

        self.episode_rewards_hist = []
        self.episode_lengths_hist = []
        self.episode_discounted_rewards_hist = []

        tracker_len = 100
        self.episode_rewards_tracker = Tracker(tracker_len)
        self.episode_lengths_tracker = Tracker(tracker_len)
        self.episode_discounted_rewards_tracker = Tracker(tracker_len)
        self.num_episodes = torch.tensor(0, dtype=int)

        # --- Timing ---
        self.timer = Timer()

    def _init_target_values(self, trajectory: RolloutTrajectory):
        """If this buffer was not created before or we changed the horizon lenght, reinitialize."""
        if self.target_values is None or self.target_values.shape[0] != len(trajectory):
            self.target_values = torch.zeros((len(trajectory), self.env.num_envs), device=self.device, dtype=torch.float32)

    def get_actions(self,
                    obs,
                    sample=True,
                    dist=False):
        """
        Retrieve an action based on the observation.

        :param obs: The observation to use for action selection.
        :param sample: Whether to sample from the action distribution or use the mean (greedy).
        :param dist: Whether to return the action distribution. If false, just returns the action
        """
        # NOTE: obs_rms.normalize(...) occurs elsewhere
        z = self.actor_encoder(obs)
        if self.actor_detach_z:
            if isinstance(z, dict):
                z = {k: v.detach() for k, v in z.items()}
            else:
                z = z.detach()
        mu, sigma, distr = self.actor(z)
        if sample:
            actions = distr.rsample()
        else:
            actions = mu

        if self.tanh_clamp:
            # clamp actions
            actions = torch.tanh(actions)

        if dist:
            return actions, mu, sigma, distr
        else:
            return actions

    @torch.no_grad()
    def evaluate_policy(self, num_episodes, sample=False):
        episode_rewards_hist = []
        episode_lengths_hist = []
        episode_discounted_rewards_hist = []
        episode_rewards = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        episode_lengths = torch.zeros(self.num_envs, dtype=int)
        episode_discounted_rewards = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        episode_gamma = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)

        obs = self.env.reset()
        obs = self._convert_obs(obs)

        episodes = 0
        while episodes < num_episodes:
            if self.obs_rms is not None:
                obs = {k: self.obs_rms[k].normalize(v) for k, v in obs.items()}

            actions = self.get_actions(obs, sample=sample)
            obs, rew, done, _ = self.env.step(actions)
            obs = self._convert_obs(obs)

            episode_rewards += rew
            episode_lengths += 1
            episode_discounted_rewards += episode_gamma * rew
            episode_gamma *= self.gamma

            done_env_ids = done.nonzero(as_tuple=False).squeeze(-1)
            if len(done_env_ids) > 0:
                for done_env_id in done_env_ids:
                    print('rew = {:.2f}, len = {}'.format(episode_rewards[done_env_id].item(), episode_lengths[done_env_id]))
                    episode_rewards_hist.append(episode_rewards[done_env_id].item())
                    episode_lengths_hist.append(episode_lengths[done_env_id].item())
                    episode_discounted_rewards_hist.append(episode_discounted_rewards[done_env_id].item())
                    episode_rewards[done_env_id] = 0.0
                    episode_lengths[done_env_id] = 0
                    episode_discounted_rewards[done_env_id] = 0.0
                    episode_gamma[done_env_id] = 1.0
                    episodes += 1

        return episode_rewards_hist, episode_lengths_hist, episode_discounted_rewards_hist

    def train(self):

        while self.agent_steps < self.max_agent_steps:
            self.epoch += 1
            if self.max_epochs > 0 and self.epoch >= self.max_epochs:
                break

            # learning rate schedule
            if self.shac_config.lr_schedule == 'linear':
                assert self.max_epochs > 0
                if self.critic_lrschedule:
                    critic_lr = (self.min_lr - self.critic_lr) * float(self.epoch / self.max_epochs) + self.critic_lr
                    for param_group in self.critic_optim.param_groups:
                        param_group['lr'] = critic_lr

                actor_lr = (self.min_lr - self.actor_lr) * float(self.epoch / self.max_epochs) + self.actor_lr
                for param_group in self.actor_optim.param_groups:
                    param_group['lr'] = actor_lr
                lr = actor_lr
            elif self.shac_config.lr_schedule == 'constant':
                lr = self.actor_lr
            elif self.shac_config.lr_schedule == 'kl':
                if self.avg_kl is not None:
                    actor_lr = adaptive_scheduler(self.last_lr, self.avg_kl.item(), **self.scheduler_kwargs)
                    if self.critic_lrschedule:
                        critic_lr = actor_lr
                        for param_group in self.critic_optim.param_groups:
                            param_group['lr'] = critic_lr
                    for param_group in self.actor_optim.param_groups:
                        param_group['lr'] = actor_lr
                    self.last_lr = actor_lr
                lr = self.last_lr
            else:
                raise NotImplementedError(self.shac_config.lr_schedule)

            # train actor
            self.timer.start("train/update_actor")
            self.actor_encoder.train()
            self.actor.train()
            # self.encoder.eval()
            self.critic.eval()
            # self.encoder_target.eval()
            self.critic_target.eval()
            actor_results = self.update_actor()
            self.timer.end("train/update_actor")

            # train critic
            # prepare dataset
            self.timer.start("train/make_critic_dataset")
            with torch.no_grad():
                self.compute_target_values()
                values_results = {
                    "target_values/mean": self.target_values.mean().item(),
                    "target_values/std": self.target_values.std().item(),
                    "target_values/max": self.target_values.max().item(),
                    "target_values/min": self.target_values.min().item(),
                }

                T, B = self.target_values.shape
                target_values = self.target_values.view(T * B, 1)
                target_values = target_values.view(T, B)

            self.encoder.train()
            self.critic.train()
            dataset = CriticDataset(self.critic_batch_size, self.obs_buf, target_values, drop_last=False)

            self.timer.start("train/update_critic")
            critic_results = self.update_critic(dataset)
            self.timer.end("train/update_critic")
            self.encoder.eval()
            self.critic.eval()

            if not self.shac_config.no_target_critic:
                # update target critic
                with torch.no_grad():
                    alpha = self.target_critic_alpha
                    soft_update(self.encoder, self.encoder_target, alpha)
                    soft_update(self.critic, self.critic_target, alpha)

            # train metrics
            results = {**actor_results, **critic_results}
            metrics = {k: torch.mean(torch.stack(v)).item() for k, v in results.items()}
            metrics.update({k: torch.mean(torch.cat(results[k]), 0).cpu().numpy() for k in ['mu', 'sigma']})  # distr
            metrics.update(values_results)
            metrics.update({"epoch": self.epoch, "lr": lr})
            if self.with_autoent:
                metrics["entropy_alpha"] = self.get_alpha(scalar=True)
            metrics = {f"train_stats/{k}": v for k, v in metrics.items()}

            # timing metrics
            timings_total_names = ("train/update_actor", "train/make_critic_dataset", "train/update_critic")
            timings = self.timer.stats(step=self.agent_steps, total_names=timings_total_names, reset=False)
            timing_metrics = {f"train_timings/{k}": v for k, v in timings.items()}
            metrics.update(timing_metrics)

            # episode metrics
            if len(self.episode_rewards_hist) > 0:
                mean_episode_rewards = self.episode_rewards_tracker.mean()
                mean_episode_lengths = self.episode_lengths_tracker.mean()
                mean_episode_discounted_rewards = self.episode_discounted_rewards_tracker.mean()

                episode_metrics = {
                    "train_scores/num_episodes": self.num_episodes.item(),
                    "train_scores/episode_rewards": mean_episode_rewards,
                    "train_scores/episode_lengths": mean_episode_lengths,
                    "train_scores/episode_discounted_rewards": mean_episode_discounted_rewards,
                }
                metrics.update(episode_metrics)
            else:
                mean_episode_rewards = -np.inf
                mean_episode_lengths = 0
                mean_episode_discounted_rewards = -np.inf

            self.writer.add(self.agent_steps, metrics)
            self.writer.write()

            self._checkpoint_save(mean_episode_rewards)

            if self.print_every > 0 and (self.epoch + 1) % self.print_every == 0:
                print(
                    f'Epochs: {self.epoch + 1} |',
                    f'Agent Steps: {int(self.agent_steps):,} |',
                    f'SPS: {timings["lastrate"]:.2f} |',  # actually totalrate since we don't reset the timer
                    f'Best: {self.best_stat if self.best_stat is not None else -float("inf"):.2f} |',
                    f'Stats:',
                    f'ep_rewards {mean_episode_rewards:.2f},',
                    f'ep_lengths {mean_episode_lengths:.2f},',
                    f'ep_discounted_rewards {mean_episode_discounted_rewards:.2f},',
                    f'value_loss {metrics["train_stats/value_loss"]:.4f},',
                    f'grad_norm_before_clip/actor {metrics["train_stats/grad_norm_before_clip/actor"]:.2f},',
                    f'grad_norm_after_clip/actor {metrics["train_stats/grad_norm_after_clip/actor"]:.2f},',
                    f'\b\b |',
                )

        timings = self.timer.stats(step=self.agent_steps)
        print(timings)

        self.save(os.path.join(self.ckpt_dir, 'final.pth'))

        # save reward/length history
        self.episode_rewards_hist = np.array(self.episode_rewards_hist)
        self.episode_lengths_hist = np.array(self.episode_lengths_hist)
        self.episode_discounted_rewards_hist = np.array(self.episode_discounted_rewards_hist)
        np.save(open(os.path.join(self.logdir, 'ep_rewards_hist.npy'), 'wb'), self.episode_rewards_hist)
        np.save(open(os.path.join(self.logdir, 'ep_lengths_hist.npy'), 'wb'), self.episode_lengths_hist)
        np.save(open(os.path.join(self.logdir, 'ep_discounted_rewards_hist.npy'), 'wb'), self.episode_discounted_rewards_hist)

    def update_actor(self):
        """Performs an actor update step."""
        results = collections.defaultdict(list)

        # zero out just in case
        with torch.no_grad():
            self.action_buf.zero_()
            self.mus.zero_()
            self.sigmas.zero_()

            if self.with_logprobs:
                self.logprobs = self.logprobs.zero_().detach()
                self.distr_ent = self.distr_ent.zero_().detach()

        def actor_closure():
            self.actor_optim.zero_grad()
            self.timer.start("train/actor_closure/actor_loss")

            self.timer.start("train/actor_closure/forward_sim")
            returns, logprobs, distr_ents = self.compute_actor_loss()
            self.timer.end("train/actor_closure/forward_sim")

            # these returns are value bootstrapped so not actually raw
            # also they may include an entropy term if self.entropy_in_return=True
            raw_returns = returns.detach().mean()
            returns = returns.view(-1, 1)

            returns /= self.horizon_len
            logprobs /= self.horizon_len
            distr_ents /= self.horizon_len

            returns = returns.squeeze(-1)
            if self.entropy_in_return or self.no_actor_entropy:  # entropy will also be discounted
                actor_loss = -returns.mean()
            elif self.with_autoent or self.entropy_coef is not None:  # here entropy is not discounted
                alpha = self.get_alpha(scalar=True) if self.with_autoent else self.entropy_coef
                entropy = distr_ents if self.use_distr_ent else -1.0 * logprobs
                actor_loss = ((alpha * -entropy) - returns).mean()
            else:
                actor_loss = -returns.mean()
            loss = actor_loss

            self.timer.start("train/actor_closure/backward_sim")
            loss.backward()
            self.timer.end("train/actor_closure/backward_sim")

            with torch.no_grad():
                if self.with_autoent:
                    self._entropy = distr_ents.detach().clone() if self.use_distr_ent else -1.0 * logprobs.detach().clone()

                # TODO: self.encoder.parameters()
                grad_norm_before_clip = grad_norm(self.actor.parameters())
                if self.shac_config.truncate_grads:
                    if self.shac_config.get("max_grad_value", None) is not None:
                        nn.utils.clip_grad_value_(self.actor.parameters(), self.shac_config.max_grad_value)
                    elif self.shac_config.max_grad_norm is not None:
                        nn.utils.clip_grad_norm_(self.actor.parameters(), self.shac_config.max_grad_norm)
                grad_norm_after_clip = grad_norm(self.actor.parameters())

                # sanity check
                if torch.isnan(grad_norm_before_clip) or grad_norm_before_clip > 1e6:
                    print('NaN gradient', grad_norm_before_clip)
                    # raise ValueError
                    raise KeyboardInterrupt

            if self.with_logprobs:
                results["entropy"].append(-1.0 * self.logprobs.mean().detach())
                results["distr_ent"].append(self.distr_ent.mean().detach())
            results["actor_loss"].append(actor_loss.detach())
            results["returns"].append(raw_returns.detach())
            results["grad_norm_before_clip/actor"].append(grad_norm_before_clip)
            results["grad_norm_after_clip/actor"].append(grad_norm_after_clip)
            self.timer.end("train/actor_closure/actor_loss")
            return actor_loss

        self.actor_optim.step(actor_closure)

        with torch.no_grad():
            obs = {k: v.view(-1, *v.shape[2:]) for k, v in self.obs_buf.items()}
            _, mu, sigma, distr = self.get_actions(obs, sample=False, dist=True)
            old_mu, old_sigma = self.mus.view(-1, self.num_actions), self.sigmas.view(-1, self.num_actions)

            # if self.with_logprobs:
            #     logprob = distr.log_prob(self.action_buf).sum(dim=-1)
            #     logprob = logprob.view(-1, 1)
            #     old_logprob = self.logprobs.view(-1, 1)
            #     # calculate approx_kl http://joschu.net/blog/kl-approx.html
            #     logratio = logprob - old_logprob
            #     kl1 = -1.0 * logratio.mean()
            #     kl2 = 0.5 * (logratio**2).mean()
            #     # kl3 = ((logratio.exp() - 1) - logratio).mean()
            #     # kl3 = ((torch.expm1(logratio) - logratio)).mean()
            #     results["actor_kl/approx_kl1"].append(kl1)
            #     results["actor_kl/approx_kl2"].append(kl2)
            #     # results["actor_kl/approx_kl3"].append(kl3)  # unstable b/c of exp

            kl_dist = policy_kl(mu.detach(), sigma.detach(), old_mu, old_sigma)
            results["mu"].append(mu)
            results["sigma"].append(sigma)
            kl_dist /= self.num_actions
            avg_kl = kl_dist.mean()
            results["avg_kl"].append(avg_kl)
            self.avg_kl = avg_kl

        if self.with_autoent:
            entropy = self._entropy
            alpha = self.get_alpha(detach=False)
            alpha_loss = (alpha * (entropy - self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            if self.shac_config.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.alpha_optim.param_groups[0]["params"], self.shac_config.max_grad_norm)
            self.alpha_optim.step()
            results["entropy_alpha_loss"].append(alpha_loss)

        return results

    def compute_actor_loss(self):
        rew_acc = torch.zeros((self.horizon_len + 1, self.num_envs), dtype=torch.float32, device=self.device)
        gamma = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
        next_values = torch.zeros((self.horizon_len + 1, self.num_envs), dtype=torch.float32, device=self.device)
        avg_next_values = torch.zeros((self.horizon_len + 1, self.num_envs), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            if self.obs_rms is not None:
                obs_rms = deepcopy(self.obs_rms)

            alpha = self.get_alpha(scalar=True) if self.with_autoent else self.entropy_coef

        # initialize trajectory to cut off gradients between episodes.
        obs = self.env.initialize_trajectory()
        obs = self._convert_obs(obs)

        if self.obs_rms is not None:
            # update obs rms
            with torch.no_grad():
                for k, v in obs.items():
                    self.obs_rms[k].update(v)
            # normalize the current obs
            obs = {k: obs_rms[k].normalize(v) for k, v in obs.items()}

        # collect trajectories and compute actor loss
        returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        logprobs = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        distr_ents = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        for i in range(self.horizon_len):
            # collect data for critic training
            with torch.no_grad():
                for k, v in obs.items():
                    self.obs_buf[k][i] = v.clone()

            # take env step
            z = self.actor_encoder(obs)
            actions, mu, sigma, distr = self.get_actions(obs, z=z, sample=True, dist=True)

            if self.with_logprobs:
                logprob = distr.log_prob(actions).sum(dim=-1)
                distr_ent = distr.entropy().sum(dim=-1)
            with torch.no_grad():
                self.action_buf[i] = actions.clone()
                self.mus[i, ...] = mu.clone()
                self.sigmas[i, ...] = sigma.clone()

            obs, rew, done, extra_info = self.env.step(actions)
            obs = self._convert_obs(obs)

            with torch.no_grad():
                raw_rew = rew.clone()
            # scale the reward
            rew = self.reward_shaper(rew)

            # update episode metrics
            with torch.no_grad():
                self.episode_rewards += raw_rew
                self.episode_lengths += 1
                self.episode_discounted_rewards += self.episode_gamma * raw_rew
                self.episode_gamma *= self.gamma

            if self.obs_rms is not None:
                # update obs rms
                with torch.no_grad():
                    for k, v in obs.items():
                        self.obs_rms[k].update(v)
                # normalize the current obs
                obs = {k: obs_rms[k].normalize(v) for k, v in obs.items()}

            # value bootstrap when episode terminates
            if self.share_encoder:
                z_target = z
            else:
                z_target = self.encoder_target(obs)
            pred_val, avg_pred_val = self.critic_target(z_target, return_type="min_and_avg")
            pred_val, avg_pred_val = pred_val.squeeze(-1), avg_pred_val.squeeze(-1)
            next_values[i + 1] = pred_val
            avg_next_values[i + 1] = avg_pred_val

            done_env_ids = done.nonzero(as_tuple=False).squeeze(-1)
            if len(done_env_ids) > 0:
                terminal_obs = extra_info['obs_before_reset']
                terminal_obs = self._convert_obs(terminal_obs)

                for env_id in done_env_ids:
                    self.get_next_values(env_id)  # TODO: deprecated

            if (next_values[i + 1] > 1e6).sum() > 0 or (next_values[i + 1] < -1e6).sum() > 0:
                print('next value error')
                raise ValueError
            if (avg_next_values[i + 1] > 1e6).sum() > 0 or (avg_next_values[i + 1] < -1e6).sum() > 0:
                print('avg next value error')
                raise ValueError

            # https://github.com/ikostrikov/rlpd/blob/c90fd4baf28c9c9ef40a81460a2e395092844f88/rlpd/agents/sac/sac_learner.py#L169
            next_vs = avg_next_values if self.actor_loss_avgcritics else next_values

            # compute actor loss
            if self.entropy_in_return:
                # operations to entropy should be out of place since cloning them further below
                entropy = distr_ent if self.use_distr_ent else -1.0 * logprob
                entropy = entropy.clone()
                rew_acc[i + 1, :] = rew_acc[i, :] + gamma * (rew + alpha * entropy)
            else:
                rew_acc[i + 1, :] = rew_acc[i, :] + gamma * rew
            if i < self.horizon_len - 1:
                rets = rew_acc[i + 1, done_env_ids] + self.gamma * gamma[done_env_ids] * next_vs[i + 1, done_env_ids]
                returns[done_env_ids] += rets
            else:
                # terminate all envs at the end of optimization iteration
                rets = rew_acc[i + 1, :] + self.gamma * gamma * next_vs[i + 1, :]
                returns += rets

            if self.with_logprobs:
                logprobs += logprob
                distr_ents += distr_ent

            # compute gamma for next step
            gamma = gamma * self.gamma

            # clear up gamma and rew_acc for done envs
            gamma[done_env_ids] = 1.0
            rew_acc[i + 1, done_env_ids] = 0.0

            # collect data for critic training
            with torch.no_grad():
                self.rew_buf[i] = rew.clone()
                if i < self.horizon_len - 1:
                    self.done_mask[i] = done.clone().to(dtype=torch.float32)
                else:
                    self.done_mask[i, :] = 1.0
                self.next_values[i] = next_values[i + 1].clone()  # this is min of critics ensemble
                self.avg_next_values[i] = avg_next_values[i + 1].clone()

                if self.with_logprobs:
                    self.logprobs[i, ...] = logprob.clone()
                    self.distr_ent[i, ...] = distr_ent.clone()

            # collect episode metrics
            with torch.no_grad():
                if len(done_env_ids) > 0:
                    done_env_ids = done_env_ids.detach().cpu()
                    self.episode_rewards_tracker.update(self.episode_rewards[done_env_ids])
                    self.episode_lengths_tracker.update(self.episode_lengths[done_env_ids])
                    self.episode_discounted_rewards_tracker.update(self.episode_discounted_rewards[done_env_ids])
                    self.num_episodes += len(done_env_ids)

                    for done_env_id in done_env_ids:
                        if self.episode_rewards[done_env_id] > 1e6 or self.episode_rewards[done_env_id] < -1e6:
                            print('ep_rewards error')
                            raise ValueError
                        self.episode_rewards_hist.append(self.episode_rewards[done_env_id].item())
                        self.episode_lengths_hist.append(self.episode_lengths[done_env_id].item())
                        self.episode_discounted_rewards_hist.append(self.episode_discounted_rewards[done_env_id].item())
                        self.episode_rewards[done_env_id] = 0.0
                        self.episode_lengths[done_env_id] = 0
                        self.episode_discounted_rewards[done_env_id] = 0.0
                        self.episode_gamma[done_env_id] = 1.0

        self.agent_steps += self.horizon_len * self.num_envs
        return returns, logprobs, distr_ents

    def get_next_values(self, trajectory: RolloutTrajectory):
        """Get the next values for states si for all observations in the trajectory."""
        if False:  # TODO: This should check for early termination. In this case the next value is 0, not the critic value
            values = 0.0
            avg_values = 0.0
        else:  # Use terminal value critic to estimate the long-term performance
            obs = trajectory.observations_tensor
            if self.obs_rms is not None:
                obs = obs  # TODO: normalize obs
            z_target = self.encoder_target(obs)
            values, avg_values = self.critic_target(z_target, return_type="min_and_avg")
        return values.squeeze(-1), avg_values.squeeze(-1)

    def update_critic(self, dataset):
        results = collections.defaultdict(list)
        j = 0
        while j < self.critic_iterations:
            total_critic_loss = 0.0
            grad_norms_before_clip = []
            grad_norms_after_clip = []
            B = len(dataset)
            for i in range(B):
                batch_sample = dataset[i]
                b_obs, b_target_values = batch_sample

                self.critic_optim.zero_grad()
                critic_loss = self.compute_critic_loss(b_obs, b_target_values)
                critic_loss.backward()

                # ugly fix for simulation nan problem
                for params in self.critic.parameters():
                    params.grad.nan_to_num_(0.0, 0.0, 0.0)

                if self.shac_config.truncate_grads:
                    # TODO: self.encoder.parameters()
                    grad_norm_before_clip = grad_norm(self.critic.parameters())
                    grad_norms_before_clip.append(grad_norm_before_clip)
                    if self.shac_config.get("max_grad_value", None) is not None:
                        nn.utils.clip_grad_value_(self.critic.parameters(), self.shac_config.max_grad_value)
                    elif self.shac_config.max_grad_norm is not None:
                        nn.utils.clip_grad_norm_(self.critic.parameters(), self.shac_config.max_grad_norm)
                    grad_norm_after_clip = grad_norm(self.critic.parameters())
                    grad_norms_after_clip.append(grad_norm_after_clip)

                self.critic_optim.step()
                total_critic_loss += critic_loss
            j += 1
            value_loss = (total_critic_loss / B).detach()
            results["value_loss"].append(value_loss)
            results["grad_norm_before_clip/critic"].append(torch.mean(torch.stack(grad_norms_before_clip)))
            results["grad_norm_after_clip/critic"].append(torch.mean(torch.stack(grad_norms_after_clip)))

        return results

    def compute_critic_loss(self, obs, target_v, mean=True, reduction='mean'):
        z = self.encoder(obs)
        pred_vs = self.critic(z, return_type='all')
        critic_loss = torch.stack([F.mse_loss(pred_v.squeeze(-1), target_v, reduction=reduction) for pred_v in pred_vs])
        if mean:
            critic_loss = critic_loss.mean()
        return critic_loss

    def compute_target_values(self,
                              trajectory: RolloutTrajectory,
                              next_values: torch.Tensor,
                              alpha: Optional[float] = None,
                              entropy: Optional[torch.Tensor] = None):
        """
        Adapted td-lambda target value computation according to SAPO (20). (soft value-bootstrapped k returns).

        Instead of a common td-lambda target value, we compute the target value as a weighted sum of the traditional
        td-lambda and a monte-carlo estimate of the return. The Monte-Carlo estimate has a high variance, esp. in
        case of sparse rewards, whereas a td-estimate is biased. The closer we are to the end of the episode (a
        potential terminal reward), the more we rely on the monte carlo estimate (this means a high lam).

        :param alpha:
        :param entropy:
        :return:
        """
        self._init_target_values(trajectory)
        entropy = entropy if entropy is not None else torch.zeros(len(trajectory), self.num_envs, dtype=torch.float32,
                                                                  device=self.device)
        alpha = alpha if alpha is not None else 0.
        rewards = trajectory.rewards_tensor
        dones = (trajectory.terminated_tensor | trajectory.truncated_tensor).to(torch.int8)
        if self.critic_method == 'one-step':
            self.target_values = (rewards + alpha * entropy) + self.gamma * next_values
        elif self.critic_method == 'td-lambda':
            td_bootstrapped = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            monte_carlo_estimates = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            lam = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
            for i in reversed(range(len(trajectory))):
                lam = lam * self.lam * (1.0 - dones[i]) + dones[i]
                reward_w_entropy = rewards[i] + alpha * entropy[i]
                adjusted_rew = (1.0 - lam) / (1.0 - self.lam) * reward_w_entropy
                td_bootstrapped = (1.0 - dones[i]) * (self.lam * self.gamma * td_bootstrapped + self.gamma * next_values[i] + adjusted_rew)
                monte_carlo_estimates = self.gamma * (next_values[i] * dones[i] + monte_carlo_estimates * (1.0 - dones[i])) + reward_w_entropy
                self.target_values[i] = (1.0 - self.lam) * td_bootstrapped + lam * monte_carlo_estimates
        else:
            raise NotImplementedError(self.critic_method)

    def get_alpha(self, detach=True, scalar=False):
        if self.shac_config.get("alpha", None) is None:
            alpha = self.log_alpha.exp()
            if detach:
                alpha = alpha.detach()
            if scalar:
                alpha = alpha.item()
        else:
            alpha = self.shac_config.alpha
        return alpha

    def eval(self):
        self.set_eval()

        episode_rewards, episode_lengths, episode_discounted_rewards = self.evaluate_policy(
            num_episodes=self.num_actors * 2, sample=True
        )

        metrics = {
            "eval_scores/num_episodes": len(episode_rewards),
            "eval_scores/episode_rewards": np.mean(np.array(episode_rewards)),
            "eval_scores/episode_lengths": np.mean(np.array(episode_lengths)),
            "eval_scores/episode_discounted_rewards": np.mean(np.array(episode_discounted_rewards)),
        }
        print(metrics)

        self.writer.add(self.agent_steps, metrics)
        self.writer.write()

        scores = {
            "epoch": self.epoch,
            "mini_epoch": self.mini_epoch,
            "agent_steps": self.agent_steps,
            "eval_scores/num_episodes": len(episode_rewards),
            "eval_scores/episode_rewards": episode_rewards,
            "eval_scores/episode_lengths": episode_lengths,
            "eval_scores/episode_discounted_rewards": episode_discounted_rewards,
        }
        json.dump(scores, open(os.path.join(self.logdir, "scores.json"), "w"), indent=4)

    def set_train(self):
        pass

    def set_eval(self):
        self.actor_encoder.eval()
        self.encoder.eval()
        self.actor.eval()
        self.critic.eval()
        self.encoder_target.eval()
        self.critic_target.eval()

    def save(self, f):
        ckpt = {
            'epoch': self.epoch,
            'mini_epoch': self.mini_epoch,
            'agent_steps': self.agent_steps,
            'obs_rms': self.obs_rms.state_dict() if self.normalize_input else None,
            'actor_encoder': self.actor_encoder.state_dict() if not self.share_encoder else None,
            'encoder': self.encoder.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'encoder_target': self.encoder_target.state_dict() if not self.shac_config.no_target_critic else None,
            'critic_target': self.critic_target.state_dict() if not self.shac_config.no_target_critic else None,
        }
        torch.save(ckpt, f)

    def load(self, f, ckpt_keys=''):
        all_ckpt_keys = ('epoch', 'mini_epoch', 'agent_steps')
        all_ckpt_keys += ('obs_rms', 'actor_encoder', 'encoder', 'actor', 'critic')
        all_ckpt_keys += ('encoder_target', 'critic_target')
        ckpt = torch.load(f, map_location=self.device)
        for k in all_ckpt_keys:
            if not re.match(ckpt_keys, k):
                print(f'Warning: ckpt skipped loading `{k}`')
                continue
            if k == 'obs_rms' and (not self.normalize_input):
                continue
            if k == 'actor_encoder' and (self.share_encoder):
                continue
            if k == 'encoder_target' and (self.shac_config.no_target_critic):
                continue
            if k == 'critic_target' and (self.shac_config.no_target_critic):
                continue

            if hasattr(getattr(self, k), 'load_state_dict'):
                getattr(self, k).load_state_dict(ckpt[k])
            else:
                setattr(self, k, ckpt[k])
