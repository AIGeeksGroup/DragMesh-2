"""
pica/a2c_agent.py - PICA v2b A2CAgent subclass.

Wraps rl_games' continuous A2CAgent and adds an auxiliary physical loss
sourced from `self.model.a2c_network.last_aux_loss`. The aux head and the
GLA temporal encoder are both trained by the joint backward pass, which is
the explicit goal of PICA v2b: teach the encoder to predict observable
physical-response signals (object joint delta, slip proxy, tracking
stress) from the history token stream.

Aux weight is scheduled by a linear warmup:
    epoch < start_epoch  -> 0
    epoch >= end_epoch   -> max_weight
    linear in between

Per-epoch aux components are accumulated across mini-epochs and printed
once at the end of every epoch via update_epoch(). They are also written
to runs/<exp>/aux_log.csv when the runs dir is available.

Registration: call `register_pica_a2c_agent(runner)` once in train.py
before `runner.load(train_config)`.
"""

from __future__ import annotations

import csv
import os

import torch
import torch.nn as nn

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.a2c_continuous import A2CAgent
from rl_games.common import common_losses


class PICAA2CAgent(A2CAgent):
    """A2CAgent + auxiliary physical-target loss with linear warmup."""

    def __init__(self, base_name, config):
        super().__init__(base_name, config)

        cfg = config.get("physical_auxiliary", {}) or {}
        self._aux_enabled = bool(cfg.get("enabled", False))
        warmup = cfg.get("warmup", {}) or {}
        self._aux_warmup_enabled = bool(warmup.get("enabled", False))
        self._aux_max_weight = float(warmup.get("max_weight", 0.01))
        self._aux_start_epoch = int(warmup.get("start_epoch", 10))
        self._aux_end_epoch = int(warmup.get("end_epoch", 40))

        # Per-minibatch accumulators reset at the end of every epoch. Keys
        # other than {"total", "n"} are populated dynamically based on which
        # component names the network reports in `last_aux_loss_components`.
        # This makes the logger backbone-agnostic between v2b ("current"
        # mode: dq/slip/track) and v2c ("causal_horizon" mode:
        # q_response_K/max_dist_K/detach_proxy_K/tracking_stress).
        self._aux_accum = {"total": 0.0, "n": 0}

        # Optional CSV log next to the existing epoch_rewards.csv. We use a
        # relative path because rl_games' agent doesn't expose its run dir
        # in a stable way; the cwd is the project root for our train.py.
        exp_name = (
            config.get("full_experiment_name")
            or config.get("name")
            or "HandDrag"
        )
        self._aux_csv_path = os.path.join("runs", exp_name, "aux_log.csv")
        self._aux_csv_header_written = False

        # Extra checkpoint safeguards for short fine-tuning runs. rl_games
        # already saves reward-best to ``HandDrag.pth``; these two keep the
        # best task-facing metrics from ``epoch_rewards.csv`` so early good
        # policies are not lost when reward/progress/success disagree.
        self._metric_best = {
            "success_mean": float("-inf"),
            "normalized_progress_mean": float("-inf"),
        }
        if self._aux_enabled:
            print(
                f"  [pica-a2c] aux loss enabled  warmup=[{self._aux_start_epoch}, "
                f"{self._aux_end_epoch}]  max_weight={self._aux_max_weight}"
            )

    # ------------------------------------------------------------------
    # Warmup schedule
    # ------------------------------------------------------------------
    def _current_aux_weight(self) -> float:
        if not self._aux_enabled:
            return 0.0
        if not self._aux_warmup_enabled:
            return self._aux_max_weight
        e = int(self.epoch_num)
        if e < self._aux_start_epoch:
            return 0.0
        if e >= self._aux_end_epoch:
            return self._aux_max_weight
        span = max(1, (self._aux_end_epoch - self._aux_start_epoch))
        frac = (e - self._aux_start_epoch) / span
        return frac * self._aux_max_weight

    # ------------------------------------------------------------------
    # Per-minibatch loss (mirrors A2CAgent.calc_gradients with one insertion)
    # ------------------------------------------------------------------
    def calc_gradients(self, input_dict):
        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        obs_batch = self._preproc_obs(obs_batch)

        lr = self.last_lr  # noqa: F841 -- kept for parity with rl_games
        kl = 1.0           # noqa: F841
        lr_mul = 1.0
        curr_e_clip = lr_mul * self.e_clip

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch,
            'obs': obs_batch,
        }

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = self.seq_len

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']

            a_loss = common_losses.actor_loss(
                old_action_log_probs_batch, action_log_probs,
                advantage, self.ppo, curr_e_clip,
            )

            if self.has_value_loss:
                c_loss = common_losses.critic_loss(
                    value_preds_batch, values, curr_e_clip,
                    return_batch, self.clip_value,
                )
            else:
                c_loss = torch.zeros(1, device=self.ppo_device)

            b_loss = self.bound_loss(mu)
            losses, sum_mask = torch_ext.apply_masks(
                [a_loss.unsqueeze(1), c_loss, entropy.unsqueeze(1), b_loss.unsqueeze(1)],
                rnn_masks,
            )
            a_loss, c_loss, entropy, b_loss = losses[0], losses[1], losses[2], losses[3]

            loss = (
                a_loss
                + 0.5 * c_loss * self.critic_coef
                - entropy * self.entropy_coef
                + b_loss * self.bounds_loss_coef
            )

            # ---- PICA v2b/v2c: aux loss insertion ----
            aux_w = self._current_aux_weight()
            aux_loss_t = getattr(self.model.a2c_network, "last_aux_loss", None)
            if aux_loss_t is not None and self._aux_enabled:
                # NOTE: aux_w can be 0 (during warmup); the multiplication is
                # still performed so the autograd graph is consistent across
                # epochs. Backward will simply produce zero gradients in
                # that case, which is what we want.
                loss = loss + aux_w * aux_loss_t
                # Per-minibatch accumulation for end-of-epoch logging. Keys
                # are picked up from whatever the network reports.
                self._aux_accum["total"] += float(aux_loss_t.detach().item())
                self._aux_accum["n"] += 1
                comps = getattr(
                    self.model.a2c_network, "last_aux_loss_components", {}
                ) or {}
                for ck, cv in comps.items():
                    if cv is None:
                        continue
                    sum_key = "comp_" + ck
                    cnt_key = "n_" + ck
                    self._aux_accum[sum_key] = self._aux_accum.get(sum_key, 0.0) + float(cv.item())
                    self._aux_accum[cnt_key] = self._aux_accum.get(cnt_key, 0) + 1

            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        if self.truncate_grads:
            if self.multi_gpu:
                self.optimizer.synchronize()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                with self.optimizer.skip_synchronize():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:
            self.scaler.step(self.optimizer)
            self.scaler.update()

        with torch.no_grad():
            reduce_kl = not self.is_rnn
            kl_dist = torch_ext.policy_kl(
                mu.detach(), sigma.detach(),
                old_mu_batch, old_sigma_batch, reduce_kl,
            )
            if self.is_rnn:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()

        self.train_result = (
            a_loss, c_loss, entropy, kl_dist, self.last_lr, lr_mul,
            mu.detach(), sigma.detach(), b_loss,
        )

    # ------------------------------------------------------------------
    # Extra task-metric checkpointing
    # ------------------------------------------------------------------
    def write_stats(self, *args, **kwargs):
        result = super().write_stats(*args, **kwargs)
        epoch = int(args[1]) if len(args) > 1 else int(self.epoch_num)
        self._save_task_metric_bests(epoch)
        return result

    def _save_task_metric_bests(self, epoch):
        if getattr(self, "rank", 0) != 0:
            return
        try:
            run_dir = os.path.dirname(self._aux_csv_path)
            epoch_csv = os.path.join(run_dir, "epoch_rewards.csv")
            if not os.path.exists(epoch_csv) or os.path.getsize(epoch_csv) == 0:
                return
            with open(epoch_csv, newline="") as f:
                rows = list(csv.DictReader(f))
            if not rows:
                return
            row = rows[-1]

            metric_specs = [
                ("success_mean", "best_success"),
                ("normalized_progress_mean", "best_progress"),
            ]
            for metric, suffix in metric_specs:
                raw = row.get(metric, "")
                if raw in ("", None):
                    continue
                value = float(raw)
                if value > self._metric_best.get(metric, float("-inf")):
                    self._metric_best[metric] = value
                    stable = os.path.join(
                        self.nn_dir, f"{self.config['name']}_{suffix}"
                    )
                    tagged = os.path.join(
                        self.nn_dir,
                        f"{self.config['name']}_{suffix}_ep{epoch:04d}_val{value:.6g}",
                    )
                    print(
                        f"saving task metric best {metric}: {value:.6g} "
                        f"-> {stable}"
                    )
                    self.save(stable)
                    self.save(tagged)
        except Exception as exc:
            print(f"  [pica-save][WARN] failed to save task metric best: {exc}")
    # ------------------------------------------------------------------
    # End-of-epoch aux logging
    # ------------------------------------------------------------------
    def update_epoch(self):
        # Print stats for the just-finished epoch BEFORE the counter is
        # incremented, so the printed epoch number matches the reported data.
        if self._aux_enabled and self._aux_accum["n"] > 0:
            n  = max(1, self._aux_accum["n"])
            tot = self._aux_accum["total"] / n
            w   = self._current_aux_weight()

            # Discover component keys dynamically (anything stored as comp_*).
            comp_keys = sorted(
                k[len("comp_"):] for k in self._aux_accum if k.startswith("comp_")
            )
            comp_strs = []
            comp_means = {}
            for ck in comp_keys:
                cn = max(1, self._aux_accum.get("n_" + ck, 0))
                cmean = self._aux_accum["comp_" + ck] / cn
                comp_means[ck] = cmean
                comp_strs.append(f"aux_loss_{ck}={cmean:.6f}")

            print(
                f"  [aux] epoch={int(self.epoch_num)} "
                f"aux_loss_total={tot:.6f} "
                + " ".join(comp_strs)
                + f" aux_weight={w:.4f}"
            )
            self._aux_log_csv(int(self.epoch_num), n, tot, comp_means, w)

        # Reset accumulators for the next epoch (keep the layout flexible).
        self._aux_accum = {"total": 0.0, "n": 0}
        return super().update_epoch()

    def _aux_log_csv(self, epoch, n, total, comp_means, weight):
        """Append-only CSV with component columns derived from comp_means.

        First write fixes the schema (column set is whatever components the
        network reported in epoch 0). If the schema later differs, the new
        columns are appended on the right and missing ones written as empty
        strings, which keeps the file usable by pandas / the postprocessor.
        Logging failures must never break training.
        """
        try:
            os.makedirs(os.path.dirname(self._aux_csv_path), exist_ok=True)
            need_header = (
                not os.path.exists(self._aux_csv_path)
                or os.path.getsize(self._aux_csv_path) == 0
            )
            comp_keys = sorted(comp_means.keys())
            if need_header:
                self._aux_csv_columns = list(comp_keys)
            else:
                # Lock to whatever the file already has. Read header once.
                if not getattr(self, "_aux_csv_columns_loaded", False):
                    try:
                        with open(self._aux_csv_path) as f:
                            existing = f.readline().strip().split(",")
                        # Strip the fixed prefix to recover component keys.
                        prefix = ["epoch", "n_minibatches", "aux_loss_total"]
                        suffix = ["aux_weight"]
                        cols = existing[len(prefix):-len(suffix)]
                        self._aux_csv_columns = [c[len("aux_loss_"):]
                                                 if c.startswith("aux_loss_") else c
                                                 for c in cols]
                    except Exception:
                        self._aux_csv_columns = list(comp_keys)
                    self._aux_csv_columns_loaded = True

            with open(self._aux_csv_path, "a") as f:
                if need_header:
                    cols = ",".join(f"aux_loss_{k}" for k in self._aux_csv_columns)
                    f.write(
                        "epoch,n_minibatches,aux_loss_total,"
                        + cols + ",aux_weight\n"
                    )
                row_vals = []
                for k in self._aux_csv_columns:
                    if k in comp_means:
                        row_vals.append(f"{comp_means[k]:.6g}")
                    else:
                        row_vals.append("")  # fill missing keys with empty
                f.write(
                    f"{epoch},{n},{total:.6g},"
                    + ",".join(row_vals)
                    + f",{weight:.6g}\n"
                )
        except Exception:
            pass


def register_pica_a2c_agent(runner):
    """Register PICAA2CAgent under the YAML algo name `pica_a2c_continuous`.

    Call once before `runner.load(train_config)` in train.py.
    """
    runner.algo_factory.register_builder(
        "pica_a2c_continuous",
        lambda **kw: PICAA2CAgent(**kw),
    )
