"""
gla_a2c_network.py -- Phase 4 GLA policy network for the hand-drag PPO task.

Architecture (separate=False, single shared backbone)
-----------------------------------------------------
                       obs  [B, obs_dim]
                        |
            +-----------+-----------+
            |                       |
    base_obs [B, base_dim]    history [B, 16, 102]
            |                       |
       MLP backbone           Linear (102 -> hidden)
            |                       |
       base_feat              GatedLinearAttention
            |                  [B, 16, hidden]
            |                       |
            |             pool (last | mean)
            |                       |
            |                  temporal_feat [B, hidden]
            +-----------+-----------+
                        |
                fused_feat [B, base_units[-1] + hidden]
                        |
              +---------+---------+
              |                   |
        actor head           value head
        (mu / logstd)         (V(s))

The temporal token at step t is `[error_t (51), a_{t-1} (51)]` (102-D),
flattened in C order at the tail of the observation by `HandDragTask`.
The network slices this off, projects to `hidden`, runs GLA over the 16
tokens, and fuses the resulting compliance summary with a flat-MLP encoding
of the base proprioceptive observation. This is the architectural
"hypothesis" row of the paper ablation.

Implementation notes
--------------------
* We register this builder under the name `gla_actor_critic`. Set
  `params.network.name: gla_actor_critic` in the train config to use it.
* `fixed_sigma: true` is honoured exactly the way the stock A2CBuilder
  does (a single learnable logstd vector), so the rest of the rl_games
  pipeline is unchanged.
* The GLA `mode` is exposed for completeness, but `fla` will internally
  fall back to `fused_recurrent` for sequence length <=64. With
  `history_length=16` we always hit the recurrent kernel; this is fine
  for our setting.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

# Make sure a local checkout of flash-linear-attention is importable when
# present. Users can also set FLA_ROOT to point to an external checkout.
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
_FLA_ROOT = os.environ.get(
    "FLA_ROOT",
    os.path.join(_PROJECT_ROOT, "flash-linear-attention"),
)
if _FLA_ROOT not in sys.path:
    sys.path.insert(0, _FLA_ROOT)

from fla.layers import GatedLinearAttention  # noqa: E402

from rl_games.algos_torch.network_builder import NetworkBuilder  # noqa: E402


class GLAA2CBuilder(NetworkBuilder):
    """rl_games NetworkBuilder for the GLA temporal-encoder policy."""

    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        return GLAA2CBuilder.Network(self.params, **kwargs)

    # =================================================================
    #  Network module
    # =================================================================
    class Network(NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.pop("actions_num")
            input_shape = kwargs.pop("input_shape")
            self.value_size = kwargs.pop("value_size", 1)
            self.num_seqs = kwargs.pop("num_seqs", 1)
            NetworkBuilder.BaseNetwork.__init__(self)
            self._load_params(params)

            assert len(input_shape) == 1, (
                f"GLAA2C expects flat observations, got input_shape={input_shape}"
            )
            obs_dim = int(input_shape[0])
            history_size = self.history_length * self.token_dim
            # PICA v2b: when an auxiliary head is enabled, the env appends
            # `aux_target_dim` regression targets to the obs tail. They are
            # transport-only -- sliced off here before any actor / GLA path
            # touches the obs.
            aux_size = self.aux_target_dim if self.aux_enabled else 0
            assert obs_dim > history_size + aux_size, (
                f"obs_dim={obs_dim} must be larger than history block "
                f"history_length*token_dim={history_size} + aux_target_dim={aux_size}"
            )
            self.obs_dim = obs_dim
            self.aux_size = aux_size
            self.base_dim = obs_dim - history_size - aux_size

            # ---- Base-obs MLP backbone (the flat-MLP "control" path) ----
            mlp_args = {
                "input_size": self.base_dim,
                "units": self.units,
                "activation": self.activation,
                "norm_func_name": self.normalization,
                "dense_func": nn.Linear,
                "d2rl": self.is_d2rl,
                "norm_only_first_layer": self.norm_only_first_layer,
            }
            self.actor_mlp = self._build_mlp(**mlp_args)
            base_out = self.units[-1] if len(self.units) > 0 else self.base_dim

            # ---- GLA temporal encoder over the [B, 16, 102] history ----
            self.token_proj = nn.Linear(self.token_dim, self.gla_hidden)
            self.gla = GatedLinearAttention(
                mode=self.gla_mode,
                hidden_size=self.gla_hidden,
                num_heads=self.gla_heads,
                num_kv_heads=self.gla_kv_heads,
                expand_k=self.gla_expand_k,
                expand_v=self.gla_expand_v,
                use_short_conv=False,
                use_output_gate=True,
                fuse_norm=False,
            )
            # Optional layer norm after pooling -- stabilises early training
            # when the temporal feature distribution is far from N(0, 1).
            self.temporal_norm = nn.LayerNorm(self.gla_hidden)

            fused_dim = base_out + self.gla_hidden
            out_size = fused_dim

            # ---- Heads (mirrors A2CBuilder.continuous fixed_sigma path) ----
            self.value = nn.Linear(out_size, self.value_size)
            self.value_act = self.activations_factory.create(self.value_activation)

            assert self.is_continuous, (
                "GLAA2C only supports the continuous action space currently "
                "(continuous_a2c_logstd model)."
            )
            self.mu = nn.Linear(out_size, actions_num)
            self.mu_act = self.activations_factory.create(
                self.space_config["mu_activation"]
            )
            mu_init = self.init_factory.create(**self.space_config["mu_init"])
            self.sigma_act = self.activations_factory.create(
                self.space_config["sigma_activation"]
            )
            sigma_init = self.init_factory.create(**self.space_config["sigma_init"])

            if self.space_config["fixed_sigma"]:
                self.sigma = nn.Parameter(
                    torch.zeros(actions_num, dtype=torch.float32),
                    requires_grad=True,
                )
            else:
                self.sigma = nn.Linear(out_size, actions_num)

            # ---- PICA v2b: optional auxiliary prediction head ----
            # Reads only the temporal_feat (the GLA-pooled summary), so
            # gradients from aux loss flow through the aux head AND back
            # through the temporal encoder -- the explicit goal of v2b.
            # Predictions live in the SAME (RunningMeanStd-normalized) space
            # as the targets, since rl_games normalizes the entire obs vector
            # before the network sees it. Net effect: MSE in z-space; the
            # learned representation still captures the right correlations.
            if self.aux_enabled and self.aux_pred_dim > 0:
                self.aux_head = nn.Sequential(
                    nn.Linear(self.gla_hidden, self.aux_hidden),
                    nn.ELU(),
                    nn.Linear(self.aux_hidden, self.aux_pred_dim),
                )
            else:
                self.aux_head = None
            self.last_aux_loss = None
            # Components dict is keyed by target name (filled in forward).
            self.last_aux_loss_components = {}
            # Per-channel weights buffer; registered so .device follows the
            # rest of the network. Default is all-ones if no weights configured.
            if self.aux_enabled and self.aux_pred_dim > 0:
                w = torch.tensor(
                    self.aux_target_weights, dtype=torch.float32,
                )
                self.register_buffer("_aux_weights_buf", w, persistent=False)
            else:
                # Allocate a 1-element buffer so attribute access is safe.
                self.register_buffer(
                    "_aux_weights_buf",
                    torch.zeros(1, dtype=torch.float32),
                    persistent=False,
                )

            mlp_init = self.init_factory.create(**self.initializer)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        nn.init.zeros_(m.bias)
            mu_init(self.mu.weight)
            if self.space_config["fixed_sigma"]:
                sigma_init(self.sigma)
            else:
                sigma_init(self.sigma.weight)

            print(
                f"  [gla] obs_dim={obs_dim} base_dim={self.base_dim} "
                f"history={self.history_length}x{self.token_dim} "
                f"hidden={self.gla_hidden} heads={self.gla_heads} "
                f"pool={self.pool} mode={self.gla_mode} "
                f"aux_enabled={self.aux_enabled} aux_pred_dim={self.aux_pred_dim} "
                f"aux_target_dim={self.aux_target_dim} "
                f"aux_mode={getattr(self, 'aux_mode', 'current')} "
                f"aux_target_keys={self.aux_target_keys}"
            )

        # -------------------------------------------------------------
        #  Forward
        # -------------------------------------------------------------
        def forward(self, obs_dict):
            obs = obs_dict["obs"]                                    # [B, obs_dim]
            states = obs_dict.get("rnn_states", None)
            is_train = obs_dict.get("is_train", True)
            B = obs.shape[0]

            # ---- PICA v2b: peel off aux-target tail BEFORE the actor split.
            # The actor / critic / GLA history operate on obs_main only.
            if self.aux_enabled and self.aux_size > 0:
                aux_targets = obs[:, -self.aux_size:]                # [B, aux_size]
                obs_main = obs[:, :-self.aux_size]
            else:
                aux_targets = None
                obs_main = obs

            base_obs = obs_main[:, : self.base_dim]                  # [B, base_dim]
            history_flat = obs_main[:, self.base_dim:]               # [B, 16*102]
            history_tokens = history_flat.view(
                B, self.history_length, self.token_dim,
            )                                                        # [B, 16, 102]

            # ---- Base path: flat MLP over proprioceptive obs ----
            base_feat = self.actor_mlp(base_obs)                     # [B, base_out]

            # ---- Temporal path: token projection -> GLA -> pool ----
            tokens = self.token_proj(history_tokens)                 # [B, 16, hidden]
            gla_out, _, _ = self.gla(tokens)                         # [B, 16, hidden]
            if self.pool == "mean":
                temporal_feat = gla_out.mean(dim=1)
            else:                                                    # "last"
                temporal_feat = gla_out[:, -1, :]
            temporal_feat = self.temporal_norm(temporal_feat)        # [B, hidden]

            # ---- Fuse and decode actor / critic ----
            fused = torch.cat([base_feat, temporal_feat], dim=-1)
            value = self.value_act(self.value(fused))

            mu = self.mu_act(self.mu(fused))
            if self.space_config["fixed_sigma"]:
                sigma = mu * 0.0 + self.sigma_act(self.sigma)
            else:
                sigma = self.sigma_act(self.sigma(fused))

            # ---- PICA v2b/v2c: aux head + per-channel weighted MSE ----
            # The custom A2CAgent reads `self.last_aux_loss` after each
            # forward and adds aux_weight * loss to the total PPO loss.
            # Per-channel components are keyed by target name so logging
            # works uniformly across v2b ("current" mode: dq/slip/track) and
            # v2c ("causal_horizon" mode: q_response/max_dist/detach/tracking).
            #
            # All channels use MSE in v2c-MVP. The user spec listed BCE as
            # "preferred" for the binary detach_proxy_K channel, but rl_games'
            # RunningMeanStd normalizes the entire obs vector before the
            # network sees it, so the binary target is z-scored by the time
            # we'd compute BCE. MSE on the normalized target is mathematically
            # consistent; BCE is not. Documented as a v2c.1 follow-up.
            if (
                self.aux_enabled
                and self.aux_head is not None
                and is_train
                and aux_targets is not None
            ):
                aux_pred = self.aux_head(temporal_feat)              # [B, P]
                target_used = aux_targets[:, : self.aux_pred_dim]    # [B, P]
                per_ch_mse = (aux_pred - target_used).pow(2).mean(dim=0)  # [P]

                # Weighted sum gives the scalar passed to PPO loss.
                # Buffer is registered so .device is correct without explicit
                # transfer; falls back to ones if no weights configured.
                weights_t = self._aux_weights_buf  # [P]
                self.last_aux_loss = (per_ch_mse * weights_t).sum()

                # Components dict keyed by target name (one entry per output).
                comps = {}
                for i, k in enumerate(self.aux_target_keys):
                    comps[k] = per_ch_mse[i].detach()
                self.last_aux_loss_components = comps
            else:
                self.last_aux_loss = torch.zeros((), device=obs.device)
                # Empty when aux is disabled; the agent handles missing keys.
                self.last_aux_loss_components = {}

            return mu, sigma, value, states

        # -------------------------------------------------------------
        #  rl_games NetworkBuilder.BaseNetwork hooks
        # -------------------------------------------------------------
        def is_separate_critic(self):
            return False

        def is_rnn(self):
            # The temporal encoder lives inside the network and consumes a
            # fixed-length history slice from the observation, so rl_games
            # itself does not need to manage RNN state.
            return False

        def get_default_rnn_state(self):
            return None

        # -------------------------------------------------------------
        #  Param parsing (mirrors A2CBuilder.Network.load semantics)
        # -------------------------------------------------------------
        def _load_params(self, params):
            mlp = params.get("mlp", {})
            self.units = mlp.get("units", [256, 256])
            self.activation = mlp.get("activation", "elu")
            self.initializer = mlp.get("initializer", {"name": "default"})
            self.is_d2rl = mlp.get("d2rl", False)
            self.norm_only_first_layer = mlp.get("norm_only_first_layer", False)
            self.value_activation = params.get("value_activation", "None")
            self.normalization = params.get("normalization", None)

            assert "space" in params and "continuous" in params["space"], (
                "GLAA2C expects a continuous action space."
            )
            self.is_continuous = True
            self.space_config = params["space"]["continuous"]

            gla_cfg = params.get("gla", {})
            self.history_length = int(gla_cfg.get("history_length", 16))
            self.token_dim = int(gla_cfg.get("token_dim", 102))
            self.gla_hidden = int(gla_cfg.get("hidden_size", 128))
            self.gla_heads = int(gla_cfg.get("num_heads", 4))
            kv_heads = gla_cfg.get("num_kv_heads", None)
            self.gla_kv_heads = int(kv_heads) if kv_heads is not None else None
            self.gla_expand_k = float(gla_cfg.get("expand_k", 0.5))
            self.gla_expand_v = float(gla_cfg.get("expand_v", 1.0))
            self.gla_mode = str(gla_cfg.get("mode", "chunk"))
            self.pool = str(gla_cfg.get("pool", "last")).lower()
            assert self.pool in ("last", "mean"), (
                f"gla.pool must be 'last' or 'mean', got {self.pool!r}"
            )

            # ---- PICA v2b/v2c: aux head config (optional) ----
            # `pred_dim` = number of regression outputs; `target_dim` = number
            # of obs-tail channels written by the env. They are equal in
            # v2b/v2c-init (no separate gate channel), but kept distinct so
            # future versions can carry extra transport channels without
            # retraining.
            #
            # `target_keys` and `target_weights` arrive from train.py, which
            # mirrors them out of `physical_auxiliary` in the config block.
            # Backwards-compat: when keys/weights are missing (older v2b
            # configs), fall back to default ordering and unit weights.
            aux = params.get("phys_aux", {}) or {}
            self.aux_enabled = bool(aux.get("enabled", False))
            self.aux_pred_dim = int(aux.get("pred_dim", 0))
            self.aux_target_dim = int(aux.get("target_dim", self.aux_pred_dim))
            self.aux_hidden = int(aux.get("hidden_size", 64))
            self.aux_mode = str(aux.get("mode", "current"))

            cfg_keys = list(aux.get("target_keys") or [])
            if not cfg_keys and self.aux_enabled:
                # v2b legacy: no keys forwarded. Use the canonical v2b order.
                cfg_keys = ["dq_obj", "slip_proxy", "tracking_stress"][: self.aux_pred_dim]
            self.aux_target_keys = cfg_keys

            cfg_weights = list(aux.get("target_weights") or [])
            if len(cfg_weights) != self.aux_pred_dim:
                cfg_weights = [1.0] * self.aux_pred_dim
            self.aux_target_weights = [float(w) for w in cfg_weights]


def register_gla_network():
    """Register this builder with rl_games' global NETWORK_REGISTRY.

    Call once at process startup, before `Runner.load(train_config)` (which
    runs ModelBuilder, which copies NETWORK_REGISTRY into its factory).
    """
    from rl_games.algos_torch import model_builder

    model_builder.register_network("gla_actor_critic", GLAA2CBuilder)
