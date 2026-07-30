import copy
from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


def _to_joint_repr(x: torch.Tensor, num_joints: int) -> torch.Tensor:
    b, t, c = x.shape
    return x.view(b, t, num_joints, 3).permute(0, 2, 1, 3).contiguous()  # B,J,T,3


def _to_flat_repr(x: torch.Tensor) -> torch.Tensor:
    b, j, t, c = x.shape
    return x.permute(0, 2, 1, 3).contiguous().view(b, t, j * c)


class MotionEncoder(nn.Module):
    """Three-layer MLP encoder for DCT-domain motion tokens."""

    def __init__(self, embed_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, motion: torch.Tensor) -> torch.Tensor:
        # motion: B,J,T,3 -> B,J,T,D
        return self.net(motion)


class GCBlock(nn.Module):
    """Lightweight GCN block with learnable spatial/temporal adjacencies."""

    def __init__(self, channels: int, num_joints: int, seq_len: int, dropout: float):
        super().__init__()
        self.aspatial = nn.Parameter(torch.eye(num_joints))
        self.atemporal = nn.Parameter(torch.eye(seq_len))
        self.bn = nn.BatchNorm2d(channels)
        self.proj = nn.Linear(channels, channels)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B,C,J,T
        y = self.bn(x)
        y = torch.einsum("ij,bcjt->bcit", self.aspatial, y)
        y = torch.einsum("tu,bcju->bcjt", self.atemporal, y)
        y = self.proj(y.permute(0, 2, 3, 1))
        y = self.drop(self.relu(y))
        y = y.permute(0, 3, 1, 2).contiguous()
        return y + x


class STCrossAttention(nn.Module):
    """Spatial-then-temporal cross attention."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.spatial = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.temporal = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_s = nn.LayerNorm(embed_dim)
        self.norm_t = nn.LayerNorm(embed_dim)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # q,kv: B,J,T,D
        b, j, t, d = q.shape
        b_kv, j_kv, t_kv, d_kv = kv.shape
        if b_kv != b or t_kv != t or d_kv != d:
            raise ValueError(
                f"q and kv must share batch/time/channel dims, got q={tuple(q.shape)}, kv={tuple(kv.shape)}"
            )

        qs = q.permute(0, 2, 1, 3).reshape(b * t, j, d)
        ks = kv.permute(0, 2, 1, 3).reshape(b * t, j_kv, d)
        spatial, _ = self.spatial(qs, ks, ks)
        spatial = self.norm_s(spatial + qs).reshape(b, t, j, d).permute(0, 2, 1, 3)

        qt = spatial.reshape(b * j, t, d)
        if j_kv == j:
            kt = kv.reshape(b * j, t, d)
        else:
            kt = kv.mean(dim=1, keepdim=True).expand(-1, j, -1, -1).reshape(b * j, t, d)
        temporal, _ = self.temporal(qt, kt, kt)
        temporal = self.norm_t(temporal + qt).reshape(b, j, t, d)
        return temporal


class IFB(nn.Module):
    """Interaction Feature Bridge for stage-2 fine-tuning."""

    def __init__(self, embed_dim: int, num_joints: int, num_heads: int, dropout: float):
        super().__init__()
        self.joint_embed = nn.Parameter(torch.randn(1, num_joints, 1, embed_dim) * 0.02)
        self.spatial = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.temporal = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B,J,T,D
        b, j, t, d = x.shape
        x = x + self.joint_embed[:, :j]
        xs = x.permute(0, 2, 1, 3).reshape(b * t, j, d)
        xs, _ = self.spatial(xs, xs, xs)
        xs = xs.reshape(b, t, j, d).permute(0, 2, 1, 3).contiguous()

        xt = xs.permute(0, 1, 3, 2).reshape(b * j, d, t)
        xt = self.temporal(xt).reshape(b, j, d, t).permute(0, 1, 3, 2)
        return self.norm(xt + x)


class JointGatedFusion(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, intra: torch.Tensor, inter: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        fuse = torch.cat([intra, inter], dim=-1)
        g = self.gate(fuse)  # B,J,T,1
        out = g * intra + (1.0 - g) * inter
        return out, g


class InteractionTokenEncoder(nn.Module):
    """Pool bidirectional cross-attention features into an interaction token.

    The point encoder is shared by both agents and the cross-attention block
    accepts different joint counts.  Consequently the same Stage-1 token
    encoder can be reused for human-human (21/21) and human-robot (21/N)
    sequences without learning a fixed joint-to-joint correspondence.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        point_hidden = max(embed_dim // 2, 32)
        self.point_encoder = nn.Sequential(
            nn.Linear(9, point_hidden),
            nn.GELU(),
            nn.Linear(point_hidden, embed_dim),
        )
        self.cross_attention = STCrossAttention(embed_dim, num_heads, dropout)
        self.frame_projection = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )
        self.relation_projection = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )
        self.temporal = nn.GRU(embed_dim, embed_dim, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pool = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def _motion_features(self, motion: torch.Tensor) -> torch.Tensor:
        # motion: B,T,J*3 in the time domain
        if motion.ndim != 3 or motion.shape[-1] % 3 != 0:
            raise ValueError(f"Expected time-domain motion [B,T,J*3], got {tuple(motion.shape)}")
        b, t, c = motion.shape
        points = motion.reshape(b, t, c // 3, 3)
        root = points[:, :, 0]
        relative = points - root.unsqueeze(2)
        velocity = torch.cat(
            [torch.zeros_like(points[:, :1]), points[:, 1:] - points[:, :-1]],
            dim=1,
        )
        # Scene coordinates preserve inter-agent displacement; body-relative
        # coordinates preserve articulation. Both, plus velocity, enter cross
        # attention before any information is pooled into the token.
        point_input = torch.cat([points, relative, velocity], dim=-1)
        # B,T,J,D -> B,J,T,D, the layout expected by STCrossAttention.
        point_features = self.point_encoder(point_input)
        point_features = point_features.permute(0, 2, 1, 3).contiguous()
        return point_features

    def _pool_cross_features(self, features: torch.Tensor) -> torch.Tensor:
        # features: B,J,T,D.  Pooling occurs only after one agent has attended
        # to the other, so every interaction token is derived from cross-agent
        # features rather than independent per-person summaries.
        pooled_mean = features.mean(dim=1)
        pooled_max = features.amax(dim=1)
        return self.frame_projection(torch.cat([pooled_mean, pooled_max], dim=-1))

    def forward(self, motion_a: torch.Tensor, motion_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if motion_a.shape[:2] != motion_b.shape[:2]:
            raise ValueError(
                f"Interaction streams must share batch/time, got {tuple(motion_a.shape)} "
                f"and {tuple(motion_b.shape)}"
            )
        point_a = self._motion_features(motion_a)
        point_b = self._motion_features(motion_b)
        cross_a = self.cross_attention(point_a, point_b)
        cross_b = self.cross_attention(point_b, point_a)
        frame_a = self._pool_cross_features(cross_a)
        frame_b = self._pool_cross_features(cross_b)
        relation = torch.cat(
            [
                frame_a,
                frame_b,
                torch.abs(frame_b - frame_a),
                frame_a * frame_b,
            ],
            dim=-1,
        )
        relation = self.relation_projection(relation)
        relation, _ = self.temporal(relation)

        query = self.query.expand(relation.shape[0], -1, -1)
        token, attention = self.pool(query, relation, relation)
        token = self.norm(token + query).squeeze(1)
        return token, attention.squeeze(1)


class AINet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = copy.deepcopy(config)
        self.human_joint = self.config.motion.dim1 // 3
        self.robot_joint = self.config.motion.dim2 // 3
        self.seq_len = self.config.motion.harper_input_length_dct
        self.pred_len = getattr(self.config.motion, "harper_target_length_train", 10)
        self.embed_dim = getattr(
            self.config.motion_mlp, "embed_dim", getattr(self.config.motion_mlp, "hidden_dim", 64)
        )
        self.dropout = getattr(self.config.motion_mlp, "dropout", 0.1)
        self.intra_layers = getattr(self.config.motion_mlp, "intra_layers", 12)
        self.inter_layers = getattr(self.config.motion_mlp, "inter_layers", 9)
        self.heads = getattr(self.config.motion_mlp, "attn_heads", 8)
        if self.embed_dim % self.heads != 0:
            raise ValueError(f"embed_dim ({self.embed_dim}) must be divisible by attn_heads ({self.heads})")
        intent_config = getattr(self.config, "intent", None)
        self.latent_dim = int(getattr(intent_config, "latent_dim", 128))
        self.intent_token_dim = int(getattr(intent_config, "token_dim", min(self.embed_dim, 256)))
        if self.intent_token_dim % self.heads != 0:
            raise ValueError(
                f"intent token_dim ({self.intent_token_dim}) must be divisible by attn_heads ({self.heads})"
            )
        self.freeze_human_backbone_stage2 = bool(
            getattr(intent_config, "freeze_human_backbone_stage2", True)
        )

        self.encoder_h = MotionEncoder(self.embed_dim, self.dropout)
        self.encoder_r = MotionEncoder(self.embed_dim, self.dropout)

        self.intra_h = nn.ModuleList(
            [GCBlock(self.embed_dim, self.human_joint, self.seq_len, self.dropout) for _ in range(self.intra_layers)]
        )
        self.intra_r = nn.ModuleList(
            [GCBlock(self.embed_dim, self.robot_joint, self.seq_len, self.dropout) for _ in range(self.intra_layers)]
        )

        self.cross_blocks = nn.ModuleList(
            [STCrossAttention(self.embed_dim, self.heads, self.dropout) for _ in range(self.inter_layers)]
        )
        self.ifb = IFB(self.embed_dim, self.robot_joint, self.heads, self.dropout)
        self.fusion_h = JointGatedFusion(self.embed_dim)
        self.fusion_r = JointGatedFusion(self.embed_dim)

        self.decoder_h = nn.Linear(self.embed_dim, 3)
        self.decoder_r = nn.Linear(self.embed_dim, 3)
        self.rec_h = nn.Linear(self.embed_dim, 3)
        self.rec_r = nn.Linear(self.embed_dim, 3)

        self.interaction_encoder = InteractionTokenEncoder(self.intent_token_dim, self.heads, self.dropout)
        self.intent_prior = nn.Sequential(
            nn.Linear(self.intent_token_dim, self.intent_token_dim),
            nn.GELU(),
            nn.Linear(self.intent_token_dim, self.latent_dim * 2),
        )
        self.intent_posterior = nn.Sequential(
            nn.Linear(self.intent_token_dim * 2, self.intent_token_dim),
            nn.GELU(),
            nn.Linear(self.intent_token_dim, self.latent_dim * 2),
        )
        self.future_token_decoder = nn.Sequential(
            nn.Linear(self.intent_token_dim + self.latent_dim, self.intent_token_dim),
            nn.GELU(),
            nn.Linear(self.intent_token_dim, self.intent_token_dim),
        )
        self.intent_film = nn.Linear(self.intent_token_dim + self.latent_dim, self.embed_dim * 2)
        self.intent_norm_h = nn.LayerNorm(self.embed_dim)
        self.intent_norm_r = nn.LayerNorm(self.embed_dim)

        self.stage = 1
        self.set_stage(1)

    @staticmethod
    def _set_trainable(module: nn.Module, trainable: bool) -> None:
        for p in module.parameters():
            p.requires_grad = trainable

    def set_stage(self, stage: int) -> None:
        if stage not in {1, 2}:
            raise ValueError(f"stage must be 1 or 2, got {stage}")
        self.stage = stage
        # Stage 1 learns the human motion and human-human interaction prior.
        # Stage 2 preserves the human backbone but adapts the robot branch,
        # cross-agent interaction, latent distribution and prediction heads.
        self._set_trainable(self, True)
        if stage == 2 and self.freeze_human_backbone_stage2:
            self._set_trainable(self.encoder_h, False)
            self._set_trainable(self.intra_h, False)

    @staticmethod
    def _split_stats(stats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = stats.chunk(2, dim=-1)
        return mu, logvar.clamp(min=-10.0, max=10.0)

    @staticmethod
    def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    @staticmethod
    def _kl_normal(
        posterior_mu: torch.Tensor,
        posterior_logvar: torch.Tensor,
        prior_mu: torch.Tensor,
        prior_logvar: torch.Tensor,
    ) -> torch.Tensor:
        variance_ratio = torch.exp(posterior_logvar - prior_logvar)
        mean_distance = (posterior_mu - prior_mu).pow(2) * torch.exp(-prior_logvar)
        return 0.5 * (
            prior_logvar - posterior_logvar + variance_ratio + mean_distance - 1.0
        ).sum(dim=-1)

    def _condition_with_intent(
        self,
        fused: torch.Tensor,
        interaction_token: torch.Tensor,
        latent: torch.Tensor,
        valid: torch.Tensor,
        norm: nn.LayerNorm,
    ) -> torch.Tensor:
        gamma, beta = self.intent_film(torch.cat([interaction_token, latent], dim=-1)).chunk(2, dim=-1)
        gamma = torch.tanh(gamma).unsqueeze(1).unsqueeze(1)
        beta = beta.unsqueeze(1).unsqueeze(1)
        conditioned = norm(fused * (1.0 + gamma) + beta)
        mask = valid.reshape(valid.shape[0], 1, 1, 1).to(dtype=fused.dtype)
        return fused + mask * (conditioned - fused)

    def load_compatible_state_dict(self, checkpoint) -> Dict[str, object]:
        """Load a Stage-1 or Stage-2 checkpoint while skipping shape changes."""

        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    state_dict = checkpoint[key]
                    break
        if not isinstance(state_dict, dict):
            raise TypeError("Checkpoint must contain a model state dictionary")

        current = self.state_dict()
        compatible = {}
        skipped = {}
        unexpected = []
        for raw_key, value in state_dict.items():
            key = raw_key[7:] if raw_key.startswith("module.") else raw_key
            if key not in current:
                unexpected.append(key)
            elif current[key].shape != value.shape:
                skipped[key] = {
                    "checkpoint": tuple(value.shape),
                    "model": tuple(current[key].shape),
                }
            else:
                compatible[key] = value
        result = self.load_state_dict(compatible, strict=False)
        return {
            "loaded": len(compatible),
            "skipped_shape": skipped,
            "missing": list(result.missing_keys),
            "unexpected": unexpected,
        }

    def _run_intra(self, feat: torch.Tensor, blocks: nn.ModuleList) -> torch.Tensor:
        # feat: B,J,T,D -> B,D,J,T
        x = feat.permute(0, 3, 1, 2).contiguous()
        for blk in blocks:
            x = blk(x)
        return x.permute(0, 2, 3, 1).contiguous()

    def _run_inter(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        x = q
        for blk in self.cross_blocks:
            x = blk(x, kv)
        return x

    def _decode_future(self, fused: torch.Tensor, head: nn.Linear, joints: int) -> torch.Tensor:
        # fused: B,J,T,D -> DCT-domain forecast coefficients
        # Every coefficient has its own temporally contextualised token. Repeating
        # the final token here made all coefficients identical and collapsed the
        # inverse-DCT trajectory to a nearly fixed basis pattern.
        pred = head(fused)  # B,J,T,3
        pred = pred.permute(0, 2, 1, 3).contiguous().view(fused.shape[0], self.seq_len, joints * 3)
        return pred

    def _decode_recon(self, fused: torch.Tensor, head: nn.Linear, joints: int) -> torch.Tensor:
        rec = head(fused)  # B,J,T,3
        rec = rec.permute(0, 2, 1, 3).contiguous().view(fused.shape[0], self.seq_len, joints * 3)
        return rec

    def forward(
        self,
        motion_input1: torch.Tensor,
        motion_input2: torch.Tensor,
        nb_iter=0,
        *,
        history_motion1: Optional[torch.Tensor] = None,
        history_motion2: Optional[torch.Tensor] = None,
        future_motion1: Optional[torch.Tensor] = None,
        future_motion2: Optional[torch.Tensor] = None,
        interaction_valid: Optional[torch.Tensor] = None,
        sample_intent: bool = False,
    ):
        # motion_input: B,T,C in DCT domain. Interaction tokens use time-domain
        # history/future streams supplied by the training and evaluation code.
        if interaction_valid is None:
            interaction_valid = motion_input1.new_ones(motion_input1.shape[0])
        else:
            interaction_valid = interaction_valid.to(
                device=motion_input1.device,
                dtype=motion_input1.dtype,
            ).reshape(-1)
            if interaction_valid.shape[0] != motion_input1.shape[0]:
                raise ValueError("interaction_valid must contain one value per batch element")

        h_pos = _to_joint_repr(motion_input1, self.human_joint)
        r_pos = _to_joint_repr(motion_input2, self.robot_joint)
        f_en_h = self.encoder_h(h_pos)
        f_en_r = self.encoder_r(r_pos)

        f_intra_h = self._run_intra(f_en_h, self.intra_h)
        f_intra_r = self._run_intra(f_en_r, self.intra_r)

        inter_r_for_h = self.ifb(f_en_r) if self.stage == 2 else f_en_r
        inter_h_for_r = f_en_h
        f_inter_h_cross = self._run_inter(f_en_h, inter_r_for_h)
        f_inter_r_cross = self._run_inter(f_en_r, inter_h_for_r)
        cross_mask = interaction_valid.reshape(-1, 1, 1, 1)
        # `interaction_valid` is a training-eligibility mask, not a
        # real-vs-synthetic provenance flag.  Stage-1 enables it for every
        # configured H-H pair so AMASS/H36M random pairs also train the
        # cross-agent path, interaction token and intent latent.
        f_inter_h = f_en_h + cross_mask * (f_inter_h_cross - f_en_h)
        f_inter_r = f_en_r + cross_mask * (f_inter_r_cross - f_en_r)

        f_fuse_h, g_h = self.fusion_h(f_intra_h, f_inter_h)
        f_fuse_r, g_r = self.fusion_r(f_intra_r, f_inter_r)

        if history_motion1 is None or history_motion2 is None:
            raise ValueError(
                "history_motion1/history_motion2 in the time domain are required for interaction encoding"
            )
        interaction_token, interaction_attention = self.interaction_encoder(history_motion1, history_motion2)
        prior_mu, prior_logvar = self._split_stats(self.intent_prior(interaction_token))

        has_future = future_motion1 is not None or future_motion2 is not None
        if has_future and (future_motion1 is None or future_motion2 is None):
            raise ValueError("future_motion1 and future_motion2 must be provided together")

        future_token = None
        posterior_mu = None
        posterior_logvar = None
        if has_future:
            future_context1 = torch.cat([history_motion1[:, -1:], future_motion1], dim=1)
            future_context2 = torch.cat([history_motion2[:, -1:], future_motion2], dim=1)
            future_token, _ = self.interaction_encoder(future_context1, future_context2)
            posterior_mu, posterior_logvar = self._split_stats(
                self.intent_posterior(torch.cat([interaction_token, future_token], dim=-1))
            )
            latent = (
                self._reparameterize(posterior_mu, posterior_logvar)
                if self.training or sample_intent
                else posterior_mu
            )
            self.last_kl_per_sample = self._kl_normal(
                posterior_mu,
                posterior_logvar,
                prior_mu,
                prior_logvar,
            )
            predicted_future_token = self.future_token_decoder(
                torch.cat([interaction_token, latent], dim=-1)
            )
            self.last_intent_token_loss_per_sample = F.smooth_l1_loss(
                predicted_future_token,
                future_token.detach(),
                reduction="none",
            ).mean(dim=-1)
        else:
            latent = self._reparameterize(prior_mu, prior_logvar) if sample_intent else prior_mu
            self.last_kl_per_sample = prior_mu.new_zeros(prior_mu.shape[0])
            self.last_intent_token_loss_per_sample = prior_mu.new_zeros(prior_mu.shape[0])

        pred_features_h = self._condition_with_intent(
            f_fuse_h,
            interaction_token,
            latent,
            interaction_valid,
            self.intent_norm_h,
        )
        pred_features_r = self._condition_with_intent(
            f_fuse_r,
            interaction_token,
            latent,
            interaction_valid,
            self.intent_norm_r,
        )

        # Prediction is intent-conditioned; reconstruction remains a pure
        # DCT-domain backbone objective and cannot leak the future posterior.
        pred_h = self._decode_future(pred_features_h, self.decoder_h, self.human_joint)
        pred_r = self._decode_future(pred_features_r, self.decoder_r, self.robot_joint)
        rec_h = self._decode_recon(f_fuse_h, self.rec_h, self.human_joint)
        rec_r = self._decode_recon(f_fuse_r, self.rec_r, self.robot_joint)

        # Keep legacy return shape style used by train.py.
        # alpha/alpha2/beta/beta2 are reused as gate diagnostics.
        alpha = g_h.mean(dim=2).permute(0, 2, 1).contiguous()   # B,1,Jh
        alpha2 = 1.0 - alpha
        beta = g_r.mean(dim=2).permute(0, 2, 1).contiguous()    # B,1,Jr
        beta2 = 1.0 - beta

        # stash recon outputs for external loss use when needed
        self.last_recon_h = rec_h
        self.last_recon_r = rec_r
        self.last_pred_partner = pred_r
        self.last_interaction_token = interaction_token
        self.last_interaction_attention = interaction_attention
        self.last_prior_mu = prior_mu
        self.last_prior_logvar = prior_logvar
        self.last_posterior_mu = posterior_mu
        self.last_posterior_logvar = posterior_logvar
        self.last_future_interaction_token = future_token
        # Keep the legacy return tuple; the symmetric person/robot prediction
        # is exposed through last_pred_partner.
        return pred_h, alpha, alpha2, beta, beta2
