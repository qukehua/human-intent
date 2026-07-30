# encoding: utf-8

import os.path as osp
import time

import yaml
from easydict import EasyDict as edict


def to_edict(value):
    if isinstance(value, dict):
        return edict({k: to_edict(v) for k, v in value.items()})
    if isinstance(value, list):
        return [to_edict(v) for v in value]
    return value


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_harper_config(path="config/harper_config.yml"):
    if not osp.isabs(path):
        path = osp.abspath(osp.join(osp.dirname(__file__), "..", path))
    raw = load_yaml(path)
    cfg = to_edict(raw)

    abs_dir = osp.dirname(osp.realpath(path))
    repo_root = osp.abspath(osp.join(abs_dir, ".."))
    exp_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    log_dir = osp.abspath(osp.join(abs_dir, cfg.paths.log_dir))

    cfg.abs_dir = abs_dir
    cfg.this_dir = osp.basename(abs_dir)
    cfg.root_dir = repo_root
    cfg.log_dir = log_dir
    cfg.snapshot_dir = osp.abspath(osp.join(log_dir, "snapshot" + exp_time))
    cfg.log_file = osp.join(log_dir, "log_" + exp_time + ".log")
    cfg.link_log_file = osp.join(log_dir, "log_last.log")
    cfg.val_log_file = osp.join(log_dir, "val_" + exp_time + ".log")
    cfg.link_val_log_file = osp.join(log_dir, "val_last.log")

    return cfg


def build_h2h_model_config(cfg):
    obs_len = int(cfg["sequence"]["obs_len"])
    pred_len = int(cfg["sequence"]["pred_len"])
    coord_dim = int(cfg["sequence"].get("coord_dim", 3))
    target_joints = int(cfg["sequence"].get("target_joints", 21))
    model_cfg = cfg.get("model", {})
    motion_mlp = model_cfg.get("motion_mlp", {})

    return to_edict(
        {
            "motion": {
                "harper_input_length": obs_len,
                "harper_input_length_dct": obs_len,
                "harper_target_length_train": pred_len,
                "harper_target_length": pred_len,
                "harper_target_length_eval": pred_len,
                "dim1": target_joints * coord_dim,
                "dim2": target_joints * coord_dim,
            },
            "motion_mlp": motion_mlp,
            "intent": cfg.get("intent", {}),
        }
    )
