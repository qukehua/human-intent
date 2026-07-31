import argparse
import copy
import json
import os
import shutil

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config.config_utils import load_harper_config
from dataset.harper_3d_2 import Harper3D
from network.model import AINet as Model
from test import test
from utils.distributed import (
    barrier,
    cleanup_distributed,
    get_world_size,
    init_distributed,
    is_main_process,
    print0,
    seed_everything,
    unwrap_model,
)
from utils.logger import get_logger, print_and_log_info
from utils.pyt_utils import ensure_dir, link_file
from utils.wandb_utils import (
    add_wandb_args,
    finish_wandb,
    init_wandb,
    resolve_wandb_settings,
    should_log,
    wandb_log,
)

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

n_joint = 21 + 23
human_joint = 21
robot_joint = 23
data_root = r"/data/user/qkh/datasets/HARPER/HARPER _3D panoptic/30hz"
d = 100


def get_dct_matrix(N):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m


def update_lr_multistep(nb_iter, total_iter, max_lr, min_lr, optimizer):
    progress = min(1.0, max(0.0, float(nb_iter) / float(max(1, total_iter - 1))))
    current_lr = min_lr + 0.5 * (max_lr - min_lr) * (1.0 + np.cos(np.pi * progress))

    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr * float(param_group.get("lr_scale", 1.0))

    return optimizer, current_lr


def gen_velocity(m):
    return m[:, 1:] - m[:, :-1]


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--exp-name", type=str, default="train", help="=exp name")
    parser.add_argument("--seed", type=int, default=888, help="=seed")
    parser.add_argument("--layer-norm-axis", type=str, default="spatial", help="=layernorm axis")
    parser.add_argument("--with-normalization", action="store_true", help="=use layernorm")
    parser.add_argument("--spatial-fc", action="store_true", help="=use only spatial fc")
    parser.add_argument("--num", type=int, default=24, help="=num of blocks")
    parser.add_argument("--weight", type=float, default=1.0, help="=loss weight")
    parser.add_argument("--work_dir", type=str, default=".", help="=work_dir")
    parser.add_argument("--stage", type=int, default=2, choices=[1, 2], help="training stage")
    parser.add_argument("--lambda-pre", type=float, default=1.0, help="prediction loss weight")
    parser.add_argument("--lambda-rec", type=float, default=0.5, help="reconstruction loss weight")
    parser.add_argument("--lambda-robot", type=float, default=None, help="robot future prediction loss weight")
    parser.add_argument("--lambda-kl", type=float, default=None, help="intent KL loss weight")
    parser.add_argument(
        "--lambda-intent-token",
        type=float,
        default=None,
        help="future interaction-token loss weight",
    )
    parser.add_argument(
        "--model-pth",
        type=str,
        default=None,
        help="Stage-1 checkpoint used for Stage-2 fine-tuning",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Per-GPU batch size (overrides config). Global batch = batch_size * world_size",
    )
    add_wandb_args(parser)
    return parser.parse_args()


def train_step(
    harper_motion_input,
    harper_motion_target,
    model,
    optimizer,
    nb_iter,
    total_iter,
    max_lr,
    min_lr,
    *,
    config,
    args,
    dct_m,
    idct_m,
    device,
    writer,
    wandb_log_every,
):
    raw_model = unwrap_model(model)
    in_features = human_joint * 3
    b, seqlen, _ = harper_motion_input.shape

    history = harper_motion_input.to(device, non_blocking=True)
    future = harper_motion_target.to(device, non_blocking=True)
    history_h, history_r = history[:, :, :in_features], history[:, :, in_features:]
    future_h, future_r = future[:, :, :in_features], future[:, :, in_features:]
    src1 = torch.matmul(dct_m[:, :, : config.motion.harper_input_length], history_h)
    src2 = torch.matmul(dct_m[:, :, : config.motion.harper_input_length], history_r)

    motion_pred1, alpha_s, alpha_t, beta_s, beta_t = model(
        src1,
        src2,
        history_motion1=history_h,
        history_motion2=history_r,
        future_motion1=future_h,
        future_motion2=future_r,
    )
    motion_pred2 = raw_model.last_pred_partner
    reg_terms = [
        torch.relu(-alpha_s) + torch.relu(alpha_s - 1),
        torch.relu(-alpha_t) + torch.relu(alpha_t - 1),
        torch.relu(-beta_s) + torch.relu(beta_s - 1),
        torch.relu(-beta_t) + torch.relu(beta_t - 1),
    ]
    reg_loss = sum(term.mean() for term in reg_terms)

    motion_pred1 = torch.matmul(idct_m[:, : config.motion.harper_input_length, :], motion_pred1)
    motion_pred2 = torch.matmul(idct_m[:, : config.motion.harper_input_length, :], motion_pred2)

    if config.deriv_output:
        offset1 = history_h[:, -1:, :]
        offset2 = history_r[:, -1:, :]
        motion_pred1 = motion_pred1[:, : config.motion.harper_target_length] + offset1
        motion_pred2 = motion_pred2[:, : config.motion.harper_target_length] + offset2
    else:
        motion_pred1 = motion_pred1[:, : config.motion.harper_target_length]
        motion_pred2 = motion_pred2[:, : config.motion.harper_target_length]

    b, n, c = harper_motion_target.shape
    motion_pred1 = motion_pred1.reshape(b, n, human_joint, 3)
    motion_pred2 = motion_pred2.reshape(b, n, robot_joint, 3)
    target_joints = future.reshape(b, n, n_joint, 3)
    motion_h_gt = target_joints[:, :, :human_joint]
    motion_r_gt = target_joints[:, :, human_joint:]
    loss_h = torch.mean(torch.norm(motion_pred1 - motion_h_gt, dim=-1))
    loss_r = torch.mean(torch.norm(motion_pred2 - motion_r_gt, dim=-1))

    if config.use_relative_loss:
        dmotion_pred = gen_velocity(motion_pred1)
        dmotion_hgt = gen_velocity(motion_h_gt)
        dlossh = torch.mean(torch.norm(dmotion_pred - dmotion_hgt, dim=-1))
        dmotion_pred_r = gen_velocity(motion_pred2)
        dmotion_rgt = gen_velocity(motion_r_gt)
        dlossr = torch.mean(torch.norm(dmotion_pred_r - dmotion_rgt, dim=-1))
        loss_h += dlossh
        loss_r += dlossr

    rec_h = raw_model.last_recon_h.reshape(-1, 3)
    rec_r = raw_model.last_recon_r.reshape(-1, 3)
    inp_h = src1.reshape(b, seqlen, human_joint, 3).reshape(-1, 3)
    inp_r = src2.reshape(b, seqlen, robot_joint, 3).reshape(-1, 3)
    rec_loss_h = torch.mean(torch.norm(rec_h - inp_h, 2, 1))
    rec_loss_r = torch.mean(torch.norm(rec_r - inp_r, 2, 1))
    rec_loss = rec_loss_h + rec_loss_r

    reg_loss = reg_loss * d
    lambda_robot = (
        float(args.lambda_robot)
        if args.lambda_robot is not None
        else float(config.intent.robot_prediction_weight)
    )
    lambda_kl = (
        float(args.lambda_kl) if args.lambda_kl is not None else float(config.intent.kl_weight)
    )
    lambda_intent_token = (
        float(args.lambda_intent_token)
        if args.lambda_intent_token is not None
        else float(config.intent.token_prediction_weight)
    )
    kl_warmup = min(1.0, float(nb_iter + 1) / float(max(1, config.intent.kl_warmup_steps)))
    kl_loss = raw_model.last_kl_per_sample.mean()
    intent_token_loss = raw_model.last_intent_token_loss_per_sample.mean()
    pred_loss = loss_h + lambda_robot * loss_r
    loss = (
        args.lambda_pre * pred_loss
        + args.lambda_rec * rec_loss
        + lambda_kl * kl_warmup * kl_loss
        + lambda_intent_token * intent_token_loss
        + reg_loss
    )

    if is_main_process() and writer is not None:
        writer.add_scalar("Loss/loss_all", loss.detach().cpu().numpy(), nb_iter)
        writer.add_scalar("Loss/loss_h", loss_h.detach().cpu().numpy(), nb_iter)
        writer.add_scalar("Loss/loss_robot", loss_r.detach().cpu().numpy(), nb_iter)
        writer.add_scalar("Loss/loss_rec", rec_loss.detach().cpu().numpy(), nb_iter)
        writer.add_scalar("Loss/loss_kl", kl_loss.detach().cpu().numpy(), nb_iter)
        writer.add_scalar("Loss/loss_intent_token", intent_token_loss.detach().cpu().numpy(), nb_iter)
        writer.add_scalar("Loss/kl_warmup", kl_warmup, nb_iter)
        writer.add_scalar("Loss/reg_loss", reg_loss.detach().cpu().numpy(), nb_iter)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    optimizer, current_lr = update_lr_multistep(nb_iter, total_iter, max_lr, min_lr, optimizer)

    if is_main_process() and writer is not None:
        writer.add_scalar("LR/train", current_lr, nb_iter)
    if is_main_process() and should_log(nb_iter + 1, wandb_log_every):
        wandb_log(
            {
                "train/loss": float(loss.detach().cpu()),
                "train/loss_h": float(loss_h.detach().cpu()),
                "train/loss_robot": float(loss_r.detach().cpu()),
                "train/loss_rec": float(rec_loss.detach().cpu()),
                "train/loss_kl": float(kl_loss.detach().cpu()),
                "train/loss_intent_token": float(intent_token_loss.detach().cpu()),
                "train/kl_warmup": kl_warmup,
                "train/reg_loss": float(reg_loss.detach().cpu()),
                "train/lr": current_lr,
                "train/world_size": get_world_size(),
            },
            step=nb_iter + 1,
        )

    return loss.item(), optimizer, current_lr, loss_h, loss_r


def main():
    args = parse_args()
    distributed, rank, local_rank, world_size = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    config = load_harper_config()
    config.motion.harper_target_length = config.motion.harper_target_length_train
    config.motion_mlp.norm_axis = args.layer_norm_axis
    config.motion_mlp.spatial_fc_only = args.spatial_fc
    config.motion_mlp.with_normalization = args.with_normalization
    config.motion_mlp.num_layers = args.num

    eval_config = copy.deepcopy(config)
    eval_config.motion.harper_target_length = eval_config.motion.harper_target_length_eval

    batch_size = int(args.batch_size if args.batch_size is not None else config.batch_size)
    seed_everything(args.seed, rank=rank)

    # Deterministic algorithms can conflict with some DDP kernels on newer GPUs.
    if not distributed:
        torch.use_deterministic_algorithms(True)

    wandb_settings = resolve_wandb_settings(
        config,
        enable=args.wandb_enable,
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
        mode=args.wandb_mode,
        log_every=args.wandb_log_every,
    )
    if not is_main_process():
        wandb_settings = dict(wandb_settings)
        wandb_settings["enable"] = False
    wandb_log_every = int(wandb_settings["log_every"])

    writer = None
    acc_log = None
    if is_main_process():
        ensure_dir("./result")
        acc_log = open("./result/" + args.exp_name, "a")
        acc_log.write("".join("Seed : " + str(args.seed) + "\n"))
        ensure_dir("./ckpt_LNv5_reproduce")
        writer = SummaryWriter("./ckpt_LNv5_reproduce")
        print0(f"reg_loss * {d}")
        print0(
            f"DDP: world_size={world_size}, per-GPU batch={batch_size}, "
            f"global batch={batch_size * world_size}"
        )

    dataset = Harper3D(
        data_path=data_root,
        split="train",
        n_input=config.motion.harper_input_length,
        n_output=config.motion.harper_target_length,
        sample=1,
        root_center=config.normalization.root_center,
    )
    train_sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
        if distributed
        else None
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=config.num_workers,
        drop_last=True,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        pin_memory=torch.cuda.is_available(),
    )

    eval_dataset = Harper3D(
        data_path=data_root,
        split="test",
        n_input=eval_config.motion.harper_input_length,
        n_output=eval_config.motion.harper_target_length,
        sample=1,
        root_center=eval_config.normalization.root_center,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=128,
        num_workers=1,
        drop_last=False,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    dct_np, idct_np = get_dct_matrix(config.motion.harper_input_length_dct)
    dct_m = torch.tensor(dct_np).float().to(device).unsqueeze(0)
    idct_m = torch.tensor(idct_np).float().to(device).unsqueeze(0)

    model = Model(config)
    model.set_stage(args.stage)
    model.train()
    model.to(device)

    init_wandb(
        wandb_settings,
        config={
            "stage": args.stage,
            "seed": args.seed,
            "exp_name": args.exp_name,
            "work_dir": args.work_dir,
            "model_pth": args.model_pth or config.model_pth,
            "lambda_pre": args.lambda_pre,
            "lambda_rec": args.lambda_rec,
            "lambda_robot": args.lambda_robot,
            "lambda_kl": args.lambda_kl,
            "lambda_intent_token": args.lambda_intent_token,
            "world_size": world_size,
            "per_gpu_batch_size": batch_size,
            "global_batch_size": batch_size * world_size,
            "harper_config": json.loads(json.dumps(config)),
        },
        job_type=f"stage{args.stage}",
    )

    try:
        raw_model = model
        semantic_modules = (
            raw_model.interaction_encoder,
            raw_model.intent_prior,
            raw_model.intent_posterior,
            raw_model.future_token_decoder,
            raw_model.intent_film,
            raw_model.intent_norm_h,
            raw_model.intent_norm_r,
        )
        semantic_parameter_ids = {
            id(parameter)
            for module in semantic_modules
            for parameter in module.parameters()
            if parameter.requires_grad
        }
        semantic_lr_scale = (
            float(getattr(config.intent, "pretrained_semantic_lr_scale", 0.1))
            if args.stage == 2
            else 1.0
        )
        if not 0.0 < semantic_lr_scale <= 1.0:
            raise ValueError("intent.pretrained_semantic_lr_scale must be in (0, 1]")

        logger = None
        if is_main_process():
            logger = get_logger(config.log_file, "train")
            link_file(config.log_file, config.link_log_file)
            print_and_log_info(logger, json.dumps(config, indent=4, sort_keys=True))

        checkpoint_path = args.model_pth or config.model_pth
        if args.stage == 2 and not checkpoint_path:
            raise RuntimeError(
                "Stage-2 fine-tuning requires --model-pth pointing to a Stage-1 checkpoint"
            )
        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            load_report = raw_model.load_compatible_state_dict(state_dict)
            critical_missing = [
                key
                for key in load_report["missing"]
                if key.startswith(
                    (
                        "interaction_encoder.",
                        "intent_prior.",
                        "intent_posterior.",
                        "future_token_decoder.",
                        "intent_film.",
                    )
                )
            ]
            if args.stage == 2 and critical_missing:
                raise RuntimeError(
                    "Stage-1 checkpoint does not contain the current cross-attention "
                    "interaction-token/intent modules. Rerun H-H pretraining with the "
                    f"current model before Stage-2. Missing: {critical_missing}"
                )
            if is_main_process() and logger is not None:
                print_and_log_info(logger, "Loading model path from {} ".format(checkpoint_path))
                print_and_log_info(logger, "Compatible checkpoint report: {}".format(load_report))

        if distributed:
            model = DDP(
                raw_model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
                broadcast_buffers=False,
            )
            # Same cross-attention blocks are reused twice in one forward.
            model._set_static_graph()

        base_parameters = [
            parameter
            for parameter in unwrap_model(model).parameters()
            if parameter.requires_grad and id(parameter) not in semantic_parameter_ids
        ]
        semantic_parameters = [
            parameter
            for parameter in unwrap_model(model).parameters()
            if parameter.requires_grad and id(parameter) in semantic_parameter_ids
        ]
        optimizer = torch.optim.Adam(
            [
                {
                    "params": base_parameters,
                    "lr": config.cos_lr_max,
                    "lr_scale": 1.0,
                },
                {
                    "params": semantic_parameters,
                    "lr": config.cos_lr_max * semantic_lr_scale,
                    "lr_scale": semantic_lr_scale,
                },
            ],
            lr=config.cos_lr_max,
            weight_decay=config.weight_decay,
        )

        nb_iter = 0
        avg_loss = 0.0
        avg_lr = 0.0
        if is_main_process():
            ensure_dir(os.path.join(config.snapshot_dir, "./model"))

        while (nb_iter + 1) < config.cos_lr_total_iters:
            if train_sampler is not None:
                train_sampler.set_epoch(nb_iter)

            data_iter = dataloader
            if is_main_process():
                data_iter = tqdm(dataloader)

            for harper_motion_input, harper_motion_target in data_iter:
                loss, optimizer, current_lr, loss_h, loss_r = train_step(
                    harper_motion_input,
                    harper_motion_target,
                    model,
                    optimizer,
                    nb_iter,
                    config.cos_lr_total_iters,
                    config.cos_lr_max,
                    config.cos_lr_min,
                    config=config,
                    args=args,
                    dct_m=dct_m,
                    idct_m=idct_m,
                    device=device,
                    writer=writer,
                    wandb_log_every=wandb_log_every,
                )
                avg_loss += loss
                avg_lr += current_lr

                if is_main_process() and (nb_iter + 1) % config.print_every == 0:
                    avg_loss = avg_loss / config.print_every
                    avg_lr = avg_lr / config.print_every
                    print_and_log_info(logger, "Iter {} Summary: ".format(nb_iter + 1))
                    print_and_log_info(logger, f"\t lr: {avg_lr} \t Training loss: {avg_loss}")
                    avg_loss = 0
                    avg_lr = 0

                if is_main_process() and (nb_iter + 1) % config.print_loss == 0:
                    print(nb_iter + 1)
                    print(f"loss {loss}, loss_h {loss_h}, loss_robot {loss_r}")

                if (nb_iter + 1) % config.save_every == 0:
                    barrier()
                    if is_main_process():
                        print(nb_iter + 1)
                        torch.save(
                            unwrap_model(model).state_dict(),
                            config.snapshot_dir
                            + "/model"
                            + "/model-iter-"
                            + str(nb_iter + 1)
                            + ".pth",
                        )
                        unwrap_model(model).eval()
                        res_dict = test(eval_config, unwrap_model(model), eval_dataloader)
                        acc_log.write("".join(str(nb_iter + 1) + "\n"))
                        line = ""
                        eval_metrics = {}
                        for key, value in res_dict.items():
                            line += str(key) + "," + ",".join([str(a) for a in value]) + "\n"
                            if hasattr(value, "__len__") and len(value) > 0:
                                eval_metrics[f"eval/{key}"] = (
                                    float(value[0])
                                    if len(value) == 1
                                    else float(sum(value) / len(value))
                                )
                                for i, v in enumerate(value):
                                    eval_metrics[f"eval/{key}/t{i}"] = float(v)
                        acc_log.write("".join(line))
                        if eval_metrics:
                            wandb_log(eval_metrics, step=nb_iter + 1)
                        unwrap_model(model).train()
                    barrier()

                if (nb_iter + 1) == config.cos_lr_total_iters:
                    break
                nb_iter += 1

        if is_main_process() and acc_log is not None:
            acc_log.close()
            shutil.copyfile(
                "./result/" + args.exp_name, os.path.join(args.work_dir, args.exp_name)
            )
            shutil.copyfile(
                "./result/" + args.exp_name, os.path.join(config.snapshot_dir, args.exp_name)
            )
    finally:
        if is_main_process():
            if writer is not None:
                writer.close()
            finish_wandb()
        cleanup_distributed()


if __name__ == "__main__":
    main()
