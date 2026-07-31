"""Optional Weights & Biases helpers for training scripts."""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Optional

_run = None


def _as_dict(cfg: Any) -> dict:
    if cfg is None:
        return {}
    if isinstance(cfg, Mapping):
        return dict(cfg)
    if hasattr(cfg, "items"):
        return dict(cfg.items())
    return {}


def resolve_wandb_settings(
    cfg: Any,
    *,
    enable: Optional[bool] = None,
    project: Optional[str] = None,
    entity: Optional[str] = None,
    run_name: Optional[str] = None,
    mode: Optional[str] = None,
    log_every: Optional[int] = None,
) -> dict:
    """Merge YAML wandb block with optional CLI overrides."""
    root = _as_dict(cfg)
    wandb_cfg = _as_dict(root.get("wandb", {}))

    settings = {
        "enable": bool(wandb_cfg.get("enable", False)),
        "project": str(wandb_cfg.get("project", "human-intent")),
        "entity": wandb_cfg.get("entity"),
        "run_name": wandb_cfg.get("run_name") or wandb_cfg.get("name"),
        "mode": str(wandb_cfg.get("mode", "online")),
        "log_every": int(wandb_cfg.get("log_every", 10)),
        "tags": list(wandb_cfg.get("tags") or []),
        "notes": wandb_cfg.get("notes"),
    }
    if enable is not None:
        settings["enable"] = bool(enable)
    if project:
        settings["project"] = project
    if entity:
        settings["entity"] = entity
    if run_name:
        settings["run_name"] = run_name
    if mode:
        settings["mode"] = mode
    if log_every is not None:
        settings["log_every"] = int(log_every)
    if settings["entity"] in {"", "null", "None"}:
        settings["entity"] = None
    if settings["run_name"] in {"", "null", "None"}:
        settings["run_name"] = None
    return settings


def init_wandb(
    settings: Mapping[str, Any],
    config: Optional[Mapping[str, Any]] = None,
    *,
    job_type: Optional[str] = None,
) -> Any:
    """Initialize a wandb run when enabled. Returns the run or None."""
    global _run
    if _run is not None:
        return _run
    if not settings.get("enable", False):
        return None
    mode = str(settings.get("mode", "online")).lower()
    if mode == "disabled":
        return None

    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "wandb is enabled but not installed. Run: pip install wandb"
        ) from exc

    init_kwargs: MutableMapping[str, Any] = {
        "project": settings.get("project", "human-intent"),
        "mode": mode,
        "config": dict(config or {}),
    }
    if settings.get("entity"):
        init_kwargs["entity"] = settings["entity"]
    if settings.get("run_name"):
        init_kwargs["name"] = settings["run_name"]
    if settings.get("tags"):
        init_kwargs["tags"] = list(settings["tags"])
    if settings.get("notes"):
        init_kwargs["notes"] = settings["notes"]
    if job_type:
        init_kwargs["job_type"] = job_type

    _run = wandb.init(**init_kwargs)
    return _run


def wandb_log(metrics: Mapping[str, Any], step: Optional[int] = None) -> None:
    """Log a metrics dict to the active wandb run, if any."""
    if _run is None:
        return
    payload = {k: v for k, v in metrics.items() if v is not None}
    if not payload:
        return
    if step is None:
        _run.log(payload)
    else:
        _run.log(payload, step=int(step))


def should_log(step: int, log_every: int) -> bool:
    every = max(1, int(log_every))
    return step % every == 0


def finish_wandb() -> None:
    global _run
    if _run is None:
        return
    _run.finish()
    _run = None


def add_wandb_args(parser) -> None:
    """Register common wandb CLI flags on an ArgumentParser."""
    group = parser.add_argument_group("wandb")
    group.add_argument(
        "--wandb",
        dest="wandb_enable",
        action="store_true",
        default=None,
        help="Enable Weights & Biases logging (overrides config)",
    )
    group.add_argument(
        "--no-wandb",
        dest="wandb_enable",
        action="store_false",
        help="Disable Weights & Biases logging (overrides config)",
    )
    group.add_argument("--wandb-project", type=str, default=None, help="wandb project name")
    group.add_argument("--wandb-entity", type=str, default=None, help="wandb entity/team")
    group.add_argument("--wandb-run-name", type=str, default=None, help="wandb run name")
    group.add_argument(
        "--wandb-mode",
        type=str,
        default=None,
        choices=["online", "offline", "disabled"],
        help="wandb mode",
    )
    group.add_argument(
        "--wandb-log-every",
        type=int,
        default=None,
        help="Log metrics to wandb every N optimizer steps",
    )
