"""Exercise a behavior-cloning experiment through one batch and backward pass."""

import runpy
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro

import wandb


@dataclass
class Args:
    experiment: Path = Path("experiments/009_simplify_arch.py")
    batch_size: int = 8
    L_ctx: int = 300
    cache_limit_gb: int = 12
    character_pair: tuple[int, int] = (1, 1)
    windows_per_replay: int = 64


def main(args: Args) -> None:
    if not args.experiment.is_file():
        raise SystemExit(f"experiment does not exist: {args.experiment}")

    module = runpy.run_path(str(args.experiment))
    cfg = module["TrainConfig"](
        L_ctx=args.L_ctx,
        batch_size=args.batch_size,
        character_pair=args.character_pair,
        windows_per_replay=args.windows_per_replay,
        num_workers=0,
    )
    device = module["DEVICE"]
    entity = wandb.Api().default_entity
    if not entity:
        raise RuntimeError("W&B credentials did not resolve an entity")
    print(f"[smoke] W&B entity={entity} device={device}", flush=True)

    stats = module["load_consolidated_stats"](Path(cfg.data_root) / "stats.json")
    loader = module["make_loader"](
        data_root=cfg.data_root,
        remote=module["streams"].remote_for_local(cfg.data_root),
        cache_limit=f"{args.cache_limit_gb}gb",
        shuffle_block_size=16,
        stats=stats,
        L_ctx=cfg.L_ctx,
        L_chunk=module["L_CHUNK"],
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        windows_per_replay=cfg.windows_per_replay,
        character_pair=cfg.character_pair,
        split="train",
        num_workers=0,
    )
    batch = next(iter(loader)).to(device)
    model = module["GPT"](cfg).to(device)
    losses = module["action_loss"](model, batch)
    loss = sum(component.mean() for component in losses.values())
    loss.backward()
    if device == "cuda":
        torch.cuda.synchronize()

    breakdown = {name: float(component.detach().mean()) for name, component in losses.items()}
    print(
        f"[smoke] experiment={args.experiment} batch={tuple(batch.target.shape)} "
        f"loss={float(loss.detach()):.4f} components={breakdown}",
        flush=True,
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
