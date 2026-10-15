import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .method import HORIZONS, PrefixNet, conformal_quantile, quality_gate, reference_rate, spectral_rate


def prepare(args):
    rows = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    groups = {}
    source_root = Path(args.manifest).resolve().parent
    for row in rows:
        record = np.load(source_root / row["file"], allow_pickle=False)
        fs = float(record["fs"])
        signal = np.asarray(record["ppg"], dtype=np.float32).reshape(-1)
        events = np.sort(np.asarray(record["breath_times"], dtype=float))
        starts = row.get("starts", [0, 96, 192, 288])
        if len(starts) != 4 or any(b - a < 64 for a, b in zip(starts, starts[1:])):
            raise ValueError("Each subject needs four fixed, nonoverlapping episodes")
        episodes, labels = [], []
        for start in starts:
            lo, hi = round(start * fs), round((start + 64) * fs)
            if hi > len(signal):
                raise ValueError(f"Short record: {row['subject']}")
            episode = signal[lo:hi]
            targets = [reference_rate(events, start + h) for h in HORIZONS]
            if not np.isfinite(targets).all():
                raise ValueError(f"Reference-ineligible subject block: {row['subject']}")
            episodes.append(episode)
            labels.append(targets)
        split = row["split"]
        if split not in {"train", "dev", "cal", "test"}:
            raise ValueError("split must be train, dev, cal or test")
        if row["subject"] in groups:
            raise ValueError("Each subject must contribute exactly one block")
        groups[row["subject"]] = (split, np.stack(episodes), np.array(labels), fs)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for subject, (split, x, y, fs) in groups.items():
        target = out / split
        target.mkdir(exist_ok=True)
        np.savez_compressed(target / f"{subject}.npz", x=x, y=y, fs=fs)


def records(path, split):
    paths = sorted((Path(path) / split).glob("*.npz"))
    if not paths:
        raise ValueError(f"No subject blocks in {split}")
    return paths


def train(args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    net = PrefixNet().to(args.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    paths = records(args.data, "train")
    best = float("inf")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(args.epochs):
        net.train()
        losses = []
        for index in rng.permutation(len(paths)):
            block = np.load(paths[index])
            x = torch.as_tensor(block["x"], device=args.device)[:, None]
            y = torch.as_tensor(block["y"], dtype=torch.float32, device=args.device)
            optimizer.zero_grad()
            loss = 0
            for k, h in enumerate(HORIZONS):
                mu, scale = net(x[..., :round(h * float(block["fs"]))])
                loss = loss + (((mu - y[:, k]).abs() / scale + scale.log()).mean() / len(HORIZONS))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        net.eval()
        with torch.no_grad():
            validation = []
            for path in records(args.data, "dev"):
                block = np.load(path)
                x = torch.as_tensor(block["x"], device=args.device)[:, None]
                for k, h in enumerate(HORIZONS):
                    mu, scale = net(x[..., :round(h * float(block["fs"]))])
                    y = torch.as_tensor(block["y"][:, k], device=args.device)
                    validation.append(float(((mu - y).abs() / scale + scale.log()).mean()))
        score = float(np.mean(validation))
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "dev_loss": score})
        if score < best:
            best = score
            torch.save(net.state_dict(), out / "model.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2))


def predictions(args, split):
    net = PrefixNet().to(args.device)
    net.load_state_dict(torch.load(args.checkpoint, map_location=args.device, weights_only=True))
    net.eval()
    outputs = []
    with torch.no_grad():
        for path in records(args.data, split):
            block = np.load(path)
            mu, scales, gates = [], [], []
            for h in HORIZONS:
                prefix = block["x"][:, :round(h * float(block["fs"]))]
                if args.predictor == "spectral":
                    mean = np.array([spectral_rate(x, float(block["fs"]))[0] for x in prefix])
                    scale = np.ones(4)
                else:
                    m, s = net(torch.as_tensor(prefix, device=args.device)[:, None])
                    mean, scale = m.cpu().numpy(), s.cpu().numpy()
                mu.append(mean)
                scales.append(np.ones(4) if args.constant_scale else scale)
                gates.append([quality_gate(x, float(block["fs"]), args.min_cycles, args.concentration) for x in prefix])
            outputs.append((path.stem, np.stack(mu, -1), np.stack(scales, -1), block["y"], np.stack(gates, -1)))
    return outputs


def evaluate(args):
    calibration = predictions(args, "cal")
    residuals = np.stack([np.abs(mu - y) / scale for _, mu, scale, y, _ in calibration])
    if args.calibration == "block":
        q = conformal_quantile(residuals.max((1, 2)), args.alpha)
    elif args.calibration == "pooled":
        q = conformal_quantile(residuals.ravel(), args.alpha)
    else:
        q = np.array([conformal_quantile(residuals[:, :, k].ravel(), args.alpha) for k in range(5)])
    subject_rows, episode_rows = [], []
    for subject, mu, scale, y, gate in predictions(args, "test"):
        radius = scale * q
        covered = np.abs(mu - y) <= radius
        subject_rows.append(bool(covered.all()))
        for e in range(4):
            eligible = (radius[e] <= args.width) & (gate[e] | args.no_gate)
            if args.policy.startswith("fixed"):
                k = HORIZONS.index(int(args.policy[5:]))
                emit = True
            else:
                if args.policy == "scale":
                    eligible = (scale[e] <= args.width) & (gate[e] | args.no_gate)
                emit = bool(eligible.any())
                k = int(np.flatnonzero(eligible)[0]) if emit else 4
            episode_rows.append({"subject": subject, "episode": e, "emitted": emit,
                                 "seconds": HORIZONS[k], "prediction": float(mu[e, k]),
                                 "target": float(y[e, k]), "covered": bool(covered[e, k]),
                                 "half_width": float(radius[e, k]) if np.isfinite(radius[e, k]) else None})
    emitted = [row for row in episode_rows if row["emitted"]]
    errors = np.array([row["prediction"] - row["target"] for row in emitted])
    metrics = {"mae_br_min": float(np.mean(np.abs(errors))) if len(errors) else None,
               "rmse_br_min": float(np.sqrt(np.mean(errors ** 2))) if len(errors) else None,
               "emitted_coverage": float(np.mean([r["covered"] for r in emitted])) if emitted else None,
               "block_joint_coverage": float(np.mean(subject_rows)),
               "output_fraction": len(emitted) / len(episode_rows),
               "mean_observation_seconds": float(np.mean([r["seconds"] for r in episode_rows]))}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    (out / "episodes.json").write_text(json.dumps(episode_rows, indent=2, allow_nan=False))
    fig, ax = plt.subplots(figsize=(6, 4))
    if emitted:
        ax.scatter([r["target"] for r in emitted], [r["prediction"] for r in emitted], s=16)
    ax.set(xlabel="Reference RR (breaths/min)", ylabel="Emitted RR (breaths/min)")
    fig.tight_layout()
    fig.savefig(out / "rates.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", default="data/prepared")
    p.set_defaults(function=prepare)
    for name in ["train", "evaluate"]:
        p = commands.add_parser(name)
        p.add_argument("--data", default="data/prepared")
        p.add_argument("--output", default=f"runs/{name}")
        p.add_argument("--device", default="cpu")
        if name == "train":
            p.add_argument("--epochs", type=int, default=40)
            p.add_argument("--seed", type=int, default=42)
            p.add_argument("--lr", type=float, default=1e-3)
            p.set_defaults(function=train)
        else:
            p.add_argument("--checkpoint", default="runs/train/model.pt")
            p.add_argument("--alpha", type=float, default=0.1)
            p.add_argument("--width", type=float, default=2)
            p.add_argument("--min-cycles", type=float, default=2)
            p.add_argument("--concentration", type=float, default=0.15)
            p.add_argument("--calibration", choices=["block", "marginal", "pooled"], default="block")
            p.add_argument("--predictor", choices=["neural", "spectral"], default="neural")
            p.add_argument("--policy", choices=["adaptive", "scale", "fixed16", "fixed32", "fixed64"], default="adaptive")
            p.add_argument("--constant-scale", action="store_true")
            p.add_argument("--no-gate", action="store_true")
            p.set_defaults(function=evaluate)
    args = parser.parse_args()
    args.function(args)
