import re
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def parse_log_file(filepath):
    steps, train_loss, train_acc = [], [], []
    epoch_summaries = []  # end-of-epoch lines with validation metrics

    step_pattern = re.compile(
        r"epoch:\s*(\d+)\s*\|\s*step:\s*(\d+)\s*\|\s*train_loss:\s*([\d.]+)\s*\|\s*train_acc:\s*([\d.]+)"
    )
    epoch_pattern = re.compile(
        r"epoch:\s*(\d+)\s*\|\s*train_loss:\s*([\d.]+)\s*\|\s*train_acc:\s*([\d.]+)"
        r"(?:\s*\|\s*valid_loss:\s*([\d.]+)\s*\|\s*valid_acc:\s*([\d.]+))?"
    )

    global_step = 0
    last_epoch = -1

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            m = step_pattern.search(line)
            if m:
                epoch, step, tl, ta = int(m.group(1)), int(m.group(2)), float(m.group(3)), float(m.group(4))
                # track absolute step across epochs
                if epoch > last_epoch:
                    last_epoch = epoch
                steps.append(global_step + step)
                train_loss.append(tl)
                train_acc.append(ta)
                continue

            m = epoch_pattern.search(line)
            if m and "step:" not in line:
                epoch = int(m.group(1))
                tl, ta = float(m.group(2)), float(m.group(3))
                vl = float(m.group(4)) if m.group(4) else None
                va = float(m.group(5)) if m.group(5) else None
                # use last known step as marker for epoch end
                epoch_end_step = steps[-1] if steps else 0
                epoch_summaries.append({
                    "epoch": epoch,
                    "step": epoch_end_step,
                    "train_loss": tl,
                    "train_acc": ta,
                    "valid_loss": vl,
                    "valid_acc": va,
                })
                # bump global step offset for next epoch
                global_step = epoch_end_step

    return steps, train_loss, train_acc, epoch_summaries


def plot_curves(filepath):
    steps, train_loss, train_acc, epoch_summaries = parse_log_file(filepath)

    has_valid = any(e["valid_loss"] is not None for e in epoch_summaries)

    fig = plt.figure(figsize=(14, 5))
    fig.suptitle("Training Curves", fontsize=15, fontweight="bold", y=1.01)
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    ax_loss = fig.add_subplot(gs[0])
    ax_acc  = fig.add_subplot(gs[1])

    # ── Loss ──────────────────────────────────────────────────────────────────
    ax_loss.plot(steps, train_loss, color="#4C72B0", linewidth=1.5, label="Train Loss")
    if has_valid:
        v_steps = [e["step"] for e in epoch_summaries if e["valid_loss"] is not None]
        v_loss  = [e["valid_loss"] for e in epoch_summaries if e["valid_loss"] is not None]
        ax_loss.plot(v_steps, v_loss, "o--", color="#DD8452", linewidth=1.8,
                     markersize=7, label="Valid Loss")

    ax_loss.set_title("Loss", fontsize=13)
    ax_loss.set_xlabel("Step")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    ax_loss.grid(True, linestyle="--", alpha=0.5)

    # ── Accuracy ──────────────────────────────────────────────────────────────
    train_acc_pct = [v * 100 for v in train_acc]
    ax_acc.plot(steps, train_acc_pct, color="#55A868", linewidth=1.5, label="Train Acc")
    if has_valid:
        v_steps   = [e["step"] for e in epoch_summaries if e["valid_acc"] is not None]
        v_acc_pct = [e["valid_acc"] * 100 for e in epoch_summaries if e["valid_acc"] is not None]
        ax_acc.plot(v_steps, v_acc_pct, "o--", color="#C44E52", linewidth=1.8,
                    markersize=7, label="Valid Acc")

    ax_acc.set_title("Accuracy", fontsize=13)
    ax_acc.set_xlabel("Step")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.legend()
    ax_acc.grid(True, linestyle="--", alpha=0.5)

    # ── Epoch boundary markers ─────────────────────────────────────────────────
    if len(epoch_summaries) > 1:
        for e in epoch_summaries[:-1]:
            ax_loss.axvline(e["step"], color="gray", linestyle=":", alpha=0.6)
            ax_acc.axvline(e["step"],  color="gray", linestyle=":", alpha=0.6)

    plt.tight_layout()
    out = filepath.rsplit(".", 1)[0] + "_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.show()


if __name__ == "__main__":
    log_path = sys.argv[1] if len(sys.argv) > 1 else "training.log"
    plot_curves(log_path)