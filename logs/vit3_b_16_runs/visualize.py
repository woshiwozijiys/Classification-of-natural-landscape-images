import pandas as pd
import matplotlib.pyplot as plt
import os

# Read CSV
csv_path = os.path.join(os.path.dirname(__file__), "./train_log.csv")
df = pd.read_csv(csv_path)

epochs = df["epoch"]
loss = df["loss"]
acc = df["acc"]

# Create figure with two y-axes
fig, ax1 = plt.subplots(figsize=(12, 6))

# Loss curve (left y-axis)
color_loss = "#E74C3C"
ax1.set_xlabel("Epoch", fontsize=13)
ax1.set_ylabel("Loss", color=color_loss, fontsize=13)
line1, = ax1.plot(epochs, loss, color=color_loss, linewidth=1.8, marker="o", markersize=3, label="Loss")
ax1.tick_params(axis="y", labelcolor=color_loss)
ax1.set_ylim(bottom=0)

# Accuracy curve (right y-axis)
ax2 = ax1.twinx()
color_acc = "#2980B9"
ax2.set_ylabel("Accuracy", color=color_acc, fontsize=13)
line2, = ax2.plot(epochs, acc, color=color_acc, linewidth=1.8, marker="s", markersize=3, label="Accuracy")
ax2.tick_params(axis="y", labelcolor=color_acc)
ax2.set_ylim(0.8, 1.0)

# Title
plt.title("vit_b_16 Training Curve", fontsize=16, fontweight="bold")

# Grid
ax1.grid(True, alpha=0.3, linestyle="--")

# Legend (merge handles from both axes)
handles, labels = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(handles + h2, labels + l2, loc="center right", fontsize=12)

total = len(epochs)

# Annotate best accuracy
best_idx = acc.idxmax()
best_epoch = epochs[best_idx]
best_acc_val = acc[best_idx]
ax2.annotate(
    f"Best Acc: {best_acc_val:.4f} (Epoch {best_epoch})",
    xy=(best_epoch, best_acc_val),
    xytext=(best_epoch, best_acc_val + 0.006),
    ha="center",
    arrowprops=dict(arrowstyle="->", color="gray"),
    fontsize=11,
    color=color_acc,
)

# Annotate final values
ax1.annotate(
    f"Final Loss: {loss.iloc[-1]:.4f}",
    xy=(total, loss.iloc[-1]),
    xytext=(total - 6, loss.iloc[-1] + 0.04),
    arrowprops=dict(arrowstyle="->", color="gray"),
    fontsize=10,
    color=color_loss,
    ha="right",
)

ax2.annotate(
    f"Final Acc: {acc.iloc[-1]:.4f}",
    xy=(total, acc.iloc[-1]),
    xytext=(total - 6, acc.iloc[-1] + 0.006),
    arrowprops=dict(arrowstyle="->", color="gray"),
    fontsize=10,
    color=color_acc,
    ha="right",
)

plt.tight_layout()

# Save
out_path = os.path.join(os.path.dirname(__file__), "train_curve.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved to: {out_path}")
plt.close()
