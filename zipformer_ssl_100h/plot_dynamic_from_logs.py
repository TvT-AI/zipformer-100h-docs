import os
import glob
import re
import matplotlib.pyplot as plt
import numpy as np

# --- Set Research Paper Style (IEEE / ACL / INTERSPEECH standard) ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif', 'Times New Roman', 'Computer Modern Roman']
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9.5
plt.rcParams['figure.titlesize'] = 13

def parse_validation_loss_from_log_dirs(candidate_dirs):
    """
    Reads all log-train* files from the given log directories and extracts
    per-epoch validation loss values.
    """
    target_dir = None
    for d in candidate_dirs:
        if os.path.exists(d) and glob.glob(os.path.join(d, "log-train*")):
            target_dir = d
            break
            
    if not target_dir:
        print(f"[WARNING] No log-train files found in: {candidate_dirs}")
        return [], []
        
    log_files = glob.glob(os.path.join(target_dir, "log-train*"))
    log_files.sort(key=os.path.getmtime, reverse=True)

    epoch_losses = {}
    for log_file in log_files:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                match = re.search(r'Epoch\s+(\d+),\s+validation:\s+loss=([0-9.]+)', line)
                if match:
                    epoch = int(match.group(1))
                    loss = float(match.group(2))
                    if loss < 50:
                        epoch_losses[epoch] = loss

    sorted_epochs = sorted(epoch_losses.keys())
    sorted_losses = [epoch_losses[e] for e in sorted_epochs]
    return sorted_epochs, sorted_losses

# --- Root directory: resolved automatically from script location, no hard-coded paths ---
curr_dir = os.path.dirname(os.path.abspath(__file__))

print(">>> Reading training log files from disk...")

# 1. Pre-training logs (resolved relative to this script's directory)
pre1_ep, pre1_loss = parse_validation_loss_from_log_dirs([os.path.join(curr_dir, "step2_ssl_iter1_pretrain", "log")])
pre2_ep, pre2_loss = parse_validation_loss_from_log_dirs([os.path.join(curr_dir, "step4_ssl_iter2_pretrain", "log")])
pre3_ep, pre3_loss = parse_validation_loss_from_log_dirs([os.path.join(curr_dir, "step6_ssl_iter3_pretrain", "log")])

# 2. Fine-tuning logs
ft1_ep, ft1_loss = parse_validation_loss_from_log_dirs([os.path.join(curr_dir, "step3_ssl_iter1_finetune", "log")])
ft2_ep, ft2_loss = parse_validation_loss_from_log_dirs([os.path.join(curr_dir, "step5_ssl_iter2_finetune", "log")])
ft3_ep, ft3_loss = parse_validation_loss_from_log_dirs([os.path.join(curr_dir, "step7_ssl_iter3_finetune", "log")])

print(f"-> Pre-train Iter 1 ({len(pre1_ep)} epochs): min loss = {min(pre1_loss) if pre1_loss else 'N/A'}")
print(f"-> Pre-train Iter 2 ({len(pre2_ep)} epochs): min loss = {min(pre2_loss) if pre2_loss else 'N/A'}")
print(f"-> Pre-train Iter 3 ({len(pre3_ep)} epochs): min loss = {min(pre3_loss) if pre3_loss else 'N/A'}")
print(f"-> Fine-tune Iter 1 ({len(ft1_ep)} epochs): min loss = {min(ft1_loss) if ft1_loss else 'N/A'}")
print(f"-> Fine-tune Iter 2 ({len(ft2_ep)} epochs): min loss = {min(ft2_loss) if ft2_loss else 'N/A'}")
print(f"-> Fine-tune Iter 3 ({len(ft3_ep)} epochs): min loss = {min(ft3_loss) if ft3_loss else 'N/A'}")

# --- Plot Figure ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

# Academic Color Palette
c_iter1 = '#1f77b4' # Navy/Blue
c_iter2 = '#ff7f0e' # Orange
c_iter3 = '#2ca02c' # Green

# Subplot (a): HuBERT Pre-training
if pre1_ep: ax1.plot(pre1_ep, pre1_loss, marker='o', color=c_iter1, label=f'Pre-training Iteration 1 ({len(pre1_ep)} Epochs)', linewidth=1.8, markersize=4)
if pre2_ep: ax1.plot(pre2_ep, pre2_loss, marker='s', color=c_iter2, label=f'Pre-training Iteration 2 ({len(pre2_ep)} Epochs)', linewidth=1.8, markersize=4)
if pre3_ep: ax1.plot(pre3_ep, pre3_loss, marker='^', color=c_iter3, label=f'Pre-training Iteration 3 ({len(pre3_ep)} Epochs)', linewidth=1.8, markersize=4)

ax1.set_title('(a) Self-Supervised Pre-training', fontweight='bold', pad=10)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Cross-Entropy Loss')
ax1.set_xlim(0, 19)
ax1.set_ylim(0, 3.0)
ax1.set_xticks(range(1, 19, 2))
ax1.grid(True, linestyle='--', alpha=0.5, linewidth=0.6)
ax1.legend(loc='upper right', frameon=True, framealpha=0.9, edgecolor='gray')

# Annotate Min Losses
if pre1_loss:
    ax1.annotate(f'min: {pre1_loss[-1]:.3f}', xy=(pre1_ep[-1], pre1_loss[-1]), xytext=(pre1_ep[-1]+0.5, pre1_loss[-1]+0.2),
                 arrowprops=dict(arrowstyle='->', color=c_iter1, lw=1), color=c_iter1, fontweight='bold', fontsize=8.5)
if pre2_loss:
    ax1.annotate(f'min: {pre2_loss[-1]:.3f}', xy=(pre2_ep[-1], pre2_loss[-1]), xytext=(pre2_ep[-1]-4, pre2_loss[-1]+0.25),
                 arrowprops=dict(arrowstyle='->', color=c_iter2, lw=1), color=c_iter2, fontweight='bold', fontsize=8.5)
if pre3_loss:
    ax1.annotate(f'min: {pre3_loss[-1]:.3f}', xy=(pre3_ep[-1], pre3_loss[-1]), xytext=(pre3_ep[-1]-4, pre3_loss[-1]+0.25),
                 arrowprops=dict(arrowstyle='->', color=c_iter3, lw=1), color=c_iter3, fontweight='bold', fontsize=8.5)

# Subplot (b): Downstream Fine-tuning
if ft1_ep: ax2.plot(ft1_ep, ft1_loss, marker='o', color=c_iter1, label=f'Fine-tuned from Iteration 1 ({len(ft1_ep)} Epochs)', linewidth=1.8, markersize=4)
if ft2_ep: ax2.plot(ft2_ep, ft2_loss, marker='s', color=c_iter2, label=f'Fine-tuned from Iteration 2 ({len(ft2_ep)} Epochs)', linewidth=1.8, markersize=4)
if ft3_ep: ax2.plot(ft3_ep, ft3_loss, marker='^', color=c_iter3, label=f'Fine-tuned from Iteration 3 ({len(ft3_ep)} Epochs)', linewidth=1.8, markersize=4)

ax2.set_title('(b) Downstream ASR Fine-tuning', fontweight='bold', pad=10)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Pruned Transducer Loss')
ax2.set_xlim(0, 19)
ax2.set_ylim(0, 10.0)
ax2.set_xticks(range(1, 19, 2))
ax2.grid(True, linestyle='--', alpha=0.5, linewidth=0.6)
ax2.legend(loc='upper right', frameon=True, framealpha=0.9, edgecolor='gray')

# Annotate Min Losses
if ft1_loss:
    ax2.annotate(f'min: {ft1_loss[-1]:.3f}', xy=(ft1_ep[-1], ft1_loss[-1]), xytext=(ft1_ep[-1]+0.5, ft1_loss[-1]+1.0),
                 arrowprops=dict(arrowstyle='->', color=c_iter1, lw=1), color=c_iter1, fontweight='bold', fontsize=8.5)
if ft2_loss:
    ax2.annotate(f'min: {ft2_loss[-1]:.3f}', xy=(ft2_ep[-1], ft2_loss[-1]), xytext=(ft2_ep[-1]-4, ft2_loss[-1]+1.2),
                 arrowprops=dict(arrowstyle='->', color=c_iter2, lw=1), color=c_iter2, fontweight='bold', fontsize=8.5)
if ft3_loss:
    ax2.annotate(f'min: {ft3_loss[-1]:.3f}', xy=(ft3_ep[-1], ft3_loss[-1]), xytext=(ft3_ep[-1]-4, ft3_loss[-1]+0.4),
                 arrowprops=dict(arrowstyle='->', color=c_iter3, lw=1), color=c_iter3, fontweight='bold', fontsize=8.5)

plt.tight_layout()

# Caption
fig.text(0.5, -0.05, 'Figure 1: Validation loss trajectories across three iterations of HuBERT pre-training and downstream ASR fine-tuning.',
         ha='center', fontsize=11, fontstyle='italic')

# Save PNG and PDF
output_png = os.path.join(curr_dir, 'research_paper_loss_plot.png')
output_pdf = os.path.join(curr_dir, 'research_paper_loss_plot.pdf')

plt.savefig(output_png, dpi=300, bbox_inches='tight')
plt.savefig(output_pdf, bbox_inches='tight')
print(f"\n[SUCCESS] Loss plot generated successfully!\n -> PNG: {output_png}\n -> PDF: {output_pdf}")
