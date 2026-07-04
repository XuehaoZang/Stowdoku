import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib import cm


def print_vessel(snap):
    """
    打印田字格，每个bay：
        left  right
deck:    X     X
hold:    X     X

    snap: Vessel实例，或snapshot()格式的dict（含"cell"）
    """
    if isinstance(snap, dict):
        cell      = snap["cell"]
        n_bay     = cell.shape[0]
        valid_arr = None
    else:
        cell      = snap.cell
        n_bay     = snap.n_bay
        valid_arr = snap.is_valid

    def cell_str(bay, lr, hd):
        if valid_arr is not None and not valid_arr[bay, lr, hd]:
            return " X "
        record = cell[bay, lr, hd]
        pod = record["POD"]
        if pod == -1:
            return " □ "
        suffix = "*" if record["type"] == "RF" else " "
        return f"{pod:2}{suffix}"

    headers = [f"  bay{b}  ".center(11) for b in range(n_bay)]
    print("        " + " | ".join(headers))

    for hd, hd_label in [(1, "deck"), (0, "hold")]:
        parts = []
        for bay in range(n_bay):
            left  = cell_str(bay, 0, hd)
            right = cell_str(bay, 1, hd)
            parts.append(f"  {left}  {right}  ")
        print(f"{hd_label:>6} " + " | ".join(parts))
    print()


def _default_port_colors(pod_values):
    """按出现的POD值自动分配颜色（tab20色板），避免硬编码港口数量。"""
    unique_pods = sorted(pod_values)
    cmap = cm.get_cmap("tab20", max(len(unique_pods), 1))
    return {pod: cmap(i) for i, pod in enumerate(unique_pods)}


def plot_bayplan(slots, title="bayplan", filename="bayplan.png", save_dir=".", port_colors=None):
    """
    绘制slot级配载图。
    slots: Vessel.proj_cell_to_vessel()的输出，含
           bay_idx, row_idx, tier_idx, lr, hd, POL, POD, type 列，
           以及can_40ft/can_20ft列（沿用full_slot_table自带的这两列，proj_cell_to_vessel透传保留）。
    port_colors: 可选{POD: color}，不传则按出现的POD值自动生成。

    颜色规则：
        can_20ft=True（现阶段求解器未决策的20ft-only槽位）  -> 浅灰
        can_40ft=True 且 POD==-1（有效但本次未分配）        -> 白色
        can_40ft=True 且 POD!=-1                            -> port_colors[POD]
        type=="RF"（分配了reefer）                          -> 叠加斜线标记

    限制：bay子图按bay_idx升序简单排列，不对应真实船体前后物理布局
    （代码库里目前没有bay_idx到物理Bay编号的反向映射工具）。
    """
    assigned_pods = slots.loc[slots["POD"] != -1, "POD"].unique().tolist()
    if port_colors is None:
        port_colors = _default_port_colors(assigned_pods)

    bay_ids = sorted(slots["bay_idx"].unique())
    n_bay = len(bay_ids)
    ncols = min(4, n_bay) if n_bay > 0 else 1
    nrows = max(1, -(-n_bay // ncols))  # ceil division

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.4 * nrows))
    axes = axes.flatten() if n_bay > 1 else [axes]

    for ax in axes:
        ax.set_visible(False)

    for i, bay_idx in enumerate(bay_ids):
        ax = axes[i]
        ax.set_visible(True)

        bay_df = slots[slots.bay_idx == bay_idx]
        rows = sorted(bay_df.row_idx.unique())
        tiers = sorted(bay_df.tier_idx.unique())
        row_pos = {r: i for i, r in enumerate(rows)}
        tier_pos = {t: i for i, t in enumerate(tiers)}

        for _, slot in bay_df.iterrows():
            x, y = row_pos[slot.row_idx], tier_pos[slot.tier_idx]

            if slot.can_20ft:
                color = "#D9D9D9"  # 浅灰：现阶段求解器未决策的20ft-only槽位
            elif slot.POD == -1:
                color = "white"    # 有效40ft槽位，本次未分配
            else:
                color = port_colors.get(slot.POD, "#3498db")

            rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=color, edgecolor="black", linewidth=0.3)
            ax.add_patch(rect)

            if slot.type == "RF":
                line = Line2D([x + 0.5, x - 0.5], [y + 0.5, y - 0.5], color="black", linewidth=0.4)
                ax.add_line(line)

        ax.set_xlim(-0.6, len(rows) - 0.4)
        ax.set_ylim(-0.6, len(tiers) - 0.4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"bay {bay_idx}", fontsize=9, fontweight="bold", pad=3)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_aspect(0.7)

    plt.suptitle(title, fontsize=12, y=0.98)

    legend_handles = [
        patches.Patch(color=color, label=f"POD={pod}", ec="black", lw=0.6)
        for pod, color in port_colors.items()
    ]
    legend_handles.append(patches.Patch(color="#D9D9D9", label="20ft(未决策)", ec="black", lw=0.5))
    legend_handles.append(patches.Patch(facecolor="white", edgecolor="black", hatch="//", label="reefer"))
    fig.legend(
        handles=legend_handles, loc="center left", bbox_to_anchor=(0.92, 0.5),
        title="POD & INFO", fontsize=8, frameon=True,
    )
    plt.subplots_adjust(right=0.9)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return save_path