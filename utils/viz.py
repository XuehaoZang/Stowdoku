import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils.vessel_io import STSE_BAY_PAIRS

plt.rcParams['font.sans-serif'] = ['SimHei']  # Use a font that supports Chinese characters
plt.rcParams['axes.unicode_minus'] = False     # Fixes minus sign rendering issues
plt.rcParams['mathtext.fontset'] = 'cm'


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
        suffix = "*" if record["RF_count"] > 0 else " "
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
    cmap = plt.get_cmap("tab20", max(len(unique_pods), 1))
    return {pod: cmap(i) for i, pod in enumerate(unique_pods)}


def _bay_grid_positions(ncols=4):
    """
    按真实配载图惯例排布bay位置（参考真实General Stowage Plan的版面顺序）：
    每个STSE_BAY_PAIRS的(b0,b1)当作一列，b0排上方子行、b1排下方子行；
    Bay01（bay_idx=0，无配对）当作一个只有b0没有b1的"伪pair"，接在序列最后（最小编号）。
    所有列按各自b0降序排列后，每ncols个一组分块；块的顺序整体倒转
    （编号小的块排在上面），块内部仍保持降序。

    例：7对pair + Bay01共8列，ncols=4时分两块：
        块1(小编号，排上方) -> row0: 6,4,2,0   row1: 7,5,3,1
        块2(大编号，排下方) -> row2: 14,12,10,8 row3: 15,13,11,9

    返回 {bay_idx: (row, col)}
    """
    groups = list(STSE_BAY_PAIRS) + [(0, None)]  # (0, None) = Bay01独立，没有b1
    groups_desc = sorted(groups, key=lambda g: g[0], reverse=True)

    chunks = [groups_desc[i:i + ncols] for i in range(0, len(groups_desc), ncols)]
    chunks.reverse()  # 小编号的块排在上面

    positions = {}
    for block_idx, chunk in enumerate(chunks):
        upper_row = block_idx * 2
        lower_row = block_idx * 2 + 1
        for col, (b0, b1) in enumerate(chunk):
            positions[b0] = (upper_row, col)
            if b1 is not None:
                positions[b1] = (lower_row, col)
    return positions


def plot_bayplan(slots, title="bayplan", filename="bayplan.png", save_dir=".", port_colors=None):
    """
    绘制slot级配载图。
    slots: Vessel.proj_cell_to_vessel()的输出，含
           bay_idx, row_idx, tier_idx, lr, hd, POL, POD, GP_count, RF_count 列，
           以及can_40ft/can_20ft/can_reefer列（沿用full_slot_table自带的这几列，
           proj_cell_to_vessel透传保留）。
    port_colors: 可选{POD: color}，不传则按出现的POD值自动生成。

    颜色规则：
        can_20ft=True（现阶段求解器未决策的20ft-only槽位）  -> 浅灰
        can_40ft=True 且 POD==-1（有效但本次未分配）        -> 白色
        can_40ft=True 且 POD!=-1                            -> port_colors[POD]
        can_reefer=True 且 该cell的RF_count>0               -> 叠加"R"标记

    注：GP_count/RF_count是cell级别的聚合数量，同一cell内所有物理槽位都共享
    这两个数值，我们不知道具体哪几个槽位实际装的是reefer——"R"标记画在
    can_reefer=True的槽位上，只要这个cell本身用到了reefer(RF_count>0)，
    这个cell里所有真正有reefer插座的槽位都会标"R"，可能比实际占用的略多，
    是田字格颗粒度下的合理近似，不是精确的逐箱位置。

    bay子图按_bay_grid_positions()算出的版面位置排布，对应真实配载图惯例
    （同一对pair的b0在上、b1在下，整体块顺序倒转，块内降序，Bay01排在末尾）。
    """
    assigned_pods = slots.loc[slots["POD"] != -1, "POD"].unique().tolist()
    if port_colors is None:
        port_colors = _default_port_colors(assigned_pods)

    present_bays = set(slots["bay_idx"].unique())
    positions = _bay_grid_positions(ncols=4)
    # 只保留真实存在于这份slots里的bay_idx（比如Bay01的伪配对b1本来就不存在）
    positions = {b: pos for b, pos in positions.items() if b in present_bays}

    nrows = max(r for r, c in positions.values()) + 1
    ncols = max(c for r, c in positions.values()) + 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.4 * nrows))
    axes = axes.reshape(nrows, ncols) if nrows * ncols > 1 else [[axes]]

    for r in range(nrows):
        for c in range(ncols):
            axes[r][c].set_visible(False)

    for bay_idx, (r, c) in positions.items():
        ax = axes[r][c]
        ax.set_visible(True)

        bay_df = slots[slots.bay_idx == bay_idx]
        rows = sorted(bay_df.row_idx.unique())
        tiers = sorted(bay_df.tier_idx.unique())
        row_pos = {rv: i for i, rv in enumerate(rows)}
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

            if slot.can_reefer and slot.RF_count > 0:
                ax.text(x, y, "R", ha="center", va="center", fontsize=6, color="black")

        ax.set_xlim(-0.6, len(rows) - 0.4)
        ax.set_ylim(-0.6, len(tiers) - 0.4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"bay {bay_idx}", fontsize=9, pad=3)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_aspect(0.7)

    plt.suptitle(title, fontsize=12, y=0.98)

    legend_handles = [
        patches.Patch(color=color, label=f"POD={pod}", ec="black", lw=0.6)
        for pod, color in port_colors.items()
    ]
    legend_handles.append(patches.Patch(color="#D9D9D9", label="20ft", ec="black", lw=0.5))
    legend_handles.append(patches.Patch(facecolor="white", edgecolor="black", label="Reefer"))
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