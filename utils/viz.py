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


def _render_bayplan(slots, title, filename, save_dir, port_colors, port_names=None, use_phy_labels=False):
    """
    实际绘图逻辑，被plot_bayplan调用一次(idx模式)或两次(idx+phy模式)。
    use_phy_labels=False：bay标题用bay_idx数字，行/层不标刻度(现状不变)。
    use_phy_labels=True ：bay标题、行刻度、层刻度都换成真实物理Bay/Row/Tier码
                          (通过utils.vessel_io.idx_to_phy_bay/STSE_ROW_LABELS/
                          STSE_TIER_LABELS反查)，只是标签换了皮，底层排布逻辑
                          (哪个格子画在哪个位置)完全不变。
    """
    from utils.vessel_io import idx_to_phy_bay, STSE_ROW_LABELS, STSE_TIER_LABELS

    assigned_pods = slots.loc[slots["POD"] != -1, "POD"].unique().tolist()
    if port_colors is None:
        port_colors = _default_port_colors(assigned_pods)

    present_bays = set(slots["bay_idx"].unique())
    positions = _bay_grid_positions(ncols=4)
    positions = {b: pos for b, pos in positions.items() if b in present_bays}

    nrows = max(r for r, c in positions.values()) + 1
    ncols = max(c for r, c in positions.values()) + 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.4 * nrows))
    axes = axes.reshape(nrows, ncols) if nrows * ncols > 1 else [[axes]]

    for r in range(nrows):
        for c in range(ncols):
            axes[r][c].set_visible(False)

    # 全局固定坐标系：所有bay子图统一用完整的row/tier范围(而不是每个bay
    # 各自实际出现过的row/tier)，保证格子大小在所有子图之间一致——
    # 某个bay没有某一层(比如没有88 tier)，对应位置就是空白，不会因为
    # 这个bay数据范围小而被拉伸放大。
    n_rows_global = len(STSE_ROW_LABELS)
    n_tiers_global = len(STSE_TIER_LABELS)
    global_row_pos = {rv: i for i, rv in enumerate(range(n_rows_global))}
    global_tier_pos = {t: i for i, t in enumerate(range(n_tiers_global))}

    for bay_idx, (r, c) in positions.items():
        ax = axes[r][c]
        ax.set_visible(True)

        bay_df = slots[slots.bay_idx == bay_idx]

        for _, slot in bay_df.iterrows():
            x, y = global_row_pos[slot.row_idx], global_tier_pos[slot.tier_idx]

            if slot.can_20ft:
                color = "#D9D9D9"  # 浅灰：现阶段求解器未决策的20ft-only槽位
            elif slot.POD == -1:
                color = "white"    # 有效40ft槽位，本次未分配
            else:
                color = port_colors.get(slot.POD, "#3498db")

            rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=color, edgecolor="black", linewidth=0.5)
            ax.add_patch(rect)

            if slot.can_reefer and slot.RF_count > 0:
                ax.text(x, y, "R", ha="center", va="center", fontsize=6, color="black")

        ax.set_xlim(-0.6, n_rows_global - 0.4)
        ax.set_ylim(-0.6, n_tiers_global - 0.4)

        if use_phy_labels:
            ax.set_xticks(list(global_row_pos.values()))
            ax.set_xticklabels(STSE_ROW_LABELS, fontsize=6)
            ax.set_yticks(list(global_tier_pos.values()))
            ax.set_yticklabels(STSE_TIER_LABELS, fontsize=6)
            ax.set_title(f"BAY {idx_to_phy_bay(bay_idx)}", fontsize=9, pad=3)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"bay {bay_idx}", fontsize=9, pad=3)

        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_aspect(0.7)

    plt.suptitle(title, fontsize=12, y=0.98)

    legend_handles = [
        patches.Patch(
            color=color,
            label=port_names.get(pod, f"POD={pod}") if port_names else f"POD={pod}",
            ec="black", lw=0.6,
        )
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


def plot_bayplan(slots, title="bayplan", filename="bayplan.png", save_dir=".",
                  port_colors=None, port_names=None, if_plot_phy=False):
    """
    绘制slot级配载图。
    slots: Vessel.proj_cell_to_vessel()的输出，含
           bay_idx, row_idx, tier_idx, lr, hd, POL, POD, GP_count, RF_count 列，
           以及can_40ft/can_20ft/can_reefer列（沿用full_slot_table自带的这几列，
           proj_cell_to_vessel透传保留）。
    port_colors: 可选{POD: color}，不传则按出现的POD值自动生成。
    if_plot_phy: False(默认)只输出idx版本(文件名不变)；
                 True则额外多输出一份物理坐标版本，文件名在原名基础上加"_phy"后缀
                 (如"6_YOK_DEP_bayplan.png" -> 同时输出"6_YOK_DEP_bayplan_phy.png")。

    颜色规则：
        can_20ft=True（现阶段求解器未决策的20ft-only槽位）  -> 浅灰
        can_40ft=True 且 POD==-1（有效但本次未分配）        -> 白色
        can_40ft=True 且 POD!=-1                            -> port_colors[POD]
        can_reefer=True 且 该cell的RF_count>0               -> 叠加"R"标记

    bay子图按_bay_grid_positions()算出的版面位置排布，对应真实配载图惯例
    （同一对pair的b0在上、b1在下，整体块顺序倒转，块内降序，Bay01排在末尾）。

    返回：文件路径列表（idx模式1个，if_plot_phy=True时2个）。
    """
    idx_path = _render_bayplan(slots, title, filename, save_dir, port_colors, port_names, use_phy_labels=False)
    paths = [idx_path]

    if if_plot_phy:
        stem, ext = os.path.splitext(filename)
        phy_filename = f"{stem}_phy{ext}"
        phy_path = _render_bayplan(slots, title, phy_filename, save_dir, port_colors, port_names, use_phy_labels=True)
        paths.append(phy_path)

    return paths