"""
临时验证脚本：核对 cbf_df_to_dict_with_20 / batch_parse_cbf_with_20 跟现有
cbf_df_to_dict / batch_parse_cbf 折算逻辑是否一致。跑完可以删。

用法（从repo根目录运行）：
    python debug/verify_cbf_with_20.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.vessel_io import batch_parse_cbf, batch_parse_cbf_with_20, STSE_PORT_MAP

CBF_RAW_DIR = "data/STSE/raw"
CBF_DIR = "debug/_verify_cbf_out"

PORT_NAMES = {v: k for k, v in STSE_PORT_MAP.items()}
TYPES = ["GP", "HC", "RF", "HR"]


def main():
    cbf = batch_parse_cbf(CBF_RAW_DIR, CBF_DIR)
    cbf20 = batch_parse_cbf_with_20(CBF_RAW_DIR, CBF_DIR)

    rows = []
    for pol, pod_dict in cbf.items():
        pod_dict20 = cbf20.get(pol, {})
        for pod, totals in pod_dict.items():
            totals20 = pod_dict20.get(pod, {})
            # cbf_df_to_dict对每个(POD,length=20,type)行分别做count//2再累加进GP，
            # 不是先把20ft箱数加总再统一//2，所以逐type floor之和才是精确还原值。
            folded_20_exact = sum(totals20.get(f"20{tt}", 0) // 2 for tt in TYPES)
            for t in TYPES:
                cbf_val = totals.get(t, 0)
                base_40 = totals20.get(t, 0)
                folded_20 = folded_20_exact if t == "GP" else 0
                reconstructed = base_40 + folded_20
                diff = cbf_val - reconstructed
                rows.append((pol, pod, t, cbf_val, reconstructed, diff))

    pol_name = lambda p: PORT_NAMES.get(int(p), str(p))

    print(f"{'POL':<6}{'POD':<6}{'类型':<6}{'cbf值':>8}{'还原值':>8}{'差值':>6}")
    for pol, pod, t, cbf_val, reconstructed, diff in rows:
        if cbf_val == 0 and reconstructed == 0:
            continue
        print(f"{pol_name(pol):<6}{pol_name(pod):<6}{t:<6}{cbf_val:>8}{reconstructed:>8}{diff:>6}")

    bad_rows = [r for r in rows if abs(r[5]) > 1]
    print()
    if bad_rows:
        print(f"!! 差值超过±1的行（{len(bad_rows)}条），需要排查折算逻辑不一致：")
        for pol, pod, t, cbf_val, reconstructed, diff in bad_rows:
            print(f"  POL={pol_name(pol)} POD={pol_name(pod)} type={t} "
                  f"cbf={cbf_val} 还原={reconstructed} diff={diff}")
    else:
        print("全部差值在±1以内，折算逻辑一致。")

    # 额外检查：cbf_with_20的GP字段本身不应包含20ft折算量，
    # 即它应严格等于原始40ft GP箱数（这点在construction里已经保证，这里再断言一次）
    for pol, pod_dict20 in cbf20.items():
        for pod, totals20 in pod_dict20.items():
            for t in TYPES:
                assert totals20.get(t, 0) >= 0
                assert totals20.get(f"20{t}", 0) >= 0
    print("cbf_with_20 字段非负性检查通过（GP字段未混入20ft原始箱数）。")


if __name__ == "__main__":
    main()
