
def print_vessel(snap: dict):
    """
    打印田字格，每个bay：
        left  right
deck:    X     X
hold:    X     X
    """
    if isinstance(snap, dict):
        pod_arr   = snap["vessel_pod"]
        type_arr  = snap["vessel_type"]
        n_bay     = pod_arr.shape[0]
        valid_arr = None
    else:
        pod_arr   = snap.vessel_pod
        type_arr  = snap.vessel_type
        n_bay     = snap.n_bay
        valid_arr = snap.is_valid
 
    def cell_str(bay, lr, hd):
        if valid_arr is not None and not valid_arr[bay, lr, hd]:
            return " X "
        pod = pod_arr[bay, lr, hd]
        if pod == -1:
            return " □ "
        suffix = "*" if type_arr[bay, lr, hd] == "RF" else " "
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