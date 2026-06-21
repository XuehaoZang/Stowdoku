
def print_bay(bay):
    '''
    横向并排打印所有 bay，每个 bay 的列用空格分隔，bay 之间用 | 分隔
    bay0       | bay1       | bay2       | bay3
    -1  1      | -1 -1      | -1 -1      | -1 -1    ← tier 1 (上层)
    -1 -1      | -1 -1      |  2 -1      | -1 -1    ← tier 0 (下层)
    '''
    n_bay, n_row, n_tier = bay.shape
    header = " | ".join(f"bay {b:<6}" * 1 for b in range(n_bay)) 
    col_w = n_row * 2  # 每个 bay 占的字符宽度估算
    titles = [f"bay{b}".center(col_w) for b in range(n_bay)]
    print(" | ".join(titles))

    for t in range(n_tier - 1, -1, -1):
        row_parts = []
        for b in range(n_bay):
            cells = [(" X" if bay[b][r][t] == -1 else f"{str(bay[b][r][t]):>2}") for r in range(n_row)]
            row_parts.append(" ".join(cells))
        print(" | ".join(row_parts))
    print()
       