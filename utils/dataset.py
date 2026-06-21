def total_containers(cbf):
    return sum(n for POD in cbf.values() for n in POD.values())

def remaining_PODs(cbf):
    """从嵌套 cbf 里取出还有剩余(count>0)的目的港POD集合"""
    return {POD for PODs in cbf.values()
              for POD, n in PODs.items() if n > 0}
