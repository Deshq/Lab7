

def split_matrix(seq, worker):
    """ 
    Split matrix into small parts according to the no of workers. These
    parts will be send to slaves by master node
    """
    rows = []
    division_value = len(seq) // worker
    remainder = len(seq) % worker
    start = 0
    end = division_value + min(1, remainder)
    for i in range(worker):
        rows.append(seq[start:end])
        remainder = max(0, remainder - 1)
        start = end
        end += division_value + min(1, remainder)

    return rows