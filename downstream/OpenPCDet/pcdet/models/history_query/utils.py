import numpy as np
def cosine_anneal(step, tau_start, tau_end, max_step):
    if step <= max_step:
        return tau_end + 0.5 * (tau_start - tau_end) * (1 + np.cos(step / max_step * np.pi))
    else:
        return tau_end