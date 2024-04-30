import numpy as np


def vertex_interp(isolevel, p1, p2, valp1, valp2):
    if abs(isolevel - valp1) < 1e-5:
        return p1
    if abs(isolevel - valp2) < 1e-5:
        return p2
    if abs(valp1 - valp2) < 1e-5:
        return p1

    mu = (isolevel - valp1) / (valp2 - valp1)
    p = np.zeros(3)
    p[0] = p1[0] + mu * (p2[0] - p1[0])
    p[1] = p1[1] + mu * (p2[1] - p1[1])
    p[2] = p1[2] + mu * (p2[2] - p1[2])

    return p
