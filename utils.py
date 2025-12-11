import numpy as np

def distance_to_colors(distances):
    distances[np.isinf(distances)] = np.nanmax(distances[np.isfinite(distances)])
    dnorm = (distances - distances.min()) / (distances.max() - distances.min())
    dnorm = dnorm**1.2
    colors = np.zeros((len(distances), 4), dtype=np.uint8)
    colors[:,0] = (255 * (1.0 - dnorm)).astype(np.uint8)  # red
    colors[:,2] = (255 * dnorm).astype(np.uint8)          # blue
    colors[:,3] = 100                                     # alpha
    return colors
