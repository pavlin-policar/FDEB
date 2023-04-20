import numpy as np


def interleave(x0, x1):
    """Interleave two arrays along the first axis."""
    assert x0.shape[1:] == x1.shape[1:], "Incompatible shapes!"

    y = np.empty((x0.shape[0] + x1.shape[0], *x0.shape[1:]), dtype=x0.dtype)
    y[::2] = x0
    y[1::2] = x1
    return y


def subdivide_edge(e, axis=1):
    e = np.swapaxes(e, 0, axis)
    midpoints = np.array([(e0 + e1) / 2 for e0, e1 in zip(e[1:], e[:-1])])
    e = interleave(e, midpoints)
    e = np.swapaxes(e, axis, 0)
    return e


def compute_forces(e, e_compat, kp):
    # Left-mid spring direction
    v_spring_l = e[:, :-1, :] - e[:, 1:, :]
    v_spring_l = np.concatenate(
        [np.zeros((v_spring_l.shape[0], 1, v_spring_l.shape[-1])), v_spring_l],
        axis=1,
    )

    # Right-mid spring direction
    v_spring_r = e[:, 1:, :] - e[:, :-1, :]
    v_spring_r = np.concatenate(
        [v_spring_r, np.zeros((v_spring_l.shape[0], 1, v_spring_l.shape[-1]))],
        axis=1,
    )

    f_spring_l = np.sum(v_spring_l ** 2, axis=-1, keepdims=True)
    f_spring_r = np.sum(v_spring_r ** 2, axis=-1, keepdims=True)

    F_spring = kp * (f_spring_l * v_spring_l + f_spring_r * v_spring_r)

    # Electrostatic force
    v_electro = np.expand_dims(e, 1) - np.expand_dims(e, 0)
    np.fill_diagonal(e_compat, 0)  # No self-interactions
    e_compat = np.expand_dims(e_compat, (2, 3))
    f_electro = e_compat / (np.sqrt(np.sum(v_electro ** 2, axis=-1, keepdims=True)) + 1e-8)

    F_electro = np.sum(f_electro * v_electro, axis=0)

    F = F_spring + F_electro
    # The first and last points are fixed
    F[:, 0, :] = F[:, -1, :] = 0

    return F


def compute_edge_compatibility(edges):
    vec = edges[:, -1] - edges[:, 0]
    vec_norm = np.sum(vec ** 2, axis=1, keepdims=True)

    # Angle comptability
    compat_angle = (vec @ vec.T) / (vec_norm @ vec_norm.T + 1e-8)

    # Length compatibility
    l_avg = (vec_norm + vec_norm.T) / 2
    compat_length = 2 / (
        l_avg / (np.minimum(vec_norm, vec_norm.T) + 1e-8) +
        np.maximum(vec_norm, vec_norm.T) / (l_avg + 1e-8) +
        1e-8
    )

    # Distance compatibility
    midpoint = (edges[:, 0] + edges[:, -1]) / 2
    midpoint_dist = np.sqrt(np.sum((midpoint[None, :] - midpoint[:, None]) ** 2, axis=-1))
    dist_compat = l_avg / (l_avg + midpoint_dist + 1e-8)

    # Visibility compatibility
    # Project point endpoints onto the line segment:
    #   t = a*p / ab*ab
    #   proj = a + t*ab
    ap = edges[None, ...] - edges[:, None, None, 0]
    t = np.sum(ap * vec[:, None, None, :], axis=-1) / (np.sum(vec ** 2, axis=-1)[:, None, None] + 1e-8)
    I = edges[:, None, 0, None] + t[..., None] * vec[:, None, None, :]

    i0, i1 = I[..., 0, :], I[..., 1, :]
    Im = (i0 + i1) / 2

    denom = np.sqrt(np.sum((i0 - i1) ** 2, axis=-1))
    num = 2 * np.linalg.norm(midpoint[:, None, ...] - Im, axis=-1)

    visibility_compat = np.maximum(0, 1 - num / (denom + 1e-8))
    visibility_compat = np.minimum(visibility_compat, visibility_compat.T)

    # Combine compatibility scores
    return compat_angle * compat_length * dist_compat * visibility_compat


def fdeb(edges, K=0.1, n_iter=50, n_iter_reduction=2 / 3, lr=0.04, lr_reduction=0.5, num_cycles=6):
    initial_edge_vecs = edges[:, 0] - edges[:, -1]

    initial_edge_lengths = np.sum(initial_edge_vecs ** 2, axis=1, keepdims=True)
    # Add a small value to avoid division by zero
    initial_edge_lengths += 1

    # Compute edge compatibilities
    edge_compatibilities = compute_edge_compatibility(edges)

    for cycle in range(num_cycles):
        edges = subdivide_edge(edges)

        num_segments = edges.shape[1] - 1
        kp = K / initial_edge_lengths * num_segments
        kp = np.expand_dims(kp, 1)

        for epoch in range(n_iter):
            F = compute_forces(edges, edge_compatibilities, kp)
            edges += F * lr

        n_iter = int(np.ceil(n_iter * n_iter_reduction))
        lr = lr * lr_reduction

    return edges
