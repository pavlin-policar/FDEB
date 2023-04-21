import numpy as np


def subdivide_edge(edges: np.ndarray, num_points: int) -> np.ndarray:
    """Subdivide edges into `num_points` segments.

     The points being uniformly distributed, as if walking along the line.

    Parameters
    ----------
    edges : array-like, shape (n_edges, current_points, 2)
        The edges to subdivide.
    num_points : int
        The number of points to generate along each edge.

    Returns
    -------
    new_points : array-like, shape (n_edges, num_points, 2)

    """
    segment_vecs = edges[:, 1:] - edges[:, :-1]
    segment_lens = np.linalg.norm(segment_vecs, axis=-1)
    cum_segment_lens = np.cumsum(segment_lens, axis=1)
    cum_segment_lens = np.hstack(
        [np.zeros((cum_segment_lens.shape[0], 1)), cum_segment_lens]
    )

    total_lens = cum_segment_lens[:, -1]

    # At which lengths do we want to generate new points
    t = np.linspace(0, 1, num=num_points, endpoint=True)
    desired_lens = t * total_lens[:, None]
    # Which segment should the new point be interpolated on
    i = np.argmax(desired_lens[:, None] < cum_segment_lens[..., None], axis=1)
    # At what percentage of the segment does this new point actually appear
    pct = (desired_lens - np.take_along_axis(cum_segment_lens, i - 1, axis=-1)) / (
        np.take_along_axis(segment_lens, i - 1, axis=-1) + 1e-8
    )

    row_indices = np.arange(edges.shape[0])[:, None]
    new_points = (
        (1 - pct[..., None]) * edges[row_indices, i - 1]
        + pct[..., None] * edges[row_indices, i]
    )

    return new_points


def compute_edge_compatibility(edges: np.ndarray) -> np.ndarray:
    """Compute pairwise-edge compatibility scores.

    Parameters
    ----------
    edges : array-like, shape (n_edges, n_points, 2)
        The edges to compute compatibility scores for.

    Returns
    -------
    compat : array-like, shape (n_edges, n_edges)
        The pairwise edge compatibility scores.

    """
    vec = edges[:, -1] - edges[:, 0]
    vec_norm = np.linalg.norm(vec, axis=1, keepdims=True)

    # Angle comptability
    compat_angle = np.abs((vec @ vec.T) / (vec_norm @ vec_norm.T + 1e-8))

    # Length compatibility
    l_avg = (vec_norm + vec_norm.T) / 2
    compat_length = 2 / (
        l_avg / (np.minimum(vec_norm, vec_norm.T) + 1e-8)
        + np.maximum(vec_norm, vec_norm.T) / (l_avg + 1e-8)
        + 1e-8
    )

    # Distance compatibility
    midpoint = (edges[:, 0] + edges[:, -1]) / 2
    midpoint_dist = np.linalg.norm(midpoint[None, :] - midpoint[:, None], axis=-1)
    compat_dist = l_avg / (l_avg + midpoint_dist + 1e-8)

    # Visibility compatibility
    # Project point endpoints onto the line segment:
    #   t = a*p / ab*ab
    #   proj = a + t*ab
    ap = edges[None, ...] - edges[:, None, None, 0]
    t = np.sum(ap * vec[:, None, None, :], axis=-1) / (
        np.sum(vec**2, axis=-1)[:, None, None] + 1e-8
    )
    I = edges[:, None, 0, None] + t[..., None] * vec[:, None, None, :]

    i0, i1 = I[..., 0, :], I[..., 1, :]
    Im = (i0 + i1) / 2

    denom = np.sqrt(np.sum((i0 - i1) ** 2, axis=-1))
    num = 2 * np.linalg.norm(midpoint[:, None, ...] - Im, axis=-1)

    compat_visibility = np.maximum(0, 1 - num / (denom + 1e-8))
    compat_visibility = np.minimum(compat_visibility, compat_visibility.T)

    # Combine compatibility scores
    return compat_angle * compat_length * compat_dist * compat_visibility


def compute_forces(e: np.ndarray, e_compat: np.ndarray, kp: np.ndarray) -> np.ndarray:
    """Compute forces on each edge point.

    Parameters
    ----------
    e : array-like, shape (n_edges, n_points, 2)
        The edge points.
    e_compat : array-like, shape (n_edges, n_edges)
        The pairwise edge compatibility scores.
    kp : array-like, shape (n_edges, 1, 1)
        The spring constant for each edge.

    Returns
    -------
    F : array-like, shape (n_edges, n_points, 2)
        The forces on each edge point.

    """
    # Left-mid spring direction
    v_spring_l = e[:, :-1] - e[:, 1:]
    v_spring_l = np.concatenate(
        [np.zeros((v_spring_l.shape[0], 1, v_spring_l.shape[-1])), v_spring_l],
        axis=1,
    )

    # Right-mid spring direction
    v_spring_r = e[:, 1:] - e[:, :-1]
    v_spring_r = np.concatenate(
        [v_spring_r, np.zeros((v_spring_l.shape[0], 1, v_spring_l.shape[-1]))],
        axis=1,
    )

    f_spring_l = np.sum(v_spring_l**2, axis=-1, keepdims=True)
    f_spring_r = np.sum(v_spring_r**2, axis=-1, keepdims=True)

    F_spring = kp * (f_spring_l * v_spring_l + f_spring_r * v_spring_r)

    # Electrostatic force
    v_electro = e[:, None, ...] - e[None, ...]
    f_electro = e_compat[..., None] / (np.linalg.norm(v_electro, axis=-1) + 1e-8)

    F_electro = np.sum(f_electro[..., None] * v_electro, axis=0)

    F = F_spring + F_electro
    # The first and last points are fixed
    F[:, 0, :] = F[:, -1, :] = 0

    return F


def fdeb(
    edges: np.ndarray,
    K: float = 0.1,
    n_iter: int = 60,
    n_iter_reduction: float = 2 / 3,
    lr: float = 0.04,
    lr_reduction: float = 0.5,
    n_cycles: int = 6,
    initial_segpoints: int = 1,
    segpoint_increase: float = 2,
    compat_threshold: float = 0.5,
) -> np.ndarray:
    """Run the Force-Directed Edge Bundling algorithm.

    Parameters
    ----------
    edges: array-like, shape (n_edges, 2, 2)
        The edge points.
    K: float
        The spring constant.
    n_iter: int
        The number of iterations to run in the first cycle.
    n_iter_reduction: float
        The factor by which to reduce the number of iterations in each cycle.
    lr: float
        The learning rate.
    lr_reduction: float
        The factor by which to reduce the learning rate in each cycle.
    n_cycles: float
        The number of cycles to run the algorithm for. In each cycle, the number
        of segments is increased by a factor `segpoint_increase`, e.g., from 1
        to 2 to 4 to 8, etc., and the learning rate is reduced by a factor of
        `lr_reduction`. Additionally, each cycle runs for a factor of
        `n_iter_reduction` less than the previous cycle.
    initial_segpoints: int
        The initial number of segments to start with, e.g., 1 corresponds to a
        single midpoint.
    segpoint_increase: float
        The factor by which to increase the number of segments in each cycle.
    compat_threshold: float
        Edge interactions with compatibility lower than a specified threshold
        are ignored.

    Returns
    -------
    edges: array-like, shape (n_edges, n_segments + 1, 2)

    References
    ----------
    .. [1] Holten, Danny, and Jarke J. Van Wijk. "Force‐directed edge bundling
       for graph visualization." Computer Graphics Forum. Vol. 28. No. 3.
       Oxford, UK: Blackwell Publishing Ltd, 2009.

    """
    initial_edge_vecs = edges[:, 0] - edges[:, -1]

    initial_edge_lengths = np.linalg.norm(initial_edge_vecs, axis=-1, keepdims=True)

    # Compute edge compatibilities
    edge_compatibilities = compute_edge_compatibility(edges)
    edge_compatibilities = (edge_compatibilities > compat_threshold).astype(np.float32)

    num_segments = initial_segpoints

    for cycle in range(n_cycles):
        edges = subdivide_edge(edges, num_segments + 2)  # Add 2 for endpoints
        num_segments = int(np.ceil(num_segments * segpoint_increase))

        kp = K / (initial_edge_lengths * num_segments + 1e-8)
        kp = kp[..., None]

        for epoch in range(n_iter):
            F = compute_forces(edges, edge_compatibilities, kp)
            edges += F * lr

        n_iter = int(np.ceil(n_iter * n_iter_reduction))
        lr = lr * lr_reduction

    return edges
