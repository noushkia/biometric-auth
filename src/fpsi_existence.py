import numpy as np


def fourier_approx_indicator_1d(x_grid, center, delta, domain_L, k):
    """Computes the 1D Fourier approximation of the indicator function."""
    a0 = delta / domain_L
    approximation = np.full_like(x_grid, a0 / 2, dtype=np.float64)
    for n in range(1, k + 1):
        an = (2 / (n * np.pi)) * np.sin(n * np.pi * delta / (2 * domain_L))
        approximation += an * np.cos(n * np.pi * (x_grid - center) / domain_L)
    return approximation


def fourier_approx_nd_point(grids, point, delta, domain_L, k):
    d = len(point)
    approx_nd = np.ones_like(grids[0], dtype=np.float64)
    for dim in range(d):
        approx_1d = fourier_approx_indicator_1d(grids[dim], point[dim], delta, domain_L, k)
        approx_nd *= approx_1d
    return approx_nd


def run_fourier_fpsi_nd(sender_set, receiver_set, domain_L, delta, k, resolution=100):
    d = len(sender_set[0])
    x_grid_1d = np.linspace(-domain_L, domain_L, resolution)
    dx = x_grid_1d[1] - x_grid_1d[0]
    dV = dx ** d
    grids = np.meshgrid(*[x_grid_1d] * d, indexing='ij')
    vector_S = np.zeros_like(grids[0], dtype=np.float64)
    for point in sender_set:
        vector_S += fourier_approx_nd_point(grids, point, delta, domain_L, k)
    vector_R = np.zeros_like(grids[0], dtype=np.float64)
    for point in receiver_set:
        vector_R += fourier_approx_nd_point(grids, point, delta, domain_L, k)
    inner_product = np.sum(vector_S * vector_R) * dV

    print(f"IP with {k} terms: {inner_product:.6f}")
    return inner_product


if __name__ == "__main__":
    DELTA = 2.0
    DOMAIN = 12.0
    RESOLUTION = 150
    sender_set = [[0.0, 0.0], [8.0, 8.0]]
    receiver_set_intersect = [[0.0, 0.5], [-8.0, -8.0]]
    receiver_set_disjoint = [[4.0, 5.0], [-8.0, -8.0]]
    print("Def intersection")
    for k_val in [5, 15, 30, 50]:
        run_fourier_fpsi_nd(sender_set, receiver_set_intersect, DOMAIN, DELTA, k=k_val, resolution=RESOLUTION)
    print("Def no intersection")
    for k_val in [5, 15, 30, 50]:
        run_fourier_fpsi_nd(sender_set, receiver_set_disjoint, DOMAIN, DELTA, k=k_val, resolution=RESOLUTION)
    print("Close intersection")
    sender_set = [[0.0, 0.0], [8.0, 8.0]]
    receiver_set_disjoint = [[0, 1.9], [-8.0, -8.0]]
    for k_val in [5, 15, 30, 50]:
        run_fourier_fpsi_nd(sender_set, receiver_set_disjoint, DOMAIN, DELTA, k=k_val, resolution=RESOLUTION)
    print("Multiple intersections")
    sender_set = [[0.0, 0.0], [8.0, 8.0], [-8.0, -8.0]]
    receiver_set_disjoint = [[0, 1.5], [7.0, 7.0], [12.0, 12.0]]
    for k_val in [5, 15, 30, 50]:
        run_fourier_fpsi_nd(sender_set, receiver_set_disjoint, DOMAIN, DELTA, k=k_val, resolution=RESOLUTION)

