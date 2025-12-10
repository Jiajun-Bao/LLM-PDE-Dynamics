import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from data_processing import (unscale_2d_array, deserialize_2d_integers)
from scipy.linalg import solve_banded

def compute_exact_solution(L, k, T, Nx, Nt, r=2.0, refine_x=10, return_fine=False):
    """
    Solves the 1D heat equation using BTCS to produce a high-accuracy reference solution. Uses the deterministic
    initial condition u(x,0)=cos(2*pi*x/L) with Neumann boundary conditions u(pm L/2, t) = 0.
    Input:
        L (float): Spatial domain length, domain is [-L/2, L/2]
        k (float): Diffusion coefficient 
        T (float): Final simulation time
        Nx (int): Number of interior spatial points in output grid
        Nt (int): Number of time steps in output
        r (float): Unused parameter (Legacy parameter carried over from Allen-Cahn)
        refine_x (int): Spatial refinement factor for accuracy (default: 10)
        return_fine (bool): Whether to return fine grid solution (default: False)
    Output:
        u_coarse (ndarray): Solution on coarse grid, shape (Nt+1, Nx)
        If return_fine=True:
            u_fine (ndarray): Solution on fine grid, shape (Nt_fine, Nx_fine)
            x_fine (ndarray): Fine spatial grid points
            t_fine (ndarray): Fine temporal grid points
    """
    dx_coarse = L / (Nx + 1)
    Nx_fine = (Nx + 1)*refine_x + 1
    dx_fine = L / (Nx_fine - 1)
    refine_t = int(np.ceil((dx_coarse/dx_fine)**2 * 1.1))
    Nt_fine = Nt * refine_t + 1
    dt_fine = T / (Nt_fine - 1)
    alpha = k * dt_fine / dx_fine**2
    # Fine grids
    x_fine = np.linspace(-L/2, L/2, Nx_fine)
    t_fine = np.linspace(0, T, Nt_fine)
    u_fine = np.zeros((Nt_fine, Nx_fine))
    u_fine[0] = np.cos(2 * np.pi * x_fine / L)    
    # Setup tridiagonal matrix for BTCS with Neumann BCs
    # We solve for all points including boundaries
    ab = np.zeros((3, Nx_fine))
    # Interior points: standard stencil
    ab[0, 2:] = -alpha        # upper diagonal
    ab[1, 1:-1] = 1 + 2*alpha # main diagonal
    ab[2, :-2] = -alpha       # lower diagonal
    # Neumann boundaries using ghost point
    ab[1, 0] = 1 + 2*alpha    # left boundary
    ab[0, 1] = -2*alpha
    ab[1, -1] = 1 + 2*alpha   # right boundary
    ab[2, -2] = -2*alpha
    # BTCS on fine grid
    for n in range(Nt_fine-1):
        rhs = u_fine[n, :].copy()
        u_fine[n+1, :] = solve_banded((1, 1), ab, rhs)
    # Downsample to coarse interior
    xi = np.arange(refine_x, Nx_fine-refine_x, refine_x)
    ti = np.arange(0, Nt_fine, refine_t)
    u_coarse = u_fine[np.ix_(ti, xi)]
    if return_fine:
        return u_coarse, u_fine, x_fine, t_fine
    else:
        return u_coarse


def compute_exact_solution_random_ic_vary_Nx(L, k, T, Nx, Nt, r=2.0, spline_obj=None, refine_x=10, return_fine=False):
    """
    Solves the 1D heat equation with a random initial condition defined by a spline interpolant.
    Uses BTCS on a refined grid with Neumann boundary conditions(pm L/2, t) = 0.
    Input:
        L (float): Spatial domain length, domain is [-L/2, L/2]
        k (float): Diffusion coefficient
        T (float): Final simulation time
        Nx (int): Number of interior spatial points in output grid
        Nt (int): Number of time steps in output
        r (float): Unused parameter (Legacy parameter carried over from Allen-Cahn)
        spline_obj (CubicSpline): Interpolant defining initial condition u(x,0)
        refine_x (int): Spatial refinement factor for accuracy (default: 10)
        return_fine (bool): Whether to return fine grid solution (default: False)
    Output:
        u_exact_all (ndarray): Solution on coarse grid, shape (Nt+1, Nx)
        If return_fine=True:
            u_fine (ndarray): Solution on fine grid, shape (Nt_fine, Nx_fine)
            x_fine (ndarray): Fine spatial grid points
            t_fine (ndarray): Fine temporal grid points
    """
    dx_coarse = L / (Nx + 1)
    Nx_fine = (Nx + 1) * refine_x + 1
    dx_fine = L / (Nx_fine - 1)
    refine_t = int(np.ceil((dx_coarse/dx_fine)**2 * 1.1))
    Nt_fine = Nt * refine_t + 1
    dt_fine = T / (Nt_fine - 1)
    alpha = k * dt_fine / dx_fine**2
    x_fine = np.linspace(-L/2, L/2, Nx_fine)
    t_fine = np.linspace(0, T, Nt_fine)
    u_fine = np.zeros((Nt_fine, Nx_fine))
    # Initialize with spline object
    u_fine[0, :] = spline_obj(x_fine)
    # Setup tridiagonal matrix for BTCS with Neumann BCs
    ab = np.zeros((3, Nx_fine))
    # Interior points
    ab[0, 2:] = -alpha        # upper diagonal
    ab[1, 1:-1] = 1 + 2*alpha # main diagonal for interior
    ab[2, :-2] = -alpha       # lower diagonal
    # Neumann boundaries
    ab[1, 0] = 1 + 2*alpha    # left boundary
    ab[0, 1] = -2*alpha
    ab[1, -1] = 1 + 2*alpha   # right boundary
    ab[2, -2] = -2*alpha
    # BTCS on fine grid
    for n in range(Nt_fine - 1):
        rhs = u_fine[n, :].copy()
        u_fine[n+1, :] = solve_banded((1, 1), ab, rhs)
    # Downsample to coarse grid
    xi = np.arange(refine_x, Nx_fine - refine_x, refine_x)
    ti = np.arange(0, Nt_fine, refine_t)
    u_exact_all = u_fine[np.ix_(ti, xi)]
    if return_fine:
        return u_exact_all, u_fine, x_fine, t_fine
    else:
        return u_exact_all


def solve_heat_ftcs(L, k, T, Nx, Nt, init_cond=None, r=2.0):
    """
    Solves the 1D heat equation using FTCS with Neumann boundary conditions.
    Input:
        L (float): Spatial domain length, domain is [-L/2, L/2]
        k (float): Diffusion coefficient
        T (float): Final simulation time
        Nx (int): Number of interior spatial points
        Nt (int): Number of time steps
        init_cond (ndarray, optional): Initial condition at interior points, shape (Nx,). 
                                     If None, uses u(x,0) = cos(2*pi*x/L)
        r (float): Unused parameter (Legacy parameter carried over from Allen-Cahn)
    Output:
        x (ndarray): Full spatial grid including boundaries, shape (Nx+2,)
        u_all (ndarray): FTCS solution at interior nodes, shape (Nt+1, Nx)
        u_exact_all (ndarray): High-accuracy reference solution if init_cond is None. Otherwise, returns a zero array placeholder.
    """
    dx = L/(Nx+1)
    x = np.linspace(-L/2, L/2, Nx+2)
    x_int = x[1:-1]
    dt = T / Nt
    s = k*dt/dx**2
    if s > 0.5:
        warnings.warn(f"Stability s={s:.4f}>0.5; FTCS may be unstable.")
    u_all = np.zeros((Nt+1, Nx))
    # Set initial condition
    if init_cond is None:
        u_all[0] = np.cos(2 * np.pi * x_int / L)
        u_exact_all = compute_exact_solution(L, k, T, Nx, Nt, r)
    else:
        init_cond = np.asarray(init_cond)
        if init_cond.shape != (Nx,):
            raise ValueError(f"init_cond must have shape ({Nx},), got {init_cond.shape}")
        u_all[0] = init_cond
        # Exact solution will be calculated using compute_exact_solution_random_ic_vary_Nx
        u_exact_all = np.zeros_like(u_all)
    # Time-stepping loop
    for n in range(Nt):
        lap = np.zeros(Nx)
        # Interior points: standard centered difference
        if Nx > 2:
            lap[1:-1] = u_all[n, 2:] - 2*u_all[n, 1:-1] + u_all[n, :-2]
        # Left boundary-adjacent point
        if Nx > 1:
            lap[0] = u_all[n, 1] - u_all[n, 0]
        else:
            lap[0] = 0.0
        # Right boundary-adjacent point
        if Nx > 1:
            lap[-1] = u_all[n, -2] - u_all[n, -1]
        else:
            lap[-1] = 0.0
        u_all[n+1] = u_all[n] + s*lap
    
    return x, u_all, u_exact_all


def solve_heat_btcs(L, k, T, Nx, Nt, init_cond=None, r=2.0):
    """
    Solves the 1D heat equation using BTCS method with Neumann BCs.
    Input:
        L (float): Spatial domain length, domain is [-L/2, L/2]
        k (float): Diffusion coefficient
        T (float): Final simulation time
        Nx (int): Number of interior spatial points
        Nt (int): Number of time steps
        init_cond (ndarray, optional): Initial condition at interior points, shape (Nx,).
                                     If None, uses u(x,0) = cos(2*pi*x/L)
        r (float): Unused parameter (Legacy parameter carried over from Allen-Cahn)
    Output:
        x (ndarray): Full spatial grid including boundaries, shape (Nx+2,)
        u_all (ndarray): BTCS solution at interior nodes, shape (Nt+1, Nx)
         u_exact_all (ndarray): High-accuracy reference solution if init_cond is None. Otherwise, returns a zero array placeholder.
    """
    dx = L / (Nx + 1)
    x = np.linspace(-L/2, L/2, Nx + 2)
    x_int = x[1:-1]
    dt = T / Nt
    alpha = k * dt / dx**2
    u_all = np.zeros((Nt + 1, Nx))
    if init_cond is None:
        u_all[0] = np.cos(2 * np.pi * x_int / L)
        u_exact_all = compute_exact_solution(L, k, T, Nx, Nt, r)
    else:
        init_cond = np.asarray(init_cond)
        if init_cond.shape != (Nx,):
            raise ValueError(f"init_cond must have shape ({Nx},), got {init_cond.shape}")
        u_all[0] = init_cond
        # Exact solution will be calculated using compute_exact_solution_random_ic_vary_Nx
        u_exact_all = np.zeros_like(u_all)

    ab = np.zeros((3, Nx))
    # Standard interior points
    ab[0, 1:] = -alpha           # upper diagonal
    ab[1, :] = 1 + 2*alpha       # main diagonal
    ab[2, :-1] = -alpha          # lower diagonal
    # First interior point (left boundary)
    ab[1, 0] = 1 + alpha
    # Last interior point (right boundary)
    ab[1, -1] = 1 + alpha
    # Time-stepping loop
    for n in range(Nt):
        rhs = u_all[n].copy()
        u_all[n+1] = solve_banded((1, 1), ab, rhs)
    return x, u_all, u_exact_all


def finite_difference_multi_predictions(full_serialized_data, input_time_steps, number_of_future_predictions,
                                        settings, vmin, vmax, L, k, Nt, Nx, T
):
    """
    Performs multi-step predictions using finite difference methods (FTCS and BTCS) with error accumulation.
    Each method starts from the same initial condition and predicts step-by-step, comparing against ground truth.
    Input:
        full_serialized_data: Serialized ground truth data for the entire simulation
        input_time_steps (int): Number of time steps used as input data
        number_of_future_predictions (int): Number of future steps to predict
        settings: Deserialization settings for the data
        vmin (float): Minimum value for data scaling
        vmax (float): Maximum value for data scaling
        L (float): Spatial domain length, domain is [-L/2, L/2]
        k (float): Diffusion coefficient
        Nt (int): Number of time steps for full simulation
        Nx (int): Number of interior spatial points
        T (float): Total simulation time
    Output:
        dict: Results dictionary containing:
            'ftcs': {'max_diff': list, 'rmse': list, 'predictions': list}
            'btcs': {'max_diff': list, 'rmse': list, 'predictions': list}
            Each containing error metrics and predictions for respective methods
    """
    # Extract full solution from serialized data
    all_rows_scaled = deserialize_2d_integers(full_serialized_data, settings)
    dt = T / Nt
    ftcs_predictions = []
    btcs_predictions = []
    ftcs_max_diff = []
    btcs_max_diff = []
    ftcs_rmse = []
    btcs_rmse = []
    # Get the initial condition from the last input step
    initial_step = input_time_steps - 1
    initial_scaled = all_rows_scaled[initial_step]
    initial_unscaled = unscale_2d_array(initial_scaled[np.newaxis, :], vmin, vmax)[0]
    current_ftcs = initial_unscaled.copy()
    current_btcs = initial_unscaled.copy()
    for step_idx in range(number_of_future_predictions):
        gt_idx = input_time_steps + step_idx
        if gt_idx >= all_rows_scaled.shape[0]:
            # Stop if we exceed the available ground truth
            break
        # Get ground truth for this step
        gt_scaled = all_rows_scaled[gt_idx]
        gt_unscaled = unscale_2d_array(gt_scaled[np.newaxis, :], vmin, vmax)[0]
        # We set T=dt and Nt=1 to evolve exactly one time step
        _, ftcs_step, _ = solve_heat_ftcs(L, k, dt, Nx, 1, init_cond=current_ftcs)
        _, btcs_step, _ = solve_heat_btcs(L, k, dt, Nx, 1, init_cond=current_btcs)
        # Extract predictions (using last time step)
        pred_ftcs = ftcs_step[-1]
        pred_btcs = btcs_step[-1]
        ftcs_predictions.append(pred_ftcs.copy())
        btcs_predictions.append(pred_btcs.copy())
        current_ftcs = pred_ftcs.copy()
        current_btcs = pred_btcs.copy()
        ftcs_max_diff.append(np.max(np.abs(pred_ftcs - gt_unscaled)))
        ftcs_rmse.append(np.sqrt(np.mean((pred_ftcs - gt_unscaled)**2)))
        btcs_max_diff.append(np.max(np.abs(pred_btcs - gt_unscaled)))
        btcs_rmse.append(np.sqrt(np.mean((pred_btcs - gt_unscaled)**2)))
        
    return {
        'ftcs': {
            'max_diff': ftcs_max_diff,
            'rmse': ftcs_rmse,
            'predictions': ftcs_predictions
        },
        'btcs': {
            'max_diff': btcs_max_diff,
            'rmse': btcs_rmse,
            'predictions': btcs_predictions
        }
    }


def visualize_spline_ic(L, Nx, init_cond):
    """
    Visualizes an initial condition and its cubic spline interpolation with Neumann BCs.
    Input:
        L (float): Spatial domain length, domain is [-L/2, L/2]
        Nx (int): Number of interior spatial points
        init_cond (ndarray): Initial condition values at interior points, shape (Nx,)
    Output:
        fig (matplotlib.figure.Figure): Plot showing discrete points and spline interpolation
        cs (CubicSpline): Cubic spline interpolant object for the initial condition
    """
    # Create grid for interior points
    dx = L / (Nx + 1)
    x_interior = np.linspace(-L/2 + dx, L/2 - dx, Nx)
    x_full = np.linspace(-L/2, L/2, Nx + 2)
    u_full = np.zeros(Nx + 2)
    u_full[1:-1] = init_cond
    # Estimate boundary values using second-order one-sided difference for zero gradient
    if Nx >= 2:
        # Left boundary
        u_full[0] = (4*init_cond[0] - init_cond[1]) / 3.0
        # Right boundary
        u_full[-1] = (4*init_cond[-1] - init_cond[-2]) / 3.0
    else:
        u_full[0] = u_full[-1] = init_cond[0]
    # Create cubic spline interpolation
    cs = CubicSpline(x_full, u_full, bc_type=((1, 0.0), (1, 0.0)))
    x_fine = np.linspace(-L/2, L/2, Nx*100)
    u_fine = cs(x_fine)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(x_full, u_full, color='red', label='Discrete Points', s=50, zorder=3)
    ax.plot(x_fine, u_fine, color='blue', label='Cubic Spline Interpolation', zorder=2)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('u(x, 0)', fontsize=12)
    ax.set_title('Initial Condition with Cubic Spline Interpolation', fontsize=10)
    ax.legend()

    return fig, cs


def plot_both_grids(L, Nx_original, Nx_new, init_cond_original):
    """
    Compares initial conditions on two different grid resolutions using cubic spline interpolation.    
    Input:
        L (float): Spatial domain length, domain is [-L/2, L/2]
        Nx_original (int): Number of interior points in original grid
        Nx_new (int): Number of interior points in new grid
        init_cond_original (ndarray): Initial condition on original grid, shape (Nx_original,)
    Output:
        fig (matplotlib.figure.Figure): Comparison plot of both grids with spline interpolation
        cs (CubicSpline): Cubic spline interpolant object
        u_coarse_new (ndarray): Initial condition sampled on new grid, shape (Nx_new,)
    """
    dx_original = L / (Nx_original + 1)
    x_interior_original = np.linspace(-L/2 + dx_original, L/2 - dx_original, Nx_original)
    x_coarse_original = np.linspace(-L/2, L/2, Nx_original + 2)
    u_coarse_full_original = np.zeros(Nx_original + 2)
    u_coarse_full_original[1:-1] = init_cond_original
    # Set boundary values using second-order one-sided difference for zero gradient
    if Nx_original >= 2:
        # Left boundary
        u_coarse_full_original[0] = (4*init_cond_original[0] - init_cond_original[1]) / 3.0
        # Right boundary
        u_coarse_full_original[-1] = (4*init_cond_original[-1] - init_cond_original[-2]) / 3.0
    else:
        u_coarse_full_original[0] = u_coarse_full_original[-1] = init_cond_original[0]
    cs = CubicSpline(x_coarse_original, u_coarse_full_original, bc_type=((1, 0.0), (1, 0.0)))
    x_coarse_new = np.linspace(-L/2, L/2, Nx_new + 2)
    u_coarse_new_full = cs(x_coarse_new)
    x_fine = np.linspace(-L/2, L/2, max(Nx_original, Nx_new)*100)
    u_fine = cs(x_fine) 
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x_fine, u_fine, 'b-', label='Cubic Spline Interpolation', linewidth=2)
    ax.scatter(x_coarse_original, u_coarse_full_original, color='red', s=50,
              label=f'Original Points (Nx={Nx_original})', zorder=3)
    ax.scatter(x_coarse_new, u_coarse_new_full, color='green', s=25, marker='d',
              label=f'Sampled Points (Nx={Nx_new})', zorder=2)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('u(x, 0)', fontsize=12)
    ax.set_title('Comparison of Original and New Grids with Cubic Spline Interpolation', fontsize=12)
    ax.legend(loc='best')
    
    return fig, cs, u_coarse_new_full[1:-1]