import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from data_processing import deserialize_2d_integers, unscale_2d_array

def compute_exact_solution(L, c, T, Nx, Nt, refine_x=10, return_fine=False):
    """
    Solves the 1D wave equation using leapfrog to produce a high-accuracy reference solution. Uses the deterministic
    initial condition u(x,0)=sin(2*pi*x/L), u_t(x,0)=0 with Dirichlet boundary conditions u(pm L/2, t) = 0.
    Input:
        L (float): Spatial domain length, domain is [-L/2, L/2]
        c (float): Wave speed
        T (float): Final simulation time
        Nx (int): Number of interior spatial points in output grid
        Nt (int): Number of time steps in output
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
    dt_coarse = T / Nt
    refine_t = int(np.ceil(c * dt_coarse / (0.9 * dx_fine)))
    Nt_fine = Nt * refine_t + 1
    dt_fine = T / (Nt_fine - 1)
    r = c * dt_fine / dx_fine
    r2 = r**2
    if r > 1.0:
        warnings.warn(f"CFL condition violated: r={r:.4f} > 1.0, solution may be unstable")
    # Fine grids and BCs
    x_fine = np.linspace(-L/2, L/2, Nx_fine)
    t_fine = np.linspace(0, T, Nt_fine)
    u_fine = np.zeros((Nt_fine, Nx_fine))
    # Apply initial conditions
    u_fine[0] = np.sin(2 * np.pi * x_fine / L)
    u_fine[:, 0] = u_fine[:, -1] = 0.0
    # First time step (using zero initial velocity)
    for i in range(1, Nx_fine-1):
        u_xx = (u_fine[0, i+1] - 2*u_fine[0, i] + u_fine[0, i-1]) / dx_fine**2
        u_fine[1, i] = u_fine[0, i] + 0.5 * dt_fine**2 * c**2 * u_xx
    # Time stepping loop
    for n in range(1, Nt_fine-1):
        for i in range(1, Nx_fine-1):
            u_fine[n+1, i] = (2*(1 - r2)*u_fine[n, i] + 
                              r2*(u_fine[n, i+1] + u_fine[n, i-1]) - 
                              u_fine[n-1, i])
    # Downsample to coarse interior
    xi = np.arange(refine_x, Nx_fine-refine_x, refine_x)
    ti = np.arange(0, Nt_fine, refine_t)
    u_coarse = u_fine[np.ix_(ti, xi)]
    if return_fine:
        return u_coarse, u_fine, x_fine, t_fine
    else:
        return u_coarse


def compute_exact_solution_random_ic_vary_Nx(L, c, T, Nx, Nt, spline_obj=None, spline_vel_obj=None, refine_x=10, return_fine=False):
    """
    Solves the 1D wave equation with a random initial condition defined by a spline interpolant.
    Uses leapfrog on a refined grid with Dirichlet boundary conditions u(pm L/2, t) = 0.
    Input:
        L (float): Spatial domain length, domain is [-L/2, L/2]
        c (float): Wave speed
        T (float): Final simulation time
        Nx (int): Number of interior spatial points in output grid
        Nt (int): Number of time steps in output
        spline_obj (CubicSpline): Initial displacement u(x,0) interpolant
        spline_vel_obj (CubicSpline): Initial velocity u_t(x,0) interpolant (None = zero)
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
    dt_coarse = T / Nt
    refine_t = int(np.ceil(c * dt_coarse / (0.9 * dx_fine)))
    Nt_fine = Nt * refine_t + 1
    dt_fine = T / (Nt_fine - 1)
    r = c * dt_fine / dx_fine
    r2 = r**2
    if r > 1.0:
        warnings.warn(f"CFL condition violated: r={r:.4f} > 1.0, solution may be unstable")
    # Fine grids and BCs
    x_fine = np.linspace(-L/2, L/2, Nx_fine)
    t_fine = np.linspace(0, T, Nt_fine)
    u_fine = np.zeros((Nt_fine, Nx_fine))
    # Apply spline initial conditions
    u_fine[0, :] = spline_obj(x_fine)
    u_fine[:, 0] = u_fine[:, -1] = 0.0  # Enforce homogeneous BCs
    # Initial velocity (zero if not provided)
    if spline_vel_obj is None:
        u_t_0 = np.zeros(Nx_fine)
    else:
        u_t_0 = spline_vel_obj(x_fine)
        u_t_0[0] = u_t_0[-1] = 0.0  # BC compatibility
    # First time step
    for i in range(1, Nx_fine-1):
        u_xx = (u_fine[0, i+1] - 2*u_fine[0, i] + u_fine[0, i-1]) / dx_fine**2
        u_fine[1, i] = u_fine[0, i] + dt_fine*u_t_0[i] + 0.5*dt_fine**2*c**2*u_xx
    # Main time stepping
    for n in range(1, Nt_fine-1):
        for i in range(1, Nx_fine-1):
            u_fine[n+1, i] = (2*(1 - r2)*u_fine[n, i] + 
                              r2*(u_fine[n, i+1] + u_fine[n, i-1]) - 
                              u_fine[n-1, i])
    # Downsample to coarse interior
    xi = np.arange(refine_x, Nx_fine - refine_x, refine_x)
    ti = np.arange(0, Nt_fine, refine_t)
    u_exact_all = u_fine[np.ix_(ti, xi)]
    if return_fine:
        return u_exact_all, u_fine, x_fine, t_fine
    else:
        return u_exact_all


def solve_wave_leapfrog(L, c, T, Nx, Nt, init_disp=None, init_vel=None):
    """
    Solves 1D wave equation using leapfrog method.
    Input:
        L (float): Spatial domain length, domain is [-L/2, L/2]
        c (float): Wave speed
        T (float): Final simulation time
        Nx (int): Number of interior spatial points
        Nt (int): Number of time steps
        init_disp (ndarray): Initial displacement u(x,0), shape (Nx,). Default: sin(2*pi*x/L)
        init_vel (ndarray): Initial velocity u_t(x,0), shape (Nx,). Default: zero
    Output:
        x (ndarray): Full spatial grid including boundaries, shape (Nx+2,)
        u_all (ndarray): Leapfrog solution at interior nodes, shape (Nt+1, Nx)
        u_exact_all (ndarray): High-accuracy reference solution if init_disp is None. Otherwise, returns a zero array placeholder.
    """
    dx = L/(Nx+1)
    x = np.linspace(-L/2, L/2, Nx+2)
    x_int = x[1:-1]
    dt = T / Nt
    r = c * dt / dx
    r2 = r**2
    if r > 1.0:
        warnings.warn(f"CFL condition violated: r={r:.4f} > 1.0, solution will be unstable")
    u_all = np.zeros((Nt+1, Nx))
    # Set initial displacement
    if init_disp is None:
        u_all[0] = np.sin(2 * np.pi * x_int / L)
        u_exact_all = compute_exact_solution(L, c, T, Nx, Nt)
    else:
        init_disp = np.asarray(init_disp)
        if init_disp.shape != (Nx,):
            raise ValueError(f"init_disp must have shape ({Nx},), got {init_disp.shape}")
        u_all[0] = init_disp
        u_exact_all = np.zeros_like(u_all)
    # Set initial velocity
    if init_vel is None:
        u_t = np.zeros(Nx)
    else:
        u_t = np.asarray(init_vel)
        if u_t.shape != (Nx,):
            raise ValueError(f"init_vel must have shape ({Nx},), got {u_t.shape}")
    u_prev = np.concatenate(([0.0], u_all[0], [0.0]))
    lap = u_prev[2:] - 2*u_prev[1:-1] + u_prev[:-2]
    u_all[1] = u_all[0] + dt*u_t + 0.5*dt**2*c**2*lap/dx**2
    # Main time stepping
    for n in range(1, Nt):
        u_curr = np.concatenate(([0.0], u_all[n], [0.0]))
        lap = u_curr[2:] - 2*u_curr[1:-1] + u_curr[:-2]
        u_all[n+1] = 2*u_all[n] - u_all[n-1] + r2*lap
    return x, u_all, u_exact_all


def solve_wave_crank_nicolson(L, c, T, Nx, Nt, init_disp=None, init_vel=None):
    """
    Solves 1D wave equation using Crank-Nicolson method.
    Converts second-order PDE to first-order system [u, v=u_t] and applies CN time stepping.
    Input:
        L (float): Spatial domain length, domain is [-L/2, L/2]
        c (float): Wave speed
        T (float): Final simulation time
        Nx (int): Number of interior spatial points
        Nt (int): Number of time steps
        init_disp (ndarray): Initial displacement u(x,0), shape (Nx,). Default: sin(2*pi*x/L)
        init_vel (ndarray): Initial velocity u_t(x,0), shape (Nx,). Default: zero
    Output:
        x (ndarray): Full spatial grid including boundaries, shape (Nx+2,)
        u_all (ndarray): Crank-Nicolson solution at interior nodes, shape (Nt+1, Nx)
        u_exact_all (ndarray): High-accuracy reference solution if init_disp is None. Otherwise, returns a zero array placeholder.
    """
    dx = L / (Nx + 1)
    x = np.linspace(-L/2, L/2, Nx + 2)
    x_int = x[1:-1]
    dt = T / Nt
    u_all = np.zeros((Nt + 1, Nx))
    v_all = np.zeros((Nt + 1, Nx))  # Velocity field
    # Initialize displacement
    if init_disp is None:
        u_all[0] = np.sin(2 * np.pi * x_int / L)
        u_exact_all = compute_exact_solution(L, c, T, Nx, Nt)
    else:
        init_disp = np.asarray(init_disp)
        if init_disp.shape != (Nx,):
            raise ValueError(f"init_disp must have shape ({Nx},), got {init_disp.shape}")
        u_all[0] = init_disp
        u_exact_all = np.zeros_like(u_all)
    # Initialize velocity
    if init_vel is None:
        v_all[0] = np.zeros(Nx)
    else:
        v_all[0] = np.asarray(init_vel)
        if v_all[0].shape != (Nx,):
            raise ValueError(f"init_vel must have shape ({Nx},), got {v_all[0].shape}")
    # Build second derivative matrix with homogeneous BCs
    D2 = np.diag(-2*np.ones(Nx)) + np.diag(np.ones(Nx-1), 1) + np.diag(np.ones(Nx-1), -1)
    D2 = D2 / dx**2
    # Construct Crank-Nicolson matrices
    I = np.eye(Nx)
    LHS_top = np.hstack([I, -0.5*dt*I])
    LHS_bot = np.hstack([-0.5*dt*c**2*D2, I])
    LHS = np.vstack([LHS_top, LHS_bot])
    RHS_top_u = I
    RHS_top_v = 0.5*dt*I
    RHS_bot_u = 0.5*dt*c**2*D2
    RHS_bot_v = I
    # Time stepping
    for n in range(Nt):
        rhs_top = RHS_top_u @ u_all[n] + RHS_top_v @ v_all[n]
        rhs_bot = RHS_bot_u @ u_all[n] + RHS_bot_v @ v_all[n]
        rhs = np.concatenate([rhs_top, rhs_bot])
        solution = np.linalg.solve(LHS, rhs)
        u_all[n+1] = solution[:Nx]
        v_all[n+1] = solution[Nx:]
    
    return x, u_all, u_exact_all


def finite_difference_multi_predictions(full_serialized_data, input_time_steps, 
                                            number_of_future_predictions, settings, 
                                            vmin, vmax, L, c, Nt, Nx, T):
    """
    Performs multi-step predictions using finite difference methods (leapfrog and Crank-Nicolson) with error accumulation.
    Each method starts from the same initial condition and predicts step-by-step, comparing against ground truth.
    Input:
        full_serialized_data: Serialized ground truth data for the entire simulation
        input_time_steps (int): Number of time steps used as input data
        number_of_future_predictions (int): Number of future steps to predict
        settings: Deserialization settings for the data
        vmin (float): Minimum value for data scaling
        vmax (float): Maximum value for data scaling
        L (float): Spatial domain length, domain is [-L/2, L/2]
        c (float): Wave speed
        Nt (int): Number of time steps for full simulation
        Nx (int): Number of interior spatial points
        T (float): Total simulation time
    Output:
        dict: Results dictionary containing:
            'leapfrog': {'max_diff': list, 'rmse': list, 'predictions': list}
            'crank_nicolson': {'max_diff': list, 'rmse': list, 'predictions': list}
            Each containing error metrics and predictions for respective methods
    """
    # Extract full solution from serialized data
    all_rows_scaled = deserialize_2d_integers(full_serialized_data, settings)
    dt = T / Nt
    dx = L / (Nx + 1)
    leapfrog_predictions = []
    crank_nicolson_predictions = []
    leapfrog_max_diff = []
    crank_nicolson_max_diff = []
    leapfrog_rmse = []
    crank_nicolson_rmse = []
    # Wave equation requires at least 2 time steps
    if input_time_steps < 2:
        raise ValueError("Wave equation needs at least 2 time steps for initial conditions")
    # Extract and unscale last two training steps
    u_nm1_scaled = all_rows_scaled[input_time_steps - 2]
    u_n_scaled = all_rows_scaled[input_time_steps - 1]
    u_nm1 = unscale_2d_array(u_nm1_scaled[np.newaxis, :], vmin, vmax)[0]
    u_n = unscale_2d_array(u_n_scaled[np.newaxis, :], vmin, vmax)[0]
    # Initialize leapfrog state
    current_leapfrog_nm1 = u_nm1.copy()
    current_leapfrog_n = u_n.copy()
    # Initialize Crank-Nicolson state (estimate velocity)
    if input_time_steps >= 3:
        u_nm2_scaled = all_rows_scaled[input_time_steps - 3]
        u_nm2 = unscale_2d_array(u_nm2_scaled[np.newaxis, :], vmin, vmax)[0]
        v_n_estimate = (3*u_n - 4*u_nm1 + u_nm2) / (2*dt)
    else:
        v_n_estimate = (u_n - u_nm1) / dt  # Fallback: backward difference
    current_cn_u = u_n.copy()
    current_cn_v = v_n_estimate.copy()
    # Build Crank-Nicolson matrices
    D2 = np.diag(-2*np.ones(Nx)) + np.diag(np.ones(Nx-1), 1) + np.diag(np.ones(Nx-1), -1)
    D2 = D2 / dx**2
    I = np.eye(Nx)
    LHS_top = np.hstack([I, -0.5*dt*I])
    LHS_bot = np.hstack([-0.5*dt*c**2*D2, I])
    LHS = np.vstack([LHS_top, LHS_bot])
    RHS_top_u = I
    RHS_top_v = 0.5*dt*I
    RHS_bot_u = 0.5*dt*c**2*D2
    RHS_bot_v = I
    for step_idx in range(number_of_future_predictions):
        gt_idx = input_time_steps + step_idx
        if gt_idx >= all_rows_scaled.shape[0]:
            # Stop if we exceed the available ground truth
            break
        # Get ground truth for this step
        gt_scaled = all_rows_scaled[gt_idx]
        gt_unscaled = unscale_2d_array(gt_scaled[np.newaxis, :], vmin, vmax)[0]
        # Leapfrog prediction
        r = c * dt / dx
        r2 = r**2
        u_curr_pad = np.concatenate(([0.0], current_leapfrog_n, [0.0]))
        lap = u_curr_pad[2:] - 2*u_curr_pad[1:-1] + u_curr_pad[:-2]
        pred_leapfrog = 2*current_leapfrog_n - current_leapfrog_nm1 + r2*lap
        # Update leapfrog state
        current_leapfrog_nm1 = current_leapfrog_n.copy()
        current_leapfrog_n = pred_leapfrog.copy()
        # Crank-Nicolson prediction
        rhs_top = RHS_top_u @ current_cn_u + RHS_top_v @ current_cn_v
        rhs_bot = RHS_bot_u @ current_cn_u + RHS_bot_v @ current_cn_v
        rhs = np.concatenate([rhs_top, rhs_bot])
        solution = np.linalg.solve(LHS, rhs)
        pred_cn = solution[:Nx]
        current_cn_v = solution[Nx:]
        current_cn_u = pred_cn.copy()
        # Store predictions and compute errors
        leapfrog_predictions.append(pred_leapfrog.copy())
        crank_nicolson_predictions.append(pred_cn.copy())
        leapfrog_max_diff.append(np.max(np.abs(pred_leapfrog - gt_unscaled)))
        leapfrog_rmse.append(np.sqrt(np.mean((pred_leapfrog - gt_unscaled)**2)))
        crank_nicolson_max_diff.append(np.max(np.abs(pred_cn - gt_unscaled)))
        crank_nicolson_rmse.append(np.sqrt(np.mean((pred_cn - gt_unscaled)**2)))
    
    return {
        'leapfrog': {
            'max_diff': leapfrog_max_diff,
            'rmse': leapfrog_rmse,
            'predictions': leapfrog_predictions
        },
        'crank_nicolson': {
            'max_diff': crank_nicolson_max_diff,
            'rmse': crank_nicolson_rmse,
            'predictions': crank_nicolson_predictions
        }
    }


def visualize_spline_ic(L, Nx, init_cond):
    """
    Visualizes an initial condition and its cubic spline interpolation.
    Input:
        L (float): Spatial domain length, domain is [-L/2, L/2]
        Nx (int): Number of interior spatial points
        init_cond (ndarray): Initial condition values at interior points, shape (Nx,)
    Output:
        fig (matplotlib.figure.Figure): Plot showing discrete points and spline interpolation
        cs (CubicSpline): Cubic spline interpolant object for the initial condition
    """
    # Create grid including boundaries
    x_coarse = np.linspace(-L/2, L/2, Nx + 2)
    u_coarse_full = np.zeros(Nx + 2)
    u_coarse_full[1:-1] = init_cond
    # Generate smooth spline interpolation
    cs = CubicSpline(x_coarse, u_coarse_full)
    x_fine = np.linspace(-L/2, L/2, Nx*100)
    u_fine = cs(x_fine)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(x_coarse, u_coarse_full, color='red', label='Discrete Points', s=50, zorder=3)
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
    x_coarse_original = np.linspace(-L/2, L/2, Nx_original + 2)
    u_coarse_full_original = np.zeros(Nx_original + 2)
    u_coarse_full_original[1:-1] = init_cond_original
    cs = CubicSpline(x_coarse_original, u_coarse_full_original)
    x_coarse_new = np.linspace(-L/2, L/2, Nx_new + 2)
    u_coarse_new = cs(x_coarse_new)
    x_fine = np.linspace(-L/2, L/2, Nx_original*100)
    u_fine = cs(x_fine)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x_fine, u_fine, 'b-', label='Cubic Spline Interpolation', linewidth=2)
    ax.scatter(x_coarse_original, u_coarse_full_original, color='red', s=50,
              label=f'Original Points (Nx={Nx_original})', zorder=3)
    ax.scatter(x_coarse_new, u_coarse_new, color='green', s=25, marker='d',
              label=f'Sampled Points (Nx={Nx_new})', zorder=2)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('u(x, 0)', fontsize=12)
    ax.set_title('Comparison of Original and New Grids with Cubic Spline Interpolation', fontsize=12)
    ax.legend(loc='best')
    
    return fig, cs, u_coarse_new[1:-1]