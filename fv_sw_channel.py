import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import xarray as xr

# shallow water equations in a channel with periodic BCs in x and no flow/free slip BCs in y.
# the top and bottom boundaries are no flow for v and free slip for u, and h doesn't change at the boundaries.
# the first set of functions are for some default parameters and grid setup, and the main function is the Lax-Wendroff timestepping for the shallow water equations.
# there are also functions for computing geostrophic velocities, vorticity, divergence, and particle trajectories.

# low resolution default grid 
nx_default=143
ny_default=29
dx_default=2.e5
dy_default=2.e5
dt_default=120

# default physical parameters
f0_default=1e-4
g_default=9.8
beta_default=1e-11

def return_default_params():
    return nx_default, ny_default, dx_default, dy_default, dt_default, f0_default, g_default, beta_default

def return_grid_initial(nx=nx_default, ny=ny_default, dx=dx_default, dy=dy_default, f0=f0_default, beta=beta_default):

    x = np.arange(0,nx*dx,dx)
    y = np.arange(0,ny*dy,dy)

    x2,y2 = np.meshgrid(x,y)

    ui = np.zeros([ny,nx])
    vi = np.zeros([ny,nx])
    hi = np.ones_like(x2)*1e5

    f2=f0 + beta*y2
    hb=np.zeros_like(hi)

    return x,y,x2,y2,f2,hb,ui,vi,hi

@njit(nopython=True, fastmath=True)
def lax_wendroff_timesteptimeStep(h, u, v, f2, hb,
             dx=dx_default, dy=dy_default, dt=dt_default,
             g=g_default,
             h_min=1e3,
             filter_fac=0.01,
             diffusion_on=False,
             diffusion_coeff=1e-2):
    """Single Lax-Wendroff timestepping for 2D shallow water.       
    Arrays shapes: h, u, v are (ny, nx).
    Note, these shapes are taken from the input arrays.
    f2 is Coriolis parameter array same shape.
    hb is bottom topography same shape.
    the first and last rows of h, u, v are the periodic BCs in x
    the first and last columns are the no flow/free slip BCs in y.
    Returns updated (h_new, u_new, v_new) with same shapes.
    """

    h_new = np.zeros_like(h)
    uh_new = np.zeros_like(u)
    vh_new = np.zeros_like(v)
    u_new = np.zeros_like(u)
    v_new = np.zeros_like(v)

    # first partial step; compute mid-point values
    uh = u * h
    vh = v * h

    # x-midpoints (axis 1) and y-midpoints (axis 0) for depth, using fluxes to update
    h_mid_xt = 0.5 * (h[1:-1, 1:] + h[1:-1, :-1]) - (0.5 * dt / dx) * (uh[1:-1, 1:] - uh[1:-1, :-1])
    h_mid_yt = 0.5 * (h[1:, 1:-1] + h[:-1, 1:-1]) - (0.5 * dt / dy) * (vh[1:, 1:-1] - vh[:-1, 1:-1])

    Ux = uh * u + 0.5 * g * h**2
    Uy = uh * v  # uvh
    uh_mid_xt = 0.5 * (uh[1:-1, 1:] + uh[1:-1, :-1]) - (0.5 * dt / dx) * (Ux[1:-1, 1:] - Ux[1:-1, :-1])
    uh_mid_yt = 0.5 * (uh[1:, 1:-1] + uh[:-1, 1:-1]) - (0.5 * dt / dy) * (Uy[1:, 1:-1] - Uy[:-1, 1:-1])

    Vx = Uy  # uvh
    Vy = vh * v + 0.5 * g * h**2
    vh_mid_xt = 0.5 * (vh[1:-1, 1:] + vh[1:-1, :-1]) - (0.5 * dt / dx) * (Vx[1:-1, 1:] - Vx[1:-1, :-1])
    vh_mid_yt = 0.5 * (vh[1:, 1:-1] + vh[:-1, 1:-1]) - (0.5 * dt / dy) * (Vy[1:, 1:-1] - Vy[:-1, 1:-1])

    # some tendencies

    h_tendency = np.zeros_like(h)
    u_tendency = np.zeros_like(u)
    v_tendency = np.zeros_like(v)

    if not diffusion_on:
        u_tendency[1:-1,1:-1] = f2[1:-1, 1:-1] * v[1:-1, 1:-1] - g/(2*dx)*(hb[1:-1, 2:]-hb[1:-1, :-2])
        v_tendency[1:-1,1:-1] = -f2[1:-1, 1:-1] * u[1:-1, 1:-1] - g/(2*dy)*(hb[2:, 1:-1]-hb[:-2, 1:-1])
    else:
        h_tendency[1:-1,1:-1] = diffusion_coeff * (
            (h[1:-1, 2:] - 2 * h[1:-1, 1:-1] + h[1:-1, :-2]) / dx**2 +
            (h[2:, 1:-1] - 2 * h[1:-1, 1:-1] + h[:-2, 1:-1]) / dy**2
        )
        u_tendency[1:-1,1:-1] = (
            f2[1:-1, 1:-1] * v[1:-1, 1:-1]  
            - g/(2*dx) * (hb[1:-1, 2:] - hb[1:-1, :-2])
            + diffusion_coeff * (
                (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) / dx**2
                + (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / dy**2
            )
        )
        v_tendency[1:-1,1:-1] = (
            -f2[1:-1, 1:-1] * u[1:-1, 1:-1]
            - g/(2*dy) * (hb[2:, 1:-1] - hb[:-2, 1:-1])
            + diffusion_coeff * (
                (v[1:-1, 2:] - 2 * v[1:-1, 1:-1] + v[1:-1, :-2]) / dx**2
                + (v[2:, 1:-1] - 2 * v[1:-1, 1:-1] + v[:-2, 1:-1]) / dy**2
            )
        )

    # finish the full step for depth (divergence of fluxes)
    h_new[1:-1, 1:-1] = (
        h[1:-1, 1:-1]
        - (dt / dx) * (uh_mid_xt[:, 1:] - uh_mid_xt[:, :-1])
        - (dt / dy) * (vh_mid_yt[1:, :] - vh_mid_yt[:-1, :])
        + dt * h_tendency[1:-1, 1:-1]
    )

    # ensure minimum depth at midpoints to avoid division by zero in momentum updates
    h_mid_xt = np.maximum(h_mid_xt, h_min)
    h_mid_yt = np.maximum(h_mid_yt, h_min)


    # momentum fluxes at midpoints; safe divisions using floored mid-depths
    Ux_mid_xt = uh_mid_xt * uh_mid_xt / h_mid_xt + 0.5 * g * h_mid_xt**2
    Uy_mid_yt = uh_mid_yt * vh_mid_yt / h_mid_yt
    uh_new[1:-1, 1:-1] = (
        uh[1:-1, 1:-1]
        - (dt / dx) * (Ux_mid_xt[:, 1:] - Ux_mid_xt[:, :-1])
        - (dt / dy) * (Uy_mid_yt[1:, :] - Uy_mid_yt[:-1, :])
        + dt * u_tendency[1:-1, 1:-1] * 0.5 * (h[1:-1, 1:-1] + h_new[1:-1, 1:-1])
    )

    Vx_mid_xt = uh_mid_xt * vh_mid_xt / h_mid_xt
    Vy_mid_yt = vh_mid_yt * vh_mid_yt / h_mid_yt + 0.5 * g * h_mid_yt**2
    vh_new[1:-1, 1:-1] = (
        vh[1:-1, 1:-1]
        - (dt / dx) * (Vx_mid_xt[:, 1:] - Vx_mid_xt[:, :-1])
        - (dt / dy) * (Vy_mid_yt[1:, :] - Vy_mid_yt[:-1, :])
        + dt * v_tendency[1:-1, 1:-1] * 0.5 * (h[1:-1, 1:-1] + h_new[1:-1, 1:-1])
    )


    # ensure minimum height before momentum updates
    h_new[1:-1, 1:-1] = np.maximum(h_new[1:-1, 1:-1], h_min)

    # compute new velocities from updated momenta and depth
    u_new[1:-1, 1:-1] = uh_new[1:-1, 1:-1] / h_new[1:-1, 1:-1]
    v_new[1:-1, 1:-1] = vh_new[1:-1, 1:-1] / h_new[1:-1, 1:-1]

    # periodic boundary conditions in x (wrap left/right)
    u_new[:, 0] = u_new[:, -2]
    u_new[:, -1] = u_new[:, 1]
    v_new[:, 0] = v_new[:, -2]
    v_new[:, -1] = v_new[:, 1]
    h_new[:, 0] = h_new[:, -2]
    h_new[:, -1] = h_new[:, 1]

    # no flow boundary in y for v
    v_new[0,:] = 0
    v_new[-1,:] = 0

    # free slip for u at top and bottom boundaries (du/dy=0)
    u_new[0,:] = u_new[1,:]
    u_new[-1,:] = u_new[-2,:]

    # h doesn't change at top and bottom boundaries
    h_new[0,:] = h[0,:]
    h_new[-1,:] = h[-1,:]

    return h_new, u_new, v_new

def runExperiment(hi,ui,vi,f2,hb,nt,
                  return_xarray=False,
                  snapshot_interval=15,
            dx=dx_default, dy=dy_default, dt=dt_default,
             g=g_default,
             h_min=1e3,
             filter_fac=0.01,
             diffusion_on=False,
             diffusion_coeff=1e-2,
             relaxation_on=False,
             pert_scale=1,
             h_eq=None,
             relaxation_timescale=24*3600):
    
    print('=== Simulation started ===')
    print('total time steps:', nt)
    print('total grid points:', hi.size)

    nx,ny = np.shape(hi)[1], np.shape(hi)[0]

    nsnapshot = nt // snapshot_interval

    # set up containers with initial conditions at time 0
    h = np.zeros([nsnapshot,ny,nx]); h[0] = hi; htemp=hi.copy()
    u = np.zeros([nsnapshot,ny,nx]); u[0] = ui; utemp=ui.copy()
    v = np.zeros([nsnapshot,ny,nx]); v[0] = vi; vtemp=vi.copy()

    if relaxation_on:
         
        if h_eq is None:
            print('need to pass heq if relaxation_on=True')
            print('simulation will end without producing output')
            return None

        h[0,0,:]=h_eq[0,:]
        h[0,-1,:]=h_eq[-1,:]

    print('=== Starting time stepping ===')

    # timestep the model forward in time
    for i in range(1,nt):
        print(i)
        hn,un,vn = lax_wendroff_timesteptimeStep(
            htemp,
            utemp,
            vtemp,
            f2,
            hb,
            dt=dt,
            dx=dx,
            dy=dy,
            g=g,
            h_min=h_min,
            filter_fac=filter_fac,
            diffusion_on=diffusion_on,
            diffusion_coeff=diffusion_coeff 
        )
        # update the temp variables for the next step
        htemp = hn.copy()
        utemp = un.copy()
        vtemp = vn.copy()

        if relaxation_on:
            w = np.exp(-dt/relaxation_timescale)
            htemp = (1-w) * h_eq + w * htemp
            utemp = w * utemp
            vtemp = w * vtemp
            htemp += pert_scale*np.random.randn(np.shape(htemp)[0],np.shape(htemp)[1])

        # if it's time to save a snapshot, store the current state in the output arrays
        if i % snapshot_interval == 0:
            h[i // snapshot_interval] = htemp
            u[i // snapshot_interval] = utemp
            v[i // snapshot_interval] = vtemp

    print('=== Simulation finished ===')

    # format the output data and return
    if return_xarray:
        ds=xr.Dataset({
            'h': (['t','y','x'], h),
            'u': (['t','y','x'], u),
            'v': (['t','y','x'], v)
        }, 
        coords={'t': np.arange(nsnapshot)*dt*snapshot_interval, 'y': np.arange(ny)*dy, 'x': np.arange(nx)*dx})
        return ds

    else:
        return h,u,v
    

def getGeostrophic(h,f2,g=g_default,dy=dy_default,dx=dx_default):
        hax = np.hstack([h[:,-1][:,np.newaxis], h, h[:,0][:,np.newaxis]])
        vg = g*(hax[:,2:]-hax[:,:-2])/(2*dx)/f2
        hay = np.vstack([h[0,:],h,h[-1,:]])
        ug = -g*(hay[2:,:]-hay[:-2,:])/(2*dy)/f2 

        # periodic BCs in x for geostrophic velocities
        ug[:,0] = ug[:,-2]
        ug[:,-1] = ug[:,1]      
        vg[:,0] = vg[:,-2]
        vg[:,-1] = vg[:,1]
        # no flow BCs in y for geostrophic velocities
        vg[0,:] = 0
        vg[-1,:] = 0
        # free slip BCs in y for geostrophic velocities
        ug[0,:] = ug[1,:]
        ug[-1,:] = ug[-2,:]

        return ug,vg  

@njit(nopython=True, fastmath=True)
def getVorticitytTimeSeries(u,v,dx=dx_default,dy=dy_default):
        z = np.zeros_like(u)
        z[:,1:-1,1:-1] = (v[:,1:-1,2:]-v[:,1:-1,:-2])/(2*dx) - (u[:,2:,1:-1]-u[:,:-2,1:-1])/(2*dy)     
        # BCs
        z[:,0,:] = z[:,1,:]
        z[:,-1,:] = z[:,-2,:]
        z[:,:,0] = z[:,:,1]
        z[:,:,-1] = z[:,:,-2]
        return z

@njit(nopython=True, fastmath=True)
def getDivergencetTimeSeries(u,v,dx=dx_default,dy=dy_default):
        d = np.zeros_like(u)
        d[:,1:-1,1:-1] = (u[:,1:-1,2:]-u[:,1:-1,:-2])/(2*dx) + (v[:,2:,1:-1]-v[:,:-2,1:-1])/(2*dy)     
        # BCs
        d[:,0,:] = d[:,1,:]
        d[:,-1,:] = d[:,-2,:]
        d[:,:,0] = d[:,:,1]
        d[:,:,-1] = d[:,:,-2]
        return d

@njit(nopython=True, fastmath=True)
def fast_interp_uv(xi,yi,x,y,u,v):

    ix = int(xi//dx)
    iy = int(yi//dy)

    nx = np.shape(u)[1]
    ny = np.shape(u)[0] 

    if not (ix>nx or iy>ny): 

        # x and y weights for interpolation
        wx = (xi - x[ix]) / dx
        wy = (yi - y[iy]) / dy

        # bilinear interpolation formulas:
        u_interp = u[iy,ix] * (1 - wx) * (1 - wy) + u[iy,ix+1] * wx * (1 - wy) + u[iy+1,ix] * (1 - wx) * wy + u[iy+1,ix+1] * wx * wy
        v_interp = v[iy,ix] * (1 - wx) * (1 - wy) + v[iy,ix+1] * wx * (1 - wy) + v[iy+1,ix] * (1 - wx) * wy + v[iy+1,ix+1] * wx * wy 


    return u_interp, v_interp

@njit(nopython=True, fastmath=True)
def RK4_step(xi, yi, x, y, u, v,dt=dt_default):
    k1u, k1v = fast_interp_uv(xi, yi, x, y, u, v)
    k2u, k2v = fast_interp_uv(xi + 0.5 * dt * k1u, yi + 0.5 * dt * k1v, x, y, u, v)
    k3u, k3v = fast_interp_uv(xi + 0.5 * dt * k2u, yi + 0.5 * dt * k2v, x, y, u, v)
    k4u, k4v = fast_interp_uv(xi + dt * k3u, yi + dt * k3v, x, y, u, v)

    ui = (k1u + 2*k2u + 2*k3u + k4u) / 6
    vi = (k1v + 2*k2v + 2*k3v + k4v) / 6

    return ui, vi

@njit(nopython=True, fastmath=True)
def particle_trajectory(x0, y0, x, y, u, v, dt=dt_default, RK4=False):

    # number of time steps is taken from the length of the velocity time series
    nt = np.shape(u)[0]
    traj_x = np.zeros(nt)
    traj_y = np.zeros(nt)

    traj_x[0] = x0
    traj_y[0] = y0

    for i in range(1, nt):

        # find the interpolated velocity at the current particle position
        if RK4:
            ui,vi = RK4_step(traj_x[i-1], traj_y[i-1], x, y, u[i-1], v[i-1], dt=dt)
        else:
            ui,vi = fast_interp_uv(traj_x[i-1], traj_y[i-1], x, y, u[i-1], v[i-1])
        
        # update the particle position using the interpolated velocity
        traj_x[i] = traj_x[i-1] + ui * dt
        traj_y[i] = traj_y[i-1] + vi * dt

        # enforce periodic BCs in x
        if traj_x[i] < x[0]: traj_x[i] = x[-1] + (traj_x[i] - x[0])  # wrap around left edge
        if traj_x[i] > x[-1]: traj_x[i] = x[0] + (traj_x[i] - x[-1])  # wrap around right edge

        # enforce no flow BCs in y (reflective)
        if traj_y[i] < y[0]: traj_y[i] = y[0] + (y[0] - traj_y[i])  # reflect off bottom edge
        if traj_y[i] > y[-1]: traj_y[i] = y[-1] - (traj_y[i] - y[-1])  # reflect off top edge

    return traj_x, traj_y