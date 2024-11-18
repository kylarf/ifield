import numpy as np
import matplotlib.pyplot as plt

from ifield.solvers import sor

def set_dirichlet(val, i, j, u, a, b, c, d, e, f):
    f[i, j] = val * e[i, j]
    u[i, j] = val
    a[i, j] = 0
    b[i, j] = 0
    c[i, j] = 0
    d[i, j] = 0

if __name__ == "__main__":
    dx = 1
    dy = 1
    ncols = 50
    nrows = 50
    l_x = ncols / dx
    l_y = nrows / dy
    u = np.zeros((nrows, ncols), dtype="float64")
    a = np.ones_like(u) / dx**2
    b = np.ones_like(u) / dx**2
    c = np.ones_like(u) / dy**2
    d = np.ones_like(u) / dy**2
    e = -2.0 * np.ones_like(u) * (1/dx**2 + 1/dy**2)
    f = np.zeros_like(u)

    # Dirichlet boundary condition
    #for _ in range(25):
    #    i = np.random.randint(1, nrows-1)
    #    j = np.random.randint(0, ncols-1)
    #    val = (100 - -100) * np.random.random_sample() - 100
    #    set_dirichlet(val, i, j, u, a, b, c, d, e, f)
    set_dirichlet(100, slice(10,20), slice(10,20), u, a, b, c, d, e, f)
    #set_dirichlet(100, slice(10,20), 25, u, a, b, c, d, e, f)

    # Neumann zero-derivative boundary conditions at edges (smooth edges)
    # top
    b[0,:] = 0
    a[0,:] = 2
    # bottom
    a[-1,:] = 0
    b[-1,:] = 2
    # left
    d[:,0] = 0
    c[:,0] = 2
    # right
    c[:,-1] = 0
    d[:,-1] = 2

    r_jac = (np.cos(np.pi/l_x) + (dx/dy)**2 * np.cos(np.pi/l_y)) / (1 + (dx/dy)**2)

    n_iter = sor(a, b, c, d, e, f, u, r_jac)
    print(f"n = {n_iter}")

    np.savetxt("sol.csv", u, delimiter=",")

    plt.contourf(u, extend="both")
    plt.show()
