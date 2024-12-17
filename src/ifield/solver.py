import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


class ElectrostaticSystem:
    def __init__(self, ni, nj, dx=1, dy=1):
        self._nu = ni * nj
        self._ni = ni
        self._nj = nj
        self._dx = dx
        self._dy = dy
        self.b = np.empty(self._nu, dtype=np.double)
        self._V = np.empty_like(self.b)
        self.V = np.empty((nj, ni), dtype=np.double)
        self.E_x = np.empty_like(self.V)
        self.E_y = np.empty_like(self.V)
        self.x_pts = np.arange(0, ni * dx, dx)
        self.y_pts = np.arange(0, nj * dy, dy)
        self.interp_x = None
        self.interp_y = None
        self.reset()

    def reset(self):
        self.b[:] = 0.0
        self._V[:] = 0.0
        self.V[:] = 0.0
        self.E_x = 0.0
        self.E_y = 0.0

        self.A = sp.sparse.lil_matrix((self._nu, self._nu), dtype=np.double)

        for j in range(self._nj):
            for i in range(self._ni):
                u = j * self._ni + i
                # assumes zero Dirichlet boundary conditions at edges
                if i == 0:
                    self.A[u, :] = 0
                    self.A[u, u] = 1
                    self.b[u] = 0.0
                elif i == self._ni - 1:
                    self.A[u, :] = 0
                    self.A[u, u] = 1
                    self.b[u] = 0.0
                elif j == 0:
                    self.A[u, :] = 0
                    self.A[u, u] = 1
                    self.b[u] = 0.0
                elif j == self._nj - 1:
                    self.A[u, :] = 0
                    self.A[u, u] = 1
                    self.b[u] = 0.0
                else:
                    self.A[u, u - self._ni] = 1 / (self._dy ** 2)
                    self.A[u, u - 1] = 1 / (self._dx ** 2)
                    self.A[u, u + 1] = 1 / (self._dx ** 2)
                    self.A[u, u + self._ni] = 1 / (self._dy ** 2)
                    self.A[u, u] = -2 / (self._dx ** 2) - 2 / (self._dy ** 2)

    def set_dirichlet(self, i, j, val):
        u = j * self._ni + i
        self.A[u, :] = 0
        self.A[u, u] = 1
        self.b[u] = val

    def solve(self):
        self._V[:] = sp.sparse.linalg.spsolve(self.A.tocsr(), self.b)
        self.V = np.reshape(self._V, (self._nj, self._ni))
        self.E_y, self.E_x = np.gradient(self.V)
        self.E_x *= -1
        self.E_y *= -1
        self.interp_x = sp.interpolate.RegularGridInterpolator(
            (self.y_pts, self.x_pts),
            self.E_x
        )
        self.interp_y = sp.interpolate.RegularGridInterpolator(
            (self.y_pts, self.x_pts),
            self.E_y
        )

    def get_field_at(self, xi, yi):
        return (self.interp_x((yi, xi)).item(), self.interp_y((yi, xi)).item())


if __name__ == "__main__":
    ni = 50
    nj = 50
    dx = 1
    dy = 1
    ep = ElectrostaticSystem(ni, nj, dx, dy)
    ep.set_dirichlet(25, 25, 100)
    ep.set_dirichlet(10, 30, -25)
    ep.solve()
    print(ep.get_field_at(25.5, 25.5))
    np.savetxt("potential.csv", ep.V, delimiter=",")
    plt.contourf(ep.V)
    plt.contour(ep.V)
    plt.colorbar
    plt.show()