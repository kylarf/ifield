import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

class ElectricPotential:
    def __init__(self, ni, nj, dx=1, dy=1):
        self._nu = ni*nj
        self._ni = ni
        self._nj = nj
        self._dx = dx
        self._dy = dy
        self.A = sp.sparse.lil_matrix((self._nu, self._nu), dtype=np.double)
        self.b = np.zeros(self._nu, dtype=np.double)
        self._V = np.zeros_like(self.b)
        self.V = np.zeros((nj, ni), dtype=np.double)

        for j in range(self._nj):
            for i in range(self._ni):
                u = j * self._ni + i
                # assumes zero Neumann boundary conditions at edges
                if i == 0:
                    self.A[u, u] = 1
                    self.A[u, u+1] = -1
                    self.b[u] = 0.0
                elif i == self._ni - 1:
                    self.A[u, u] = 1
                    self.A[u, u-1] = -1
                    self.b[u] = 0.0
                elif j == 0:
                    self.A[u, u] = 1
                    self.A[u, u+self._ni] = -1
                    self.b[u] = 0.0
                elif j == self._nj - 1:
                    self.A[u, u] = 1
                    self.A[u, u-self._ni] = -1
                    self.b[u] = 0.0
                else:
                    self.A[u, u-self._ni] = 1/(self._dy**2)
                    self.A[u, u-1] = 1/(self._dx**2)
                    self.A[u, u+1] = 1/(self._dx**2)
                    self.A[u, u+self._ni] = 1/(self._dy**2)
                    self.A[u, u] = -2/(self._dx**2) - 2/(self._dy**2)
    
    def set_dirichlet(self, i, j, val):
        u = j*self._ni + i
        self.A[u, :] = 0
        self.A[u, u] = 1
        self.b[u] = val

    def solve(self):
        self._V[:] = sp.sparse.linalg.spsolve(self.A, self.b)
        self.V = np.reshape(self._V, (self._nj, self._ni))


if __name__ == "__main__":
    ni = 50
    nj = 50
    dx = 1
    dy = 1
    ep = ElectricPotential(ni, nj, dx, dy)
    ep.set_dirichlet(25, 25, 100)
    ep.set_dirichlet(10, 30, -25)
    ep.solve()
    np.savetxt("potential.csv", ep.V, delimiter=",")
    plt.contourf(ep.V)
    plt.contour(ep.V)
    plt.colorbar
    plt.show()
