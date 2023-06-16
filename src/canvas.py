import time
import numpy as np
from PIL import Image
from math import sin, cos, sqrt
from functools import reduce
import sys
from random import random

nx = 60
ny = 150
niter = 1

class Cell:
    def __init__(self, canvas, x, y):
        self.canvas = canvas
        self.x = x
        self.y = y

    @property
    def neighbors(self):
        return [self.canvas._cells[(self.x + dx) % self.canvas.nx][(self.y + dy) % self.canvas.ny]
                for dx, dy in [(-1, 0), (0, 1), (1, 0), (0, -1)]]
    
    @property
    def edges(self):
        return [self.canvas._edges[(self.x+dx) % self.canvas.nx+1][(self.y+dy) % self.canvas.ny+1][direction]
                for dx, dy, direction in [(0, 0, Edge.VERTICAL), (1, 0, Edge.HORIZONTAL), (0, 1, Edge.VERTICAL), (0, 0, Edge.HORIZONTAL)]]
    
    def neighbor_in(self, direction):
        if direction == Edge.LEFT:
            return self.canvas._cells[(self.x - 1) % self.canvas.nx][self.y]
        if direction == Edge.TOP:
            return self.canvas._cells[self.x % self.canvas.nx][(self.y + 1) % self.canvas.ny]
        if direction == Edge.RIGHT:
            return self.canvas._cells[(self.x + 1) % self.canvas.nx][self.y]
        if direction == Edge.BOTTOM:
            return self.canvas._cells[self.x % self.canvas.nx][(self.y - 1) % self.canvas.ny]

    def edge(self, direction):
        if direction == Edge.LEFT:
            return self.canvas._edges[self.x][self.y][Edge.VERTICAL]
        if direction == Edge.TOP:
            return self.canvas._edges[self.x][(self.y+1) % (self.canvas.ny+1)][Edge.HORIZONTAL]
        if direction == Edge.RIGHT:
            return self.canvas._edges[(self.x+1) % (self.canvas.nx+1)][self.y][Edge.VERTICAL]
        if direction == Edge.BOTTOM:
            return self.canvas._edges[self.x][self.y][Edge.HORIZONTAL]

    def is_boundary(self):
        return (self.x == 0) or (self.x == self.canvas.nx - 1) or (self.y == 0) or (self.y == self.canvas.ny - 1)

class Edge:
    VERTICAL = 0
    HORIZONTAL = 1
    LEFT   = 0
    TOP    = 1
    RIGHT  = 2
    BOTTOM = 3

    def __init__(self, canvas, x, y, direction):
        self.canvas = canvas
        self.x = x
        self.y = y
        self.direction = direction
    
    @property
    def cells(self):
        if self.is_boundary():
            return [self.canvas._cells[self.x][self.y]]
        
        return [self.canvas._cells[self.x][self.y], self.canvas._cells[self.x][self.y].neighbor_in(self.direction)]

    def is_boundary(self):
        if self.direction == Edge.VERTICAL:
            return (self.x == 0) or (self.x == self.canvas.nx)
        if self.direction == Edge.HORIZONTAL:
            return (self.y == 0) or (self.y == self.canvas.ny)

class SimCell(Cell):
    def __init__(self, canvas, x, y):
        super().__init__(canvas, x, y)
        self.T = 0.0
        self.p = 1.0

    '''
    @property
    def divergence(self):
        div = 0
        for direction, sign in [(Edge.LEFT, 1), (Edge.TOP, -1), (Edge.RIGHT, -1), (Edge.BOTTOM, 1)]:
            div += sign * self.edge(direction).v
        return div
    '''

    def diffuse(self, dt, property_grabber):
        return dt * (sum([property_grabber(cell) for cell in self.neighbors]) - 4 * property_grabber(self))
    
    @property
    def v(self):
        return ((self.edge(Edge.LEFT).v + self.edge(Edge.RIGHT).v) / 2, (self.edge(Edge.TOP).v + self.edge(Edge.BOTTOM).v) / 2)

    def advect(self, dt, property_grabber):
        v = self.v
        x = self.x + v[0] * dt
        y = self.y + v[1] * dt
        return (self.canvas.cells_weighted_average(x, y, property_grabber) - property_grabber(self))

class SimEdge(Edge):
    def __init__(self, canvas, x, y, direction):
        super().__init__(canvas, x, y, direction)
        self.v = 0.0
        self.dv = 0.0

class Canvas:
    def __init__(self, nx, ny):
        self.nx = nx
        self.ny = ny
        self._cells = [[SimCell(self, x, y) for y in range(ny)] for x in range(nx)]
        self._edges = [[[SimEdge(self, x, y, direction) for direction in [Edge.VERTICAL, Edge.HORIZONTAL]] for y in range(ny+1)] for x in range(nx+1)]

    def cells_weighted_average(self, x, y, property_grabber):
        x0 = int(x)
        x1 = x0 + 1
        y0 = int(y)
        y1 = y0 + 1
        return (x1 - x) * (y1 - y)   * property_grabber(self._cells[x0 % self.nx][y0 % self.ny]) + \
               (x1 - x) * (y  - y0)  * property_grabber(self._cells[x0 % self.nx][y1 % self.ny]) + \
               (x  - x0) * (y1 - y)  * property_grabber(self._cells[x1 % self.nx][y0 % self.ny]) + \
               (x  - x0) * (y  - y0) * property_grabber(self._cells[x1 % self.nx][y1 % self.ny])
    
    @property
    def cells(self):
        for x in range(self.nx):
            for y in range(self.ny):
                yield self._cells[x][y]

    def apply_to_each_cell(self, func):
        for x in range(self.nx):
            for y in range(self.ny):
                self._cells[x][y] = func(self._cells[x][y])

def fluid_simulation(dt, canvas):

    '''
    def turbulence(cell):
        if cell.T >= 100:
            cell.edge(Edge.BOTTOM).v += (random()-0.5) * 0.1 * dt
            cell.edge(Edge.LEFT).v += (random()-0.5) * 0.4 * dt
        return cell
    '''
    
    def gravity(cell):
        if cell.T > 50:
            cell.edge(Edge.BOTTOM).v -= dt * 2.1 * cell.T * cell.d
        return cell

    def diffusion(cell):
        # vapor
        if cell.T >= 15:
            cell.dd += 0.2 * cell.diffuse(dt, lambda _cell: _cell.d)
        # combustion and evaporation
        if cell.T >= 80 and cell.d > 0:
            cell.dT += 230.0 * cell.d * 10 * dt
            cell.dd -= 600.0 * cell.dT * dt
        # heat diffusion
        cell.dT += 0.8 * cell.diffuse(dt, lambda _cell: _cell.heat_conduction * _cell.T)
        return cell

    def apply_iteration(cell):
        cell.d += cell.dd
        cell.dd = 0
        if cell.d < 0:
            cell.d = 0
        cell.T += cell.dT
        cell.dT = 0
        return cell

    def advection(cell):
        cell.dd += 0.2 * cell.advect(dt, lambda _cell: _cell.d)
        #cell.dT -= 0.02 * CELL.advect(dt, lambda _cell: _cell.d) * cell.advect(dt, lambda _cell: _cell.T)
        return cell

    #canvas.apply_to_each_cell(turbulence)
    canvas.apply_to_each_cell(gravity)
    canvas.apply_to_each_cell(diffusion)
    canvas.apply_to_each_cell(apply_iteration)
    canvas.apply_to_each_cell(advection)
    canvas.apply_to_each_cell(apply_iteration)

    return canvas

t_last = time.time()
def renderer():
    global t_last
    cnv = Canvas(nx, ny)

    # candle
    for x in range(int(0.35*nx), int(0.65*nx)):
        for y in range(int(0.8*ny)):
            cnv._cells[x][y].d = 1.0
            cnv._cells[x][y].material = 1

    # ignition source
    for x in range(int(nx/2)-2, int(nx/2)+2):
        for y in range(int(0.8*ny),int(0.82*ny)):
            cnv._cells[x][y].T = 160

    while True:
        # calculating time step
        t = time.time()
        dt = t - t_last
        t_last = t

        # walls
        for y in range(ny):
            cnv._cells[0][y].T = 0
            cnv._cells[nx-1][y].T = 0

        # running the simulation
        cnv = fluid_simulation(dt, cnv)
    
        # converting canvas to an image
        data = np.zeros((ny, nx, 3))
        for x in range(nx):
            for y in range(ny):
                data[-y-1, x, 2] = min(max(cnv._cells[x][y].d * (cnv._cells[x][y].T/50), 0.0), 1)
                data[-y-1, x, 1] = min(max(cnv._cells[x][y].d, 0.0), 1)
                data[-y-1, x, 0] = min(max(cnv._cells[x][y].T / 100, 0.0), 1)
            
        #data = np.divide(data, max(np.amax(data), 0.1))
        data = np.multiply(data, 255)
        
        yield Image.fromarray(data.astype('uint8'), 'RGB')
