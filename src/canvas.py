import time
import numpy as np
from PIL import Image
from math import sin, sqrt
from functools import reduce
import sys

nx = 10
ny = 10
niter = 1

class Cell:
    def __init__(self, canvas, x, y):
        self.canvas = canvas
        self.x = x
        self.y = y

    @property
    def neighbors(self):
        return [self.canvas.cells[(self.x + dx) % self.canvas.nx][(self.y + dy) % self.canvas.ny]
                for dx, dy in [(-1, 0), (0, 1), (1, 0), (0, -1)]]
    
    @property
    def edges(self):
        return [self.canvas.edges[(self.x+dx) % self.canvas.nx+1][(self.y+dy) % self.canvas.ny+1][direction]
                for dx, dy, direction in [(0, 0, Edge.VERTICAL), (1, 0, Edge.HORIZONTAL), (0, 1, Edge.VERTICAL), (0, 0, Edge.HORIZONTAL)]]
    
    def neighbor_in(self, direction):
        if direction == Edge.LEFT:
            return self.canvas.cells[(self.x - 1) % self.canvas.nx][self.y]
        if direction == Edge.TOP:
            return self.canvas.cells[self.x % self.canvas.nx][(self.y + 1) % self.canvas.ny]
        if direction == Edge.RIGHT:
            return self.canvas.cells[(self.x + 1) % self.canvas.nx][self.y]
        if direction == Edge.BOTTOM:
            return self.canvas.cells[self.x % self.canvas.nx][(self.y - 1) % self.canvas.ny]

    def edge(self, direction):
        if direction == Edge.LEFT:
            return self.canvas.edges[self.x][self.y][Edge.VERTICAL]
        if direction == Edge.TOP:
            return self.canvas.edges[self.x][(self.y+1) % (self.canvas.ny+1)][Edge.HORIZONTAL]
        if direction == Edge.RIGHT:
            return self.canvas.edges[(self.x+1) % (self.canvas.nx+1)][self.y][Edge.VERTICAL]
        if direction == Edge.BOTTOM:
            return self.canvas.edges[self.x][self.y][Edge.HORIZONTAL]

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
            return [self.canvas.cells[self.x][self.y]]
        
        return [self.canvas.cells[self.x][self.y], self.canvas.cells[self.x][self.y].neighbor_in(self.direction)]

    def is_boundary(self):
        if self.direction == Edge.VERTICAL:
            return (self.x == 0) or (self.x == self.canvas.nx)
        if self.direction == Edge.HORIZONTAL:
            return (self.y == 0) or (self.y == self.canvas.ny)

class SimCell(Cell):
    def __init__(self, canvas, x, y):
        super().__init__(canvas, x, y)
        self.s = 1.0
        self.p = 1.0
        self.d = 0.0
        self.dd = 0.0
    
    @property
    def divergence(self):
        div = 0
        for direction, sign in [(Edge.LEFT, 1), (Edge.TOP, -1), (Edge.RIGHT, -1), (Edge.BOTTOM, 1)]:
            div += sign * self.edge(direction).v
        return div

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
        self.cells = [[SimCell(self, x, y) for y in range(ny)] for x in range(nx)]
        self.edges = [[[SimEdge(self, x, y, direction) for direction in [Edge.VERTICAL, Edge.HORIZONTAL]] for y in range(ny+1)] for x in range(nx+1)]

    def cells_weighted_average(self, x, y, property_grabber):
        x0 = int(x)
        x1 = x0 + 1
        y0 = int(y)
        y1 = y0 + 1
        return (x1 - x) * (y1 - y)   * property_grabber(self.cells[x0 % self.nx][y0 % self.ny]) + \
               (x1 - x) * (y  - y0)  * property_grabber(self.cells[x0 % self.nx][y1 % self.ny]) + \
               (x  - x0) * (y1 - y)  * property_grabber(self.cells[x1 % self.nx][y0 % self.ny]) + \
               (x  - x0) * (y  - y0) * property_grabber(self.cells[x1 % self.nx][y1 % self.ny])

def fluid_simulation(dt, canvas):

    # Diffusion
    for x in range(nx):
        for y in range(ny):
            canvas.cells[x][y].dd += 0.2 * canvas.cells[x][y].diffuse(dt, lambda cell: cell.d)
    
    for x in range(nx):
        for y in range(ny):
            canvas.cells[x][y].d += canvas.cells[x][y].dd
            canvas.cells[x][y].dd = 0.0

    # Advection
    for x in range(nx):
        for y in range(ny):
            canvas.cells[x][y].dd += canvas.cells[x][y].advect(dt, lambda cell: cell.d)

    for x in range(nx):
        for y in range(ny):
            canvas.cells[x][y].d += canvas.cells[x][y].dd
            canvas.cells[x][y].dd = 0.0


    return canvas

t_last = time.time()
def renderer():
    global t_last
    cnv = Canvas(nx, ny)

    for x in range(nx):
        for y in range(ny):
            cnv.cells[x][y].edge(Edge.BOTTOM).v = 1.5
            cnv.cells[x][y].edge(Edge.LEFT).v = 1.2
            if y == 0: cnv.cells[x][y].d = 250.0
    
    while True:
        # calculating time step
        t = time.time()
        dt = t - t_last
        t_last = t

        # running the simulation
        cnv = fluid_simulation(dt, cnv)
    
        # converting canvas to an image
        data = np.zeros((nx, ny, 3))
        for x in range(nx):
            for y in range(ny):
                data[-y-1, x, :] = cnv.cells[x][y].d
            
        print(sum(data.flatten()) / 3)
        #data = np.divide(data, max(np.amax(data), 0.1))
        #data = np.multiply(data, 255)
        
        yield Image.fromarray(data.astype('uint8'), 'RGB')
