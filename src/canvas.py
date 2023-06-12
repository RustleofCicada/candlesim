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
        try:
            if direction == Edge.LEFT:
                return self.canvas.edges[self.x][self.y][Edge.VERTICAL]
            if direction == Edge.TOP:
                return self.canvas.edges[self.x][(self.y+1) % (self.canvas.ny+1)][Edge.HORIZONTAL]
            if direction == Edge.RIGHT:
                return self.canvas.edges[(self.x+1) % (self.canvas.nx+1)][self.y][Edge.VERTICAL]
            if direction == Edge.BOTTOM:
                return self.canvas.edges[self.x][self.y][Edge.HORIZONTAL]
        except:
            return None

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

def fluid_simulation(dt, canvas):

    for _ in range(niter):
        for x in range(canvas.nx):
            for y in range(canvas.ny):
                for e in [Edge.LEFT, Edge.TOP, Edge.RIGHT, Edge.BOTTOM]:
                    canvas.cells[x][y].edge(e).dv += (canvas.cells[x][y].p - canvas.cells[x][y].neighbor_in(canvas.cells[x][y].edge(e).direction).p) / (2 * niter * 20)

    for x in range(canvas.nx):
        for y in range(canvas.ny):
                for e in [Edge.VERTICAL, Edge.HORIZONTAL]:
                    canvas.cells[x][y].p -= canvas.edges[x][y][e].dv * 0.2
                    canvas.edges[x][y][e].v += canvas.edges[x][y][e].dv
                    canvas.edges[x][y][e].dv = 0

    return canvas

t_last = time.time()
def renderer():
    global t_last
    cnv = Canvas(nx, ny)
    
    cnv.cells[4][4].p = 2.0
    
    while True:

        dt = time.time() - t_last
        t_last = time.time()
        cnv = fluid_simulation(dt, cnv)
        data = np.zeros((nx, ny, 3))

        t = time.time()
        for x in range(nx):
            for y in range(ny):
                data[-y-1, x, :] = cnv.cells[x][y].p
        
        data = np.divide(data, max(np.amax(data), 0.1))
        data = np.multiply(data, 255)
        
        yield Image.fromarray(data.astype('uint8'), 'RGB')
