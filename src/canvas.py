import time
import numpy as np
from PIL import Image
from math import sin, sqrt
from functools import reduce

nx = 40
ny = 40
niter = 1
overrelaxation = 1.9

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
        return [self.canvas.edges[(self.x + dx) % self.canvas.nx+1][(self.y + dy) % self.canvas.ny+1][direction]
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
                return self.canvas.edges[self.x][(self.y + 1) % (self.canvas.ny+1)][Edge.HORIZONTAL]
            if direction == Edge.RIGHT:
                return self.canvas.edges[(self.x + 1) % (self.canvas.nx+1)][self.y][Edge.VERTICAL]
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
        
        return [self.canvas.cells[self.x][self.y], self.canvas.cells[self.x][self.y].neighbor_at(self)]

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
        self.pf = 1.0
        self.d = 1.0
        self.df = 1.0
    
    @property
    def divergence(self):
        ret = 0
        for e, si in [(Edge.LEFT, 1), (Edge.TOP, -1), (Edge.RIGHT, -1), (Edge.BOTTOM, 1)]:
            ret += si * self.edge(e).v
        return ret
    
class SimEdge(Edge):
    def __init__(self, canvas, x, y, direction):
        super().__init__(canvas, x, y, direction)
        self.v = 0.0
        self.vf = 0.0

class Canvas:
    def __init__(self, nx, ny):
        self.nx = nx
        self.ny = ny
        self.cells = [[SimCell(self, x, y) for y in range(ny)] for x in range(nx)]
        self.edges = [[[SimEdge(self, x, y, direction) for direction in [Edge.VERTICAL, Edge.HORIZONTAL]] for y in range(ny+1)] for x in range(nx+1)]

def fluid_simulation(dt, canvas):
    
    '''
    for x in range(canvas.nx):
        for y in range(canvas.ny):
            canvas.cells[x][y].edge(Edge.BOTTOM).v -= 0.1
    '''
    
    for iteration in range(niter):
        for x in range(canvas.nx):
            for y in range(canvas.ny):
                
                if canvas.cells[x][y].is_boundary():
                    canvas.cells[x][y].s = 0
                    continue

                divergence = canvas.cells[x][y].edge(Edge.LEFT).v - canvas.cells[x][y].edge(Edge.RIGHT).v - canvas.cells[x][y].edge(Edge.TOP).v + canvas.cells[x][y].edge(Edge.BOTTOM).v
                ss = {direction: canvas.cells[x][y].neighbor_in(direction).s for direction in [Edge.LEFT, Edge.TOP, Edge.RIGHT, Edge.BOTTOM]}
                s = sum(ss.values())

                canvas.cells[x][y].edge(Edge.LEFT).vf   += divergence * ss[Edge.LEFT] / s
                canvas.cells[x][y].edge(Edge.TOP).vf    += divergence * ss[Edge.TOP] / s
                canvas.cells[x][y].edge(Edge.RIGHT).vf  += divergence * ss[Edge.RIGHT] / s
                canvas.cells[x][y].edge(Edge.BOTTOM).vf += divergence * ss[Edge.BOTTOM] / s

                canvas.cells[x][y].pf += divergence / s

        for x in range(canvas.nx):
            for y in range(canvas.ny):
                for e in [Edge.LEFT, Edge.TOP, Edge.RIGHT, Edge.BOTTOM]:
                    canvas.cells[x][y].edge(e).v = canvas.cells[x][y].edge(e).vf
                canvas.cells[x][y].p = canvas.cells[x][y].pf

    return canvas

t_last = time.time()
def renderer():
    global t_last
    cnv = Canvas(nx, ny)
        
    cnv.cells[5][5].edge(Edge.BOTTOM).v = 0.1
    #cnv.cells[5][7].edge(Edge.TOP).v = 10
    
    while True:
        
        dt = time.time() - t_last
        t_last = time.time()
        cnv = fluid_simulation(dt, cnv)
        data = np.zeros((nx, ny, 3))

        t = time.time()
        for x in range(nx):
            for y in range(ny):
                data[-y-1, x, :] = cnv.cells[x][y].p
        
        m = np.amax(data)
        print(m)
        data = np.divide(data, np.amax(data))
        data = np.multiply(data, 255)
        
        yield Image.fromarray(data.astype('uint8'), 'RGB')
