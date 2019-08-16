from math import pi, sin, cos, floor, ceil
import numpy as np
from random import random
from matplotlib.patches import Circle, Rectangle

# all angles in this file are in RADIANS unless otherwise noted

LINEWIDTH = 50.0

def angle_diff(a, b):
  """Calculate the difference in angles between -pi and pi"""
  delta = a-b
  while delta > pi:
    delta -= 2*pi
  while delta <= -pi:
    delta += 2*pi
  return delta

class Tile:
  def __init__(self, start, end):
    self.start = np.array(start)
    self.end = np.array(end)

class StraightTile(Tile):
  def distance_angle(self, x, y, theta):
    """Distance and angle from nearest road centerline"""
    dist = np.cross(self.vec(), np.array((x,y))-self.start).item()
    ang = angle_diff(theta, self.angle())
    return dist, ang

  def vec(self):
    return self.end-self.start

  def angle(self):
    v = self.vec()
    return np.arctan2(v[1], v[0])

  def plot(self, ax, xy, linewidth=LINEWIDTH):
    p1 = self.start + xy
    p2 = self.end + xy
    ax.plot([p1[0],p2[0]], [p1[1],p2[1]], linewidth=linewidth, color="gray")

class CurveTile(Tile):
  def distance_angle(self, x, y, theta):
    """Distance and angle from nearest road centerline"""
    # this is probably more complicated than necessary. Don't stare at it too hard.
    c2 = self.center2()
    c3 = self.center3()
    pos = np.array((x, y))
    cpos = pos - c2
    cpos3 = np.array((x-c2[0], y-c2[1], 0.0))
    if np.linalg.norm(cpos) <= 1e-6:
      closest = self.start
    else:
      closest = c2 + 0.5*cpos/np.linalg.norm(cpos)
    # tangent = np.cross((0., 0., c3[2]), cpos3)/np.linalg.norm(cpos3)
    closest3 = np.array((closest[0], closest[1], 0.0))
    tangent = np.cross((0., 0., c3[2]), closest3-c3)/0.5
    assert np.abs(np.linalg.norm(tangent) - 1.0) < 1e-5
    dist = np.cross(tangent[0:2], pos-closest).item()
    ang = angle_diff(theta, np.arctan2(tangent[1], tangent[0]))
    return dist, ang

  def center2(self):
    """2d array at the corner the road curves around."""
    cx = 1.0 if max(self.start[0], self.end[0]) > 1-1e-5 else 0.0
    cy = 1.0 if max(self.start[1], self.end[1]) > 1-1e-5 else 0.0
    return np.array((cx, cy))

  def center3(self):
    """3d array with x, y at center2. z=-1 if clockwise; z=1 o.w."""
    center = self.center2()
    relstart = self.start - center
    relend = self.end - center
    cz = np.cross(relstart, relend).item()
    assert cz != 0.0
    cz = 1.0 if cz > 0.0 else -1.0
    return np.array((center[0], center[1], cz))

  def plot(self, ax, xy, linewidth=LINEWIDTH):
    c = self.center2() + np.array(xy)
    circ = Circle(c, 0.5, linewidth=linewidth, fill=False, edgecolor="gray")
    rect = Rectangle(xy, 1.0, 1.0, facecolor="none", edgecolor="none")
    ax.add_artist(rect)
    ax.add_artist(circ)
    circ.set_clip_path(rect)


class RoadMap:
  """Collection of tiles that makes a map. tiles argument is a 2d numpy array of tiles"""

  def __init__(self, tiles):
    self.tiles = tiles
    
  def distance_angle_deg(self, x, y, theta_deg):
    tile = self.get_tile(x, y)
    rx, ry = self.tile_relative(x, y)
    theta_rad = theta_deg*pi/180
    d, a = tile.distance_angle(rx, ry, theta_rad)
    return d, a*180/pi

  def distance_angle(self, x, y, theta_rad):
    tile = self.get_tile(x, y)
    rx, ry = self.tile_relative(x, y)
    d, a = tile.distance_angle(rx, ry, theta_rad)
    return d, a

  def get_tile(self, x, y):
    """Return the tile that this falls into"""
    row = self.tiles.shape[0]-ceil(y)
    row = np.clip(row, 0, self.tiles.shape[0]-1)
    col = np.clip(floor(x), 0, self.tiles.shape[1]-1)
    return self.tiles[row, col]

  def tile_relative(self, x, y):
    if x >= self.tiles.shape[1]:
      x = x - (self.tiles.shape[1] - 1)
    elif x > 0: # within the tiles
      x = x%1
    if y >= self.tiles.shape[0]:
      y = y - (self.tiles.shape[0] - 1)
    elif y > 0: # within the tiles
      y = y%1
    return x, y

  def sample(self):
    return random()*self.tiles.shape[1], random()*self.tiles.shape[0]

  def plot(self, ax, **kwargs):
    ax.set_facecolor("lightgreen")
    for i in range(self.tiles.shape[0]):
      for j in range(self.tiles.shape[1]):
        self.tiles[i,j].plot(ax, (j, self.tiles.shape[0]-i-1), **kwargs)

coords = {'left':(0.0, 0.5),
          'right':(1.0, 0.5),
          'top':(0.5, 1.0),
          'bottom':(0.5, 0.0)}

def make_oval():
  l = coords['left']
  r = coords['right']
  t = coords['top']
  b = coords['bottom']

  rl = StraightTile(r, l)
  rb = CurveTile(r, b)
  tr = CurveTile(t, r)
  lr = StraightTile(l, r)
  lt = CurveTile(l, t)
  bl = CurveTile(b, l)
  return RoadMap(np.array(((rb, rl, bl),
                           (tr, lr, lt))))

def make_maze():
  l = coords['left']
  r = coords['right']
  t = coords['top']
  b = coords['bottom']

  rl = StraightTile(r, l)
  rb = CurveTile(r, b)
  tr = CurveTile(t, r)
  lr = StraightTile(l, r)
  lt = CurveTile(l, t)
  bl = CurveTile(b, l)
  tb = StraightTile(t, b)
  br = CurveTile(b, r)
  lb = CurveTile(l, b)
  rt = CurveTile(r, t)
  tl = CurveTile(t, l)
  bt = StraightTile(b, t)
  return RoadMap(np.array([[rb, rl, rl, bl, rb, bl],
                           [tb, br, lb, rt, tl, bt],
                           [tb, bt, tr, lr, lr, lt],
                           [tb, rt, rl, rl, rl, bl],
                           [tb, br, lb, br, lb, bt],
                           [tr, lt, tr, lt, tr, lt]]))
