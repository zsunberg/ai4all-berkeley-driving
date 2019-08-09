from math import pi, sin, cos
import numpy as np
# for later?
# coords = {'left': (0.0, 0.5),
#           'right': (1.0, 0.5),
#           'top': (0.5, 1.0),
#           'bottom': (0.5, 0.0)}

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

class CurveTile(Tile):
  def distance_angle(self, x, y, theta):
    """Distance and angle from nearest road centerline"""
    # this is probably more complicated than necessary. Don't stare at it too hard.
    c2 = self.center2()
    c3 = self.center3()
    pos = np.array((x, y))
    cpos = pos - c2
    cpos3 = np.array((x-c2[0], y-c2[1], 0.0))
    closest = c2 + 0.5*cpos/np.linalg.norm(cpos)
    tangent = np.cross((0., 0., c3[2]), cpos3)/np.linalg.norm(cpos3)
    assert np.abs(np.linalg.norm(tangent) - 1.0) < 1e-5
    dist = np.cross(tangent[0:2], pos-closest).item()
    print(np.arctan2(tangent[1], tangent[0]))
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

class RoadMap:
  """Collection of tiles that makes a map. tiles argument is a"""

  def __init__(self, tiles):
    pass
    
  def distance_angle_deg(self, x, y, theta):
    pass

  def tile_of(self, x, y):
    pass

  def tile_relative(self, x, y):
    pass

