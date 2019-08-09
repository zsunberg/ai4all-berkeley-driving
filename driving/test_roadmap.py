from roadmap import *
from math import pi, sqrt
import pytest

def test_straight_horiz():
  tile = StraightTile((0.0, 0.5), (1.0, 0.5))
  assert tile.angle() == 0.0
  d, a = tile.distance_angle(0.5, 1.0, 0.1)
  assert d == 0.5
  assert a == 0.1
  d, a = tile.distance_angle(0.2, 0.8, -0.3)
  assert d == pytest.approx(0.3)
  assert a == pytest.approx(-0.3)

def test_straight_down():
  tile = StraightTile((0.5, 1.0), (0.5, 0.0))
  assert tile.angle() == pytest.approx(-pi/2)
  d, a = tile.distance_angle(0.5, 1.0, 0.1)
  assert d == pytest.approx(0.0)
  assert a == pytest.approx(0.1+pi/2)
  d, a = tile.distance_angle(0.2, 0.8, -0.3)
  assert d == pytest.approx(-0.3)
  assert a == pytest.approx(pi/2-0.3)

def test_bottom_left():
  tile = CurveTile((0.5, 0.0), (0.0, 0.5))
  assert all(tile.center2() == (0.0, 0.0))
  assert all(tile.center3() == (0.0, 0.0, 1.0))
  d, a = tile.distance_angle(0.5, 0.5, 0.1)
  assert d == pytest.approx(-(sqrt(0.5)-0.5))
  assert a == pytest.approx(0.1-3*pi/4)
  d, a = tile.distance_angle(0.2, 0.8, -0.3)
  xy = np.array((0.2, 0.8))
  assert d == pytest.approx(-np.linalg.norm(xy-0.5*xy/np.linalg.norm(xy)))
  assert a == pytest.approx(angle_diff(-0.3, np.arctan2(0.8, 0.2)+pi/2))

def test_right_top():
  tile = CurveTile((1.0, 0.5), (0.5, 1.0))
  assert all(tile.center2() == (1.0, 1.0))
  assert all(tile.center3() == (1.0, 1.0, -1.0))
  d, a = tile.distance_angle(0.5, 0.5, 0.1)
  assert d == pytest.approx(sqrt(0.5)-0.5)
  assert a == pytest.approx(angle_diff(0.1, 3*pi/4))
  d, a = tile.distance_angle(0.2, 0.8, -0.3)
  c = np.array((1.0, 1.0))
  xy = np.array((0.2, 0.8))
  cpos = xy - c
  assert d > 0.0
  assert d == pytest.approx(np.linalg.norm(xy - (0.5*cpos/np.linalg.norm(cpos)+c)))
  assert a == pytest.approx(angle_diff(-0.3, np.arctan2(cpos[1], cpos[0])-pi/2))

def test_get_tile():
  l = coords['left']
  r = coords['right']
  t = coords['top']
  b = coords['bottom']
  oval = make_oval()
  bl = oval.get_tile(0.0, 0.0)
  assert all(bl.start == t)
  assert all(bl.end == r)
  bl = oval.get_tile(-1.0, -1.0)
  assert all(bl.start == t)
  assert all(bl.end == r)
  tl = oval.get_tile(0.1, 1.1)
  assert all(tl.start == r)
  assert all(tl.end == b)
  tm = oval.get_tile(1.1, 1.1)
  assert all(tm.start == r)
  assert all(tm.end == l)

def test_global_distance_angle():
  oval = make_oval()
  d, ar = oval.distance_angle(1.1, 0.7, 0.1)
  assert d == pytest.approx(0.2)
  assert ar == pytest.approx(0.1)

# if __name__ == "__main__":
#   test_straight_horiz()
#   test_straight_down()
#   test_bottom_left()
