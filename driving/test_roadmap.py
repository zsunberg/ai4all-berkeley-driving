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

# if __name__ == "__main__":
#   test_straight_horiz()
#   test_straight_down()
#   test_bottom_left()
