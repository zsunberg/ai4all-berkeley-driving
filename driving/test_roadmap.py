from roadmap import *
from math import pi
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
