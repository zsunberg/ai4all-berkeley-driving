from car import *
import numpy as np
import pytest

def test_car():
  car = DubinsCarModel(speed=1.0, max_turn_rate=pi)
  s = (0.0, 0.5, -0.1)
  sp = car.dynamics(s, 0.0, 1.0)
  assert sp[0] == pytest.approx(cos(-0.1))
  assert sp[1] == pytest.approx(0.5+sin(-0.1))
  assert sp[2] == pytest.approx(-0.1)
  s = (0.0, 0.5, 0.0)
  dt = pi/4
  sp = car.dynamics(s, -(pi/2)/dt, dt)
  print('sp', sp)
  assert sp[0] == pytest.approx(0.5)
  assert sp[1] == pytest.approx(0.0)
  assert sp[2] == pytest.approx(-pi/2)
