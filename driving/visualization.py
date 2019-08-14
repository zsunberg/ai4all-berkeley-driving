import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np
from IPython.display import HTML
from celluloid import Camera

def animate(env, policy, time=20.0, dt=0.1):
    fig = plt.figure(figsize=(11,8))
    ax = fig.gca()
    ax.set_aspect(1)
    cam = Camera(fig)
    s = env.reset()
    n_steps = round(time/dt)
    for i in range(n_steps):
        # print(f'step {i} of {n_steps}', end='\r')
        env.map.plot(ax)
        car = Rectangle((s[0],s[1]), 0.1, 0.07, s[2], zorder=10, color="orange")
        ax.add_patch(car)
        cam.snap()
        a = policy(*s)
        s, r, done, info = env.step_a_deg(a)
    # return cam.animate()
    anim = cam.animate(interval=dt*1000)
    return HTML(anim.to_html5_video())

def sim(env, policy, n_steps=99):
    s = env.reset()
    history = []
    for i in range(n_steps):
        a = policy(*s)
        sp, r, done, info = env.step_a_deg(a)
        history.append((s, a, r, sp))
        s = sp
        if done:
            break
    return history

def plot_episode(env, policy, n_steps=100, ax=None):
    if ax == None:
        ax = plt.gca()
        figure(figsize(11,8))
    history = sim(env, policy, n_steps)
    xs = [step[0][0] for step in history]
    ys = [step[0][1] for step in history]
    reward = sum([step[2] for step in history])
    env.map.plot(ax)
    return ax.plot(xs, ys, color='red')
