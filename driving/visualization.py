import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib import gridspec
import numpy as np
from IPython.display import HTML, display
from celluloid import Camera
from driving.env import DrivingEnv, LINEWIDTH
from math import sin, pi
import ipywidgets as widgets

def record(env, policy, time=20.0, dt=0.1, initial_state=None, fix_limits=True):
    fig = plt.figure(figsize=(11,8))
    ax = fig.gca()
    ax.set_aspect(1)
    if fix_limits:
        ax.set_xlim(-0.0, 3.0)
        ax.set_ylim(-0.0, 2.0)
    cam = Camera(fig)
    s = env.reset()
    if initial_state is not None:
        s = initial_state
        env.state = initial_state
    n_steps = round(time/dt)
    for i in range(n_steps):
        # print(f'step {i} of {n_steps}', end='\r')
        env.map.plot(ax)
        car = Rectangle((s[0],s[1]), 0.1, 0.07, s[2], zorder=10, color="orange")
        ax.add_patch(car)
        cam.snap()
        a = policy(*s)
        s, r, done, info = env.step_a_deg(a)
    return cam

def animate(env, policy, dt=0.1, **kwargs):
    cam = record(env, policy, dt=dt, **kwargs)
    anim = cam.animate(interval=dt*1000)
    return HTML(anim.to_jshtml())

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

def plot_episode(env, policy, n_steps=100, ax=None, fix_limits=True, buff=0.0, linewidth=LINEWIDTH):
    if ax == None:
        fig = plt.figure(figsize=(11,8))
        ax = fig.gca()
        ax.set_aspect(1)
    if fix_limits:
        ax.set_xlim(-0.0-buff, 3.0+buff)
        ax.set_ylim(-0.0-buff, 2.0+buff)
    history = sim(env, policy, n_steps)
    xs = [step[0][0] for step in history]
    ys = [step[0][1] for step in history]
    reward = sum([step[2] for step in history])
    env.map.plot(ax, linewidth=linewidth)
    return ax.plot(xs, ys, color='red')

# Returns the plot object itself
def plot_episode_training(env, policy, n_steps=100, ax=None):
    if ax == None:
        ax = plt.gca()
        plt.figure(figsize=(11,8))
    history = sim(env, policy, n_steps)
    xs = [step[0][0] for step in history]
    ys = [step[0][1] for step in history]
    reward = sum([step[2] for step in history])
    env.map.plot(ax)
    ax.plot(xs, ys, color='red')
    return ax

def xy_contourf(f, theta=0.0, a=0.0, xmax=3, ymax=2, ax=None, **kwargs):
    ax = ax or plt.gca()
    if 'cmap' not in kwargs:
        kwargs['cmap'] = plt.get_cmap('RdYlGn')
    N = 100
    xs = np.linspace(0.0, xmax, N)
    ys = np.linspace(0.0, ymax, N)
    X, Y = np.meshgrid(xs, ys)
    Z = np.empty(X.shape)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            xx = X[i,j]
            yy = Y[i,j]
            Z[i,j] = f(xx, yy, theta, a)
            
    mappable = ax.contourf(X,Y,Z,
                           alpha=1.0,
                           zorder=8,
                           **kwargs
                          )
    return mappable

def r_q_ep(model, theta=0.0, a=0.0, fig=None):
    fig = fig or plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(2,2)
    rax = fig.add_subplot(gs[0, 0])
    rax.set_aspect(1)
    qax = fig.add_subplot(gs[0, 1])
    qax.set_aspect(1)
    epax = fig.add_subplot(gs[1, :])
    epax.set_aspect(1)
    # rax, qax, epax = fig.subplots(1,3)
    
    env = model.env
    xmax = env.map.tiles.shape[1]
    ymax = env.map.tiles.shape[0]
    xy_contourf(env.reward, theta, a, xmax, ymax, rax)
    rax.set_title('R(x, y, θ={}, a={})'.format(theta, a))
    mappable = xy_contourf(model.q_value, theta, a, xmax, ymax, qax, cmap=plt.get_cmap('viridis'))
    qax.set_title('Q(x, y, θ={}, a={})'.format(theta, a))
    fig.colorbar(mappable)
    plot_episode(env, model.basicPolicy, ax=epax, buff=1.5, linewidth=25.0)
    return fig

def view_sa_func(f, env=DrivingEnv()):
    x_slider = widgets.FloatSlider(description='x', min=0.0, max=3.0, value=1.0, continuous_update=False)
    y_slider = widgets.FloatSlider(description='y', min=0.0, max=2.0, value=0.5, continuous_update=False)
    theta_slider = widgets.FloatSlider(description='θ', min=0.0, max=360.0, continuous_update=False)
    # theta_slider = widgets.FloatSlider(description='θ', min=0.0, max=360.0,)
    a_slider = widgets.SelectionSlider(options=env.actions, description="a", continuous_update=False)
    ui = widgets.HBox([widgets.VBox([x_slider, y_slider]),
                       widgets.VBox([theta_slider, a_slider])])

    def show_function(f, x, y, theta, a):
        xmax = env.map.tiles.shape[1]
        ymax = env.map.tiles.shape[0]
        
        gs = gridspec.GridSpec(2, 1, height_ratios=[3,1])
        fig = plt.figure()
        fig.set_size_inches(10, 8)
        ax1 = plt.subplot(gs[0])
        # ax1 = fig.gca()
        ax1.set_aspect(1)
        env.map.plot(ax1)
        car = Rectangle((x, y), 0.1, 0.07, theta,
                        zorder=10, color="orange")
        ax1.add_patch(car)
        ax1.set_xlim(0.0, xmax)
        ax1.set_ylim(0.0, ymax)
        
        mappable = xy_contourf(f, theta, a, xmax, ymax, ax1)        
        fig.colorbar(mappable)
        
        ax2 = plt.subplot(gs[1])
        vals = [f(x, y, theta, aa) for aa in env.actions]
        cmap = mappable.get_cmap()
        norm = mappable.norm
        colors = [cmap(norm(v)) for v in vals]
        
        ax2.bar(range(len(env.actions)),
                vals,
                color = colors,
                tick_label=env.actions,
               )
        
    interfunc = lambda x, y, theta, a: show_function(f, x, y, theta, a)

    out = widgets.interactive_output(interfunc,
                                     {'x':x_slider,
                                      'y':y_slider,
                                      'theta':theta_slider,
                                      'a':a_slider
                                     })
    display(ui, out)
