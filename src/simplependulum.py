import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# Constants
G = 9.81  # Gravitational acceleration (m/s^2)
L = 1.0   # Length of the pendulum (m)
M = 1.0   # Mass of the pendulum bob (kg)

# Differential equation for the simple pendulum
def derivs(t, state):
    theta, omega = state
    theta_dot = omega
    omega_dot = -(G / L) * np.sin(theta)
    return [theta_dot, omega_dot]

# Initial conditions: [theta, omega]
init_state = [np.pi / 4, 0]  # Pendulum starts at 45 degrees with zero angular velocity
t_span = np.linspace(0, 10, 500)  # Time span for the simulation
# Solve the differential equation
sol = solve_ivp(derivs, [t_span[0], t_span[-1]], init_state, t_eval=t_span, method="RK45")
theta_vals = sol.y[0]
# Compute the position of the pendulum
x_vals = L * np.sin(theta_vals)
y_vals = -L * np.cos(theta_vals)
# Set up the plot
fig, ax = plt.subplots(figsize=(5, 5))
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
line, = ax.plot([], [], 'o-', lw=2, color='blue')  # Pendulum rod and bob

# Animation functions
def init():
    line.set_data([], [])
    return line,

def animate(i):
    line.set_data([0, x_vals[i]], [0, y_vals[i]])
    return line,

if __name__ == '__main__':
    ani = FuncAnimation(fig, animate, frames=len(t_span), init_func=init, interval=1000 / 60, blit=True)
    plt.show()
