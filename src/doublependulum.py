import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# Constants
G = 9.81  # Gravitational acceleration (m/s^2)
M1, M2 = 1.0, 1.0  # Masses of the pendulums
L1, L2 = 1.0, 1.0  # Lengths of the pendulums

# Differential equations for the double pendulum
def derivs(t, state):
    theta1, omega1, theta2, omega2 = state
    delta_theta = theta1 - theta2

    den1 = (2 * M1 + M2 - M2 * np.cos(2 * delta_theta))
    theta1_ddot = (-G * (2 * M1 + M2) * np.sin(theta1)
                   - M2 * G * np.sin(theta1 - 2 * theta2)
                   - 2 * M2 * np.sin(delta_theta) * (omega2**2 * L2 + omega1**2 * L1 * np.cos(delta_theta))) / (L1 * den1)

    theta2_ddot = (2 * np.sin(delta_theta)
                   * (omega1**2 * L1 * (M1 + M2)
                      + G * (M1 + M2) * np.cos(theta1)
                      + omega2**2 * L2 * M2 * np.cos(delta_theta))) / (L2 * den1)

    return [omega1, theta1_ddot, omega2, theta2_ddot]

# Initial conditions: [theta1, omega1, theta2, omega2]
init_state = [np.pi / 2, 0, np.pi / 2, 0]
t_span = np.linspace(0, 20, 2000)  # Time span with finer resolution for smoother animation

# Solve the system
sol = solve_ivp(derivs, [t_span[0], t_span[-1]], init_state, t_eval=t_span, method="RK45")
theta1_vals, theta2_vals = sol.y[0], sol.y[2]

# Compute the positions of the pendulums
x1 = L1 * np.sin(theta1_vals)
y1 = -L1 * np.cos(theta1_vals)
x2 = x1 + L2 * np.sin(theta2_vals)
y2 = y1 - L2 * np.cos(theta2_vals)

# Set up the plot
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.set_aspect('equal')
line, = ax.plot([], [], 'o-', lw=2, color='blue')
trail, = ax.plot([], [], '-', lw=1, color='orange', alpha=0.5)

# Animation setup
trail_length = 100  # Number of points in the trail
trail_x, trail_y = [], []

def init():
    line.set_data([], [])
    trail.set_data([], [])
    return line, trail

def animate(i):
    global trail_x, trail_y
    line.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])
    trail_x.append(x2[i])
    trail_y.append(y2[i])
    if len(trail_x) > trail_length:
        trail_x = trail_x[-trail_length:]
        trail_y = trail_y[-trail_length:]
    trail.set_data(trail_x, trail_y)
    return line, trail

if __name__ == '__main__':
    ani = FuncAnimation(fig, animate, frames=len(t_span), init_func=init, interval=1000 / 60, blit=True)
    plt.show()