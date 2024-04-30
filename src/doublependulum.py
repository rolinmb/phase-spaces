import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

G = 9.81
M1, M2 = 1, 1 # pendulum masses
L1, L2 = 1, 1 # pendulum lengths

def derivs(t, state):
    theta1, theta1_dot, theta2, theta2_dot = state
    theta1_ddot = (-G * (2 * M1 + M2) * np.sin(theta1) - M2 * G * np.sin(theta1 - 2 * theta2) 
        - 2 * np.sin(theta1 - theta2) * M2 * (theta2_dot**2 * L2 + theta1_dot**2 * L1 * np.cos(theta1 - theta2))) / (L1 * (2 * M1 + M2 - M2 * np.cos(2 * theta1 - 2 * theta2)))
    theta2_ddot = (2 * np.sin(theta1 - theta2) * (theta1_dot**2 * L1 * (M1 + M2) * np.cos(theta1) + theta2_dot**2 * L2 * M2 * np.cos(theta1 - theta2))) / (L2 * (2 * M1 + M2 - M2 *         np.cos(2 * theta1 - 2 * theta2)))
    return [theta1_dot, theta1_ddot, theta2_dot, theta2_ddot]

init_state = [np.pi / 2, 0, np.pi / 2, 0]
t_span = np.linspace(0, 20, 1000)
sol = solve_ivp(derivs, [t_span[0], t_span[-1]], init_state, t_eval=t_span)
theta1_vals = sol.y[0]
theta2_vals = sol.y[2]
x1 = L1 * np.sin(theta1_vals)
y1 = -L1 * np.cos(theta1_vals)
x2 = x1 + L2 * np.sin(theta2_vals)
y2 = y1 - L2 * np.cos(theta2_vals)
fig, ax = plt.subplots()
ax.set_aspect("equal")
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
line, = ax.plot([], [], lw=2)

def init():
    line.set_data([], [])
    return line,

def animate(i):
    line.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])
    return line,

ani = FuncAnimation(fig, animate, frames=len(sol.t), init_func=init, blit=True)

if __name__ == "__main__":
    plt.show()
