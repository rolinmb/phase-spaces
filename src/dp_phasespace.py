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
theta1_vals, omega1_vals = sol.y[0], sol.y[1]
theta2_vals, omega2_vals = sol.y[2], sol.y[3]

# Normalize angles to [-pi, pi] for cleaner phase space plots
theta1_vals = (theta1_vals + np.pi) % (2 * np.pi) - np.pi
theta2_vals = (theta2_vals + np.pi) % (2 * np.pi) - np.pi

# Set up the plot
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
for ax in axs:
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-10, 10)
    ax.set_xlabel("Theta")
    ax.set_ylabel("Omega")
axs[0].set_title("Phase Space: Pendulum 1 (\u03B8\u2081, \u03C9\u2081)")
axs[1].set_title("Phase Space: Pendulum 2 (\u03B8\u2082, \u03C9\u2082)")

# Scatter plots for phase space
scatter1 = axs[0].scatter([], [], s=1, c=[], cmap='viridis', vmin=-np.pi, vmax=np.pi)
scatter2 = axs[1].scatter([], [], s=1, c=[], cmap='viridis', vmin=-np.pi, vmax=np.pi)
cbar = plt.colorbar(scatter1, ax=axs, location='right', pad=0.1)
cbar.set_label("Initial Theta")

# Animation initialization
def init():
    scatter1.set_offsets(np.empty((0, 2)))  # Set empty 2D array
    scatter2.set_offsets(np.empty((0, 2)))
    scatter1.set_array([])
    scatter2.set_array([])
    return scatter1, scatter2

# Animation update function
def animate(i):
    theta1_points = theta1_vals[:i]
    omega1_points = omega1_vals[:i]
    theta2_points = theta2_vals[:i]
    omega2_points = omega2_vals[:i]

    scatter1.set_offsets(np.c_[theta1_points, omega1_points])
    scatter1.set_array(theta1_points)
    
    scatter2.set_offsets(np.c_[theta2_points, omega2_points])
    scatter2.set_array(theta2_points)

    return scatter1, scatter2

if __name__ == '__main__':
    ani = FuncAnimation(fig, animate, frames=len(t_span), init_func=init, interval=1000 / 60, blit=False)
    plt.show()
