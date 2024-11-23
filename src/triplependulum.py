import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
G = 9.81  # Gravitational acceleration (m/s^2)
L1, L2, L3 = 1.0, 1.0, 1.0  # Lengths of the pendulums
M1, M2, M3 = 1.0, 1.0, 1.0  # Masses of the pendulums
TIME_SPAN = 20  # Total simulation time (s)
FPS = 60  # Frames per second for animation

# Initial conditions: [theta1, theta2, theta3, omega1, omega2, omega3]
y0 = [np.pi / 2, np.pi / 4, np.pi / 6, 0, 0, 0]  # Angles in radians, angular velocities

# Equations of motion for the triple pendulum
def equations(t, y):
    theta1, theta2, theta3, omega1, omega2, omega3 = y
    cos12 = np.cos(theta1 - theta2)
    sin12 = np.sin(theta1 - theta2)
    cos23 = np.cos(theta2 - theta3)
    sin23 = np.sin(theta2 - theta3)
    cos13 = np.cos(theta1 - theta3)
    sin13 = np.sin(theta1 - theta3)
    # Equations derived from Lagrangian mechanics
    denom1 = M1 + M2 * sin12**2
    denom2 = M2 + M3 * sin23**2

    d_omega1 = (-G * (2 * M1 + M2 + M3) * np.sin(theta1)
                - M2 * G * np.sin(theta1 - 2 * theta2)
                - 2 * sin12 * M2 * (omega2**2 * L2 + omega1**2 * L1 * cos12)) / (L1 * denom1)

    d_omega2 = (2 * sin12 * (omega1**2 * L1 * (M1 + M2 + M3) 
                + G * (M1 + M2 + M3) * np.cos(theta1)
                + omega2**2 * L2 * M3 * cos12)) / (L2 * denom1)

    d_omega3 = (2 * sin23 * (omega2**2 * L2 * (M2 + M3) 
                + G * (M2 + M3) * np.cos(theta2)
                + omega3**2 * L3 * M3 * cos23)) / (L3 * denom2)

    return [omega1, omega2, omega3, d_omega1, d_omega2, d_omega3]

# Time grid for integration
t_eval = np.linspace(0, TIME_SPAN, TIME_SPAN * FPS)
# Solve the differential equations
solution = solve_ivp(equations, [0, TIME_SPAN], y0, t_eval=t_eval, method="RK45")
# Extract angles from solution
theta1, theta2, theta3 = solution.y[0], solution.y[1], solution.y[2]
# Calculate positions of the pendulum masses
x1 = L1 * np.sin(theta1)
y1 = -L1 * np.cos(theta1)
x2 = x1 + L2 * np.sin(theta2)
y2 = y1 - L2 * np.cos(theta2)
x3 = x2 + L3 * np.sin(theta3)
y3 = y2 - L3 * np.cos(theta3)

# Animation
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_aspect('equal')
line, = ax.plot([], [], 'o-', lw=2, color="blue")
trail, = ax.plot([], [], '-', lw=1, color="orange", alpha=0.5)

# Initialize animation
def init():
    line.set_data([], [])
    trail.set_data([], [])
    return line, trail

# Update animation
trail_length = 100  # Number of points in the trail
trail_x, trail_y = [], []

def update(frame):
    global trail_x, trail_y
    line.set_data([0, x1[frame], x2[frame], x3[frame]],
                  [0, y1[frame], y2[frame], y3[frame]])
    trail_x.append(x3[frame])
    trail_y.append(y3[frame])
    if len(trail_x) > trail_length:
        trail_x = trail_x[-trail_length:]
        trail_y = trail_y[-trail_length:]
    trail.set_data(trail_x, trail_y)
    return line, trail

if __name__ == '__main__':
    ani = FuncAnimation(fig, update, frames=len(t_eval), init_func=init, interval=1000 / FPS, blit=True)
    plt.show()