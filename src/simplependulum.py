import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

G = 9.81 # m/s^2
L = 1.0 # length of the pendulum
NPOINTS = 100
NFRAMES = 100
theta_range = np.linspace(-2*np.pi, 2*np.pi, NPOINTS) # initial theta values
p_range = np.linspace(-3, 3, NPOINTS) # initial momentum values
theta_space, p_space = np.meshgrid(theta_range, p_range)

def hamiltonian(theta, p):
    return p**2 / (2 *  L**2) - G ** L * (1 - np.cos(theta))

H = hamiltonian(theta_space, p_space)
fig, ax = plt.subplots()
plt.xlabel("Theta")
plt.ylabel("P")
plt.title("Pendulum Phase Space")
color_map = plt.cm.jet
scatter = ax.scatter(theta_space, p_space, c=H, cmap=color_map)

def update(frame):
    theta_next = theta_range + 0.1 * frame
    p_next = p_range
    h_next = hamiltonian(theta_next, p_next)
    scatter.set_array(h_next.flatten())
    return scatter,

if __name__ == "__main__":
    ani = FuncAnimation(fig, update, frames=range(NFRAMES), interval = 50, blit=True)
    plt.colorbar(scatter, label="System Hamiltonian")
    plt.show()
