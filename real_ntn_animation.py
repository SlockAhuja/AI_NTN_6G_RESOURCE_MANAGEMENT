import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --------------------------
# Simulation Parameters
# --------------------------
PATH_LOSS_EXPONENT = 2.2
NOISE_POWER = 0.5

# --------------------------
# Setup Figure
# --------------------------
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_title("Satellite ↔ Ground NTN Communication")
ax.set_facecolor("black")

# --------------------------
# Ground Towers
# --------------------------
ground_positions = np.array([
    [2, 1],
    [4, 1],
    [6, 1],
    [8, 1],
    [5, 1]
])

ground = ax.scatter(
    ground_positions[:, 0],
    ground_positions[:, 1],
    s=200,
    color="cyan",
    label="Ground Towers"
)

# --------------------------
# Satellite
# --------------------------
satellite, = ax.plot([], [], 'o', markersize=15, color="yellow", label="Satellite")

# Communication lines
lines = []
for _ in range(len(ground_positions)):
    line, = ax.plot([], [], linewidth=2)
    lines.append(line)

ax.legend()

# --------------------------
# Update Function
# --------------------------
def update(frame):

    sat_x = (frame * 0.05) % 10
    sat_y = 8

    satellite.set_data([sat_x], [sat_y])

    for i, tower in enumerate(ground_positions):

        dx = tower[0] - sat_x
        dy = tower[1] - sat_y
        distance = np.sqrt(dx**2 + dy**2)

        channel_gain = 1 / (distance ** PATH_LOSS_EXPONENT)
        snr = channel_gain / NOISE_POWER

        throughput = np.log2(1 + snr)

        # Beam thickness based on throughput
        beam_width = 1 + 5 * throughput

        lines[i].set_data([tower[0], sat_x], [tower[1], sat_y])
        lines[i].set_linewidth(beam_width)

        # Color intensity based on signal strength
        intensity = min(throughput / 3, 1.0)
        lines[i].set_color((1, intensity, 0))

    return [satellite] + lines


ani = animation.FuncAnimation(
    fig,
    update,
    frames=400,
    interval=30,
    blit=True
)

plt.show()