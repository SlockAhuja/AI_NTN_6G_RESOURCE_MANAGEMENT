import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

R = 6371
SAT_ALT = 1200
SAT_RADIUS = R + SAT_ALT

fig = plt.figure(figsize=(11,9))
ax = fig.add_subplot(111, projection='3d')

ax.set_xlim([-7500,7500])
ax.set_ylim([-7500,7500])
ax.set_zlim([-7500,7500])

ax.set_title("Global NTN Communication Simulation")

# --------------------------
# EARTH
# --------------------------

u = np.linspace(0,2*np.pi,60)
v = np.linspace(0,np.pi,60)

x = R*np.outer(np.cos(u),np.sin(v))
y = R*np.outer(np.sin(u),np.sin(v))
z = R*np.outer(np.ones(np.size(u)),np.cos(v))

ax.plot_surface(
    x,
    y,
    z,
    color='blue',
    alpha=0.05,      # more transparent
    edgecolor='none'
)

# --------------------------
# GLOBAL USERS
# --------------------------

num_users = 100

users = []

for _ in range(num_users):

    lat = np.random.uniform(-np.pi/2,np.pi/2)
    lon = np.random.uniform(0,2*np.pi)

    x = R*np.cos(lat)*np.cos(lon)
    y = R*np.cos(lat)*np.sin(lon)
    z = R*np.sin(lat)

    users.append([x,y,z])

users = np.array(users)

user_scatter = ax.scatter(
    users[:,0],
    users[:,1],
    users[:,2],
    color='cyan',
    s=25,
    label="Global Users"
)

# --------------------------
# SATELLITES
# --------------------------

num_sats = 20
sat_angles = np.linspace(0,2*np.pi,num_sats)

sat_positions = np.zeros((num_sats,3))

sat_scatter = ax.scatter([],[],[],color='yellow',s=80,label="Satellites")

# --------------------------
# CONNECTION LINES
# --------------------------

lines = []

for _ in range(num_users):
    line, = ax.plot([],[],[],linewidth=1)
    lines.append(line)

# --------------------------
# UPDATE FUNCTION
# --------------------------

def update(frame):

    global sat_positions

    for i in range(num_sats):

        angle = sat_angles[i] + frame*0.02

        x = SAT_RADIUS*np.cos(angle)
        y = SAT_RADIUS*np.sin(angle)
        z = SAT_RADIUS*0.3*np.sin(angle*0.5)

        sat_positions[i] = [x,y,z]

    sat_scatter._offsets3d = (
        sat_positions[:,0],
        sat_positions[:,1],
        sat_positions[:,2]
    )

    for i,user in enumerate(users):

        distances = np.linalg.norm(sat_positions-user,axis=1)

        nearest = np.argmin(distances)

        sat = sat_positions[nearest]

        color = plt.cm.jet(nearest/num_sats)

        lines[i].set_data(
            [user[0],sat[0]],
            [user[1],sat[1]]
        )

        lines[i].set_3d_properties(
            [user[2],sat[2]]
        )

        lines[i].set_color(color)

    return lines

ani = FuncAnimation(
    fig,
    update,
    frames=600,
    interval=40
)

plt.legend()
plt.show()