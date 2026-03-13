import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ----------------------------
# Setup Figure
# ----------------------------
fig, ax = plt.subplots(figsize=(12,6))
ax.set_xlim(0,20)
ax.set_ylim(0,12)
ax.set_facecolor("black")
ax.set_title("Multi-User Global NTN Communication",color="white")

# ----------------------------
# Users in India
# ----------------------------
india_users = np.array([
    [2,1],
    [3,1],
    [4,1],
    [2.5,1],
    [3.5,1]
])

# ----------------------------
# Users in USA
# ----------------------------
usa_users = np.array([
    [16,1],
    [17,1],
    [18,1],
    [16.5,1],
    [17.5,1]
])

ax.scatter(india_users[:,0],india_users[:,1],s=120,color="cyan",label="India Users")
ax.scatter(usa_users[:,0],usa_users[:,1],s=120,color="magenta",label="USA Users")

# ----------------------------
# Satellites
# ----------------------------
sat_positions = [
    [0,9],
    [7,10],
    [14,9]
]

sat_markers=[]

for pos in sat_positions:
    marker, = ax.plot([],[], 'o',markersize=15,color="yellow")
    sat_markers.append(marker)

# Lines for users
india_lines=[]
usa_lines=[]

for _ in india_users:
    l, = ax.plot([],[],linewidth=2,color="red")
    india_lines.append(l)

for _ in usa_users:
    l, = ax.plot([],[],linewidth=2,color="red")
    usa_lines.append(l)

# inter satellite link
isl_line, = ax.plot([],[],linewidth=3,color="green")

ax.legend()

# ----------------------------
# Helper Function
# ----------------------------
def nearest_sat(user,satellites):

    d=[np.sqrt((user[0]-s[0])**2+(user[1]-s[1])**2) for s in satellites]

    return np.argmin(d)

# ----------------------------
# Update
# ----------------------------
def update(frame):

    # move satellites
    for i in range(len(sat_positions)):
        sat_positions[i][0]+=0.05

        if sat_positions[i][0]>20:
            sat_positions[i][0]=0

    # update satellite markers
    for i,m in enumerate(sat_markers):
        m.set_data([sat_positions[i][0]],[sat_positions[i][1]])

    # India users → satellites
    india_sat_indices=[]

    for i,user in enumerate(india_users):

        sat_i=nearest_sat(user,sat_positions)

        india_sat_indices.append(sat_i)

        sat=sat_positions[sat_i]

        india_lines[i].set_data(
            [user[0],sat[0]],
            [user[1],sat[1]]
        )

    # USA users → satellites
    usa_sat_indices=[]

    for i,user in enumerate(usa_users):

        sat_i=nearest_sat(user,sat_positions)

        usa_sat_indices.append(sat_i)

        sat=sat_positions[sat_i]

        usa_lines[i].set_data(
            [user[0],sat[0]],
            [user[1],sat[1]]
        )

    # choose first connection pair for inter satellite routing
    sat_a=sat_positions[india_sat_indices[0]]
    sat_b=sat_positions[usa_sat_indices[0]]

    isl_line.set_data(
        [sat_a[0],sat_b[0]],
        [sat_a[1],sat_b[1]]
    )

    return sat_markers + india_lines + usa_lines + [isl_line]


ani=animation.FuncAnimation(
    fig,
    update,
    frames=600,
    interval=30,
    blit=False
)

plt.show()