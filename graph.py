import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Network Parameters
# ----------------------------

num_users = 20
num_sats = 3
area = 20

B = 10e6                 # bandwidth (10 MHz)
noise = 1e-9
tx_power = 1
c = 3e8                  # speed of light

time_steps = 200

# ----------------------------
# Generate Users
# ----------------------------

users = np.random.rand(num_users,2) * area

# ----------------------------
# Satellites
# ----------------------------

sats = np.array([
    [2,10],
    [10,11],
    [18,10]
], dtype=float)
# ----------------------------
# Metrics Storage
# ----------------------------

throughput_history=[]
delay_history=[]
packet_loss_history=[]
fairness_history=[]

# ----------------------------
# Helper Functions
# ----------------------------

def distance(a,b):

    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


def sinr(user,sat):

    d = distance(user,sat)

    path_loss = 1/(d**2 + 1)

    return (tx_power * path_loss) / noise


def throughput(user,sat):

    s = sinr(user,sat)

    rate = B * np.log2(1+s)

    return rate/1e6


def delay(user,sat):

    d = distance(user,sat)*1000

    return d/c


def nearest_sat(user):

    d=[distance(user,s) for s in sats]

    return np.argmin(d)

# ----------------------------
# Simulation Loop
# ----------------------------

for t in range(time_steps):

    total_throughput=0
    delays=[]
    rates=[]
    packet_loss=0

    # satellite movement
    sats[:,0]+=0.05

    for user in users:

        sat_i = nearest_sat(user)

        sat = sats[sat_i]

        r = throughput(user,sat)

        rates.append(r)

        total_throughput+=r

        d = delay(user,sat)

        delays.append(d)

        # simple packet loss model
        if sinr(user,sat) < 5:
            packet_loss+=1

    avg_delay = np.mean(delays)

    packet_loss_ratio = packet_loss/num_users

    # fairness index
    rates=np.array(rates)

    fairness = (np.sum(rates)**2)/(num_users*np.sum(rates**2))

    # store metrics
    throughput_history.append(total_throughput)

    delay_history.append(avg_delay)

    packet_loss_history.append(packet_loss_ratio)

    fairness_history.append(fairness)

# ----------------------------
# Plot Results
# ----------------------------

plt.figure()
plt.plot(throughput_history)
plt.title("Network Throughput vs Time")
plt.xlabel("Time Step")
plt.ylabel("Throughput (Mbps)")
plt.grid()

plt.figure()
plt.plot(delay_history)
plt.title("Average Delay vs Time")
plt.xlabel("Time Step")
plt.ylabel("Delay (seconds)")
plt.grid()

plt.figure()
plt.plot(packet_loss_history)
plt.title("Packet Loss Ratio vs Time")
plt.xlabel("Time Step")
plt.ylabel("Packet Loss")
plt.grid()

plt.figure()
plt.plot(fairness_history)
plt.title("Jain Fairness Index")
plt.xlabel("Time Step")
plt.ylabel("Fairness")
plt.grid()

plt.show()