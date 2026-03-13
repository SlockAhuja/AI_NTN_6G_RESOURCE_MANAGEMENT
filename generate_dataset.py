import pandas as pd
import numpy as np

num_samples = 12000

data = {
    "user_lat": np.random.uniform(-90,90,num_samples),
    "user_lon": np.random.uniform(-180,180,num_samples),
    "sat_lat": np.random.uniform(-90,90,num_samples),
    "sat_lon": np.random.uniform(-180,180,num_samples),
    "sat_altitude": np.random.uniform(500,1200,num_samples),
    "channel_gain": np.random.uniform(0,1,num_samples),
    "sinr": np.random.uniform(0,30,num_samples),
    "bandwidth": np.random.uniform(5,100,num_samples),
    "power": np.random.uniform(0.1,1,num_samples),
    "delay": np.random.uniform(10,200,num_samples),
    "packet_loss": np.random.uniform(0,0.1,num_samples)
}

df = pd.DataFrame(data)
df.to_csv("ntn_dataset.csv", index=False)

print("Dataset generated:", len(df))
