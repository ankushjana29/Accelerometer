import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# read the accelerometer data from a CSV file
df = pd.read_csv("C:/Users/admin/Desktop/internship/accelerometer/accelerometer.csv")

# Select only the accelerometer data (x, y, and z columns)
accel_data = df[['x', 'y', 'z']]

# Plot the accelerometer data
accel_data.plot()
plt.title('Accelerometer Data')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (g)')
plt.show()


# Select only the accelerometer data (x, y, and z columns)
accel_data = df[['x', 'y', 'z']]

# Calculate the magnitude of the acceleration vector for each row
accel_mag = np.sqrt(np.square(accel_data).sum(axis=1))

# Calculate the mean amplitude of the acceleration data
mean_amplitude = accel_mag.mean()

# Print the result
print('Mean Amplitude: {:.2f} g'.format(mean_amplitude))

# Define a function to convert weight configuration IDs to configuration types
def get_config_type(config_id):
    if config_id == 1:
        return 'normal'
    elif config_id == 2:
        return 'perpendicular'
    elif config_id == 3:
        return 'opposite'
    else:
        return 'unknown'

# Apply the function to the wconfid column to create a new configuration type column
df['config_type'] = df['wconfid'].apply(get_config_type)

# Filter the dataframe to show only "normal" configuration type
df_normal = df[df['wconfid'] == 1]

# Display the filtered dataframe
print(df_normal)

# Filter the dataframe to show only "perpendicular" configuration type
df_perp = df[df['wconfid'] == 2]

# Display the filtered dataframe
print(df_perp)

# Filter the dataframe to show only "opposite" configuration type
df_opp = df[df['wconfid'] == 3]

# Display the filtered dataframe
print(df_opp)

