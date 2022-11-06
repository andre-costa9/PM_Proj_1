# task 2

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')
# print(df.head())

robot_x = []
robot_y = []
robot_theta = []

for index in range(len(df.index)):

    row = df.iloc[index]

    # Get initial values
    if index == 0:
        robot_x.append(row['x'])
        robot_y.append(row['y'])
        robot_theta.append(row['theta'])

    else:
        prev_row = df.iloc[index-1]

        robot_x.append(robot_x[index-1] + prev_row['dx'])
        robot_y.append(robot_y[index-1] + prev_row['dy'])
        robot_theta.append(math.atan(prev_row['dy']/prev_row['dx']))

approx = pd.DataFrame(
    list(zip(robot_x, robot_y, robot_theta)), columns=['x', 'y', 'theta'])
print(approx)

# walls_x = np.array([])
# walls_y = np.array([])

# for n in range(len(df.index)):
#     for i in range(61):
#         aux = -30+i
#         angle = math.radians(df.loc[n, "theta"]-aux)
#         walls_x = np.append(
#             walls_x, df.loc[n, "x"]+math.cos(angle)*df.loc[n, str(aux)])
#         walls_y = np.append(
#             walls_y, df.loc[n, "y"]+math.sin(angle)*df.loc[n, str(aux)])


# # Não sei pq q a label não está a ficar
plt.figure()
plt.plot(approx.loc[:, "x"], approx.loc[:, "y"], 'r', label='oddometry')
plt.plot(df.loc[:, "x"], df.loc[:, "y"], 'g', label='real')
plt.show()
# plt.scatter(walls_x, walls_y)
