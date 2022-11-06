from dis import dis
from cv2 import threshold
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
#import cv2

lidar_readings = 61
linReg_TH = 5

df = pd.read_csv('data.csv')
df.head()

angle=[]
dist=[]


#TASK 1
# TODO:
# Guardar a informação num doc csv para mostrar ao prof
# alterar o intervalo em que faço plot das linear regression
# Validar os resultados ao fazer outro plot com as posições relativas à posição do robo para realmente ver se é um canto ou não


#for n in range(len(df.index)-1):
for n in range(1):
  #Para passar a primeira leitura à frente faço n+1
  #como estou a remover uma linha tenho que fazer len(...)-1 
  angle=[]
  dist=[]

  #Recolher leitura lidar
  sweep = df.iloc[n+1,6:]

  #sweep = sweep.rolling(4).mean() 
  
  for i in range(lidar_readings):
    angle.append(-30+i)
    dist.append(sweep[i])

  
  plt.scatter(angle,dist,label='Lidar sweep',zorder=1)

  angle = np.array(angle)
  dist = np.array(dist)

  angle = angle[~np.isnan(angle)]
  dist = dist[~np.isnan(dist)]

  maxpos=np.argmax(dist)

  if 3 <= maxpos <= lidar_readings-3:
    #corner found
    
    #calcular as retas à direita e À esquerda para descobrir o canto
    left_angle = angle[maxpos-linReg_TH:maxpos]
    left_angle = left_angle.reshape((-1, 1))
    left_dist = dist[maxpos-linReg_TH:maxpos]

    left_model = LinearRegression().fit(left_angle, left_dist)
    left_b=left_model.intercept_
    left_m=left_model.coef_

    left_line= left_m * angle[maxpos-linReg_TH:maxpos+linReg_TH] + left_b
    plt.plot(angle[maxpos-linReg_TH:maxpos+linReg_TH], left_line, '-r', label='linReg_left',zorder=2)



    right_angle=angle[maxpos+1:maxpos+linReg_TH]
    right_angle = right_angle.reshape((-1, 1))
    right_dist=dist[maxpos+1:maxpos+linReg_TH]

    right_model = LinearRegression().fit(right_angle, right_dist)
    right_b=right_model.intercept_
    right_m=right_model.coef_

    right_line= right_m * angle[maxpos-linReg_TH:maxpos+linReg_TH] + right_b
    plt.plot(angle[maxpos-linReg_TH:maxpos+linReg_TH], right_line, '-g', label='linReg_right',zorder=3)

    #calcular intercessão

    corner_x = (right_b-left_b)/(left_m-right_m)
    corner_y = left_m * corner_x + left_b


    corner_plt=plt.scatter(corner_x,corner_y,color="black",s=75,marker='X',label='Corner (alpha,dist)= ('+str(corner_x)+','+str(corner_y)+')',zorder=4)
    print(str(n+1)+" : Corner found -> (alpha,dist)= ("+str(corner_x)+","+str(corner_y)+")")


  else:
    print(str(n+1)+" : Corner not found")

  
  plt.legend()
  plt.show()


#TASK 2



robot_x = []
robot_y = []
robot_theta = []

time = np.arange(0,len(df.index),1)

for n in range(len(df.index)):
  
  row = df.iloc[n]

  if n == 0:
    robot_x.append(row['x'])
    robot_y.append(row['y'])
    robot_theta.append(row['theta'])

  else:
    prev_row = df.iloc[n-1]

    robot_x.append(robot_x[n-1] + prev_row['dx'])
    robot_y.append(robot_y[n-1] + prev_row['dy'])
    robot_theta.append(robot_theta[n-1] + prev_row['dtheta'])

plt.subplot(1, 4, 1)
plt.plot(time, df.loc[:,'x'], label = 'X -> Real movement')
plt.plot(time, robot_x, label = 'X -> Movement with error')
plt.legend()

plt.subplot(1, 4, 2)
plt.plot(time, df.loc[:,'y'], label = 'Y -> Real movement')
plt.plot(time, robot_y, label = 'Y -> Movement with error')
plt.legend()

plt.subplot(1, 4, 3)
plt.plot(time, df.loc[:,'theta'], label = 'Theta -> Real movement')
plt.plot(time, robot_theta, label = 'THETA -> Movement with error')
plt.legend()

plt.subplot(1, 4, 4)
plt.plot(df.loc[:,'x'], df.loc[:,'y'], label = 'Robot -> Real movement')
plt.plot(robot_x, robot_y, label = 'Robot -> Movement with error')
plt.legend()

plt.show()



#angle = df.loc[n+1,"theta"]+math.radians(aux)
#walls_x.append(df.loc[n+1,"x"]+math.cos(angle)*sweep[i])
#walls_y.append(df.loc[n+1,"y"]+math.sin(angle)*sweep[i])

  #print(len(dist))
  #print(dist)
  #angle = angle.reshape((-1, 1))
  #
  #model = LinearRegression().fit(angle, dist)
  #r_sq = model.score(angle, dist)
  #print(f"coefficient of determination: {r_sq}")
  #print(f"intercept: {model.intercept_}")
  #print(f"slope: {model.coef_}")