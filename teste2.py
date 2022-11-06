import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

def estimate_coef(x, y):
	# number of observations/points
	n = np.size(x)

	# mean of x and y vector
	m_x = np.mean(x)
	m_y = np.mean(y)

	# calculating cross-deviation and deviation about x
	SS_xy = np.sum(y*x) - n*m_y*m_x
	SS_xx = np.sum(x*x) - n*m_x*m_x

	# calculating regression coefficients
	b_1 = SS_xy / SS_xx
	b_0 = m_y - b_1*m_x

	return (b_0, b_1)

def plot_regression_line(x, y, b):
	# plotting the actual points as scatter plot
	plt.scatter(x, y, color = "m",
			marker = "o", s = 30)

	# predicted response vector
	y_pred = b[0] + b[1]*x

	# plotting the regression line
	plt.plot(x, y_pred, color = "g")

	# putting labels
	plt.xlabel('x')
	plt.ylabel('y')

	# function to show plot
	plt.show()


# observations / data
x = np.empty([61])
y = np.empty([61])

df = pd.read_csv('data.csv')
df.head()
sweep = df.iloc[1,6:]
sweep = sweep.rolling(4).mean() 

for i in range(61):
    #passar pontos para (x,y)
    aux=-30+i
    angle = df.loc[1,"theta"]+math.radians(aux)
	x = np.append(x,df.loc[1,"x"]+math.cos(angle)*sweep[i])
	y = np.append(y,df.loc[1,"y"]+math.sin(angle)*sweep[i])

# estimating coefficients
b = estimate_coef(x, y)
print("Estimated coefficients:\nb_0 = {} nb_1 = {}".format(b[0], b[1]))

# plotting regression line
plot_regression_line(x, y, b)