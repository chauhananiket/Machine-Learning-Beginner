import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
import tensorflow as tf
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10, 6)

#Let's define the independent variable:
X = np.arange(0.0, 5.0, 0.1)
X
##You can adjust the slope and intercept to verify the changes in the graph
a = 1
b = 0

Y = a*X+b

plt.plot(X, Y) 
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()

df = pd.read_csv(r"https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv")

train_x = np.asanyarray(df[['ENGINESIZE']])
train_y = np.asanyarray(df[['CO2EMISSIONS']])

a = tf.Variable(1139.0)
b = tf.Variable(11125.0)
y = a * train_x + b

#To find value of our loss, we use <b>tf.reduce_mean()</b>. This function finds the mean of a multidimensional tensor, and the result can have a different dimension.
loss = tf.reduce_mean(tf.square(y - train_y))

optimizer = tf.train.GradientDescentOptimizer(0.05)

train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

loss_values = []
train_data = []
for step in range(1000):
    _, loss_val, a_val, b_val = sess.run([train, loss, a, b])
    loss_values.append(loss_val)
    if step % 5 == 0:
        print(step, loss_val, a_val, b_val)
        train_data.append([a_val, b_val])
        
plt.plot(loss_values, 'ro')      
plt.plot(train_data)        