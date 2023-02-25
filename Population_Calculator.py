import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

##part 1
###for population 1
def f1(params,t):
    initial_poulation=params[0]
    growth_rate=0.08;max_population=10000;
    p=initial_poulation
    dpdt = growth_rate*(1-p/max_population)*p
    return dpdt
t = np.linspace(0,10) ## t from 0 to 10
initial_poulation=3000;
params1=[initial_poulation]
s1= odeint(f1,params1,t)

##part2
###for population 2
def f(params,t):
    initial_poulation=params[0]
    growth_rate=0.17;max_population=10000;
    p=initial_poulation
    dpdt = growth_rate*(1-p/max_population)*p
    return dpdt
initial_poulation=3000;
params2=[initial_poulation]
s2= odeint(f,params2,t)

plt.plot(t,s1[:,0],'r-', linewidth=2.0)
plt.plot(t,s2[:,0],'b-', linewidth=2.0)
plt.xlabel("time")
plt.ylabel("Population")
plt.legend(["population1 "," population2"])
plt.show()

