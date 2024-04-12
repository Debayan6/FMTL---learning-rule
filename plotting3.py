import numpy as np 
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

file = open("Data_Payoff1.txt","r")

x = file.readlines()
a = []

for i in range(len(x)):
    a.append([float(b) for b in x[i].split("\t")])

a = np.array(a)
#x_new = np.linspace(a[:,0].min(), a[:,0].max(),200)

"""f = interp1d(a[:,0], a[:,1], kind='quadratic')
y_smooth1=f(x_new)
f = interp1d(a[:,0], a[:,1], kind='quadratic')
y_smooth2=f(x_new)
f = interp1d(a[:,0], a[:,1], kind='quadratic')
y_smooth3=f(x_new)
f = interp1d(a[:,0], a[:,1], kind='quadratic')
y_smooth4=f(x_new)
"""

plt.ylim([0,1])
plt.xscale("log")
plt.plot([60,60],[0,1],"-",label = r"x = 60",color = "grey",linewidth = 1.1)
plt.plot([19,19],[0,1],"-",label = r"x = 19",color = "grey",linewidth = 1.1)
plt.plot(a[:,0],a[:,1],"-o",label = r"$a_{FS}$")
plt.plot(a[:,0],a[:,2],"-o",label = r"$a_{SF}$")
plt.plot(a[:,0],a[:,3],"-o",label = r"$a_{SS}$")
plt.plot(a[:,0],a[:,4],"-o",label = r"$a_{FF}$")
plt.xlabel("Number of Rounds")
plt.ylabel("Payoff")
plt.legend()
plt.grid(True)
plt.savefig("Evolution_of_strats.jpg",dpi = 800)
plt.show()

file.close()


