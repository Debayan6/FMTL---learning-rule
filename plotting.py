import numpy as np
import matplotlib.pyplot as plt


learning_rule_x = "FMTL"
learning_rule_y = "SELFISH"
"""
a=[]
b=[]
for i in range(50,450,50):
    x = np.load("p_1_donation_"+str(learning_rule_x)+"_"+str(learning_rule_y)+str(i)+".npy")
    y = np.load("q_1_donation_"+str(learning_rule_x)+"_"+str(learning_rule_y)+str(i)+".npy")
    a.append(x)
    b.append(y)

x = np.load("p_1_donation_"+str(learning_rule_x)+"_"+str(learning_rule_y)+str(600)+".npy")
y = np.load("q_1_donation_"+str(learning_rule_x)+"_"+str(learning_rule_y)+str(600)+".npy")
a.append(x)
b.append(y)

for i in range(len(a)-1):
    plt.scatter(a[i],b[i],s=10,label =str((i+1)*50))
plt.scatter(a[len(a)-1],b[len(a)-1],s=20,label =str(600))
plt.legend()
plt.savefig("plot1.jpg",dpi=500)
"""
a = np.load("donation3"+str(learning_rule_x)+"_"+str(learning_rule_y)+str(800)+".npy")
print(a[:,1])
#plt.scatter(a[:,0],a[:,1])

#ax = plt.axes(projection = '2d')
plt.xlim(-1.2,2.2)
plt.ylim(-1.2,2.2)
#plt.rc('axes', titlesize=25)
plt.scatter(a[:,1],a[:,0],c = a[:,2],s = 20,cmap = 'plasma',label = "FMTL vs SELFISH")#,linewidth = 0.5)
plt.plot([0,-1],[0,2],"b-",linewidth = 0.6)
plt.plot([0,2],[0,-1],"b-",linewidth = 0.6)
plt.plot([-1,1],[2,1],"b-",linewidth = 0.6)
plt.plot([2,1],[-1,1],"b-",linewidth = 0.6)
plt.plot([0,1],[0,1],"b--",linewidth = 0.6)
plt.colorbar(label = "update steps")
plt.xlabel(r"$\pi_x$")
plt.ylabel(r"$\pi_y$")
plt.legend()
plt.grid(True,linestyle = "--",linewidth = "0.2")
plt.savefig("FPlot.jpg",dpi = 800)
plt.show()
