import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import Localvol
from matplotlib import cm
import LVOptimizeCurve

'this is a script for testing purpose'
'''delta=-0.05
kappa=3
gamma=0.3
Xaxis=np.linspace(-2,2,100)
lvar = list(map(lambda x: 0.04 + delta * np.tanh(kappa * x) / kappa + \
                      gamma / 2 * (np.tanh(kappa * x) / kappa) ** 2, Xaxis))
Yaxis = list(map(lambda x: x ** 0.5, lvar))

plt.plot(Xaxis, Yaxis)
plt.title("Fitted")
plt.xlabel('K/S')
plt.show()'''

Yaxis = np.linspace(0.0, 3, 400)  # 400 steps for maturity/365
Xaxis = np.linspace(1000, 7000, 100)  # 100 steps for spot price is just right for graph showing
underlying = Localvol.LocalVol('SPX')

fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y, Z = [], [], []
for i in Xaxis:
    for j in Yaxis:
        X.append(i) # Strike
        Y.append(j) # TTM
        Z.append(underlying.localvolsur(i, j))
        #Z.append(LVOptimizeCurve.IV_cubicspline(Y)) # cannot do this because it it only a function TTM not spot

print(len(X))
print(len(Y))
print(len(Z))

# Normalize the colors based on Z value
norm = plt.Normalize(np.min(Z), np.max(Z))
colors = cm.jet(norm(Z))


plt.title('Local vol surface SPX call')
plt.xlabel('Strike')
plt.ylabel('Maturity')
ax.scatter(X, Y, Z, facecolors=colors, s=0.2).set_facecolor((0,0,0,0))
plt.show()


# def plot3D(X, Y, Z):
#     fig = plt.figure()
#     ax = Axes3D(fig, azim=-29, elev=50)
#     ax.plot(X, Y, Z, 'o')
#     plt.xlabel("expiry")
#     plt.ylabel("strike")
#     plt.show()