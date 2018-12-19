import matplotlib.pyplot as plt
import matplotlib.ticker

fig = plt.figure()
ax = fig.add_subplot(111)

plt.scatter((1, 10, 5, 18, 12), (9, 2, 8, 4, 10))
plt.ylim(-35, 20)
plt.xlim(-25, 35)
ax2 = ax.twinx().twiny()

plt.ylim(-70, 40)
plt.xlim(-50, 70)

l = ax.get_ylim()
l2 = ax2.get_ylim()
f = lambda x : l2[0]+(x-l[0])/(l[1]-l[0])*(l2[1]-l2[0])
ticks = f(ax.get_yticks())
print(ticks<0)
ax2.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks[ticks<0]))

plt.show()

