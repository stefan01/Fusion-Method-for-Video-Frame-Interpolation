import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 20, 0.1)
t0 = np.sin(x)
t1 = np.sin(x - np.pi/3)
t2 = np.sin(x - np.pi/6)
t3 = np.sin(x - np.pi/6 + np.pi)

plt.figure(figsize=(10, 2))
plt.plot(x, t0, label='t0 = sin(x)')
plt.plot(x, t1, label='t1 = sin(x - pi/3)')
plt.plot(x, t2, label='t1 = sin(x - pi/3)')
plt.plot(x, t3, '--')
#plt.legend()
plt.show()