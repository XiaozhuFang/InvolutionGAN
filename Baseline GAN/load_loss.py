import os
import json
import matplotlib.pyplot as plt
import numpy as np



file = os.path.join("..", "Baseline GAN", "logs", "celeb_igan_0.3",'loss.log')
with open(file, 'r') as f:
    s = f.readline()
s = s.strip()

log = json.loads(s)



# print(len(log['lossG']))
# print(len(log['lossD']))
lossg,=plt.plot(np.array(log['lossg']), color="red")
lossd,=plt.plot(np.array(log['lossd']), color="blue")
gp,=plt.plot(np.array(log['gp']), color="yellow")
# for i in range(30, 1000):
#     if log['lossD'][i] > -5:
#         log['lossD'][i] = 0.6*log['lossD'][i]+0.4*log['lossD'][i-1]
plt.xlabel('Training Epochs')
plt.ylabel('Loss estiamte')
plt.legend(['loss G, loss D, penalty'])
# plt.ylim([-200000, -10])

# plt.plot(np.array(log['lossD'])[150:], color='r')
# plt.plot(log['lossG'], color='g')
# plt.yscale('symlog')
plt.plot()
plt.show()