import numpy as np 
import matplotlib.pyplot as plt

"""[[9.8708636e-01 6.7503056e-06 8.3038548e-11 8.1807224e-07 1.0659893e-06
  6.3416767e-03 1.4644398e-07 6.3858549e-03 6.5349928e-07 1.7679541e-04]]
(1, 10)"""
confidence = np.arange(0,50).reshape(5,10)
print(confidence)
print(confidence[0])

X = np.arange(confidence.shape[1])+1

fig, ax = plt.subplots()
index = np.arange(X.shape[0])
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, confidence[1], bar_width, alpha=opacity, color='royalblue', label='Predicted - Epoch 50')
rects1 = plt.bar(index, confidence[0], bar_width, alpha=opacity, color='cornflowerblue', label='Correct - Epoch 50')

rects2 = plt.bar(index + bar_width, confidence[3], bar_width, alpha=opacity, color='g', label='Predicted - Epoch 100')
rects2 = plt.bar(index + bar_width, confidence[2], bar_width, alpha=opacity, color='m', label='Correct - Epoch 100')

plt.xlabel('Class')
plt.ylabel('Number of samples')
plt.title('Confidence distribution')
plt.legend()

plt.tight_layout()
plt.show()