import torch
from torch.nn.functional import softmax
import matplotlib.pyplot as plt

random_tensor = torch.randn(20)
dist = softmax(random_tensor)

fig, ax = plt.subplots(1,5, figsize=(12, 3),
                       sharey=True)

for i, T in enumerate([0.5, 0.8, 1, 1.25, 2]):
    scaled_tensor = random_tensor / T
    dist = softmax(scaled_tensor)
    ax[i].bar(range(1, len(dist)+1), dist, width=1)
    
    if T != 1:
        ax[i].set_title(r"$p(\omega|\boldsymbol{x})', Temp =$" + str(T))
    else:
        ax[i].set_title(r"$p(\omega|\boldsymbol{x})$")

plt.savefig('temp_anneal')
