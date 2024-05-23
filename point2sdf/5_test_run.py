import numpy as np
import os



sigmas = np.linspace(0.2, 1.5, 6)

print(sigmas)

for sigma in sigmas:
    si = float(sigma)
    ok = os.system(f'python train.py -e {si}')
    if ok != 0:
        print('error on ', si)
        break