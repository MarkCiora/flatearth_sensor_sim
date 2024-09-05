import numpy as np
import torch

floattype = np.float64

ER = 6371000
EM = 5.9722 * (10**24)
G = 6.6743 * (10**-11)

ER_km = 6371
EM_km = 5.9722 * (10**24)
G_km = 6.6743 * (10**-20)

G_EM_km = EM_km * G_km

dt = 1

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    print(device)
