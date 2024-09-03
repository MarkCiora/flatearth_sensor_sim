import numpy as np

floattype = np.float64

ER = 6371000
EM = 5.9722 * (10**24)
G = 6.6743 * (10**-11)

ER_km = 6371
EM_km = 5.9722 * (10**24)
G_km = 6.6743 * (10**-20)

G_EM_km = EM_km * G_km

dt = .5

if __name__ == "__main__":
    import numpy as np
    altitude = 1000
    speed = np.sqrt(G_EM_km / (ER_km + altitude))
    print("speed:", speed)

    period = 2*np.pi*(ER_km + altitude) / speed
    print("period:", period)

    print(3600*24)
