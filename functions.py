import numpy as np

def random3d():
    x = np.random.uniform(-1,1,(3))
    while np.linalg.norm(x) > 1.0 or np.linalg.norm(x) < 1e-6:
        x = np.random.uniform(-1,1,(3))
    return x / np.linalg.norm(x)

def random2d():
    x = np.random.uniform(-1,1,(2))
    while np.linalg.norm(x) > 1.0 or np.linalg.norm(x) < 1e-6:
        x = np.random.uniform(-1,1,(2))
    return x / np.linalg.norm(x)

def get_basis(x): # x cross b1 = b2 / x:forward b1:up b2:right
    b1 = random3d()
    while (b1 == x).all():
        b1 = random3d()
    b1 = np.cross(x,b1)
    b1 = b1 / np.linalg.norm(b1)
    b2 = np.cross(x,b1)
    return b1,b2

def project(x,y): #projects x vector onto y plane
    y_normalized = y / np.linalg.norm(y)
    y_component = y_normalized * np.dot(y_normalized, x)
    return x - y_component

def normalize(x):
    v = np.linalg.norm(x)
    if v == 0:
        return x
    else:
        return x / v

if __name__ == "__main__":

    x = random3d() * 2.5
    y = random3d()

    z = project(x,y)

    print(np.linalg.norm(z))
    print(np.dot(y,z))