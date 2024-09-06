import numpy as np
import torch

import globals
from globals import floattype
import functions


class Sensor:
    ppdt = 5
    def __init__(self):
        self.p = np.zeros((3), dtype=floattype)
        self.v = np.zeros((3), dtype=floattype)
        self.F = np.zeros((3), dtype=floattype)#FxU=R
        self.U = np.zeros((3), dtype=floattype)
        self.R = np.zeros((3), dtype=floattype)
        self.F[2] = -1
        b1, b2 = functions.get_basis(self.F)
        self.U = b1
        self.R = b2
        self.fov = np.array([4 * np.pi / 180, 4 * np.pi]).astype(floattype)
        self.slew_rate = 6.0 * np.pi / 180
        self.slew_observe_cap = self.slew_rate * 0.5

    def propagate(self, target_pos):
        dt = globals.dt / Target.ppdt
        self.p += self.v * dt
        d = functions.normalize(target_pos - self.p)
        axis = functions.normalize(np.cross(self.F, d))
        for i in range(Target.ppdt):
            angle_to_go = np.arccos(min(np.dot(self.F, d), 1.0))
            angle = min(self.slew_rate * dt, angle_to_go)
            if angle <= self.slew_observe_cap * dt:
                can_observe = True
                self.F = d
            else:
                can_observe = False
                turn_dir = functions.normalize(np.cross(axis, self.F))
                self.F += turn_dir * angle
            self.U = functions.normalize(functions.project(self.U, self.F))
            self.R = np.cross(self.F, self.U)
        return can_observe

    def check_in_fov(self, target_pos):
        d = target_pos - self.p
        d = d / np.linalg.norm(d)
        proj_b1_plane = functions.project(d, self.U)
        proj_b2_plane = functions.project(d, self.R)
        angle1 = np.arccos(min(np.dot(proj_b1_plane, self.F), 1.0))
        angle2 = np.arccos(min(np.dot(proj_b2_plane, self.F), 1.0))
        return (angle1 <= self.fov[0] and angle2 <= self.fov[1])
    
    def angle_between(self, target_pos):
        d = functions.normalize(target_pos - self.p)
        return np.arccos(min(np.dot(self.F, d), 1.0))
    
    def copy(self):
        scopy = Sensor()
        scopy.p[:] = self.p
        scopy.v[:] = self.v
        scopy.F[:] = self.F
        scopy.U[:] = self.U
        scopy.R[:] = self.R
        return scopy

class Target:
    ppdt = 2
    Q = np.diag([1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4]).astype(floattype)
    Q = np.array([  [(globals.dt/ppdt)**3/3, 0, 0,  (globals.dt/ppdt)**2/2, 0, 0],\
                    [0, (globals.dt/ppdt)**3/3, 0,  0, (globals.dt/ppdt)**2/2, 0],\
                    [0, 0, (globals.dt/ppdt)**3/3,  0, 0, (globals.dt/ppdt)**2/2],\
                    [(globals.dt/ppdt)**2/2, 0, 0,  (globals.dt/ppdt), 0, 0],\
                    [0, (globals.dt/ppdt)**2/2, 0,  0, (globals.dt/ppdt), 0],\
                    [0, 0, (globals.dt/ppdt)**2/2,  0, 0, (globals.dt/ppdt)]]) * 1e-3
    R = np.diag([1e-3, 1e-3]) ** 2
    def __init__(self):
        self.x = np.zeros((6), dtype=floattype)#first 3 pos, next 3 vel
        self.x_ = np.zeros((6), dtype=floattype)
        self.P = np.zeros((6,6), dtype=floattype)
        self.P[0:3,0:3] += np.eye(3, dtype=floattype) * 1**2
        self.P[3:6,3:6] += np.eye(3, dtype=floattype) * 0.5**2
    
    def propagate(self):
        dt = globals.dt / Target.ppdt
        for i in range(Target.ppdt):
            F = np.eye(6, dtype=floattype)
            F[0:3, 3:6] += np.eye(3) * dt
            self.x = F@self.x
            self.x_ = F@self.x_
            self.P = F@self.P@F.T + Target.Q
    
    def update(self, sensor_pos):
        d = self.x[0:3] - sensor_pos #displacement (vector)
        r = np.linalg.norm(d) #radius (scalar)
        b1, b2 = functions.get_basis(d)
        H = np.zeros((2,6), dtype=floattype)
        H[0,0:3] = b1 * np.sqrt(Target.R[0,0]) * r
        H[1,0:3] = b2 * np.sqrt(Target.R[1,1]) * r
        z = np.random.multivariate_normal(H@self.x, Target.R)
        y = z - H@self.x_
        S = H@self.P@H.T + Target.R
        K = self.P@H.T@np.linalg.pinv(S)
        self.x_ = self.x_ + K@y
        self.P = self.P - K@H@self.P
    
    def predict_update_FI(self, sensor_pos):
        d = self.x[0:3] - sensor_pos #displacement (vector)
        r = np.linalg.norm(d) #radius (scalar)
        b1, b2 = functions.get_basis(d)
        H = np.zeros((2,6), dtype=floattype)
        H[0,0:3] = b1 * np.sqrt(Target.R[0,0]) * r
        H[1,0:3] = b2 * np.sqrt(Target.R[1,1]) * r
        S = H@self.P@H.T + Target.R
        K = self.P@H.T@np.linalg.pinv(S)
        return np.trace(K@H@self.P)
    
    def copy(self):
        tcopy = Target()
        tcopy.x[:] = self.x_ #because algorithm doesnt actually know
        tcopy.x_[:] = self.x_
        tcopy.P[:] = self.P
        return tcopy

target_height_bound = [20, 400]
target_radius_bound = 1500
target_speed_range = [2,2.5]

sensor_height_bound = [1000, 2000]
sensor_radius_bound = 2500

class Simulation:
    def __init__(self, S=3, T=5, sim_duration=100):
        sensors = []
        targets = []
        self.S = S
        self.T = T
        self.time = 0
        self.dt = globals.dt
        self.sim_duration = sim_duration
        for i in range(S):
            angle = i * 2*np.pi / S
            b1 = np.cos(angle)
            b2 = np.sin(angle)
            sensor = Sensor()
            sensor.p[0] = b1 * sensor_radius_bound
            sensor.p[1] = b2 * sensor_radius_bound
            sensor.p[2] = 1500
            sensors.append(sensor)
        for i in range(T):
            target = Target()
            height = np.random.uniform(target_height_bound[0], target_height_bound[1], 1)[0]
            pos01 = functions.random2d() * target_radius_bound
            speed = np.random.uniform(target_speed_range[0], target_speed_range[1], 1)
            dir01 = functions.random2d()
            target.x[0:2] = pos01
            target.x[2] = height
            target.x[3:5] = dir01 * speed
            target.x_[:] = np.random.multivariate_normal(target.x[:], target.P)
            target.x_[3:6] *= 0
            targets.append(target)
        self.sensors = sensors
        self.targets = targets

    def step(self, action):
        targets = self.targets
        sensors = self.sensors
        reward = 0
        for target in targets:
            reward += np.sqrt(np.trace(target.P))
            target.propagate()
        for i in range(self.S):
            if sensors[i].propagate(targets[action[i]].x[0:3]):
                for j in range(self.T):
                    if sensors[i].check_in_fov(targets[j].x[0:3]):
                        # print("update:", np.trace(targets[j].P), end="")
                        targets[j].update(sensors[i].p)
                        # print("--->", np.trace(targets[j].P))
        for target in targets:
            reward -= np.sqrt(np.trace(target.P))
        self.time += self.dt
        return reward

    def get_state_vector(self):
        # x_ for each target
        # P for each target
        # p for each sensor
        # F for each sensor
        x1 = []
        x2 = []
        x3 = []
        x4 = []
        s1 = np.sqrt(3)
        s2 = 6
        
        cov_trace_max = 1e-8
        distances_max = 1e-8
        for i in range(self.T):
            cov_trace_max = max(np.trace(self.targets[i].P), cov_trace_max)
            distances_max = max(np.linalg.norm(self.targets[i].x_), distances_max)
        for i in range(self.S):
            distances_max = max(np.linalg.norm(self.sensors[i].p), distances_max)
        for i in range(self.T):
            x1.append(self.targets[i].x_)
            x2.append(self.targets[i].P)
        for i in range(self.S):
            x3.append(self.sensors[i].p)
            x4.append(self.sensors[i].F)
        s1 /= distances_max
        s2 /= cov_trace_max
        x1 = np.stack(x1).flatten() * s1-0.5
        x2 = np.stack(x2).flatten() * s2-0.5
        x3 = np.stack(x3).flatten() * s1-0.5
        x4 = np.stack(x4).flatten()
        return np.hstack((x1,x2,x3,x4,np.log(s1*1e3),np.log(s2)*2e-1)).ravel()
    
    def get_uncertainty(self):
        s = 0
        for t in self.targets:
            s += np.sqrt(np.trace(t.P))
        return s

    def is_done(self):
        return self.time >= self.sim_duration

if __name__ == "__main__":
    T = 1
    S = 3

    sensors = []
    targets = []

    for i in range(S):
        sensors.append(Sensor())
    for i in range(T):
        targets.append(Target())
        targets[-1].x[2] = 100
        targets[-1].x[3] = 3
        targets[-1].x[4] = 4
        targets[-1].x_[:] = np.random.multivariate_normal(targets[-1].x[:], targets[-1].P)

    print(np.sqrt(np.trace(targets[0].P)))
    for i in range(10):
        targets[0].propagate()
        print(np.sqrt(np.trace(targets[0].P)))

    # # print(targets[0].P, targets[0].x_, targets[0].x)
    # tgt = targets[0]
    # # print(tgt.P, tgt.x, tgt.x_)
    # # print(np.trace(tgt.P))
    # for i in range(10):
    #     print("next iter")
    #     tgt.propagate()
    #     tgt.update(np.array([500,100,1000]).astype(floattype))
    #     print(np.linalg.norm(tgt.x-tgt.x_))

    # print("\n next sat")
    # for i in range(10):
    #     print("next iter")
    #     tgt.propagate()
    #     tgt.update(np.array([-500,-100,1000]).astype(floattype))
    #     print(np.linalg.norm(tgt.x-tgt.x_))