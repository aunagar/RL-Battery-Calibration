
"""
@ Yuan : YUTIAN@ETHZ.CH

"""

import math
import matplotlib.pyplot as plt
import gym
from gym import spaces, logger
from gym.utils import seeding

import numpy as np


class Battery_Management(gym.Env):

    def __init__(self):
        # Define a structure with fields for each model parameter

        ################################## Basics

        self.sampleTime = 1
        self.qMobile = 7600
        self.xnMax = 0.6  # maximum mole fraction (neg electrode)
        self.xnMin = 0  # minimum mole fraction (neg electrode)
        self.xpMax = 1.0  # maximum mole fraction (pos electrode)
        self.xpMin = 0.4  # minimum mole fraction (pos electrode) -> note xn+xp=1
        self.qMax = self.qMobile / (self.xnMax - self.xnMin) # note qMax = qn+qp
        self.Ro = 0.117215 # for Ohmic drop (current collector resistances plus electrolyte resistance plus solid phase resistances at anode and cathode)

        ################################## Constants of nature
        self.R = 8.3144621 # universal gas constant, J/K/mol
        self.F = 96487 # Faraday's constant, C/mol

        ################################## Li-ion self
        self.alpha = 0.5 # anodic/cathodic electrochemical transfer coefficient
        self.Sn = 0.000437545 # surface area (- electrode)
        self.Sp = 0.00030962  # surface area (+ electrode)
        self.kn = 2120.96 # lumped constant for BV (- electrode)
        self.kp = 248898 # lumped constant for BV (+ electrode)
        self.Vol = 2e-5 # total interior battery volume/2 (for computing concentrations)
        self.VolSFraction = 0.1 # fraction of total volume occupied by surface volume

        # Volumes (total volume is 2*P.Vol), assume volume at each electrode is the
        # same and the surface/bulk split is the same for both electrodes
        self.VolS = self.VolSFraction * self.Vol # surface volume
        self.VolB = self.Vol - self.VolS # bulk volume

        # set up charges (Li ions)
        self.qpMin = self.qMax * self.xpMin # min charge at pos electrode
        self.qpMax = self.qMax * self.xpMax # max charge at pos electrode
        self.qpSMin = self.qpMin * self.VolS / self.Vol # min charge at surface, pos electrode
        self.qpBMin = self.qpMin * self.VolB / self.Vol # min charge at bulk, pos electrode
        self.qpSMax = self.qpMax * self.VolS / self.Vol # max charge at surface, pos electrode
        self.qpBMax = self.qpMax * self.VolB / self.Vol # max charge at bulk, pos electrode
        self.qnMin = self.qMax * self.xnMin # max charge at neg electrode
        self.qnMax = self.qMax * self.xnMax # max charge at neg electrode
        self.qnSMax = self.qnMax * self.VolS / self.Vol # max charge at surface, neg electrode
        self.qnBMax = self.qnMax * self.VolB / self.Vol # max charge at bulk, neg electrode
        self.qnSMin = self.qnMin * self.VolS / self.Vol # min charge at surface, neg electrode
        self.qnBMin = self.qnMin * self.VolB / self.Vol # min charge at bulk, neg electrode
        self.qSMax = self.qMax * self.VolS / self.Vol # max charge at surface (pos and neg)
        self.qBMax = self.qMax * self.VolB / self.Vol # max charge at bulk (pos and neg)

        ################################## time constants
        self.tDiffusion = 7e6 # diffusion time constant (increasing this causes decrease in diffusion rate)
        self.to = 6.08671 # for Ohmic voltage
        self.tsn = 1001.38 # for surface overpotential (neg)
        self.tsp = 46.4311 # for surface overpotential (pos)

        # Redlich-Kister self(positive electrode)
        self.U0p = 4.03
        self.Ap0 = -31593.7
        self.Ap1 = 0.106747
        self.Ap2 = 24606.4
        self.Ap3 = -78561.9
        self.Ap4 = 13317.9
        self.Ap5 = 307387
        self.Ap6 = 84916.1
        self.Ap7 = -1.07469e+06
        self.Ap8 = 2285.04
        self.Ap9 = 990894
        self.Ap10 = 283920
        self.Ap11 = -161513
        self.Ap12 = -469218

        ################################### Redlich-Kister self (negative electrode)

        self.U0n = 0.01
        self.An0 = 86.19
        self.An1 = 0
        self.An2 = 0
        self.An3 = 0
        self.An4 = 0
        self.An5 = 0
        self.An6 = 0
        self.An7 = 0
        self.An8 = 0
        self.An9 = 0
        self.An10 = 0
        self.An11 = 0
        self.An12 = 0
        ################################## End of discharge voltage threshold
        self.VEOD = 3.0

        ################################## Default initial conditions (fully charged)
        self.x0_qpS = self.qpSMin
        self.x0_qpB = self.qpBMin
        self.x0_qnS = self.qnSMax
        self.x0_qnB = self.qnBMax
        self.x0_Vo = 0
        self.x0_Vsn = 0
        self.x0_Vsp = 0
        self.x0_Tb = 292.1

        ################################## Process noise variances
        self.v_qpS = 1e-5
        self.v_qpB = 1e-3
        self.v_qnS = 1e-5
        self.v_qnB = 1e-3
        self.v_Vo = 1e-10
        self.v_Vsn = 1e-10
        self.v_Vsp = 1e-10
        self.v_Tb = 1e-6
        self.V=[[self.v_qpS],[self.v_qpB],[self.v_qnS],[self.v_qnB],[self.v_Vo],[self.v_Vsn],[self.v_Vsp],[self.v_Tb]]

        ################################# Sensor noise variances
        self.n_Vm = 1e-3
        self.n_Tbm = 1e-3
        self.N=[[self.n_Vm],[self.n_Tbm]]


        ################################UAV setting
        self.mass=1.357
        self.payload_mass=0.3
        self.gravity=9.8
        self.weight=self.mass*self.gravity
        self.payload_weight=0.3*self.gravity
        self.air_density=1.15
        self.propeller_diameter=0.2413
        self.disc_actuator_area=0.1829

        ################################# State
        self.high=np.array([5.,5.,5.,5.,20.])
        self.low = np.array([0.,0., 0., 0., 0.])
        self.action_space = spaces.Box(low=np.array([0,0,0,0]), high=np.array([10.,10.,10.,10.]), dtype=np.float32)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)
        self.B_init = np.array([292.1, 0, 0, 0, 6840, 760, 4.5600e+03, 506.6667])
        self.B_X_profile=[]
        self.B_Z_profile = []
        #
        # self.seed()
        # self.viewer = None
        # self.state = None
        # self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def InputEqn(self,t,inputself=[]):

        #  InputEqn   Compute model inputs for the given time and input self
        #
        #    U = InputEqn(self,t,inputself) computes the inputs to the
        #    battery model given the self structure, the current time, and the
        #    set of input self. The input self are optional - if not
        #    provided a default input of 8 Watts is used. If inputself is
        #    provided, it should be a matrix, numInputself x numSamples.
        #
        #    For the battery model, the input self are a list of numbers
        #    specifying a sequence of load segments, with each segment defined by a
        #    magnitude and a duration. So, for example, the following input
        #    self vector:
        #       [5 100 2 200 3 300]
        #    captures a set of three segments, the first of 5 W lasting 100 seconds,
        #    the second 2 W lasting 200 s, the third 3 W lasting 300 s. The initial
        #    time is assumed to be 0, so if t is given as 150 s, for example, then
        #    the load magnitude will be 2 W (second segment).


        if inputself==[]:
            P=[8]
        else:
            P = np.zeros(np.shape(inputself)[1])
            for i in range(np.shape(inputself)[1]):
                u = np.reshape(inputself[:, i],(len(inputself),1))


                loads = u[0:np.shape(inputself)[0]:2]
                durations = u[1:np.shape(inputself)[0]:2]
                times = np.hstack(([0], np.cumsum(durations).tolist()))

                #  Find which load corresponds to given time
                loadIndex = np.where(times >= t)[0][0] -1

                if loadIndex > len(loads):
                    P[i] = loads[-1]
                else:
                    P[i] = loads[loadIndex]




        U = [P]

        return U

    def StateEqn(self,t,X,U,N,dt=1):
        # StateEqn   Compute the new states of the battery model
        #
        #    XNew = StateEqn(self,t,X,U,N,dt) computes the new states of the
        #    battery model given the self strcucture, the current time, the
        #    current states, inputs, process noise, and the sampling time.

        # Extract states

        Tb = X[0,:]
        Vo = X[1,:]
        Vsn = X[2,:]
        Vsp = X[3,:]
        qnB = X[4,:]
        qnS = X[5,:]
        qpB = X[6,:]
        qpS = X[7,:]
        #  Extract inputs
        P = U[:]
        # Constraints
        Tbdot = 0
        CnBulk = qnB/ self.VolB
        CnSurface = qnS/ self.VolS
        CpSurface = qpS/ self.VolS
        xnS = min(max(qnS/ self.qSMax,0.01),0.99)
        Ven5 = self.An5* ((2* xnS - 1)** (5 + 1) - (2* xnS* 5* (1 - xnS))/ (2* xnS - 1)** (
                    1 - 5))/ self.F
        xpS = min(max(qpS/ self.qSMax,0.01),0.99)
        Vep3 = self.Ap3* ((2* xpS - 1)** (3 + 1) - (2* xpS* 3* (1 - xpS))/ (2* xpS - 1)** (
                    1 - 3))/ self.F
        Vep12 = self.Ap12* ((2* xpS - 1)** (12 + 1) - (2* xpS* 12* (1 - xpS))/ (2* xpS - 1)** (
                    1 - 12))/ self.F
        Vep4 = self.Ap4* ((2* xpS - 1)** (4 + 1) - (2* xpS* 4* (1 - xpS))/ (2* xpS - 1)** (
                    1 - 4))/ self.F
        Vep11 = self.Ap11* ((2* xpS - 1)** (11 + 1) - (2* xpS* 11* (1 - xpS))/ (2* xpS - 1)** (
                    1 - 11))/ self.F
        Vep2 = self.Ap2* ((2* xpS - 1)** (2 + 1) - (2* xpS* 2* (1 - xpS))/ (2* xpS - 1)** (
                    1 - 2))/ self.F
        Vep7 = self.Ap7* ((2* xpS - 1)** (7 + 1) - (2* xpS* 7* (1 - xpS))/ (2* xpS - 1)** (
                    1 - 7))/ self.F
        CpBulk = qpB/ self.VolB
        Vep8 = self.Ap8* ((2* xpS - 1)** (8 + 1) - (2* xpS* 8* (1 - xpS))/ (2* xpS - 1)** (
                    1 - 8))/ self.F
        qdotDiffusionBSn = (CnBulk - CnSurface)/ self.tDiffusion
        qnBdot = - qdotDiffusionBSn
        Jn0 = self.kn* (1 - xnS)** self.alpha* (xnS)** self.alpha
        Ven3 = self.An3* ((2* xnS - 1)** (3 + 1) - (2* xnS* 3* (1 - xnS))/ (2* xnS - 1)** (
                    1 - 3))/ self.F
        qdotDiffusionBSp = (CpBulk - CpSurface)/ self.tDiffusion
        Ven0 = self.An0* ((2* xnS - 1)** (0 + 1))/ self.F

        Jp0 = self.kp* (1 - xpS)** self.alpha* (xpS)** self.alpha
        Ven10 = self.An10* ((2* xnS - 1)** (10 + 1) - (2* xnS* 10* (1 - xnS))/ (2* xnS - 1)** (
                    1 - 10))/ self.F
        Ven7 = self.An7* ((2* xnS - 1)** (7 + 1) - (2* xnS* 7* (1 - xnS))/ (2* xnS - 1)** (
                    1 - 7))/ self.F
        Ven2 = self.An2* ((2* xnS - 1)** (2 + 1) - (2* xnS* 2* (1 - xnS))/ (2* xnS - 1)** (
                    1 - 2))/ self.F
        Ven11 = self.An11* ((2* xnS - 1)** (11 + 1) - (2* xnS* 11* (1 - xnS))/ (2* xnS - 1)** (
                    1 - 11))/ self.F
        Ven8 = self.An8* ((2* xnS - 1)** (8 + 1) - (2* xnS* 8* (1 - xnS))/ (2* xnS - 1)** (
                    1 - 8))/ self.F
        Ven12 = self.An12* ((2* xnS - 1)** (12 + 1) - (2* xnS* 12* (1 - xnS))/ (2* xnS - 1)** (
                    1 - 12))/ self.F
        Ven1 = self.An1* ((2* xnS - 1)** (1 + 1) - (2* xnS* 1* (1 - xnS))/ (2* xnS - 1)** (
                    1 - 1))/ self.F
        Ven4 = self.An4* ((2* xnS - 1)** (4 + 1) - (2* xnS* 4* (1 - xnS))/ (2* xnS - 1)** (
                    1 - 4))/ self.F
        Ven6 = self.An6* ((2* xnS - 1)** (6 + 1) - (2* xnS* 6* (1 - xnS))/ (2* xnS - 1)** (
                    1 - 6))/ self.F
        Ven9 = self.An9* ((2* xnS - 1)** (9 + 1) - (2* xnS* 9* (1 - xnS))/ (2* xnS - 1)** (
                    1 - 9))/ self.F
        Vep0 = self.Ap0* ((2* xpS - 1)** (0 + 1))/ self.F
        Vep5 = self.Ap5* ((2* xpS - 1)** (5 + 1) - (2* xpS* 5* (1 - xpS))/ (2* xpS - 1)** (
                    1 - 5))/ self.F
        Vep6 = self.Ap6* ((2* xpS - 1)** (6 + 1) - (2* xpS* 6* (1 - xpS))/ (2* xpS - 1)** (
                    1 - 6))/ self.F
        Vep1 = self.Ap1* ((2* xpS - 1)** (1 + 1) - (2* xpS* 1* (1 - xpS))/ (2* xpS - 1)** (
                    1 - 1))/ self.F
        Vep10 = self.Ap10* ((2* xpS - 1)** (10 + 1) - (2* xpS* 10* (1 - xpS))/ (2* xpS - 1)** (
                    1 - 10))/ self.F
        Vep9 = self.Ap9* ((2* xpS - 1)** (9 + 1) - (2* xpS* 9* (1 - xpS))/ (2* xpS - 1)** (
                    1 - 9))/ self.F
        qpBdot = - qdotDiffusionBSp
        Ven = self.U0n + self.R* Tb/ self.F* np.log((
                                                                                     1 - xnS)/ xnS) + Ven0 + Ven1 + Ven2 + Ven3 + Ven4 + Ven5 + Ven6 + Ven7 + Ven8 + Ven9 + Ven10 + Ven11 + Ven12
        Vep = self.U0p + self.R* Tb/ self.F* np.log((
                                                                                     1 - xpS)/ xpS) + Vep0 + Vep1 + Vep2 + Vep3 + Vep4 + Vep5 + Vep6 + Vep7 + Vep8 + Vep9 + Vep10 + Vep11 + Vep12
        V = Vep - Ven - Vo - Vsn - Vsp
        i = P/ V
        qpSdot = i + qdotDiffusionBSp
        Jn = i/ self.Sn
        VoNominal = i* self.Ro
        Jp = i/ self.Sp
        qnSdot = qdotDiffusionBSn - i
        VsnNominal = self.R* Tb/ self.F/ self.alpha* np.arcsinh(Jn/ (2* Jn0))
        Vodot = (VoNominal - Vo)/ self.to
        VspNominal = self.R* Tb/ self.F/ self.alpha* np.arcsinh(Jp/ (2* Jp0))
        Vsndot = (VsnNominal - Vsn)/ self.tsn
        Vspdot = (VspNominal - Vsp)/ self.tsp

        # Update state
        XNew = np.zeros(np.shape(X))
        XNew[0,:] = Tb + Tbdot*dt
        XNew[1,:] = Vo + Vodot*dt
        XNew[2,:] = Vsn + Vsndot*dt
        XNew[3,:] = Vsp + Vspdot*dt
        XNew[4,:] = qnB + qnBdot*dt
        XNew[5,:] = qnS + qnSdot*dt
        XNew[6,:] = qpB + qpBdot*dt
        XNew[7,:] = qpS + qpSdot*dt

        # Add process noise
        XNew = XNew + dt*N
        return XNew

    def OutputEqn(self,t,X,U,N):
        #  OutputEqn   Compute the outputs of the battery model
        #
        #    Z = OutputEqn(self,t,X,U,N) computes the outputs of the battery
        #    model given the self structure, time, the states, inputs, and
        #    sensor noise. The function is vectorized, so if the function inputs are
        #    matrices, the funciton output will be a matrix, with the rows being the
        #    variables and the columns the samples.


        # Extract states
        Tb = X[0,:]
        Vo = X[1,:]
        Vsn = X[2,:]
        Vsp = X[3,:]
        qnB = X[4,:]
        qnS = X[5,:]
        qpB = X[6,:]
        qpS = X[7,:]

        # Extract inputs

        P = U[:]

        # Constraints
        # 点除是对应元素相除
        Tbm = Tb - 273.15

        xpS = min(max(qpS/ self.qSMax,0.01),0.99)
        # print(qpS,xpS)
        Vep3 = self.Ap3* ((2* xpS - 1)** (3 + 1) - (2* xpS* 3* (1 - xpS))/ (2* xpS - 1)** (
                    1 - 3))/ self.F
        Vep8 = self.Ap8* ((2* xpS - 1)** (8 + 1) - (2* xpS* 8* (1 - xpS))/ (2* xpS - 1)** (
                    1 - 8))/ self.F
        Vep6 = self.Ap6* ((2* xpS - 1)** (6 + 1) - (2* xpS* 6* (1 - xpS))/ (2* xpS - 1)** (
                    1 - 6))/ self.F
        Vep5 = self.Ap5* ((2* xpS - 1)** (5 + 1) - (2* xpS* 5* (1 - xpS))/ (2* xpS - 1)** (
                    1 - 5))/ self.F
        Vep10 = self.Ap10* ((2* xpS - 1)** (10 + 1) - (2* xpS* 10* (1 - xpS))/ (2* xpS - 1)** (
                    1 - 10))/ self.F
        Vep9 = self.Ap9* ((2* xpS - 1)** (9 + 1) - (2* xpS* 9* (1 - xpS))/ (2* xpS - 1)** (
                    1 - 9))/ self.F
        Vep12 = self.Ap12* ((2* xpS - 1)** (12 + 1) - (2* xpS* 12* (1 - xpS))/ (2* xpS - 1)** (
                    1 - 12))/ self.F
        Vep4 = self.Ap4* ((2* xpS - 1)** (4 + 1) - (2* xpS* 4* (1 - xpS))/ (2* xpS - 1)** (
                    1 - 4))/ self.F
        Vep11 = self.Ap11* ((2* xpS - 1)** (11 + 1) - (2* xpS* 11* (1 - xpS))/ (2* xpS - 1)** (
                    1 - 11))/ self.F
        Vep2 = self.Ap2* ((2* xpS - 1)** (2 + 1) - (2* xpS* 2* (1 - xpS))/ (2* xpS - 1)** (
                    1 - 2))/ self.F
        Vep7 = self.Ap7* ((2* xpS - 1)** (7 + 1) - (2* xpS* 7* (1 - xpS))/ (2* xpS - 1)** (
                    1 - 7))/ self.F
        Vep0 = self.Ap0* ((2* xpS - 1)** (0 + 1))/ self.F
        Vep1 = self.Ap1* ((2* xpS - 1)** (1 + 1) - (2* xpS* 1* (1 - xpS))/ (2* xpS - 1)** (
                    1 - 1))/ self.F
        xnS =min(max(qnS/ self.qSMax,0.01),0.99)

        Ven5 = self.An5* ((2* xnS - 1)** (5 + 1) - (2* xnS* 5* (1 - xnS))/ (2* xnS - 1)** (
                    1 - 5))/ self.F
        Ven1 = self.An1* ((2* xnS - 1)** (1 + 1) - (2* xnS* 1* (1 - xnS))/ (2* xnS - 1)** (
                    1 - 1))/ self.F
        Ven10 = self.An10* ((2* xnS - 1)** (10 + 1) - (2* xnS* 10* (1 - xnS))/ (2* xnS - 1)** (
                    1 - 10))/ self.F
        Ven7 = self.An7* ((2* xnS - 1)** (7 + 1) - (2* xnS* 7* (1 - xnS))/ (2* xnS - 1)** (
                    1 - 7))/ self.F
        Ven2 = self.An2* ((2* xnS - 1)** (2 + 1) - (2* xnS* 2* (1 - xnS))/ (2* xnS - 1)** (
                    1 - 2))/ self.F
        Ven8 = self.An8* ((2* xnS - 1)** (8 + 1) - (2* xnS* 8* (1 - xnS))/ (2* xnS - 1)** (
                    1 - 8))/ self.F
        Ven4 = self.An4* ((2* xnS - 1)** (4 + 1) - (2* xnS* 4* (1 - xnS))/ (2* xnS - 1)** (
                    1 - 4))/ self.F
        Ven3 = self.An3* ((2* xnS - 1)** (3 + 1) - (2* xnS* 3* (1 - xnS))/ (2* xnS - 1)** (
                    1 - 3))/ self.F

        Vep = self.U0p + self.R* Tb/ self.F* np.log((1 - xpS)/ xpS) + Vep0 + Vep1 + Vep2 + Vep3 + Vep4 + Vep5 + Vep6 + Vep7 + Vep8 + Vep9 + Vep10 + Vep11 + Vep12
        Ven0 = self.An0* ((2* xnS - 1)** (0 + 1))/ self.F
        Ven11 = self.An11* ((2* xnS - 1)** (11 + 1) - (2* xnS* 11* (1 - xnS))/ (2* xnS - 1)** (
                    1 - 11))/ self.F
        Ven12 = self.An12* ((2* xnS - 1)** (12 + 1) - (2* xnS* 12* (1 - xnS))/ (2* xnS - 1)** (
                    1 - 12))/ self.F
        Ven6 = self.An6* ((2* xnS - 1)** (6 + 1) - (2* xnS* 6* (1 - xnS))/ (2* xnS - 1)** (
                    1 - 6))/ self.F
        Ven9 = self.An9* ((2* xnS - 1)** (9 + 1) - (2* xnS* 9* (1 - xnS))/ (2* xnS - 1)** (
                    1 - 9))/ self.F
        Ven = self.U0n + self.R* Tb/ self.F* np.log((1 - xnS)/ xnS) + Ven0 + Ven1 + Ven2 + Ven3 + Ven4 + Ven5 + Ven6 + Ven7 + Ven8 + Ven9 + Ven10 + Ven11 + Ven12
        V = Vep - Ven - Vo - Vsn - Vsp
        Vm = V

        # set outputs


        Z = np.zeros([2,np.max([np.shape(X)[1],np.shape(U)[1],1])])
        Z[0,:] = Tbm
        Z[1,:] = Vm


        # Add sensor noise
        Z = Z + N


        return Z

    def ThresholdEqn(self,t,X,U):
        # ThresholdEqn   Compute whether the battery has reached end-of-discharge
        #
        #    B = ThresholdEqn(parameters,t,X,U) computes whether the battery model
        #    for the given time, states, and inputs, has crossed the voltage
        #    threshold defining end-of-discharge.

        # Extract states

        Tb = X[0,:]
        Vo = X[1,:]
        Vsn = X[2,:]
        Vsp = X[3,:]
        qnB = X[4,:]
        qnS = X[5,:]
        qpB = X[6,:]
        qpS = X[7,:]

        # Extract inputs
        P = U[:]

        # Constraints
        xpS = qpS/self.qSMax
        Vep3 = self.Ap3* ((2* xpS - 1)** (3 + 1) - (2* xpS* 3* (1 - xpS))/(2* xpS - 1)** (
                    1 - 3))/self.F
        Vep8 = self.Ap8* ((2* xpS - 1)** (8 + 1) - (2* xpS* 8* (1 - xpS))/(2* xpS - 1)** (
                    1 - 8))/self.F
        Vep6 = self.Ap6* ((2* xpS - 1)** (6 + 1) - (2* xpS* 6* (1 - xpS))/(2* xpS - 1)** (
                    1 - 6))/self.F
        Vep5 = self.Ap5* ((2* xpS - 1)** (5 + 1) - (2* xpS* 5* (1 - xpS))/(2* xpS - 1)** (
                    1 - 5))/self.F
        Vep10 = self.Ap10* ((2* xpS - 1)** (10 + 1) - (2* xpS* 10* (1 - xpS))/(2* xpS - 1)** (
                    1 - 10))/self.F
        Vep9 = self.Ap9* ((2* xpS - 1)** (9 + 1) - (2* xpS* 9* (1 - xpS))/(2* xpS - 1)** (
                    1 - 9))/self.F
        Vep12 = self.Ap12* ((2* xpS - 1)** (12 + 1) - (2* xpS* 12* (1 - xpS))/(2* xpS - 1)** (
                    1 - 12))/self.F
        Vep4 = self.Ap4* ((2* xpS - 1)** (4 + 1) - (2* xpS* 4* (1 - xpS))/(2* xpS - 1)** (
                    1 - 4))/self.F
        Vep11 = self.Ap11* ((2* xpS - 1)** (11 + 1) - (2* xpS* 11* (1 - xpS))/(2* xpS - 1)** (
                    1 - 11))/self.F
        Vep2 = self.Ap2* ((2* xpS - 1)** (2 + 1) - (2* xpS* 2* (1 - xpS))/(2* xpS - 1)** (
                    1 - 2))/self.F
        Vep7 = self.Ap7* ((2* xpS - 1)** (7 + 1) - (2* xpS* 7* (1 - xpS))/(2* xpS - 1)** (
                    1 - 7))/self.F
        Vep0 = self.Ap0* ((2* xpS - 1)** (0 + 1))/self.F
        Vep1 = self.Ap1* ((2* xpS - 1)** (1 + 1) - (2* xpS* 1* (1 - xpS))/(2* xpS - 1)** (
                    1 - 1))/self.F
        xnS = qnS/self.qSMax
        Ven5 = self.An5* ((2* xnS - 1)** (5 + 1) - (2* xnS* 5* (1 - xnS))/(2* xnS - 1)** (
                    1 - 5))/self.F
        Ven1 = self.An1* ((2* xnS - 1)** (1 + 1) - (2* xnS* 1* (1 - xnS))/(2* xnS - 1)** (
                    1 - 1))/self.F
        Ven10 = self.An10* ((2* xnS - 1)** (10 + 1) - (2* xnS* 10* (1 - xnS))/(2* xnS - 1)** (
                    1 - 10))/self.F
        Ven7 = self.An7* ((2* xnS - 1)** (7 + 1) - (2* xnS* 7* (1 - xnS))/(2* xnS - 1)** (
                    1 - 7))/self.F
        Ven2 = self.An2* ((2* xnS - 1)** (2 + 1) - (2* xnS* 2* (1 - xnS))/(2* xnS - 1)** (
                    1 - 2))/self.F
        Ven8 = self.An8* ((2* xnS - 1)** (8 + 1) - (2* xnS* 8* (1 - xnS))/(2* xnS - 1)** (
                    1 - 8))/self.F
        Ven4 = self.An4* ((2* xnS - 1)** (4 + 1) - (2* xnS* 4* (1 - xnS))/(2* xnS - 1)** (
                    1 - 4))/self.F
        Ven3 = self.An3* ((2* xnS - 1)** (3 + 1) - (2* xnS* 3* (1 - xnS))/(2* xnS - 1)** (
                    1 - 3))/self.F
        Vep = self.U0p + self.R* Tb/self.F* np.log((
                                                                                     1 - xpS)/xpS) + Vep0 + Vep1 + Vep2 + Vep3 + Vep4 + Vep5 + Vep6 + Vep7 + Vep8 + Vep9 + Vep10 + Vep11 + Vep12
        Ven0 = self.An0* ((2* xnS - 1)** (0 + 1))/self.F
        Ven11 = self.An11* ((2* xnS - 1)** (11 + 1) - (2* xnS* 11* (1 - xnS))/(2* xnS - 1)** (
                    1 - 11))/self.F
        Ven12 = self.An12* ((2* xnS - 1)** (12 + 1) - (2* xnS* 12* (1 - xnS))/(2* xnS - 1)** (
                    1 - 12))/self.F
        Ven6 = self.An6* ((2* xnS - 1)** (6 + 1) - (2* xnS* 6* (1 - xnS))/(2* xnS - 1)** (
                    1 - 6))/self.F
        Ven9 = self.An9* ((2* xnS - 1)** (9 + 1) - (2* xnS* 9* (1 - xnS))/(2* xnS - 1)** (
                    1 - 9))/self.F
        Ven = self.U0n + self.R* Tb/self.F* np.log((
                                                                                     1 - xnS)/xnS) + Ven0 + Ven1 + Ven2 + Ven3 + Ven4 + Ven5 + Ven6 + Ven7 + Ven8 + Ven9 + Ven10 + Ven11 + Ven12
        V = Vep - Ven - Vo - Vsn - Vsp
        Vm = V

        # Return true if voltage is less than the voltage threshold
        B = Vm < self.VEOD


        return B

    def getDefaultInitialization(self,t0=0,inputself=None):
        # getDefaultInitialization Get default initial states, inputs,
        # and outputs.
        # This method returns the initial states, inputs, and outputs
        # for the given initial time, and, if provided, a set of
        # input self. If no arguments are given to this method,
        # an initial time of 0 is assumed.

        # Set up initial state from the state struct array
        x0 = np.zeros((len(self.B_init),1))
        for i in range(len(self.B_init)):
            x0[i] = self.B_init[i]

        # Compute inputs for initial time

        if inputself==None:
            u0 = self.InputEqn(t0)
        else:
            u0 = self.InputEqn(t0,inputself)

        # Compute outputs for initial time, states, and inputs
        z0 = self.OutputEqn(t0, x0, u0, 0)
        # print(self.B_init)
        # print(x0)
        # print(u0)
        # print(z0)


        return x0,u0,z0

    def generateSensorNoise(self,numSamples=1):
        # generateSensorNoise   Generate sensor noise samples
        # Generate uncorrelated sensor noise from normal
        # distributions with zero-mean. The variances are defined
        # from the outputs structure of the model.

        n = np.tile(self.N, (1, numSamples))*np.random.random((len(self.N),numSamples))

        return n

    def generateProcessNoise(self,numSamples=1):
        # generateProcessNoise   Generate process noise samples
        # Generate uncorrelated process noise from normal
        # distributions with zero-mean. The variances are defined
        # from the states structure of the model.
        v = np.tile(self.V,(1,numSamples))*np.random.random((len(self.V),numSamples))
        return v

    def simulate(self,tFinal,printTime,input=[]):

        # Simulate model for a given time interval
        #  [T,X,U,Z] = simulate(M,tFinal,varargin) simulates the model
        #  from time 0 to tFinal, returning time, state, input, and
        #  output arrays. A first order Euler solver is used.
        #
        #  The additional arguments are optional and come in
        #  string/value pairs. The optional arguments are as follows:
        #  - enableSensorNoise: flag indicating whether to consider
        #    sensor noise in the simulation
        #  - enableProcessNoise: flag indicating whether to consider
        #    process noise in the simulation
        #  - printTime: the number of time points after which to print
        #    the current status of the simulation
        #  - x0: initial state with which to begin the simulation
        #  - t0: initial time with which to begin the simulation


        # Set default options
        options_enableProcessNoise = 0
        options_enableSensorNoise = 0
        options_printTime = 1e10
        options_x0 = []
        options_t0 = 0

        # Create a structure from the string, value pairs

        #Initialize
        x0, u0, z0 = self.getDefaultInitialization(0)
        if options_x0 != []:
            x0 = options_x0
            z0 = self.OutputEqn(options_t0, x0, u0, options_enableSensorNoise * self.generateSensorNoise())

        # Preallocate output data: columns represent times
        T = list(range(options_t0,tFinal+1,self.sampleTime))

        X = np.zeros((len(x0), len(T)))
        Z = np.zeros((len(z0), len(T)))
        U = np.zeros((len(u0), len(T)))


        X[:,0:1] = x0
        U[:,0:1] = u0
        Z[:,0:1] = z0

        # Simulate
        x = x0
        u = u0
        for i in range(1,len(T)):
            # Update state from time t-dt to time t
            x = self.StateEqn(T[i - 1], x, u, options_enableProcessNoise * self.generateProcessNoise())
            # Get inputs for time t
            u = self.InputEqn(T[i],inputself=input)

            # Compute outputs for time t
            z = self.OutputEqn(T[i], x, u, options_enableSensorNoise * self.generateSensorNoise())
            X[:, i:i+1] = x
            U[:, i:i+1] = u
            Z[:, i:i+1] = z
        return T,X,U,Z

    def reset(self):
        if self.B_X_profile==[]:
            T1, X1, U1, Z1 = self.simulate(3150, 1e10)
            self.B_X_profile = X1
            self.B_Z_profile = Z1


        list = [np.random.randint(1000, 1200), np.random.randint(1500, 1800),np.random.randint(2200, 2300),np.random.randint(2500, 3100)]
        np.random.shuffle(list)
        t1,t2,t3,t4 = list
        # t1 = np.random.randint(1000, 1200)
        # t2 = np.random.randint(1500, 1800)
        # t3 = np.random.randint(2200, 2300)
        # t4 = np.random.randint(2500, 3100)
        # print(t1,t2,t3,t4)
        self.B1 = np.array(self.B_X_profile[:, t1])
        self.B2 = np.array(self.B_X_profile[:, t2])
        self.B3 = np.array(self.B_X_profile[:, t3])
        self.B4 = np.array(self.B_X_profile[:, t4])
        self.B1_baseline = np.array(self.B_X_profile[:, t1])
        self.B2_baseline = np.array(self.B_X_profile[:, t2])
        self.B3_baseline = np.array(self.B_X_profile[:, t3])
        self.B4_baseline = np.array(self.B_X_profile[:, t4])
        z1 = self.B_Z_profile[1][t1]
        z2 = self.B_Z_profile[1][t2]
        z3 = self.B_Z_profile[1][t3]
        z4 = self.B_Z_profile[1][t4]
        load_demand=60*max(np.random.rand(),0.4)
        self.state = np.array([z1,z2,z3,z4,load_demand])
        return self.state

    def eval_reset(self):
        if self.B_X_profile==[]:
            T1, X1, U1, Z1 = self.simulate(3150, 1e10)
            self.B_X_profile = X1
            self.B_Z_profile = Z1

        t1 = 1001
        t2 = 1555
        t3 = 2222
        t4 = 2555
        self.B1 = np.array(self.B_X_profile[:, t1])
        self.B2 = np.array(self.B_X_profile[:, t2])
        self.B3 = np.array(self.B_X_profile[:, t3])
        self.B4 = np.array(self.B_X_profile[:, t4])
        self.B1_baseline = np.array(self.B_X_profile[:, t1])
        self.B2_baseline = np.array(self.B_X_profile[:, t2])
        self.B3_baseline = np.array(self.B_X_profile[:, t3])
        self.B4_baseline = np.array(self.B_X_profile[:, t4])
        z1 = self.B_Z_profile[1][t1]
        z2 = self.B_Z_profile[1][t2]
        z3 = self.B_Z_profile[1][t3]
        z4 = self.B_Z_profile[1][t4]
        load_demand= 32
        self.state = np.array([z1,z2,z3,z4,load_demand])
        return self.state

    def test_reset(self,t1,t2,t3,t4):
        if self.B_X_profile==[]:
            T1, X1, U1, Z1 = self.simulate(3150, 1e10)
            self.B_X_profile = X1
            self.B_Z_profile = Z1

        self.B1 = np.array(self.B_X_profile[:, t1])
        self.B2 = np.array(self.B_X_profile[:, t2])
        self.B3 = np.array(self.B_X_profile[:, t3])
        self.B4 = np.array(self.B_X_profile[:, t4])
        self.B1_baseline = np.array(self.B_X_profile[:, t1])
        self.B2_baseline = np.array(self.B_X_profile[:, t2])
        self.B3_baseline = np.array(self.B_X_profile[:, t3])
        self.B4_baseline = np.array(self.B_X_profile[:, t4])
        z1 = self.B_Z_profile[1][t1]
        z2 = self.B_Z_profile[1][t2]
        z3 = self.B_Z_profile[1][t3]
        z4 = self.B_Z_profile[1][t4]
        load_demand= 8
        self.state = np.array([z1,z2,z3,z4,load_demand])
        return self.state

    def simulateToThreshold(self,input=[]):


        # Set default options
        options_enableProcessNoise = 0
        options_enableSensorNoise = 0
        options_printTime = 1e10
        options_x0 = []
        options_t0 = 0

        # Create a structure from the string, value pairs

        # Initialize
        x0, u0, z0 = self.getDefaultInitialization(0)
        if options_x0 != []:
            x0 = options_x0
            z0 = self.OutputEqn(options_t0, x0, u0, options_enableSensorNoise * self.generateSensorNoise())

        # Setup output data: columns represent times
        t = options_t0
        T = [t]

        X = np.zeros((len(x0), 1))
        Z = np.zeros((len(z0), 1))
        U = np.zeros((len(u0), 1))

        X[:, 0:1] = x0
        U[:, 0:1] = u0
        Z[:, 0:1] = z0
        # Simulate
        x = x0
        u = u0
        i = 1
        while not self.ThresholdEqn(t,x,u):
            # Update state from time t-dt to time t
            x = self.StateEqn(t, x, u, options_enableProcessNoise * self.generateProcessNoise())
            # Update time
            t = t + self.sampleTime
            # Get inputs for time t
            u = self.InputEqn(t,inputself=input)
            # Compute outputs for time t
            z = self.OutputEqn(t, x, u, options_enableSensorNoise * self.generateSensorNoise())
            T.append(t)
            X[:, i:i + 1] = x
            U[:, i:i + 1] = u
            Z[:, i:i + 1] = z
        return T, X, U, Z

    def step(self, action,load_demand):

        done = False


        u1 = [action[0]*self.state[-1]]
        u2 = [action[1]*self.state[-1]]
        u3 = [action[2]*self.state[-1]]
        u4 = [action[3]*self.state[-1]]



        x1 = np.reshape(self.B1, (8, 1))
        x2 = np.reshape(self.B2, (8, 1))
        x3 = np.reshape(self.B3, (8, 1))
        x4 = np.reshape(self.B4, (8, 1))

        last_state=self.state


        for i in range(1):
            x1 = self.StateEqn(0, x1, u1, 0 * self.generateProcessNoise())
            # z1 = self.OutputEqn(0, x1, [u1], 1 * self.generateSensorNoise())
            # self.B1 = np.reshape(x1, (8))

            x2 = self.StateEqn(0, x2, u2, 0 * self.generateProcessNoise())
            # z2 = self.OutputEqn(0, x2, [u2], 1 * self.generateSensorNoise())
            # self.B2 = np.reshape(x2, (8))

            x3 = self.StateEqn(0, x3, u3, 0 * self.generateProcessNoise())
            # z3 = self.OutputEqn(0, x3, [u3], 1 * self.generateSensorNoise())
            # self.B3 = np.reshape(x3, (8))

            x4 = self.StateEqn(0, x4, u4, 0 * self.generateProcessNoise())
            # z4 = self.OutputEqn(0, x4, [u4], 1 * self.generateSensorNoise())
            # self.B4 = np.reshape(x4, (8))

        z1 = self.OutputEqn(0, x1, [u1], 0 * self.generateSensorNoise())
        self.B1 = np.reshape(x1, (8))

        z2 = self.OutputEqn(0, x2, [u2], 0 * self.generateSensorNoise())
        self.B2 = np.reshape(x2, (8))

        z3 = self.OutputEqn(0, x3, [u3], 0 * self.generateSensorNoise())
        self.B3 = np.reshape(x3, (8))

        z4 = self.OutputEqn(0, x4, [u4], 0 * self.generateSensorNoise())
        self.B4 = np.reshape(x4, (8))

        self.state = [z1[1][0], z2[1][0], z3[1][0], z4[1][0], load_demand]

        # coef_1 = 1/(np.exp(last_state[0]) / (
        #             np.exp(last_state[0]) + np.exp(last_state[1]) + np.exp(last_state[2]) + np.exp(last_state[3])))
        # coef_2 = 1/(np.exp(last_state[1]) / (
        #             np.exp(last_state[0]) + np.exp(last_state[1]) + np.exp(last_state[2]) + np.exp(last_state[3])))
        # coef_3 = 1/(np.exp(last_state[2]) / (
        #         np.exp(last_state[0]) + np.exp(last_state[1]) + np.exp(last_state[2]) + np.exp(last_state[3])))
        # coef_4 = 1/(np.exp(last_state[3]) / (
        #         np.exp(last_state[0]) + np.exp(last_state[1]) + np.exp(last_state[2]) + np.exp(last_state[3])))

        # reward_consumption_1 = (self.state[0] - last_state[0])* coef_1
        # reward_consumption_2 = (self.state[1] - last_state[1])* coef_2
        # reward_consumption_3 = (self.state[2] - last_state[2])* coef_3
        # reward_consumption_4 = (self.state[3] - last_state[3])* coef_4


        # reward_consumption_1 = np.exp(-(self.state[0] - last_state[0]))
        # reward_consumption_2 = np.exp(-(self.state[1] - last_state[1]))
        # reward_consumption_3 = np.exp(-(self.state[2] - last_state[2]))
        # reward_consumption_4 = np.exp(-(self.state[3] - last_state[3]))



        ratio_reward_1 = - action[np.argmin(last_state[:-1])]
        ratio_reward_2 = action[np.argmax(last_state[:-1])]
        survival_reward = self.state[np.argmin(self.state)]-3

        # print(reward_ratio)


        # reward_consumption = reward_consumption_1+reward_consumption_2+reward_consumption_3+reward_consumption_4

        # reward_consumption = np.exp(last_state[np.argmin(self.state)]-self.state[np.argmin(self.state)])/(reward_consumption_1+reward_consumption_2+reward_consumption_3+reward_consumption_4)



        reward = ratio_reward_1+ratio_reward_2+survival_reward

        if z1[1][0]<=3 or z2[1][0]<=3 or z3[1][0]<=3 or z4[1][0]<=3:
            done = True
            # reward = -100

        return reward,self.state,done

    def test_step(self, action,load_demand):

        action = np.exp(action)/sum(np.exp(action))

        done = False


        u1 = [action[0]*self.state[-1]]
        u2 = [action[1]*self.state[-1]]
        u3 = [action[2]*self.state[-1]]
        u4 = [action[3]*self.state[-1]]


        x1 = np.reshape(self.B1, (8, 1))
        x2 = np.reshape(self.B2, (8, 1))
        x3 = np.reshape(self.B3, (8, 1))
        x4 = np.reshape(self.B4, (8, 1))

        last_state=self.state


        for i in range(1):
            x1 = self.StateEqn(0, x1, u1, 1 * self.generateProcessNoise())
            # z1 = self.OutputEqn(0, x1, [u1], 1 * self.generateSensorNoise())
            # self.B1 = np.reshape(x1, (8))

            x2 = self.StateEqn(0, x2, u2, 1 * self.generateProcessNoise())
            # z2 = self.OutputEqn(0, x2, [u2], 1 * self.generateSensorNoise())
            # self.B2 = np.reshape(x2, (8))

            x3 = self.StateEqn(0, x3, u3, 1 * self.generateProcessNoise())
            # z3 = self.OutputEqn(0, x3, [u3], 1 * self.generateSensorNoise())
            # self.B3 = np.reshape(x3, (8))

            x4 = self.StateEqn(0, x4, u4, 1 * self.generateProcessNoise())
            # z4 = self.OutputEqn(0, x4, [u4], 1 * self.generateSensorNoise())
            # self.B4 = np.reshape(x4, (8))

        z1 = self.OutputEqn(0, x1, [u1], 1 * self.generateSensorNoise())
        self.B1 = np.reshape(x1, (8))

        z2 = self.OutputEqn(0, x2, [u2], 1 * self.generateSensorNoise())
        self.B2 = np.reshape(x2, (8))

        z3 = self.OutputEqn(0, x3, [u3], 1 * self.generateSensorNoise())
        self.B3 = np.reshape(x3, (8))

        z4 = self.OutputEqn(0, x4, [u4], 1 * self.generateSensorNoise())
        self.B4 = np.reshape(x4, (8))

        self.state = [z1[1][0], z2[1][0], z3[1][0], z4[1][0], load_demand]

        # coef_1 = 1/(np.exp(last_state[0]) / (
        #             np.exp(last_state[0]) + np.exp(last_state[1]) + np.exp(last_state[2]) + np.exp(last_state[3])))
        # coef_2 = 1/(np.exp(last_state[1]) / (
        #             np.exp(last_state[0]) + np.exp(last_state[1]) + np.exp(last_state[2]) + np.exp(last_state[3])))
        # coef_3 = 1/(np.exp(last_state[2]) / (
        #         np.exp(last_state[0]) + np.exp(last_state[1]) + np.exp(last_state[2]) + np.exp(last_state[3])))
        # coef_4 = 1/(np.exp(last_state[3]) / (
        #         np.exp(last_state[0]) + np.exp(last_state[1]) + np.exp(last_state[2]) + np.exp(last_state[3])))

        # reward_consumption_1 = (self.state[0] - last_state[0])* coef_1
        # reward_consumption_2 = (self.state[1] - last_state[1])* coef_2
        # reward_consumption_3 = (self.state[2] - last_state[2])* coef_3
        # reward_consumption_4 = (self.state[3] - last_state[3])* coef_4


        # reward_consumption_1 = np.exp(-(self.state[0] - last_state[0]))
        # reward_consumption_2 = np.exp(-(self.state[1] - last_state[1]))
        # reward_consumption_3 = np.exp(-(self.state[2] - last_state[2]))
        # reward_consumption_4 = np.exp(-(self.state[3] - last_state[3]))



        ratio_reward_1 = - action[np.argmin(last_state[:-1])]
        ratio_reward_2 = action[np.argmax(last_state[:-1])]
        survival_reward = self.state[np.argmin(self.state)]-3

        # print(reward_ratio)


        # reward_consumption = reward_consumption_1+reward_consumption_2+reward_consumption_3+reward_consumption_4

        # reward_consumption = np.exp(last_state[np.argmin(self.state)]-self.state[np.argmin(self.state)])/(reward_consumption_1+reward_consumption_2+reward_consumption_3+reward_consumption_4)



        reward = survival_reward + ratio_reward_2 + ratio_reward_1

        if z1[1][0]<=3 or z2[1][0]<=3 or z3[1][0]<=3 or z4[1][0]<=3:
            done = True
            reward = -100

        return reward,self.state,done

    def baseline_step(self, s_baseline, load_demand):
        done = False

        u1 = [0.25 * s_baseline[-1]]
        u2 = [0.25 * s_baseline[-1]]
        u3 = [0.25 * s_baseline[-1]]
        u4 = [0.25 * s_baseline[-1]]

        x1 = np.reshape(self.B1_baseline, (8, 1))
        x2 = np.reshape(self.B2_baseline, (8, 1))
        x3 = np.reshape(self.B3_baseline, (8, 1))
        x4 = np.reshape(self.B4_baseline, (8, 1))


        for i in range(1):
            x1 = self.StateEqn(0, x1, u1, 0 * self.generateProcessNoise())
            # z1 = self.OutputEqn(0, x1, [u1], 1 * self.generateSensorNoise())
            # self.B1_baseline = np.reshape(x1, (8))

            x2 = self.StateEqn(0, x2, u2, 0 * self.generateProcessNoise())
            # z2 = self.OutputEqn(0, x2, [u2], 1 * self.generateSensorNoise())
            # self.B2_baseline = np.reshape(x2, (8))

            x3 = self.StateEqn(0, x3, u3, 0 * self.generateProcessNoise())
            # z3 = self.OutputEqn(0, x3, [u3], 1 * self.generateSensorNoise())
            # self.B3_baseline = np.reshape(x3, (8))

            x4 = self.StateEqn(0, x4, u4, 0 * self.generateProcessNoise())
            # z4 = self.OutputEqn(0, x4, [u4], 1 * self.generateSensorNoise())
            # self.B4_baseline = np.reshape(x4, (8))


        z1 = self.OutputEqn(0, x1, [u1], 0 * self.generateSensorNoise())
        z2 = self.OutputEqn(0, x2, [u2], 0 * self.generateSensorNoise())
        z3 = self.OutputEqn(0, x3, [u3], 0 * self.generateSensorNoise())
        z4 = self.OutputEqn(0, x4, [u4], 0 * self.generateSensorNoise())
        self.B1_baseline = np.reshape(x1, (8))
        self.B2_baseline = np.reshape(x2, (8))
        self.B3_baseline = np.reshape(x3, (8))
        self.B4_baseline = np.reshape(x4, (8))

        s_baseline = [z1[1][0], z2[1][0], z3[1][0], z4[1][0], load_demand]

        if z1[1][0] <= 3 or z2[1][0] <= 3 or z3[1][0] <= 3 or z4[1][0] <= 3:
            done = True

        return s_baseline, done



def test_code():
    # Create battery model
    B_test = Battery()
    B_test.reset()
    # Simulate for default conditions
    T1, X1, U1, Z1 = B_test.simulate(3150, 1e10)
    # Determine end-of-discharge time for these conditions
    T,_,_,_ = B_test.simulateToThreshold()
    print("EOD time is : ",T[-1]," s")


    # Simulate for a variable load profile
    # Specified by a sequence of pairs of numbers, where the first is the load
    # (in Watts) and the second is the duration (in seconds).
    B_test.reset()
    loads = np.array([[8], [10*60], [4], [5*60], [12], [15*60], [5], [20*60], [10], [10*60]])
    T2, X2, U2, Z2 = B_test.simulate(3150, 60,input=loads)
    T,_,_,_ = B_test.simulateToThreshold(input=loads)
    # Determine end-of-discharge time for these conditions


    # Plot
    print("EOD time is : ", T[-1], " s")
    plt.subplot(211)
    plt.title("Tbm")
    x = np.linspace(0, np.shape(T2)[0] - 1, np.shape(T2)[0])
    plt.xlim(0,3150)
    plt.ylim(18, max(Z2[0])+0.01)
    plt.plot(x, Z2[0], color='red',linestyle='-')
    plt.plot(x, Z1[0], color='blue', linestyle='-')
    plt.subplot(212)
    plt.title("Vm")
    plt.xlim(0, 3150)
    plt.ylim(min(Z2[1]), max(Z2[1]))
    plt.plot(x, Z2[1], color='red')
    plt.plot(x, Z1[1], color='blue')
    plt.show()


def RL_test():
    # Create battery model
    battery = Battery_Management()
    battery.reset()
    Z1=[18.95]
    Z2=[4.19135025]
    for i in range(3150):
        action = [[8]]
        r,s_,done,z=battery.step(action)
    
        Z1.append(z[0][0].tolist())
        Z2.append(z[1][0].tolist())
    
    battery.reset()
    # Simulate for default conditions
    T1, X1, U1, Z3 = battery.simulate(3150, 1e10)
    print(Z3[1]-Z2)
    
    x = np.linspace(0, np.shape(Z1)[0] - 1, np.shape(Z1)[0])
    plt.subplot(211)
    plt.title("Tbm")
    plt.xlim(0, 3150)
    plt.ylim(18, max(Z1) + 0.01)
    plt.plot(x, Z1, color='red', linestyle='-')
    plt.plot(x, Z3[0], color='blue', linestyle='-')
    plt.subplot(212)
    plt.title("Vm")
    plt.xlim(0, 3150)
    plt.ylim(min(Z2), max(Z2))
    plt.plot(x, Z3[1], color='red')
    plt.plot(x, Z2, color='blue')
    plt.show()

if __name__ == '__main__':
    test_code()
# RL_test()
# #
# #
# #
# #
