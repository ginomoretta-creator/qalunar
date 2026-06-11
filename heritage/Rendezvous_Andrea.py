import matplotlib.pyplot as plt
import minorminer
import dimod
import time
import dwave_networkx as dnx

from neal  import SimulatedAnnealingSampler
from dwave.system import DWaveSampler, FixedEmbeddingComposite
from dwave.preprocessing import ScaleComposite

import numpy as np
# from scipy.linalg import lstsq
# import time
from scipy.optimize import lsq_linear

###############################################################################
###############################################################################
###############################################################################
def act(x, w, b, type_activation):
    z = w * x + b
    
    if type_activation == 1:  # Logistic
        h = 1 / (1 + np.exp(-z))
        hd = h * (1 - h) * w
        hdd = hd * (1 - 2 * h) * w
    elif type_activation == 2:  # Tanh
        h = np.tanh(z)
        hd = w * (1 - h**2)
        hdd = -2 * w**2 * h * (1 - h**2)
    elif type_activation == 3:  # Sine
        h = np.sin(z)
        hd = w * np.cos(z)
        hdd = -w**2 * np.sin(z)
    elif type_activation == 4:  # Cosine
        h = np.cos(z)
        hd = -w * np.sin(z)
        hdd = -w**2 * np.cos(z)
    elif type_activation == 5:  # Gaussian
        h = np.exp(-z**2)
        hd = -2 * w * z * h
        hdd = -2 * w**2 * h * (1 - 2 * z**2)
    elif type_activation == 6:  # ArcTan
        h = np.arctan(z)
        hd = w / (1 + z**2)
        hdd = -2 * w**2 * z / (1 + z**2)**2
    elif type_activation == 7:  # Sinh
        h = np.sinh(z)
        hd = w * np.cosh(z)
        hdd = w**2 * np.sinh(z)
    elif type_activation == 8:  # SoftPlus
        h = np.log1p(np.exp(z))
        hd = w * (1 / (1 + np.exp(-z)))
        hdd = hd * (1 - hd / w)
    elif type_activation == 9:  # Bent Identity
        h = (np.sqrt(z**2 + 1) - 1)/2 + z
        hd = (z / (2 * np.sqrt(z**2 + 1)) + 1) * w
        hdd = (w**2 / (2 * np.sqrt(z**2 + 1)**3))
    elif type_activation == 10:  # asinh
        h = np.arcsinh(z)
        hd = w / np.sqrt(z**2 + 1)
        hdd = -w**2 * z / (z**2 + 1)**1.5
    elif type_activation == 11:  # Softsign
        h = z / (1 + np.abs(z))
        hd = w / (1 + np.abs(z))**2
        hdd = -2 * w**2 * np.sign(z) / (1 + np.abs(z))**3
    else:
        raise ValueError("Unknown activation type.")
    
    return h, hd, hdd

###############################################################################
def Bianrization(j0, nBit, nVar, signed: bool = True, mode: str = "twos"): 
    
    base_single = np.array([2.0**(j0 - j) for j in range(1, nBit + 1)], dtype=float)
   
    if signed:

        if mode.lower() == "twos":
            sign_weight = -2.0**(j0)
        elif mode.lower() == "ones":
            sign_weight = -2.0**(j0) + 2.0**(j0 - nBit + 1)
        else:
            raise ValueError("mode deve essere 'twos' o 'ones'.")
        
        base_single = np.concatenate(([sign_weight], base_single))        
        
    I = np.eye(nVar)
    B = np.kron(I, base_single)

    return B
    
#==============================================================================
def QUBO_sol(A, b, j0, nBit):    
    
    K = Bianrization(j0,nBit,A.shape[1])    
    
    Q = (K.T @ (A.T @ A) @ K)
    L = ( -2*(b.T @ A) @ K).squeeze()

    Q[np.abs(Q) < 1e-9] = 0.0
    Q[np.abs(Q) > 1e+9] = 0.0
    L[np.abs(L) < 1e-9] = 0.0
    L[np.abs(L) > 1e+9] = 0.0
  
    Q += np.diag(L)    
    Q = np.triu(Q) + np.triu(Q,1)

    bqm = dimod.BinaryQuadraticModel(Q,'BINARY')  
    SQA = SimulatedAnnealingSampler()   
    res = SQA.sample(bqm,num_reads=10, seed=42) # mettere quella con Xp e Up 
    
    q = np.array(list(res.first.sample.items()))[:,1:]
    
    xi = K @ q
    
    # IF YOU WANT TRY TO IMPLEMENT ON REAL QUBO
    # n = np.ceil(np.sqrt( (Q!=0).sum() / 24) /2 )
    # bqm_graph = dimod.to_networkx_graph(bqm)
    # target_graph = dnx.pegasus_graph(int(n)) 
    # embedding = minorminer.find_embedding(bqm_graph, target_graph)
    
    
    return xi

###############################################################################
###############################################################################
###############################################################################

np.random.seed(0)

if __name__ == '__main__':
    
    #==========================================================================
    # problem fixed inputs ====================================================
    #==========================================================================
    mu     = 3.986004418*(10**14) #  [m^3/s^2] - gravitational parameter
    r_csp  = 7500*(10**3)         #  [m]       - chief satellite position
    
    ToF = 2000  # [s] - time of flight 
    
    # initial states
    r0 = np.array([ 7047,  5136, 5013 ]) # [m] - position 
    v0 = np.array([ -2.4, -13.7, 4.08 ]) # [m/s] - velocity 
    
    # final states
    rf = np.array([ 0, 0, 0 ]) # [m] - position
    vf = np.array([ 0, 0, 0 ]) # [m/s] - velocity
    
    #--------------------------------------------------------------------------
    w  = np.sqrt(mu/(r_csp**3))   #  [1/s] - angular velocity of the chief satellite
    
    M  = np.array([ [3*w**2, 0, 0] , [0, 0, 0],  [0, 0, -w**2] ]) * (1/w**2)
    MT = M.T
     
    N  = np.array([ [0, 2*w, 0] , [-2*w, 0, 0],  [0, 0, 0] ])  * (1/w)
    NT = N.T
    
    #--------------------------------------------------------------------------
    # Normalization
    
    ToF = ToF*w  # [s] - time of flight 
    
    r0 = r0/r_csp      # [m] - position 
    v0 = v0/(r_csp*w); # [m/s] - velocity 
    
    rf = rf/r_csp                # [m] - position
    vf = vf/(r_csp*w)            # [m/s] - velocity
    
    #==========================================================================
    # Neural Network Input ====================================================
    #==========================================================================   
    type_trainingPoints = 1 # 1 = linspace, 2 = random (unif. dist.) 
    type_act = 5            # choose activation (TanH)    
   
    QUBO = 1
        
    n  = 5000  # training points 20, 30 
    L  = 5     # number of neurons 80, 50
    nt = 200   # test points
    Ub =  10
    Lb = -10
    
    nBit = 8   # number of bits
    
    #--------------------------------------------------------------------------
    # define training points
    
    t0, tf =  0, ToF    
    z0, zf = -1, 1
    
    if type_trainingPoints == 1:
        z = np.linspace(z0,zf,n)[:,np.newaxis]
    elif type_trainingPoints == 2:
        z =  np.random.uniform(z0,zf,n)
        z[0], z[-1] = z0, zf
    else:
        print(f'Warinign type_trainingPoints = {type_trainingPoints}, not valid value. Default 1 implemented!')
        z = np.linspace(z0,zf,n)
    
    
    c  = ( z[-1] - z[0] )/ (tf - t0)  # mapping coeff.
    c2 = c**2; 
    t  = t0 + (1/c)*( z - z[0] )      # t span
    
    #--------------------------------------------------------------------------
    # constraints    
    (r01, r02, r03), (v01, v02, v03) = r0 , v0
    (rf1, rf2, rf3), (vf1, vf2, vf3) = rf, vf    
            
    #--------------------------------------------------------------------------
    # switchig functions 
    dz, dz2, dz3 = (z[-1] - z[0]) , (z[-1] - z[0])**2, (z[-1] - z[0])**3
    
    om1 = 1 + 2*(z - z[0])**3/dz3 - 3*(z - z[0])**2/dz2
    om2 = -2*(z - z[0])**3/dz3 + 3*(z - z[0])**2/dz2
    om3 = (z - z[0]) + (z - z[0])**3/dz2 - 2*(z - z[0])**2/dz
    om4   = (z - z[0])**3/dz2 - (z - z[0])**2/dz
    
    om1d = 6*(z - z[0])**2/dz3 - 6*(z - z[0])/dz2
    om2d = -6*(z - z[0])**2/dz3 + 6*(z - z[0])/dz2
    om3d = 1 + 3*(z - z[0])**2/dz2 - 4*(z - z[0])/dz
    om4d = 3*(z - z[0])**2/dz2 - 2*(z - z[0])/dz
    
    om1dd = 12*(z - z[0])/dz3 - 6/dz2
    om2dd = -12*(z - z[0])/dz3 + 6/dz2
    om3dd = 6*(z - z[0])/dz2 - 4/dz
    om4dd = 6*(z - z[0])/dz2 - 2/dz    
    
    #--------------------------------------------------------------------------
    # compute activation functions
    weight = np.random.uniform(Lb, Ub, (L,1))
    bias   = np.random.uniform(Lb, Ub, (L,1))
    
    #--------------------------------------------------------------------------
    # training
    h   = np.zeros((n,L))
    hd  = np.zeros((n,L))
    hdd = np.zeros((n,L))    
    for i in range(n):
        for j in range(L):
            h[i,j], hd[i,j], hdd[i,j] = act(z[i], weight[j], bias[j], type_act)
    
    #--------------------------------------------------------------------------
    # extract boundary values
    h0  = h[0,:]
    hf  = h[-1,:]
    hd0 = hd[0,:]
    hdf = hd[-1,:]
        
    # states
    F   = h  - om1*h0 - om2*hf - om3*hd0 - om4*hdf
    Fd  = c  * (hd - om1d*h0 - om2d*hf - om3d*hd0 - om4d*hdf)
    Fdd = c2 * (hdd - om1dd*h0 - om2dd*hf - om3dd*hd0 - om4dd*hdf)
    
    # r1
    C1   = om1*r01 + om2*rf1 + (1/c)*(om3*v01 + om4*vf1)
    C1d  = c*(om1d*r01 + om2d*rf1 + (1/c)*(om3d*v01 + om4d*vf1))
    C1dd = c2*(om1dd*r01 + om2dd*rf1 + (1/c)*(om3dd*v01 + om4dd*vf1))
    
    # r2
    C2   = om1*r02 + om2*rf2 + (1/c)*(om3*v02 + om4*vf2)
    C2d  = c*(om1d*r02 + om2d*rf2 + (1/c)*(om3d*v02 + om4d*vf2))
    C2dd = c2*(om1dd*r02 + om2dd*rf2 + (1/c)*(om3dd*v02 + om4dd*vf2))
    
    # r3
    C3   = om1*r03 + om2*rf3 + (1/c)*(om3*v03 + om4*vf3)
    C3d  = c*(om1d*r03 + om2d*rf3 + (1/c)*(om3d*v03 + om4d*vf3))
    C3dd = c2*(om1dd*r03 + om2dd*rf3 + (1/c)*(om3dd*v03 + om4dd*vf3))
    
    # costates
    H  = h
    Hd = c*hd
    
    # Least Squares

    z0 = np.zeros((n,L))
    zz = np.zeros((6*n,1))
    
    A1 = np.vstack([
        np.hstack([Fdd-M[0,0]*F, -N[0,1]*Fd, z0, z0, z0, z0]),
        np.hstack([-N[1,0]*Fd, Fdd, z0, z0, z0, z0]),
        np.hstack([z0, z0, Fdd-M[2,2]*F, z0, z0, z0]),
        np.hstack([z0, z0, z0, Hd, z0, z0]),
        np.hstack([z0, z0, z0, z0, Hd, z0]),
        np.hstack([z0, z0, z0, z0, z0, Hd]),
        np.hstack([z0, z0, z0, H, z0, z0]),
        np.hstack([z0, z0, z0, z0, H, z0]),
        np.hstack([z0, z0, z0, z0, z0, H])
    ])
    
    A2 = np.vstack([
        np.hstack([H, z0, z0]),
        np.hstack([z0, H, z0]),
        np.hstack([z0, z0, H]),
        np.hstack([MT[0,0]*H, z0, z0]),
        np.hstack([z0, z0, z0]),
        np.hstack([z0, z0, MT[2,2]*H]),
        np.hstack([Hd, NT[0,1]*H, z0]),
        np.hstack([NT[1,0]*H, Hd, z0]),
        np.hstack([z0, z0, Hd])
    ])
    
    A = np.hstack((A1, A2))
    
    B = np.vstack([
        -C1dd + 3*C1 + 2*C2d,
        -C2dd - 2*C1d,
        -C3dd - C3,
        zz
    ]) 
       
    #-------------------------------------------------------------------------- 
    # The QUBO solution appears to perform better in terms of the cost function, 
    # as the resulting control follows a smoother and cleaner trajectory.
    #
    # However, a key difference is that in the QUBO formulation the solution 
    # must satisfy specific constraints, since it is defined over a discrete domain. 
    # In contrast, the classical approach allows the optimization variables 
    # ("res") to take values over the entire continuous real domain.
    
    if QUBO:
        j0 = int(np.ceil(np.log2(0.01)))
        xi = QUBO_sol(A, B, j0, nBit)
    else:
        res = lsq_linear(A, B.squeeze(), bounds=(Lb,Ub)) # with boundaries 
        # res = lsq_linear(A, B.squeeze())
        xi = res.x[:,np.newaxis]  
        
    loss = A @ xi - B
    absL = np.abs(loss)

    # extract xi
    xis1 = xi[0:L]
    xis2 = xi[L:2*L]
    xis3 = xi[2*L:3*L]
    
    xicr1 = xi[3*L:4*L]
    xicr2 = xi[4*L:5*L]
    xicr3 = xi[5*L:6*L]
    
    xicv1 = xi[6*L:7*L]
    xicv2 = xi[7*L:8*L]
    xicv3 = xi[8*L:9*L]
    
    # build CE
    r1 = F @ xis1 + C1
    v1 = Fd @ xis1 + C1d
    a1 = Fdd @ xis1 + C1dd
    
    r2 = F @ xis2 + C2
    v2 = Fd @ xis2 + C2d
    a2 = Fdd @ xis2 + C2dd
    
    r3 = F @ xis3 + C3
    v3 = Fd @ xis3 + C3d
    a3 = Fdd @ xis3 + C3dd
    
    cr1 = H @ xicr1
    cr2 = H @ xicr2
    cr3 = H @ xicr3
    
    cv1 = H @ xicv1
    cv2 = H @ xicv2
    cv3 = H @ xicv3
    
    # Hamiltonian
    r  = np.column_stack((r1,r2,r3))
    v  = np.column_stack((v1,v2,v3))
    cr = np.column_stack((cr1,cr2,cr3))
    cv = np.column_stack((cv1,cv2,cv3))
    
    ham = np.zeros(n)
    
    for i in range(n):
        ham[i] = (
            -0.5 * (cv[i] @ cv[i])
            + cr[i] @ v[i]
            + cv[i] @ (M @ r[i] + N @ v[i])
        )
    
    ham = np.abs(ham)
    
    # unnormalized quantities
    r1 *= r_csp
    r2 *= r_csp
    r3 *= r_csp
    
    v1 *= r_csp*w
    v2 *= r_csp*w
    v3 *= r_csp*w
    
    a1 *= r_csp*w**2
    a2 *= r_csp*w**2
    a3 *= r_csp*w**2
    
    cr1 *= r_csp*w**3
    cr2 *= r_csp*w**3
    cr3 *= r_csp*w**3
    
    cv1 *= r_csp*w**2
    cv2 *= r_csp*w**2
    cv3 *= r_csp*w**2
    
    # cost
    J = np.sum((cv1[:-1,:]**2 + cv2[:-1,:]**2 + cv3[:-1,:]**2) * np.diff(t/w,axis=0) )/2
    
    # time rescale
    t = t/w
    
    print("***************************************************************")
    print(f"TRAINING {'QUBO'*QUBO}")
    print("***************************************************************")
    
    print(f"The number of training points is {n}")
    print(f"The number of neurons is {L}")    
    print(f"Cost Function {J}")

    print("***************************************************************")
    print(f"The max abs. loss is {np.max(absL)}")
    print(f"The mean abs. loss is {np.mean(absL)}")
    print(f"The std abs. loss is {np.std(absL)}")
    print(f"The norm abs. loss is {np.linalg.norm(absL)}")
    
    print("***************************************************************")
    print(f"The abs. Hamiltonian at final time is {ham[-1]}")
    print(f"The mean abs. Hamiltonian is {np.mean(ham)}")
    print(f"The std abs. Hamiltonian is {np.std(ham)}")
    
    
    # constraint errors
    err_r10 = r1[0] - r01*r_csp
    err_r20 = r2[0] - r02*r_csp
    err_r30 = r3[0] - r03*r_csp
    
    err_r1f = r1[-1] - rf1*r_csp
    err_r2f = r2[-1] - rf2*r_csp
    err_r3f = r3[-1] - rf3*r_csp
    
    err_v10 = v1[0] - v01*r_csp*w
    err_v20 = v2[0] - v02*r_csp*w
    err_v30 = v3[0] - v03*r_csp*w
    
    err_v1f = v1[-1] - vf1*r_csp*w
    err_v2f = v2[-1] - vf2*r_csp*w
    err_v3f = v3[-1] - vf3*r_csp*w
    
    err_constraints = np.array([
        err_r10,err_r20,err_r30,
        err_v10,err_v20,err_v30,
        err_r1f,err_r2f,err_r3f,
        err_v1f,err_v2f,err_v3f
    ])
    
    print("***************************************************************")
    print(f"Mean constraint error {np.mean(err_constraints)}")
    print(f"Std constraint error {np.std(err_constraints)}")
    
    # # plots
    # plt.figure()
    # plt.plot(t,r1,label="r1")
    # plt.plot(t,r2,label="r2")
    # plt.plot(t,r3,label="r3")
    # plt.grid()
    # plt.legend()
    # plt.xlabel("t")
    # plt.title("Position-Time history")
    
    # plt.figure()
    # plt.plot(t,v1,label="v1")
    # plt.plot(t,v2,label="v2")
    # plt.plot(t,v3,label="v3")
    # plt.grid()
    # plt.legend()
    # plt.xlabel("t")
    # plt.title("Velocity-Time history")
    
    # plt.figure()
    # plt.plot(t,a1,label="a1")
    # plt.plot(t,a2,label="a2")
    # plt.plot(t,a3,label="a3")
    # plt.grid()
    # plt.legend()
    # plt.xlabel("t")
    # plt.title("Acceleration-Time history")
    
    plt.figure()
    plt.plot(t,-cv1,label="u1")
    plt.plot(t,-cv2,label="u2")
    plt.plot(t,-cv3,label="u3")
    plt.grid()
    plt.legend()
    plt.xlabel("t")
    plt.title("Control-Time history")
    
    # plt.figure()
    # plt.plot(t,cr1,label="cr1")
    # plt.plot(t,cr2,label="cr2")
    # plt.plot(t,cr3,label="cr3")
    # plt.grid()
    # plt.legend()
    # plt.xlabel("t")
    # plt.title("Costate Position-Time history")
    
    # plt.figure()
    # plt.plot(t,cv1,label="cv1")
    # plt.plot(t,cv2,label="cv2")
    # plt.plot(t,cv3,label="cv3")
    # plt.grid()
    # plt.legend()
    # plt.xlabel("t")
    # plt.title("Costate Velocity-Time history")
    
    fig = plt.figure()
    ax = fig.add_subplot(111,projection="3d")
    ax.plot(r1,r2,r3)
    ax.set_xlabel("r1")
    ax.set_ylabel("r2")
    ax.set_zlabel("r3")
    ax.set_title("Trajectories")    
    ax.view_init(elev=25, azim=-130) 
    plt.show()