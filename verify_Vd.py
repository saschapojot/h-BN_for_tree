from sympy import *
import random
import numpy as np



r00,r01,r02=symbols("r00,r01,r02",cls=Symbol,real=True)
r10,r11,r12=symbols("r10,r11,r12",cls=Symbol,real=True)
r20,r21,r22=symbols("r20,r21,r22",cls=Symbol,real=True)
# Set a seed for reproducibility (optional)
random.seed(42)

half = Rational(1,2)
quarter = Rational(1,4)



# Generate a random orthogonal matrix using QR decomposition
np.random.seed(42)
random_matrix = np.random.randn(3, 3)
R_numpy, _ = np.linalg.qr(random_matrix)

# Convert to SymPy
# r00, r01, r02 = R_numpy[0, :]
# r10, r11, r12 = R_numpy[1, :]
# r20, r21, r22 = R_numpy[2, :]

# Convert to SymPy for symbolic computation
# r00, r01, r02 = Float(r00), Float(r01), Float(r02)
# r10, r11, r12 = Float(r10), Float(r11), Float(r12)
# r20, r21, r22 = Float(r20), Float(r21), Float(r22)



alpha=pi/3
r00=cos(alpha)
r01=-sin(alpha)
r10=sin(alpha)
r11=cos(alpha)
r22=1

r02=0
r12=0
r20=0
r21=0


N0=sqrt(15/(4*pi))

N1=sqrt(15/(4*pi))

N2=sqrt(15/(4*pi))

N3=sqrt(15/(16*pi))

N4=sqrt(5/(16*pi))

# u00,u10,u20,u30,u40=symbols("u00,u10,u20,u30,u40",cls=Symbol)

u00=r00*r11+r01*r10
u10=r10*r21+r11*r20
u20=r00*r21+r01*r20
u30=-r20*r21-2*r10*r11
u40=sqrt(3)*r20*r21

u01=r01*r12+r02*r11
u11=r11*r22+r12*r21
u21=r01*r22+r02*r21
u31=-r21*r22-2*r11*r12
u41=sqrt(3)*r21*r22

u02=r00*r12+r02*r10
u12=r10*r22+r12*r20
u22=r00*r22+r02*r20
u32=-r20*r22-2*r10*r12
u42=sqrt(3)*r20*r22

u03=r00*r10-r01*r11
u13=r10*r20-r11*r21
u23=r00*r20-r01*r21
u33=-half*r20**2+half*r21**2-r10**2+r11**2
u43=sqrt(3)/2*(r20**2-r21**2)


u04=-1/sqrt(3)*r00*r10-1/sqrt(3)*r01*r11+2/sqrt(3)*r02*r12
u14=-1/sqrt(3)*r10*r20-1/sqrt(3)*r11*r21+2/sqrt(3)*r12*r22

u24=-1/sqrt(3)*r00*r20-1/sqrt(3)*r01*r21+2/sqrt(3)*r02*r22
u34=1/(2*sqrt(3))*(r20**2+r21**2)-1/sqrt(3)*r22**2\
    +1/sqrt(3)*(r10**2+r11**2)-2/sqrt(3)*r12**2

u44=-half*r20**2-half*r21**2+r22**2


Vd=Matrix([
    [u00,u01,u02,u03,u04],
    [u10,u11,u12,u13,u14],
    [u20,u21,u22,u23,u24],
    [u30,u31,u32,u33,u34],
    [u40,u41,u42,u43,u44]
])
pprint(Vd)
# print("Vd * Vd.T:")
# pprint((Vd * Vd.T))


def GetSymD(R):
    R_11, R_12, R_13 = R[0,0], R[0,1], R[0,2]
    R_21, R_22, R_23 = R[1,0], R[1,1], R[1,2]
    R_31, R_32, R_33 = R[2,0], R[2,1], R[2,2]
    RD = zeros(5,5)
    sr3 = sqrt(3)
    #
    RD[0,0] = R_11*R_22+R_12*R_21
    RD[0,1] = R_21*R_32+R_22*R_31
    RD[0,2] = R_11*R_32+R_12*R_31
    RD[0,3] = 2*R_11*R_12+R_31*R_32
    RD[0,4] = sr3*R_31*R_32
    #
    RD[1,0] = R_12*R_23+R_13*R_22
    RD[1,1] = R_22*R_33+R_23*R_32
    RD[1,2] = R_12*R_33+R_13*R_32
    RD[1,3] = 2*R_12*R_13+R_32*R_33
    RD[1,4] = sr3*R_32*R_33
    #
    RD[2,0] = R_11*R_23+R_13*R_21
    RD[2,1] = R_21*R_33+R_23*R_31
    RD[2,2] = R_11*R_33+R_13*R_31
    RD[2,3] = 2*R_11*R_13+R_31*R_33
    RD[2,4] = sr3*R_31*R_33
    #
    RD[3,0] = R_11*R_21-R_12*R_22
    RD[3,1] = R_21*R_31-R_22*R_32
    RD[3,2] = R_11*R_31-R_12*R_32
    RD[3,3] = (R_11**2-R_12**2 )+1/2*(R_31**2-R_32**2 )
    RD[3,4] = sr3/2*(R_31**2-R_32**2 )
    #
    RD[4,0] = 1/sr3*(2*R_13*R_23-R_11*R_21-R_12*R_22)
    RD[4,1] = 1/sr3*(2*R_23*R_33-R_21*R_31-R_22*R_32)
    RD[4,2] = 1/sr3*(2*R_13*R_33-R_11*R_31-R_12*R_32)
    RD[4,3] = 1/sr3*(2*R_13**2-R_11**2-R_12**2 )+1/sr3/2*(2*R_33**2-R_31**2-R_32**2 )
    RD[4,4] = 1/2*(2*R_33**2-R_31**2-R_32**2 )

    return RD.T
R=Matrix([[r00,r01,r02],
          [r10,r11,r12],
          [r20,r21,r22]
          ])

pprint(GetSymD(R))