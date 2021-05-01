'''
Created on Apr 20, 2021

@author: alienware
'''
import numpy as np
def Gam_Ruin_P(p,mn,r1 = None,rn = None):
    OUT = np.zeros((mn+1,mn+1))
    for row in range(mn):
        for col in range(mn):
            if 0 < row < mn:
                if row-1 == col:
                    OUT[row,col] = (1-p) 
                elif row + 1 == col:
                    OUT[row,col] = p
    if r1 is None:
        OUT[0,0] = 1
    else:
        OUT[0,:] = r1
    if rn is None:
        OUT[-1,-1] = 1
    else:
        OUT[-1,:] = rn
    return OUT

def PT(Transition_Matrix):
    return Transition_Matrix[1:-1,1:-1]

def Get_S(Pt):
    "Mean time spent in transient state"
    I = np.eye(Pt.shape[0])
    return np.linalg.inv((I - Pt))

def f_ij(i,j,S):
    "Probability that j will ever be reached from i"
    i -= 1
    j -= 1
    kd_ij = (1 if i == j else 0)
    return (S[i,j] - kd_ij)/S[j,j]

def F_matrix(S):
    F = np.zeros(S.shape)
    for i in range(F.shape[0]):
        for j in range(F.shape[1]):
            F[i,j] = f_ij(i, j, S)
    return F
    

if __name__ == '__main__':
    Pt = PT(Gam_Ruin_P(0.5, 12))
    S = Get_S(Pt)
    P = np.ones((5,5))
    P = 1/6*P 
    for row in range(5):
        for col in range(5):
            if row == col:
                P[row,col] = (row+1)/6
            elif row > col:
                P[row,col] = 0
    S = Get_S(P)
    print(S)
    # print(S[3-1,2-1])
    # print(S[3-1,5-1])
    #print(f_ij(3, 1, S))
    #print(F_matrix(S))
    