'''
Created on May 1, 2021

@author: alienware
'''
import numpy as np

def Fn(SN,j,S,Px,Psx,P):
    """
    Recursively calculate:
    Fn(j):= Pr(Sn = sn, Xn = j) 
    where sn is a vector of observations
    -------------------Variables--------------------
    @SN: A list of the Observed states 
    @j: the state of the hidden Markov Chain
    @S: the number of states of {Xn}n
    @Px: Transition matrix of hidden Markov Chain (Xn)n (Known)
    @Psx: Matrix where Pxs_ij representing Pr(Sn =  i | Xn = j)
    @P: A vector representing the random distribution of the initial state of (Xn)1 
    """
    #Base Case:
    assert len(P) == S, 'Initial Distribution of Xn must equal to its number of states S'
    assert Px.shape[0] == S, 'The Transition matrix must be nxn matrix where n is equal to S, the number of states'
    assert all( np.array([sum(Psx[:,i]) for i in range(Psx.shape[1])]) == 1 ), 'The columns of Psx must add up to 1'
    n = SN[-1]
    if len(SN) == 1:
        return P[j]*Psx[n,j]
    else:
        SN_next = list(SN)
        SN_next.pop(-1)
        SUM = sum([Fn(SN_next,i,S,Px,Psx,P) * Px[i,j] for i in range(S)])
        return Psx[n,j]*SUM
    
def Pr_Xn_j (j,SN,S,Px,Psx,P):
    "Calculate the Probability of Xn in state j given SN (n = N)"
    num = Fn(SN,j,S,Px,Psx,P)
    denom = sum([Fn(SN,i,S,Px,Psx,P) for i in range(S)])
    return num/denom
    
def P_X_next(i,SN,S,Px,Psx,P): 
    "Calculate the Probability of Xn+1 in state i given SN (n = N)"
    temp = [Px[j,i]*Pr_Xn_j(j, SN, S, Px, Psx, P) for j in range(S)]
    return sum(temp)

def P_S_next(i,SN,S,Px,Psx,P):
    "Calculate the Probability of Sn in state i given SN (n = N)"
    temp = [Psx[i,j]*P_X_next(j, SN, S, Px, Psx, P) for j in range(S)]
    return sum(temp)
    
if __name__ == '__main__':
    #Initialization
    S = 2
    Px = np.array([0.9,0.1,0,1]).reshape(2,2)
    Psx = np.array([0.01,0.04,0.99,0.96]).reshape(2,2)
    P = [0.8,0.2]
    # #Checking Base case
    # SN = [1]
    # print("Start")
    # print(Fn(SN,0,S,Px,Psx,P))
    # print(Fn(SN,1,S,Px,Psx,P))
    # #Check Next Phase:
    # SN = [1,0]
    # print(Fn(SN,0,S,Px,Psx,P))
    # print(Fn(SN,1,S,Px,Psx,P))
    #Check Next Phase, n = 3:
    SN = [1,0,1] #0 means good,1 means defect
    # print(Fn(SN,0,S,Px,Psx,P))
    # print(Fn(SN,1,S,Px,Psx,P))
    #Checking Pr_Xn_j:
    print(f"Probability that Xn is in State 0: i.e, Good: {Pr_Xn_j(0,SN,S,Px,Psx,P)},\n Given the observation being made:{SN}") 
    print(f"Probability that Xn is in State 1: i.e, Bad: {Pr_Xn_j(1,SN,S,Px,Psx,P)},\n Given the observation being made:{SN}")
    print(f"pr(X{len(SN)+1} = 0|S{len(SN)} = {SN}) = {P_X_next(0,SN,S,Px,Psx,P)}") 
    print(f"pr(s{len(SN)+1} = 1|S{len(SN)} = {SN}) = {P_S_next(1,SN,S,Px,Psx,P)}")
    