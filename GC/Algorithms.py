# -*- coding: utf-8 -*-

"""
@author: Sushant Kuchankar
2018
Dept: UG

Content
 
1) VAR Estimator:
    With QR factorization by Numpy least sq solver
    With Moore–Penrose inverse Cholesky factorization by numpy Moore–Penrose inverse solver
    
2) VAR Model Selection:
     Akaike information criterion
     Bayesian information criterion
     final prediction error (FPE) criterion

3) Wald Test for Granger-Causality:
    F-random variable estimation
    Hypothesis test for Granger-Causality
    
4) Data organization functions

5) Matrix Vectorization funtion
    
    
"""
#Imports
import scipy.io as sci
from scipy.stats import f
import statsmodels.tsa.stattools as stat
import numpy as np


"""

Notation Reference 
Y := (y1, . . . , yT)        (K × T-p),
B := (ν, A1, . . . , Ap)     (K×(Kp + 1)),
Zt :=⎡ 1 yt...  yt−p+1]       ((Kp + 1) × 1),
Z := (Z0, . . . , ZT −1)     ((Kp + 1) × T-p),

Y = BZ + U
Here K is the dimension of the vector
We will Use Least Sqaure solution of the above eqaution
to obtain VAR

"""

def Organise(data,lag):
    #input data:= K*T
    p = lag
    data =  np.asarray(data)
    K,T = np.shape(data)
    #######################Y########################
    Y = data[:,lag].reshape((K,1)) 
    for i in range(lag+1,T):
        Y = np.hstack((Y,data[:,i].reshape((K,1))))
        
    #######################Z#######################
    
    for i in range(p,T):
        for j in range(i-p,i):
            if(j == i-p):
                Zt = data[:,j].reshape((K,1))
            else:
                Zt = np.vstack((Zt,data[:,j].reshape((K,1))))
        if(i == p):
            Z = Zt
        else:
            Z = np.hstack((Z,Zt))
        del Zt
    n = np.shape(Z)[1]
    Z = np.vstack( (np.ones((1,n)),Z) )
    
    ########lets do the sanity check before going ahead####################
    if (K*p+1,T-p) != np.shape(Z):
        raise "Z shape invalide"
    if (K,T-p) != np.shape(Y):
        raise "Y shape invalide"

    return Y,Z

def Organise2(data,lag):
    try:
        trailnumber = np.shape(data)[2]
        v=0
    except IndexError:
        v=1
        
    if(v==1):
        Y,Z = Organise(data,lag)
        return Y,Z 
    
    else:
        for i in range(0,trailnumber):
            if i ==0:
                Y,Z = Organise(data[:,:,i],lag)
            else:
                Y0,Z0 = Organise(data[:,:,i],lag)
                Y = np.hstack((Y,Y0))
                Z = np.hstack((Z,Z0))
        return Y,Z
    
def Var_fit(data,lag):
    
    Y,Z = Organise2(data,lag)
    YZdash = np.matmul(Y,Z.T)
    invZZdash = np.linalg.pinv(np.matmul(Z,Z.T))
    B = np.matmul(YZdash,invZZdash)
    
    return B

def Var_fit_LS(data,lag):
    
    Y,Z = Organise2(data,lag)
    B = np.linalg.lstsq(np.matmul(Z,Z.T),np.matmul(Z,Y.T))
    return B

def AIC(data,B,lag):
    Y,Z = Organise2(data,lag)
    k = np.shape(Y)[0]
    if(len(np.shape(data))>2):
        x = np.shape(data)
        t = x[1]*x[2]
    else:
        t = np.shape(data)[1]
    Y_BZ = Y - np.matmul(B,Z) 
    sigma = np.matmul(Y_BZ,Y_BZ.T)/t
    aic = np.log(np.linalg.det(sigma)) + 2*lag*( (k)**2 )/t
    return aic

def FPE(data,B,lag):
    Y,Z = Organise2(data,lag)
    k = np.shape(Y)[0]
    if(len(np.shape(data))>2):
        x = np.shape(data)
        t = x[1]*x[2]
    else:
        t = np.shape(data)[1]
    Y_BZ = Y - np.matmul(B,Z) 
    sigma = np.matmul(Y_BZ,Y_BZ.T)/t
    fpe = (((t+k*lag+1)/(t-k*lag-1))**k)*np.linalg.det(sigma)
    return fpe
   
def BIC(data,B,lag):
    Y,Z = Organise2(data,lag)
    k = np.shape(Y)[0]
    if(len(np.shape(data))>2):
        x = np.shape(data)
        t = x[1]*x[2]
    else:
        t = np.shape(data)[1]
    Y_BZ = Y - np.matmul(B,Z) 
    sigma = np.matmul(Y_BZ,Y_BZ.T)/t
    bic = 2*np.log(np.linalg.det(sigma)) + 2*((k)**2)*lag*np.log(t)/t
    return bic
    

def orderselection(data,maxlag, result = 0):
    aic = np.empty((maxlag,1))
    fpe = np.empty((maxlag,1))
    bic = np.empty((maxlag,1))
    for i in range(1,maxlag+1):
        print(i)
        lag = i
        B = Var_fit(data,lag)
        aic[i-1] = AIC(data,B,lag)
        fpe[i-1] = FPE(data,B,lag)
        bic[i-1] = BIC(data,B,lag)
    if(result ==0):
        return aic,fpe,bic
    else:
        pass

def vectorize(B4):
    return B4.reshape((np.shape(B4)[0]*np.shape(B4)[1],1))


def Cmatrix(k,kx,lag,direction = 0):
    # y' :Kx1 = [ x :kxX1; z :(K-k_x)X1 ]
    B =  np.zeros((k, k*lag) )
    for i in range(0,lag):
        for j in range(1,k+1):
            if direction ==0:
                if(j > kx ):
                    for n in range(0,kx):
                        B[n,(j-1)+i*k] = 1
            if direction ==1:
                if(j <= kx ):
                    for n in range(kx,k):
                        B[n,(j-1)+i*k] = 1
                
    B = np.hstack( (np.zeros((k,1)),B) )
    B1 = vectorize(B)
    N  = int( sum(B1) )
    C = np.zeros( (int(N),len(B1) ) )
    index = np.where(B1==1)[0]
    for i in range(0,N):
        C[i,index[i]]=1    
    
    return C

def GrangerTest(data,lag,kx,alpha=0.05):
    B0 = Var_fit(data,lag)
    B = vectorize(B0)
    Y,Z = Organise2(data,lag)
    k = np.shape(Y)[0]
    T = np.shape(Y)[1] + lag
    if(len(np.shape(data))>2):
        x = np.shape(data)
        t = x[1]*x[2]
    else:
        t = np.shape(data)[1]
    Y_BZ = Y - np.matmul(B0,Z) 
    sigma = np.matmul(Y_BZ,Y_BZ.T)/t
    del Y,Y_BZ,B0
    
    C = Cmatrix(k,kx,lag,0)
    C_B = np.matmul(C,B)
    ZZ_t  = np.linalg.pinv(np.matmul(Z,Z.T))
    M  = np.kron(ZZ_t,sigma)
    CM = np.matmul(C,M)
    CMCinv = np.linalg.pinv(np.matmul(CM,C.T))
    d1 = np.shape(C)[0]
    del CM,M,C
    temp1 = np.matmul(C_B.T,CMCinv)
    lambdaf = np.matmul(temp1,C_B)
    d2 = T - k*lag -1
    pvalue = 1- f.cdf(lambdaf[0]/d1,d1,d2)[0]
    result  = pvalue < alpha 
    return pvalue,result,lambdaf[0]/d1



"""

B2 = Var_fit(data[:3,:,:20],4)
B2_hat = vectorize(B2)

aic,fpe,bic  = orderselection(data[:2,:,:10],50)
import matplotlib.pyplot as plt
#plt.plot(aic)
"""
