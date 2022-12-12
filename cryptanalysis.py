import numpy as np
from scipy import stats
from simulate_SCA import EstimateAccurracy
from collections import Counter
import math

from tqdm import trange

#Solution of CDCG
def Psi(ResTemplate,mat,s,t):
    #compute Syndrome
	#TODO: recenter syndrome if needed
    Syn = np.sum(np.abs(np.diff(np.argmax(ResTemplate,axis=-1), axis=1)), axis=1)+np.argmax(ResTemplate[:,0],axis=-1)
    for i in range(len(Syn)):
        if(Syn[i]%2!=s[i]):
            Syn[i]+=1
    # Complement of the parity check matrix
    mat_c = 1 - mat
    # Complement of the syndrome
    s_c = t - Syn
    # Computation of the cost function vector
    phi = np.dot(Syn, mat) + np.dot(s_c, mat_c)
    # Get the permutation that sorts phi, -1 to get perm in decreasing order
    perm = np.argsort(-1*phi) 
    return perm
    
#Solution section 4.1    
def Puncture(ResTemplate,word_size,sigma):
    acc=EstimateAccurracy(sigma)
    n=np.shape(ResTemplate)[0]
    k=np.shape(ResTemplate)[0]-np.shape(ResTemplate)[1]
    #beta1=(n-k)*acc-np.sqrt(2*acc*(n-k)*np.log(n-k))    
    #betai=(n-k)*(acc**2+((1-acc)**2)/2)-np.sqrt((2*acc**2+(1-acc)**2)*(n-k)*np.log(n-k))
   
    beta1=(n-k)*(1-acc)+np.sqrt(2*acc*(n-k)*np.log(n-k))    
    betai=(n-k)*(1-acc**2+((1-acc)**2)/2)+np.sqrt((2*acc**2+(1-acc)**2)*(n-k)*np.log(n-k))
    
    
    diff=np.zeros((np.shape(ResTemplate)[0],np.shape(ResTemplate)[1]))
    #compute difference of difference to obtain something close to weight leakage 
    #Is there a difference between consacutive value of hamming weight
    diff[:,0]=np.argmax(ResTemplate[:,0],axis=-1)
    diff[:,1:] =np.abs(np.diff(np.argmax(ResTemplate,axis=-1), axis=1))
    sci=np.repeat(np.sum(diff!=0, axis=0)>betai,word_size)
    sci[0:word_size]=np.repeat(np.sum(diff[:,0]!=0, axis=0)>beta1,word_size)
    #sci=np.repeat(np.sum(diff, axis=0)>betai,word_size)
    #sci[0:word_size]=np.repeat(np.sum(diff[:,0], axis=0)>beta1,word_size)
    return sci

    
#Solution 4.1 + CDCG    
#Input
#ResTemplate The result od the side-channel attack
#mat The public encryption matrix
#s The syndrome (encapsulated key)
#t The public weight of the key to encapsulate    
#word_size The size of register/word considered
#alpha the value to distinguish 1 or 0
def PsiPoinconner(ResTemplate,mat,s,t,word_size,sigma):
    #compute Syndrome
    Syn = np.sum(np.abs(np.diff(np.argmax(ResTemplate,axis=-1), axis=1)), axis=1)+np.argmax(ResTemplate[:,0],axis=-1)
    retmp=np.copy(ResTemplate)    
    retmp=np.insert(retmp, 0, 0, axis=1)
    #TODO: check threshold 
    acc=EstimateAccurracy(sigma)
    n=np.shape(ResTemplate)[0]
    k=np.shape(ResTemplate)[0]-np.shape(ResTemplate)[1]
    
    beta1=(n-k)*(1-acc)+np.sqrt(2*acc*(n-k)*np.log(n-k))    
    betai=(n-k)*(1-acc**2+((1-acc)**2)/2)+np.sqrt((2*acc**2+(1-acc)**2)*(n-k)*np.log(n-k))
   
    diff=np.zeros((np.shape(ResTemplate)[0],np.shape(ResTemplate)[1]))
    #compute difference of difference to obtain something close to weight leakage 
    #Is there a difference between consacutive value of hamming weight
    diff[:,0]=np.argmax(ResTemplate[:,0],axis=-1)
    diff[:,1:] =np.abs(np.diff(np.argmax(ResTemplate,axis=-1), axis=1))
    #case 0
    sci=np.repeat(np.sum(diff!=0, axis=0)>betai,word_size)
    sci[0:word_size]=np.repeat(np.sum(diff[:,0]!=0, axis=0)>beta1,word_size)  
    for i in range(len(Syn)):
        if(Syn[i]%2!=s[i]):
            Syn[i]+=1
    matpoin=mat[:,sci[:np.shape(mat)[1]]]    
    # Complement of the parity check matrix
    matpoin_c = 1 - matpoin
    # Complement of the syndrome
    s_c = t - Syn
    # Computation of the cost function vector
    psi = np.dot(Syn, matpoin) + np.dot(s_c, matpoin_c)
    # Get the permutation that sorts phi, -1 to get perm in decreasing order
    perm = np.argsort(-1*psi) 
    return sci[:np.shape(mat)[1]],perm

    
    
#Solution 4.2    
def Ttesteval(ResTemplate,mat,word_size):   
    diff=np.zeros((np.shape(ResTemplate)[0],np.shape(ResTemplate)[1]))
    scoretest=np.zeros(np.shape(mat)[1])
    #compute difference of difference to obtain something close to weight leakage 
    #Is there a difference between consacutive value of hamming weight
    diff[:,0]=np.argmax(ResTemplate[:,0],axis=-1)
    diff[:,1:] =np.abs(np.diff(np.argmax(ResTemplate,axis=-1), axis=1))
    for j in range(np.shape(diff)[1]):
        #separate the line of the matrix if there is a change in 
        set0=[]
        set1=[]
        for i in range(np.shape(diff)[0]):
            if diff[i][j]==0:
                set0.append(mat[i,j*word_size:(j+1)*word_size])
            else:
                set1.append(mat[i,j*word_size:(j+1)*word_size])
        if len(set0)>3 and len(set1)>3:
            t_test=stats.ttest_ind(set1, set0)
            scoretest[j*word_size:(j+1)*word_size]=t_test[0]
    perm = np.argsort(-1*scoretest) 
    return perm    


 
