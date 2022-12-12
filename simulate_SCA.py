import numpy as np
from scipy.stats import norm
import math



#This function Generate trace for matrix(S) vector(L) product
#The leakage function is the hamming weight over word_size bits plus a normally distributate noise of mean 0 and standard deviation sigma
#The only available leakages are the successive xor (line 7 of algorithm 4 of https://eprint.iacr.org/2022/125)
#word_size do not need to divide len(L)  
def GenerateTrace(S,L,sigma,word_size):
    #compute intermetiade result
    AND=S&L
    hwXOR=np.zeros((np.shape(S)[0],int(np.ceil(np.shape(S)[1]/word_size))))
    XOR=AND[:,0:word_size]  
    hwXOR[:,0]=np.sum(XOR,axis=1)+np.random.normal(0,sigma,np.shape(S)[0])
    for i in range(1,(np.shape(S)[1]//word_size)):
        XOR=AND[:,i*word_size:(i+1)*word_size]^XOR
        hwXOR[:,i]=[np.sum(XOR,axis=1)]+np.random.normal(0,sigma,np.shape(S)[0])
    if(int(np.ceil(np.shape(S)[1]/word_size))!=(np.shape(S)[1]//word_size)):
        tmp=np.concatenate([AND[:,(np.shape(S)[1]//word_size)*word_size:],np.zeros((np.shape(XOR)[0],(len(XOR[1])-len(AND[0,(np.shape(S)[1]//word_size)*word_size:])) ))],axis=1).astype(np.uint8)
        XOR=tmp^XOR
        hwXOR[:,(np.shape(S)[1]//word_size)]=[np.sum(XOR,axis=1)]+np.random.normal(0,sigma,np.shape(S)[0])
    return hwXOR

#This function build perfect model for Gaussian template attacks
def BuildPerfectModel(sigma,word_size):
    mean=range(word_size+1)
    std=[sigma]*(word_size+1)
    return mean,std	

#assume one model for all leak
#DistLeak,HamLeak 1d array PoI of each intermediate value Probably works for more PoI
def ComputeTemplateAttack(Leak,ModelMean,ModelStd,word_size):
    alpha=np.zeros((len(Leak),len(Leak[0]),word_size+1))
    for int in range(len(ModelMean)):
        alpha[:,:,int]=norm.pdf(Leak,loc=ModelMean[int],scale=ModelStd[int])
    for row in range(len(Leak)):
        for col in range(len(Leak[0])):
            alpha[row][col]=alpha[row][col]/np.sum(alpha[row][col])    
    return alpha				

#Estimate accuracy using 3-sigma rule    
def EstimateAccurracy(sigma):
    return math.erf(1/(2*sigma*math.sqrt(2)))
    
#Compute accuracy from experiments    
def ComputeAccurracy(S,L,ResTemplate,word_size):
    AND=S&L
    hwXOR=np.zeros((np.shape(S)[0],int(np.ceil(np.shape(S)[1]/word_size))))
    XOR=AND[:,0:word_size]  
    hwXOR[:,0]=np.sum(XOR,axis=1)
    for i in range(1,(np.shape(S)[1]//word_size)):
        XOR=AND[:,i*word_size:(i+1)*word_size]^XOR
        hwXOR[:,i]=np.sum(XOR,axis=1)
    if(int(np.ceil(np.shape(S)[1]/word_size))!=(np.shape(S)[1]//word_size)):
        tmp=np.concatenate([AND[:,(np.shape(S)[1]//word_size)*word_size:],np.zeros((np.shape(XOR)[0],(len(XOR[1])-len(AND[0,(np.shape(S)[1]//word_size)*word_size:])) ))],axis=1).astype(np.uint8)
        XOR=tmp^XOR
        hwXOR[:,(np.shape(S)[1]//word_size)]=[np.sum(XOR,axis=1)]
    guess=np.argmax(ResTemplate,axis=-1)
    print(np.shape(guess))
    print(np.shape(hwXOR))
    return np.sum(guess==hwXOR)/(np.shape(guess)[0]*np.shape(guess)[1])

    
if __name__=="__main__":
    print("####################################################")
    print("#                     Test                         #")
    print("####################################################")
    sigma=0.2
    print(EstimateAccurracy(sigma))
    S = np.random.randint(0, 2, size=(96, 128), dtype="uint32")
    ## Error vector of weight t
    L = np.array(16 * [1] + (128 - 16) * [0], dtype="uint32")
    np.random.shuffle(L)
    (MeanHW,stdHW)=BuildPerfectModel(sigma,8)
    Leak=GenerateTrace(S,L,sigma,8)
    ResTA=ComputeTemplateAttack(Leak,MeanHW,stdHW,8)
    print(ComputeAccurracy(S,L,ResTA,8))