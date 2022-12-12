import numpy as np


import simulate_SCA
import cryptanalysis

import time

from tqdm import trange



              
              
if __name__=="__main__":

    NB_REPS = 100
    parameters = [
                  #(3488, 2720, 64,8,0.1),
                  #(3488, 2720, 64,8,0.18),
                  #(3488, 2720, 64,8,0.26),
                  (3488, 2720, 64,8,0.16),
                  (3488, 2720, 64,32,0.16),
                  (3488, 2720, 64,64,0.16),
                  #(4608, 3360, 96,8,0.13),
                  #(6688, 5024, 128,8,0.13),
                  #(6960, 5413, 119,8,0.13),
                  #(8192, 6528, 128,8,0.24),
     
                  ]

    for (N, k, t,size_word,sigma) in parameters:
        print(N,k,t,size_word,sigma)
        success=0

        #build perfect template model One for all
        (MeanHW,stdHW)=simulate_SCA.BuildPerfectModel(sigma,size_word)
        res_psi=np.zeros(NB_REPS)
        res_psipoin=np.zeros(NB_REPS)
        res_poin=np.zeros(NB_REPS)
        res_ttest=np.zeros(NB_REPS)
        res_puc=np.zeros(NB_REPS)
        for repeat in trange(NB_REPS):
            '''
            #Generate data
            ## Parity-check matrix
            '''
            #tic=time.time()
            H = np.random.randint(0, 2, size=((N - k), N), dtype="uint64")
            ## Error vector of weight t
            e = np.array(t * [1] + (N - t) * [0], dtype="uint64")
            np.random.shuffle(e)
            if N%size_word!=0:
                edivsizeword=np.concatenate([e,np.zeros(size_word-(N%size_word))])
            else:
                edivsizeword=np.copy(e)
            s = np.matmul(H, e)%2
            #Generate simulated trace
            HammingLeakages=simulate_SCA.GenerateTrace(H,e,sigma,size_word)
            '''
            #Perform template attack
            '''
            ResTemplateHamming=simulate_SCA.ComputeTemplateAttack(HammingLeakages,MeanHW,stdHW,size_word)
            '''
            #T-test method
            '''
            tic=time.time()
            perm=cryptanalysis.Ttesteval(ResTemplateHamming,H,size_word)
            toc=time.time()
            res_ttest[repeat]=np.sum(e[perm][:(N-k)])
            '''
            Score psi
            '''
            tic=time.time()
            permu=cryptanalysis.Psi(ResTemplateHamming,H,s,t)
            toc=time.time()
            res_psi[repeat]=np.sum(edivsizeword[permu][:(N-k)])
            '''
            Score psi punctred
            '''
            tic=time.time()
            sci,permv=cryptanalysis.PsiPoinconner(ResTemplateHamming,H,s,t,size_word,sigma)
            toc=time.time()
            res_psipoin[repeat]=np.sum(e[sci][permv][:(N-k)])
            res_poin[repeat]=np.sum(e[sci][:(N-k)])
            '''
            Pucture test
            tic=time.time()
            puc=cryptanalysis.Puncture(ResTemplateHamming,size_word,sigma)
            toc=time.time()
            #print(np.shape(puc))
            #print(puc)
            if (np.dot(edivsizeword,puc)==t):
                res_puc[repeat]=((np.sum(puc)))
            else:
                res_puc[repeat]=N+1
            '''
            
            
        #print(N,k,t,size_word,sigma)        
        print('resTtest'+str(N)+'_'+str(size_word)+'_'+str(sigma).replace('.','')+'=',res_ttest)        
        print('resPoinpsi'+str(N)+'_'+str(size_word)+'_'+str(sigma).replace('.','')+'=',res_psipoin)
        print('resPsi'+str(N)+'_'+str(size_word)+'_'+str(sigma).replace('.','')+'=',res_psi)
        #print('respoin'+str(N)+'_'+str(size_word)+'_'+str(sigma).replace('.','')+'=',res_poin)
        #print(res_puc)
        