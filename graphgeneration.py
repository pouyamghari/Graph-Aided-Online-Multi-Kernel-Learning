import numpy as np
import math

def graphgeneration(gamma, m):
    KL = np.zeros((gamma.shape[0],gamma.shape[0]))
    gamma_ind = []
    n_ind = []
    for i in range(0,gamma.shape[0]):
        for j in range(0,gamma.shape[0]):
            KL[i,j] = np.sqrt(math.pi)*( gamma[i]+gamma[j] \
                                            - ((2*np.sqrt(2)*gamma[i]*gamma[j])/np.sqrt(gamma[i]**2 + gamma[j]**2)) )
    ind_KL = np.zeros((gamma.shape[0],gamma.shape[0]))
    A_t = np.zeros((gamma.shape[0],gamma.shape[0]))
    for i in range(0,gamma.shape[0]):
        n_gg = []
        n_gn = []
        A_t[i,i] = 1
        n_gg.append(gamma[int(i)])
        n_gn.append(int(i))
        if m>1:
            for j in range(0,m-1):
                mnn = np.sum(KL[n_gn,:],axis=0)
                for k in n_gn:
                    mnn[k] = 0
                m_arg = np.argmax(mnn)
                A_t[i,int(m_arg)] = 1
                n_gg.append(gamma[int(m_arg)])
                n_gn.append(int(m_arg))
        n_gg = np.array(n_gg)
        n_gn = np.array(n_gn)
        gamma_ind.append(n_gg)
        n_ind.append(n_gn)
    return A_t, KL, gamma_ind, n_ind