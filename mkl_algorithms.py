import numpy as np
from numpy import linalg as LA

class OMKR:
    def __init__(self, eta, gamma, kernel_list):
        self.eta = eta
        self.gamma = np.array(gamma)
        self.kernel_list = kernel_list
        
    def predict(self, X, w, theta):
        M, N = X.shape
        b = self.gamma.shape[0]
        f_RF_p = np.zeros((b,1))
        for j in range(0,self.gamma.shape[0]):
            if M > 1:
                if self.kernel_list[j] == 'Gaussian':
                    for k in range(0,M-1):
                        f_RF_p[j,0] = f_RF_p[j,0] + theta[j,k+1]*np.exp(-(LA.norm(X[M-1,:]-X[k,:])**2)/self.gamma[j])
                elif self.kernel_list[j] == 'Laplacian':
                    for k in range(0,M-1):
                        f_RF_p[j,0] = f_RF_p[j,0] + theta[j,k+1]*np.exp(-(LA.norm((X[M-1,:]-X[k,:]), ord=1))/self.gamma[j])
        w_bar = w/np.sum(w)
        f_RF = w_bar.dot(f_RF_p)
        return f_RF, f_RF_p
    
    def update(self, f_RF_p, Y, theta, w):
        l = np.zeros((1,self.gamma.shape[0]))
        for j in range(0,self.gamma.shape[0]):
            l[0,j] = (f_RF_p[j,0]-Y)**2
            w[0,j] = w[0,j]*(.5**(l[0,j]))
        theta = np.concatenate((theta,-self.eta*(f_RF_p-Y*np.ones((self.gamma.shape[0],1)))),axis=1)
        return w, theta
    
class RBF:
    def __init__(self, eta, gamma):
        self.eta = eta
        self.gamma = np.array(gamma)
        
    def predict(self, X, theta):
        M, N = X.shape
        b = self.gamma.shape[0]
        f_RF = np.zeros((1,1))
        for j in range(0,self.gamma.shape[0]):
            if M > 1:
                for k in range(0,M-1):
                    f_RF += theta[j,k+1]*np.exp(-(LA.norm(X[M-1,:]-X[k,:])**2)/self.gamma[j])
        return f_RF
    
    def update(self, f_RF_p, Y, theta):
        l = (f_RF_p-Y)**2
        theta = np.concatenate((theta,-self.eta*(f_RF_p-Y)),axis=1)
        return theta
    
class POLY:
    def __init__(self, eta, gamma):
        self.eta = eta
        self.gamma = np.array(gamma)
        
    def predict(self, X, theta):
        M, N = X.shape
        b = self.gamma.shape[0]
        f_RF = np.zeros((1,1))
        for j in range(0,self.gamma.shape[0]):
            if M > 1:
                for k in range(0,M-1):
                    f_RF += theta[j,k+1]*( (X[M-1,:].dot(np.transpose(X[k,:])))**self.gamma[j] )
        return f_RF
    
    def update(self, f_RF_p, Y, theta):
        l = (f_RF_p-Y)**2
        theta = np.concatenate((theta,-self.eta*(f_RF_p-Y)),axis=1)
        return theta

class RFOMKR:
    def __init__(self, rf_feature, eta):
        self.eta = eta
        self.rf_feature = np.array(rf_feature)
        
    def predict(self, X, theta, w):
        M, N = X.shape
        a, n_components, b = self.rf_feature.shape
        f_RF_p = np.zeros((b,1))
        X_f = np.zeros((b,n_components))
        X_features = np.zeros((b,2*n_components))
        for j in range(0,b):
            X_f[j,:] = X.dot(self.rf_feature[:,:,j])
        X_features = (1/np.sqrt(n_components))*np.concatenate((np.sin(X_f),np.cos(X_f)),axis=1)
        for j in range(0,b):
            f_RF_p[j,0] = X_features[j,:].dot(theta[:,j])
        f_RF = w.dot(f_RF_p)
        return f_RF, f_RF_p, X_features
    
    def update(self, f_RF_p, Y, theta, w, X_features):
        b, n_components = X_features.shape
        l = np.zeros((1,b))
        for j in range(0,b):
            theta[:,j] = theta[:,j] - self.eta*(2*(f_RF_p[j,0] - Y)*np.transpose(X_features[j,:]))
            l[0,j] = (f_RF_p[j,0]-Y)**2
            w[0,j] = w[0,j]*(.5**(l[0,j]))
        return w, theta

    

class Raker:
    def __init__(self, lam, rf_feature, eta):
        self.lam = lam
        self.eta = eta
        self.rf_feature = np.array(rf_feature)
        
    def predict(self, X, theta, w):
        M, N = X.shape
        a, n_components, b = self.rf_feature.shape
        f_RF_p = np.zeros((b,1))
        X_f = np.zeros((b,n_components))
        X_features = np.zeros((b,2*n_components))
        for j in range(0,b):
            X_f[j,:] = X.dot(self.rf_feature[:,:,j])
        X_features = (1/np.sqrt(n_components))*np.concatenate((np.sin(X_f),np.cos(X_f)),axis=1)
        for j in range(0,b):
            f_RF_p[j,0] = X_features[j,:].dot(theta[:,j])
        w_bar = w/np.sum(w)
        f_RF = w_bar.dot(f_RF_p)
        return f_RF, f_RF_p, X_features
    
    def update(self, f_RF_p, Y, theta, w, X_features):
        b, n_components = X_features.shape
        l = np.zeros((1,b))
        for j in range(0,b):
            theta[:,j] = theta[:,j] - self.eta*( (2*(f_RF_p[j,0] - Y)*np.transpose(X_features[j,:]))                                                     +2*self.lam*theta[:,j] )
            l[0,j] = (f_RF_p[j,0]-Y)**2+self.lam*(LA.norm(theta[:,j])**2)
            w[0,j] = w[0,j]*np.exp(-self.eta*l[0,j])
        return w, theta
    
    
    
class OMKLGF:
    def __init__(self, lam, rf_feature, gamma, eta, eta_e, M, J):
        self.lam = lam
        self.eta = eta
        self.eta_e = eta_e
        self.rf_feature = np.array(rf_feature)
        self.gamma = np.array(gamma)
        self.M = M
        self.J = J
        
    def graph_gen(self, w):
        p_k = np.zeros((self.gamma.shape[0],self.J))
        p_kk = np.zeros((self.gamma.shape[0],self.J))
        p_c = np.zeros((1,self.J))
        w_bar = w/np.sum(w)
        a, n_components, c = self.rf_feature.shape
        A_t = np.zeros((c,self.J))
        for j in range(0,self.J):
            p_k[:,j:j+1] = (1-self.eta_e**(j+1))*np.transpose(w_bar)+(1/c)*(self.eta_e**(j+1))*np.ones((c,1))
            for k in range(0,c):
                p_kk[k:k+1,j:j+1] = 1-((1-p_k[k:k+1,j:j+1])**self.M)
            for k in range(0,self.M):
                n = 0
                rr = np.random.rand()
                while rr>np.sum(p_k[0:n,j]) and n<c-1:
                    n = n+1
                A_t[n,j] = 1
        return A_t, p_kk
    
    def predict(self, X, theta, w, A_t):
        m, N = X.shape
        gamma_n = []
        n_n = []
        u = w.dot(A_t)
        u_bar = u/np.sum(u)
        p_c = (1-self.eta_e)*u_bar+(1/self.J)*self.eta_e*np.ones((1,self.J))
        c = 0
        rr = np.random.rand()
        while rr>np.sum(p_c[0,:c+1]) and c<self.J-1:
            c = c+1
        I_t = c
        for n in range(0,self.gamma.shape[0]):
            if A_t[n,I_t]==1:
                gamma_n.append(self.gamma[n])
                n_n.append(n)
        gamma_n = np.array(gamma_n)
        n_n = np.array(n_n)
        a, n_components, c = self.rf_feature.shape
        f_RF_p = np.zeros((self.gamma.shape[0],1))
        X_f = np.zeros((self.gamma.shape[0],n_components))
        X_features = np.zeros((self.gamma.shape[0],2*n_components))
        f_RF_p = np.zeros((self.gamma.shape[0],1))
        for j in n_n:
            X_f[j,:] = X.dot(self.rf_feature[:,:,j])
        X_features = (1/np.sqrt(n_components))*np.concatenate((np.sin(X_f),np.cos(X_f)),axis=1)
        for j in n_n:
            f_RF_p[j,0] = X_features[j,:].dot(theta[:,j])
        w_n = np.zeros((1,self.gamma.shape[0]))
        f_RF_p_n = np.zeros((self.gamma.shape[0],1))
        for j in range(0,gamma_n.shape[0]):
            w_n[0,j] = w[0,n_n[j]]
            f_RF_p_n[j,0] = f_RF_p[n_n[j],0]
        w_bar = w_n/np.sum(w_n)
        f_RF = w_bar.dot(f_RF_p_n)
        return f_RF, f_RF_p, X_features, n_n, p_c, I_t
    
    def update(self, f_RF_p, Y, theta, w, X_features, n_n, p_kk, p_c, ter):
        c, n_components = X_features.shape
        l = np.zeros((1,c))
        q = p_kk.dot(np.transpose(p_c))
        for j in n_n:
            if np.sum(q[j,:])>.05:
                theta[:,j] -=  self.eta*( (2*(f_RF_p[j,0] - Y)*np.transpose(X_features[j,:]))+2*self.lam*theta[:,j] )/np.sum(q[j,:])
                l[0,j] = ( (f_RF_p[j,0]-Y)**2+self.lam*(LA.norm(theta[:,j])**2) )/np.sum(q[j,:])
            else:
                theta[:,j] -=  self.eta*20*( (2*(f_RF_p[j,0] - Y)*np.transpose(X_features[j,:]))+2*self.lam*theta[:,j] )
                l[0,j] = 20*( (f_RF_p[j,0]-Y)**2+self.lam*(LA.norm(theta[:,j])**2) )
            w[0,j] *= np.exp(-self.eta*l[0,j])
        return w, theta
    
    
class OMKLSFG:
    def __init__(self, lam, rf_feature, gamma, A_t, KL, gamma_ind, n_ind):
        self.lam = lam
        self.rf_feature = np.array(rf_feature)
        self.gamma = np.array(gamma)
        self.A_t = np.mat(A_t)
        self.KL = np.mat(KL)
        self.gamma_ind = gamma_ind
        self.n_ind = n_ind
    
    def greedysetcover(self):
        B_t = self.A_t
        d_set = []
        u_set = []
        for i in range(0,self.A_t.shape[0]):
            u_set.append(i)
        while len(u_set)>0:
            deg_n = B_t.dot(np.ones((B_t.shape[0],1)))
            m = np.argmax(deg_n)
            d_set.append(u_set[m])
            u_set_update = []
            u_set_array = np.array(u_set)
            for i in u_set_array:
                if self.A_t[u_set[m],i]==0:
                    u_set_update.append(i)
            u_set = u_set_update
            if len(u_set)>0:
                B_t = np.zeros((len(u_set),len(u_set)))
                for i in range(0,len(u_set)):
                    for j in range(0,len(u_set)):
                        B_t[i,j] = self.A_t[u_set[i],u_set[j]]
        return d_set
    
    def graphrefinement(self,ep,u,eta):
        d_set = []
        d_set_a = []
        u_bar = u/np.sum(u)
        for i in range(0,self.gamma.shape[0]):
            if u_bar[0,i] >= (ep-(eta/self.gamma.shape[0]))/(1-eta):
                d_set.append(i)
        d_set = np.array(d_set)
        B_t = np.zeros((self.gamma.shape[0],self.gamma.shape[0]))
        B_t += self.A_t
        m_d = np.zeros((1,self.gamma.shape[0]))
        gamma_ind_ref = []
        n_ind_ref = []
        for i in range(0,self.gamma.shape[0]):
            B_t[d_set[np.argmax(self.KL[d_set,i])],i] = 1
            m_d[0,i] = np.argmax(self.KL[d_set,i])
            gamma_ind_ref.append(self.gamma_ind[i])
            n_ind_ref.append(self.n_ind[i])
        for i in range(0,self.gamma.shape[0]):
            if self.A_t[d_set[int(m_d[0,i])],i]==0:
                gamma_ind_ref[d_set[int(m_d[0,i])]] = np.append(gamma_ind_ref[d_set[int(m_d[0,i])]],self.gamma[i])
                n_ind_ref[d_set[int(m_d[0,i])]] = np.append(n_ind_ref[d_set[int(m_d[0,i])]],i)
        d_set_ref = d_set
        return d_set_ref, B_t, n_ind_ref, gamma_ind_ref
        
    def predict(self, X, theta, w, u, eta, d_set, ter, s_t):
        m, N = X.shape
        if ter==1:
            s_n = s_t
            u_bar = u/np.sum(u)
            p = (1-eta)*u_bar
            d_set = np.array(d_set)
            for i in d_set:
                p[0,i] += eta/d_set.shape[0]
        else:
            u_bar = u/np.sum(u)
            p = (1-eta)*u_bar
            d_set = np.array(d_set)
            for i in d_set:
                p[0,i] += eta/d_set.shape[0]
            s_n = 0
            rr = np.random.rand()
            while rr>np.sum(p[0,0:s_n]) and s_n<self.gamma.shape[0]-1:
                s_n+=1
        q = p.dot(self.A_t)
        gamma_n = np.array(self.gamma_ind[s_n])
        n_n = np.array(self.n_ind[s_n])
        a, n_components, c = self.rf_feature.shape
        f_RF_p = np.zeros((self.gamma.shape[0],1))
        X_f = np.zeros((self.gamma.shape[0],n_components))
        X_features = np.zeros((self.gamma.shape[0],2*n_components))
        f_RF_p = np.zeros((self.gamma.shape[0],1))
        for j in n_n:
            X_f[j,:] = X.dot(self.rf_feature[:,:,j])
        X_features = (1/np.sqrt(n_components))*np.concatenate((np.sin(X_f),np.cos(X_f)),axis=1)
        for j in n_n:
            f_RF_p[j,0] = X_features[j,:].dot(theta[:,j])
        w_n = np.zeros((1,self.gamma.shape[0]))
        f_RF_p_n = np.zeros((self.gamma.shape[0],1))
        for j in range(0,gamma_n.shape[0]):
            w_n[0,j] = w[0,n_n[j]]
            f_RF_p_n[j,0] = f_RF_p[n_n[j],0]
        w_bar = w_n/np.sum(w_n)
        f_RF = w_bar.dot(f_RF_p_n)
        return f_RF, f_RF_p, X_features, n_n, s_n, p, q
    
    def predict_refinement(self, X, theta, w, u, eta, d_set_ref, B_t, n_ind_ref, gamma_ind_ref, ter, s_t):
        m, N = X.shape
        d_set = np.array(d_set_ref)
        if ter==1:
            s_n = s_t
            u_bar = u/np.sum(u)
            p = (1-eta)*u_bar
            d_set = np.array(d_set)
            for i in d_set:
                p[0,i] += eta/d_set.shape[0]
        else:
            u_bar = u/np.sum(u)
            p = (1-eta)*u_bar
            for i in d_set:
                p[0,i] += eta/d_set.shape[0]
            s_n = 0
            rr = np.random.rand()
            while rr>np.sum(p[0,0:s_n]) and s_n<self.gamma.shape[0]-1:
                s_n+=1
        q = p.dot(B_t)
        s_n = 0
        rr = np.random.rand()
        while rr>np.sum(p[0,0:s_n]) and s_n<self.gamma.shape[0]-1:
            s_n+=1
        gamma_n = np.array(gamma_ind_ref[s_n])
        gamma_n_rf = np.array(self.gamma_ind[s_n])
        n_n = np.array(n_ind_ref[s_n])
        n_n_rf = np.array(self.n_ind[s_n])
        a, n_components, c = self.rf_feature.shape
        f_RF_p = np.zeros((self.gamma.shape[0],1))
        X_f = np.zeros((self.gamma.shape[0],n_components))
        X_features = np.zeros((self.gamma.shape[0],2*n_components))
        f_RF_p = np.zeros((self.gamma.shape[0],1))
        for j in n_n:
            X_f[j,:] = X.dot(self.rf_feature[:,:,j])
        X_features = (1/np.sqrt(n_components))*np.concatenate((np.sin(X_f),np.cos(X_f)),axis=1)
        for j in n_n:
            f_RF_p[j,0] = X_features[j,:].dot(theta[:,j])
        w_n = np.zeros((1,self.gamma.shape[0]))
        f_RF_p_n = np.zeros((self.gamma.shape[0],1))
        for j in range(0,gamma_n_rf.shape[0]):
            w_n[0,j] = w[0,n_n_rf[j]]
            f_RF_p_n[j,0] = f_RF_p[n_n_rf[j],0]
        w_bar = w_n/np.sum(w_n)
        f_RF = w_bar.dot(f_RF_p_n)
        return f_RF, f_RF_p, X_features, n_n, s_n, p, q
    
    def update(self, f_RF, f_RF_p, Y, theta, w, u, X_features, n_n, s_n, p, q, eta):
        c, n_components = X_features.shape
        l = np.zeros((1,c))
        w_n = np.zeros((1,self.gamma.shape[0]))
        for j in n_n:
            w_n[0,j] = w[0,j]
            if q[0,j]>.1:
                theta[:,j] -=  eta*( (2*(f_RF_p[j,0] - Y)*np.transpose(X_features[j,:]))+2*self.lam*theta[:,j] )/q[0,j]
                l[0,j] = ( (f_RF_p[j,0]-Y)**2+self.lam*(LA.norm(theta[:,j])**2) )/q[0,j]
            else:
                theta[:,j] -=  eta*( (2*(f_RF_p[j,0] - Y)*np.transpose(X_features[j,:]))+2*self.lam*theta[:,j] )/.1
                l[0,j] = ( (f_RF_p[j,0]-Y)**2+self.lam*(LA.norm(theta[:,j])**2) )/.1
            w[0,j] *= np.exp(-eta*l[0,j])
        w_n/=np.sum(w_n)
        fin_loss = (f_RF-Y)**2
        if p[0,s_n]<.2:
            u[0,s_n] *= np.exp(-eta*5*fin_loss)
        else:
            u[0,s_n] *= np.exp(-eta*(fin_loss/p[0,s_n]))
        return w, u, theta