# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 17:25:38 2020

@author: tom9m
"""
from scipy.special import gammaln
import scipy.special
from sklearn import preprocessing
import numpy as np
import sys
import os
import pickle
import random
from scipy import stats
import matplotlib.pyplot as plt
import collections

digamma=scipy.special.digamma
def dircheck():
    if not os.path.isdir('C:/Users/tom/src/hdpsmlda/output'):
        os.makedirs('C:/Users/tom/src/hdpsmlda/output') 
class DefaultDict(dict):
    def __init__(self, v):
        self.v = v
        dict.__init__(self)
    def __getitem__(self, k):
        return dict.__getitem__(self, k) if k in self else self.v
    def update(self, d):
        dict.update(self, d)
        return self
    
class HDPSMLDA:
    def __init__(self, alpha,  gamma,beta, docs, V ,frequency, w, dim,kernel):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.V = V
        self.L = len(docs)
        self.dim=dim
        self.w = w
        self.perp=[]

        # t : table index for document j
        #     t=0 means to draw a new table
        self.using_t = [[0] for j in range(self.L)]

        # k : dish(topic) index
        #     k=0 means to draw a new dish
        self.using_k = [0]

        self.w_ji = docs # vocabulary for each document and term
        self.x_ji = frequency
        self.k_jt = [np.zeros(1 ,dtype=int) for j in range(self.L)]   # topics of document and table
        self.n_jt = [np.zeros(1 ,dtype=int) for j in range(self.L)]   # number of terms for each table of document
        self.n_jtv = [[None] for j in range(self.L)]
        self.m_jt = [np.zeros([1,self.dim] ,dtype=int) for j in range(self.L)]   # number of terms for each table of document
       

        self.l = 0
        self.l_k = np.ones(1 ,dtype=int)  # number of tables for each topic
        self.n_k = np.array([self.beta * self.V]) # number of terms for each topic ( + beta * V )
        self.n_kv =[DefaultDict(0)]            # number of terms for each topic and vocabulary ( + beta )

        self.s_jn= frequency*0
        
        self.epoch=0
        
        self.mu=[np.ones(1 ,dtype=float)*np.average(frequency) for j in range(self.dim)]
        self.lambda_=[np.ones(1 ,dtype=float) for j in range(self.dim)]
        self.eta_init=0.1
        self.a_init=0.1
        self.b_init=0.1
        self.m_init=0.1
        self.m_k=np.zeros([1,self.dim])
        self.eta=np.zeros([1,self.dim])
        self.a=np.zeros([1,self.dim])
        self.b=np.zeros([1,self.dim])
        self.m=np.zeros([1,self.dim])
        self.kernel=kernel



        # table for each document and term (-1 means not-assigned)
        self.t_ji = [np.zeros(len(w_i), dtype=int) - 1 for w_i in docs]
        self.tx_ji=np.zeros([len(frequency),self.kernel,dim],dtype='int')-1


    def inference(self):
        for j, w_i in enumerate(self.w_ji):
            for i in range(len(w_i)):
                self.sampling_t(j, i)             
        for j, x_i in enumerate(self.x_ji):  
            for q in range(self.dim):
                for i in range(len(x_i)):
                    sample_x=preprocessing.normalize(self.w[j,:,q].reshape(1, -1), norm="l1")[0]
                    idx=np.argmax(np.random.multinomial(1, sample_x, size=1))
                    self.sampling_tx(j, idx, q)     
        print(self.m_k)
        print(self.n_k)                
        for j in range(self.L):
            for t in self.using_t[j]:
                if t != 0: self.sampling_k(j, t)
       
        self.optimize_beta()
        self.optimize_alpha()
        self.optimize_mu_sigma()
                
                
    def optimize_beta(self):

        numerator=0
        denominator=0
        for k in self.using_k:
            if k==0:
                continue
            for v in range(self.V):
                numerator += (digamma(self.n_kv[k][v]) - digamma(self.beta))
                
            denominator += (digamma(self.n_k[k]) - digamma(self.beta*self.V))
        if(denominator!=0):            
            newbeta = self.beta*numerator / self.V / denominator
            
            for k in self.using_k:
                if k==0:
                    continue                
                self.n_k[k]=self.n_k[k]-self.beta*self.V+newbeta*self.V
                for v in range(self.V):
                    self.n_kv[k][v]=self.n_kv[k][v]-self.beta+newbeta
            self.beta=newbeta
        
    def optimize_alpha(self):
        K=len(self.using_k)-1
        
        numerator=0
        for i in range(len(self.w_ji)):
            numerator+=len(self.w_ji[i])
            numerator+=len(self.x_ji[i])*self.kernel
        denominator=self.alpha+numerator
        p=numerator/denominator
        s=stats.bernoulli.rvs(p, size=1)
        
        a=self.alpha+1
        b=numerator
        pi=a/(a+b)

        c_1=1
        c_2=1

        newalpha=(c_1+K-s)/(c_2-np.log(pi))
        self.alpha=newalpha
        
    def optimize_mu_sigma(self):
        self.epoch+=1
        K=len(self.using_k)-1
        for q in range(self.dim):
            for k in self.using_k:
                if k==0:
                    continue   
                #μとΣの更新
                self.eta[k,q]=self.eta_init+self.m_k[k,q]
                sum_x=0
                sum_x_2=0
                for ii in range(len(self.w_ji)):
                    for jj in range(self.kernel):
                        if(self.s_jn[ii,jj,q]==k):
                            sum_x += self.x_ji[ii,jj,q]
                            sum_x_2 +=pow(self.x_ji[ii,jj,q],2)
                            
                self.m[k,q]=(sum_x+self.m_init*self.eta_init)/self.eta[k,q]
                
                self.a[k,q]=self.a_init+self.m_k[k,q]/2
                self.b[k,q]=self.b_init+(sum_x_2+self.eta_init*pow(self.m_init,2)-self.eta[k,q]*pow(self.m[k,q],2))/2
                
                #あきらめ
                self.lambda_[k,q]=1/pow(((np.max(self.x_ji[:,:,q])-np.min(self.x_ji[:,:,q]))/(6*K)),2)

                #if(self.epoch>100):
                    #分散計算
                    #あきらめない
                #    self.lambda_[k,q]=self.a[k,q]/self.b[k,q]#random.gammavariate(a[q,k],1/b[q,k])
                    
                self.mu[k,q]=np.random.normal(self.m[k,q],1/(self.eta[k,q]*self.lambda_[k,q]), 1)            
        
    def worddist(self):
        """return topic-word distribution without new topic"""
        return [DefaultDict(self.beta / self.n_k[k]).update(
            (v, n_kv / self.n_k[k]) for v, n_kv in self.n_kv[k].items())
                for k in self.using_k if k != 0]

    def docdist(self):
        """return document-topic distribution with new topic"""

        # am_k = effect from table-dish assignment
        am_k = np.array(self.l_k, dtype=float)
        am_k[0] = self.gamma
        am_k *= self.alpha / am_k[self.using_k].sum()

        theta = []
        for j, n_jt in enumerate(self.n_jt):
            p_jk = am_k.copy()
            for t in self.using_t[j]:
                if t == 0: continue
                k = self.k_jt[j][t]
                p_jk[k] += n_jt[t]            
                p_jk[k] += sum(self.m_jt[j][t])
            p_jk = p_jk[self.using_k]
            theta.append(p_jk / p_jk.sum())

        return np.array(theta)

    def perplexity(self):
        phi = [DefaultDict(1.0/self.V)] + self.worddist()
        theta = self.docdist()
        log_likelihood = 0
        N = 0
        for w_ji, p_jk in zip(self.w_ji, theta):
            for v in w_ji:
                word_prob = sum(p * p_kv[v] for p, p_kv in zip(p_jk, phi))
                log_likelihood -= np.log(word_prob)
            N += len(w_ji)
        p=np.exp(log_likelihood / N)
        self.perp.append(p)
        return p


                
    def sampling_t(self, j, i):
        """sampling t (table) from posterior"""
        self.leave_from_table(j, i)

        v = self.w_ji[j][i]
        f_k = self.calc_f_k(v)

        assert f_k[0] == 0 # f_k[0] is a dummy and will be erased

        # sampling from posterior p(t_ji=t)
        p_t = self.calc_table_posterior(j, f_k)

        if len(p_t) > 1 and p_t[1] < 0: self.dump()
        t_new = self.using_t[j][np.random.multinomial(1, p_t).argmax()]
        if t_new == 0:
            p_k = self.calc_dish_posterior_w(f_k)
            k_new = self.using_k[np.random.multinomial(1, p_k).argmax()]
            if k_new == 0:
                k_new = self.add_new_dish()
            t_new = self.add_new_table(j, k_new)

        # increase counters
        self.seat_at_table(j, i, t_new)
   

    def leave_from_table(self, j, i):

        t = self.t_ji[j][i]
        
        if t  > 0:
            k = self.k_jt[j][t]
            assert k > 0

            # decrease counters
            v = self.w_ji[j][i]
            self.n_kv[k][v] -= 1
            self.n_k[k] -= 1
            self.n_jt[j][t] -= 1
            self.n_jtv[j][t][v] -= 1

            if self.n_jt[j][t]+sum(self.m_jt[j][t]) == 0:
               
                self.remove_table(j, t)
                
                
    def sampling_tx(self, j, i, q):
        """sampling t (table) from posterior"""

        self.leave_from_tablex(j, i ,q)

        x = self.x_ji[j,i,q]
        f_kx = self.calc_f_kx(x,q)
        

        assert f_kx[0] == 0 # f_k[0] is a dummy and will be erased

        # sampling from posterior p(t_ji=t)
        p_t = self.calc_table_posteriorx(j, f_kx)
        if len(p_t) > 1 and p_t[1] < 0: self.dump()
        t_new = self.using_t[j][np.random.multinomial(1, p_t).argmax()]
        if t_new == 0:
            p_k = self.calc_dish_posterior_x(f_kx,x)
            k_new = self.using_k[np.random.multinomial(1, p_k).argmax()]
            if k_new == 0:
                k_new = self.add_new_dish()
                self.s_jn[j,i,q]=k_new
            t_new = self.add_new_table(j, k_new)

        # increase counters
        self.seat_at_tablex(j, i, q, t_new)
   
        
    def leave_from_tablex(self, j, i, q):
        t = self.tx_ji[j, i, q]
        
        if t  > 0:
            k = self.k_jt[j][t]
            assert k > 0

            # decrease counters
            self.m_k[k,q] -= 1
            self.m_jt[j][t][q] -= 1
            

            if self.n_jt[j][t]+sum(self.m_jt[j][t]) == 0:
                self.remove_table(j, t)
                
    def remove_table(self, j, t):
        """remove the table where all guests are gone"""
        k = self.k_jt[j][t]
        self.using_t[j].remove(t)
        self.l_k[k] -= 1
        self.l -= 1
        assert self.l_k[k] >= 0
        if self.l_k[k] == 0:
            # remove topic (dish) where all tables are gone
            self.using_k.remove(k)

    def calc_f_k(self, v):
        return [n_kv[v] for n_kv in self.n_kv] / self.n_k

    def calc_f_kx(self, x,q):
        sigma=1/self.lambda_[:,q]
        mu=self.mu[:,q]
        
        p=np.zeros(len(self.l_k))
        for k in self.using_k:
            p[k]=np.exp(-1*pow(x-mu[k],2)/(2*sigma[k]))/(np.sqrt(2*np.pi*sigma[k]))        
        p[0]=0
        if(all([i==0 for i in p])):
            p[1:]=1
 
        
        return p/np.sum(p)

    def calc_table_posterior(self, j, f_k):
        #文書で生きてるテーブル
        using_t = self.using_t[j]


        p_t =  (self.n_jt[j][using_t]+np.sum(self.m_jt[j][using_t],1)) * f_k[self.k_jt[j][using_t]]

        p_x_ji = np.inner(self.l_k, f_k) + self.gamma / self.V
        p_t[0] = p_x_ji * self.alpha / (self.gamma + self.l)

        #print("un-normalized p_t = ", p_t)
        
        return p_t / p_t.sum()
    
    def calc_table_posteriorx(self, j, f_kx):
        using_t = self.using_t[j]
        #print(using_t)
        #print(self.n_jt[j])
        #print(np.sum(self.m_jt[j],1))
        p_t = (self.n_jt[j][using_t]+np.sum(self.m_jt[j][using_t],1)) * f_kx[self.k_jt[j][using_t]]
     
        p_x_ji = np.inner(self.l_k, f_kx) + self.gamma / self.V
        p_t[0] = p_x_ji * self.alpha / (self.gamma + self.l)
        #print("un-normalized p_t = ", p_t)
        return p_t / p_t.sum()
    
    def seat_at_table(self, j, i, t_new):
        assert t_new in self.using_t[j]
        self.t_ji[j][i] = t_new
        self.n_jt[j][t_new] += 1

        k_new = self.k_jt[j][t_new]
        self.n_k[k_new] += 1

        v = self.w_ji[j][i]
        self.n_kv[k_new][v] += 1
        self.n_jtv[j][t_new][v] += 1

    def seat_at_tablex(self, j, i,q, t_new):
        
        assert t_new in self.using_t[j]
        self.tx_ji[j][i][q] = t_new
        self.m_jt[j][t_new][q] += 1
        
        k_new = self.k_jt[j][t_new]
     
        self.m_k[k_new,q] += 1
        self.s_jn[j][i][q]=k_new


    # Assign guest x_ji to a new table and draw topic (dish) of the table
    def add_new_table(self, j, k_new):
        assert k_new in self.using_k
        for t_new, t in enumerate(self.using_t[j]):
            if t_new != t: break
        else:
            t_new = len(self.using_t[j])
            self.n_jt[j].resize(t_new+1)  
            self.m_jt[j].resize([t_new+1,self.dim])
                           
            self.k_jt[j].resize(t_new+1)
            self.n_jtv[j].append(None)

        self.using_t[j].insert(t_new, t_new)
        self.n_jt[j][t_new] = 0  # to make sure
        self.m_jt[j][t_new]=0

        
        self.n_jtv[j][t_new] = DefaultDict(0)

        self.k_jt[j][t_new] = k_new
        self.l_k[k_new] += 1
        self.l += 1

        return t_new

    def calc_dish_posterior_w(self, f_k):
        "calculate dish(topic) posterior when one word is removed"
        p_k = (self.l_k * f_k)[self.using_k]
        p_k[0] = self.gamma / self.V
        return p_k / p_k.sum()

    def calc_dish_posterior_x(self, f_kx,x):
        "calculate dish(topic) posterior when one word is removed"
        p_k = (self.l_k * f_kx)[self.using_k]
        mu_=random.gauss(np.average(self.x_ji), 1)
        sigma=1/np.average(self.lambda_)

        p_newk=np.exp(-1*pow(x-mu_,2)/(2*sigma))/np.sqrt(2*np.pi*sigma)
        
        p_k[0] = self.gamma *p_newk
        return p_k / p_k.sum()

    def sampling_k(self, j, t):
        """sampling k (dish=topic) from posterior"""
        self.leave_from_dish(j, t)

        # sampling of k
        p_k = self.calc_dish_posterior_t(j, t)
        k_new = self.using_k[np.random.multinomial(1, p_k).argmax()]
        if k_new == 0:
            k_new = self.add_new_dish()

        self.seat_at_dish(j, t, k_new)

    def leave_from_dish(self, j, t):
        """
        This makes the table leave from its dish and only the table counter decrease.
        The word counters (n_k and n_kv) stay.
        """
        k = self.k_jt[j][t]
        assert k > 0
        assert self.l_k[k] > 0
        self.l_k[k] -= 1
        self.l -= 1
        if self.l_k[k] == 0:
            self.using_k.remove(k)
            self.k_jt[j][t] = 0

    def calc_dish_posterior_t(self, j, t):
        "calculate dish(topic) posterior when one table is removed"
        k_old = self.k_jt[j][t]     # it may be zero (means a removed dish)
        Vbeta = self.V * self.beta
        n_k = self.n_k.copy()
        m_k = self.m_k.copy()
        n_jt = self.n_jt[j][t]
        m_jt = self.m_jt[j][t]
        n_k[k_old] -= n_jt
        n_k = n_k[self.using_k]
        m_k[k_old] -= m_jt
        m_k = np.sum(m_k[self.using_k])
        log_p_k = np.log(self.l_k[self.using_k]) + gammaln(n_k+m_k) - gammaln(n_k + m_k + n_jt + sum(m_jt))
        log_p_k = np.log(self.l_k[self.using_k]) + gammaln(n_k) - gammaln(n_k + n_jt )
        log_p_k_new = np.log(self.gamma) + gammaln(Vbeta) - gammaln(Vbeta + n_jt)

        gammaln_beta = gammaln(self.beta)
        for w, n_jtw in self.n_jtv[j][t].items():
            assert n_jtw >= 0
            if n_jtw == 0: continue
            n_kw = np.array([n.get(w, self.beta) for n in self.n_kv])
            n_kw[k_old] -= n_jtw
            n_kw = n_kw[self.using_k]
            n_kw[0] = 1 # dummy for logarithm's warning
            if np.any(n_kw <= 0): print(n_kw) # for debug
            log_p_k += gammaln(n_kw + n_jtw) - gammaln(n_kw)
            log_p_k_new += gammaln(self.beta + n_jtw) - gammaln_beta
            
        log_p_k[0] = log_p_k_new
        p_k = np.exp(log_p_k - log_p_k.max())
        return p_k / p_k.sum()

    def seat_at_dish(self, j, t, k_new):
        self.l += 1
        self.l_k[k_new] += 1

        k_old = self.k_jt[j][t]     # it may be zero (means a removed dish)
        if k_new != k_old:
            self.k_jt[j][t] = k_new

            n_jt = self.n_jt[j][t]
            m_jt = self.m_jt[j][t]
            
            if k_old != 0: 
                self.n_k[k_old] -= n_jt
                self.m_k[k_old] -= m_jt
            self.n_k[k_new] += n_jt
            self.m_k[k_new] += m_jt
            for v, n in self.n_jtv[j][t].items():
                if k_old != 0: self.n_kv[k_old][v] -= n
                self.n_kv[k_new][v] += n


    def add_new_dish(self):
        "This is commonly used by sampling_t and sampling_k."
        for k_new, k in enumerate(self.using_k):
            if k_new != k: break
        else:
            k_new = len(self.using_k)
            if k_new >= len(self.n_kv):
                self.n_k = np.resize(self.n_k, k_new + 1)
                self.m_k = np.resize(self.m_k, (k_new + 1,self.dim))
                
                self.eta =np.resize(self.eta, (k_new + 1,self.dim))
                self.a = np.resize(self.a, (k_new + 1,self.dim))
                self.b = np.resize(self.b, (k_new + 1,self.dim))
                self.m = np.resize(self.m, (k_new + 1,self.dim))
                
                self.mu = np.resize(self.mu, (k_new + 1,self.dim))
                self.lambda_ = np.resize(self.lambda_, (k_new + 1,self.dim))
                
                self.l_k = np.resize(self.l_k, k_new + 1)
                self.n_kv.append(None)
            assert k_new == self.using_k[-1] + 1
            assert k_new < len(self.n_kv)

        self.using_k.insert(k_new, k_new)
        self.n_k[k_new] = self.beta * self.V
        self.l_k[k_new] = 0
        self.n_kv[k_new] = DefaultDict(self.beta)
        return k_new
               
def hdplda_learning(hdplda, iteration):
    for i in range(iteration):
        hdplda.inference()
        print("-%d K=%d p=%f" % (i + 1, len(hdplda.using_k)-1, hdplda.perplexity()))
    return hdplda

def test(hdplda,iteration,outname, docs, V, frequency, w, dim,kernel):
    phi = hdplda.worddist()
    documents_stem_id=docs   
        
    documents=frequency

    K=len(hdplda.using_k) - 1
    D=len(documents_stem_id)

    N_dk = np.zeros([D,K])
    N_kv = np.zeros([K,V])        
    N_k  = np.zeros([K,1])
    N_d=np.zeros([D,1])
    for d in range(D):
        N_d[d]=len(documents_stem_id[d])
        theta_k_=np.zeros([K], dtype=np.float64)
    z_dn=[]
    for d in range(D):
        z_dn.append(np.random.randint(0, K,len(documents_stem_id[d])) )

        for i in range(len(z_dn[d])):
            N_dk[d,z_dn[d][i]]+=1
            N_k[z_dn[d][i]]+=1
   
        for v,k in zip(documents_stem_id[d],z_dn[d]):
            N_kv[k,v]+=1

    M_dk = np.zeros([dim,D,K],dtype='int') 
    M_k  = np.zeros([dim,K],dtype='int')
    M_d=np.zeros([dim,D],dtype='int')
    for d in range(D):
        for q in range(dim):
            M_d[q,d]=len(documents[d,:,q])
                
    theta_d=np.zeros([K])
    
    
    s_dn=np.zeros([D,kernel,dim],dtype='int')
    for q in range(dim):
        for d in range(D):
            s_dn[d,:,q]=(np.random.randint(0, K,len(documents[d,:,q])) )

            for i in range(len(s_dn[d,:,q])):
                M_dk[q,d,s_dn[d,i,q]]+=1
                M_k[q,s_dn[d,i,q]]+=1


    k_list=hdplda.using_k[1:]
    mu=hdplda.mu[k_list].T
    lambda_=hdplda.lambda_[k_list].T

    p_s_dn=np.zeros([K])
    
    def f(x,mu,sigma):
        p=np.exp(-1*pow(x-mu,2)/(2*sigma))/np.sqrt(2*np.pi*sigma)
        return p
    numerator=0
    for d in range(D):
        sys.stdout.write("\r%d / %d" % (d+1, D))
        sys.stdout.flush()
        sgoto=0

        for i in range(iteration):

            
            for q in range(dim):

                for n in np.random.permutation(len(documents[d,:,q])):
                    current_topic = s_dn[d,n,q]
                    M_dk[q,d, current_topic] -= 1
                    M_k[q,current_topic] -= 1

                    sample_x=preprocessing.normalize(w[d,:,q].reshape(1, -1), norm="l1")[0]
                    documents[d,n,q]=documents[d,np.argmax(np.random.multinomial(1, sample_x, size=1)),q]                            
             
                    for k in range(K):
                        theta_d=(N_dk[d,k]+np.sum(M_dk[:,d,k])+0.1)/(N_d[d]+np.sum(M_d[:,d])-1+0.1*K)
                        temp=f(documents[d,n,q],mu[q,k],1/lambda_[q,k])

                        p_s_dn[k]=np.exp(np.log(theta_d)+np.log(temp))

                    p_s_dn = preprocessing.normalize(p_s_dn.reshape(1, -1), norm="l1")[0] 
                    new_topic=np.argmax(np.random.multinomial(1, p_s_dn, size=1))
                    s_dn[d,n,q]=new_topic
                          
                    M_dk[q,d, new_topic] += 1
                    M_k[q,new_topic] += 1

            loglikelihood=0
            for n in np.random.permutation(len(documents_stem_id[d])):
                current_topic = z_dn[d][n]
                v=documents_stem_id[d][n]
               
                N_dk[d, current_topic] -= 1
                N_kv[current_topic, v] -= 1
                N_k[current_topic] -= 1
                theta_phi=0
                if (N_kv[current_topic, v]<0):
                    print(N_kv[current_topic, v])
                        
                                
                   
                p_z_dn = np.zeros(K)
                theta_phi=0
    
                for k, phi_k in enumerate(phi):        

                    p = (N_dk[d,k] + sum(M_dk[:,d,k]) + 0.1) * phi_k[v]
          
                    if(p  < 0):
                        break
                    p_z_dn[k] = p
                                        
                    theta_k = (N_dk[d,k]+sum(M_dk[:,d,k])+0.1) / (N_d[d]-1+sum(M_d[:,d])+0.1*K)                       
                    theta_k_[k]=theta_k[0]


                    
                for k, phi_k in enumerate(phi): 
                    theta_k_=preprocessing.normalize(theta_k_.reshape(1, -1), norm="l1")[0]
 
                    theta_phi +=np.exp(np.log(theta_k_[k])+np.log(phi_k[v]))

                
                loglikelihood += np.log(theta_phi)
                p_z_dn = preprocessing.normalize(p_z_dn.reshape(1, -1), norm="l1")[0] 
                new_topic=np.argmax(np.random.multinomial(1, p_z_dn, size=1))
                z_dn[d][n]=new_topic
                                                  
                N_dk[d, new_topic] += 1
                N_kv[new_topic, v] += 1
                N_k[new_topic] += 1
            
            sgoto += np.exp(loglikelihood)
        numerator+=np.log(sgoto)
    perplexityw=np.exp(-1*numerator/np.sum(N_d))
    print('\n'+'test_perplexity:{}'.format(perplexityw))    
    
def output_summary(hdplda, data,outname):
    fp=open('C:/Users/tom/src/hdpsmlda/output/summary_{}.txt'.format(outname),'w',encoding='utf-8')
    V=len(data)
    K = len(hdplda.using_k) - 1
    kmap = dict((k,i-1) for i, k in enumerate(hdplda.using_k))
    dishcount = np.zeros(K, dtype=int)
    wordcount = [DefaultDict(0) for k in range(K)]
    for j, w_ji in enumerate(hdplda.w_ji):
        for v, t in zip(w_ji, hdplda.t_ji[j]):
            k = kmap[hdplda.k_jt[j][t]]
            dishcount[k] += 1
            wordcount[k][v] += 1

    phi = hdplda.worddist()
    for k, phi_k in enumerate(phi):
        fp.write("\n-- topic: %d (%d words)\n" % (hdplda.using_k[k+1], dishcount[k]))
        for w in sorted(phi_k, key=lambda w:-phi_k[w])[:]:
            if(wordcount[k][w]==0):
                continue
            fp.write("%s: %f (%d)\n" % (data[w], phi_k[w], wordcount[k][w]))

    fp.write("--- document-topic distribution\n")
    theta = hdplda.docdist()
    for j, theta_j in enumerate(theta):
        fp.write("%d\t%s\n" % (j, "\t".join("%.3f" % p for p in theta_j[1:])))

    fp.write("--- dishes for document\n")
    for j, using_t in enumerate(hdplda.using_t):
        fp.write("%d\t%s\n" % (j, "\t".join(str(hdplda.k_jt[j][t]) for t in using_t if t>0)))        
    fp.close()
    
    k_list=hdplda.using_k[1:]
    np.savetxt('C:/Users/tom/src/hdpsmlda/output/mu_{}.txt'.format(outname),hdplda.mu[k_list])
    np.savetxt('C:/Users/tom/src/hdpsmlda/output/lambda_{}.txt'.format(outname),hdplda.lambda_[k_list])
    phi_kv=np.zeros([K,V])
    for k, phi_k in enumerate(phi):
        for v in range(V):
            phi_kv[k,v]=phi_k[v]
    np.savetxt('C:/Users/tom/src/hdpsmlda/output/phi_{}.txt'.format(outname),phi_kv)
    fig, ax = plt.subplots()
    ax.plot(hdplda.perp)
    plt.show()

def advgene(hdplda,V, frequency_test, w_test, dim,kernel,voca,test_num,outname):
    fp=open('C:/Users/tom/src/hdpsmlda/output/test_{}.txt'.format(outname),'w',encoding='utf-8')
    K=len(hdplda.using_k) - 1
    phi_=hdplda.worddist()
    phi=np.zeros([K,V])
    for v in range(V):
        for k, phi_k in enumerate(phi_):
            phi[k,v]=phi_k[v]

    epoch=100
    V=len(voca)
    tmpp=np.ones([K,V])

    id_word={}            
    for i in range(V):
        id_word.update([(i,voca[i])])


    k_list=hdplda.using_k[1:]
    mu=hdplda.mu[k_list].T
    lambda_=hdplda.lambda_[k_list].T
    smkdata=frequency_test
    w=w_test
    def f(x,mu,sigma):
        p=np.exp(-1*pow(x-mu,2)/(2*sigma))/np.sqrt(2*np.pi*sigma)
        return p

    for i in range(len(smkdata)):
        for d in range(dim):
            p_s_dn=np.zeros(K)
            theta_=np.zeros(K)
            topi=np.zeros(len(smkdata[i])*20,dtype='int')
            topic_num=K
            phitheta=np.ones([K,V])
            sample_x=preprocessing.normalize(w[i,:,d].reshape(1, -1), norm="l1")[0]
            for e in range(epoch):
                for x in range(kernel):         
                    target=smkdata[i,np.argmax(np.random.multinomial(1, sample_x, size=1)),d]
                  
                
                    for k in range(K):
                        p_s_dn[k]=f(target,mu[d,k],1/lambda_[d,k])
                    
                    p=preprocessing.normalize(p_s_dn.reshape(1, -1), norm="l1")[0]
                    topi[20*x:20*(x+1)]=np.argmax(np.random.multinomial(1, p, size=20))
            
                c=collections.Counter(topi)
            
                for k in range(K):
                    theta_[k]=c[k]+1
                theta=preprocessing.normalize(theta_.reshape(1, -1), norm="l1")[0]
            
                for k in range(K):        
                    phitheta[k,:]*=phi[k,:]*theta[k]
                    
                
            tmp=np.power(phitheta,1/epoch)
            tmpp*=tmp

        tmpp=np.power(tmpp,1/dim)
        
        fp.write('*******{}につけられた副詞*******{}\n'.format(test_num[i]))
        for j in range(10):
            idx=np.unravel_index(np.argmax(tmp), tmp.shape)
            fp.write(id_word[idx[1]].replace('\n',''))
            tmp[idx]=0
        fp.write('\n')
    fp.close()
    


    
def main ():
    dircheck()
    outname="aist10p2"#input()
    
    file1="C:/Users/tom/src/lda/temp/documents/aist_documents_data.pickle"#input()
    file2="C:/Users/tom/src/lda/temp/vocabulary/aist_vocabulary_data.txt"#input()
    with open(file1,"rb")as f0:
        documents=pickle.load(f0)
        lines=len(documents)

    dim=3
    kernel=10
  
    tmp=np.zeros([lines,kernel,dim]) 
    x_mu=np.zeros([dim])
    wtmp=np.zeros([lines,kernel,dim])
    for i in range(dim):
        a=np.loadtxt("C:/Users/tom/src/input/smkdata/aist10_{}d.txt".format(i+1))
        w=np.loadtxt("C:/Users/tom/src/input/smkdata/aist10_w{}d.txt".format(i+1))
       
        tmp[:,:,i]=a
        wtmp[:,:,i]=w
        x_mu[i]=np.average(a)  
    
        
    #train_num=random.sample(range(lines),int(lines*1/12))#0.95))
    train_num=np.loadtxt('C:/Users/tom/src/f_lda/train_num/train_aist10.txt',dtype='int')
    test_num=list(set(range(lines))-set(train_num))
    np.savetxt('C:/Users/tom/src/f_lda/train_num/train_{}.txt'.format(outname),train_num)
    frequency=tmp[train_num]
    w=wtmp[train_num]
    frequency_test=tmp[test_num]
    w_test=wtmp[test_num]

    docs=[]
    docs_test=[]
    for i in train_num:
        docs.append(documents[i])
    for i in test_num:
        docs_test.append(documents[i])
    lines=len(docs)
    with open(file2,"r",encoding='utf-8')as f1:
        voca=f1.readlines()
        V=len(voca)    
    iteration=3000
    iteration_test=200
    beta=0.1
    alpha=0.1#np.random.gamma(1, 1)
    gamma=0.01#np.random.gamma(1, 1)

    hdplda = HDPSMLDA(alpha, gamma, beta, docs, V, frequency, w, dim,kernel)
    print("corpus=%d words=%d alpha=%.3f gamma=%.3f beta=%.3f" % (lines, V, alpha, gamma, beta))
    
    hdplda_learning(hdplda, iteration)
    output_summary(hdplda,voca,outname)
    test(hdplda,iteration_test,outname, docs_test, V, frequency_test, w_test, dim,kernel)
    advgene(hdplda,V, frequency_test, w_test, dim,kernel,voca,test_num,outname)

    
    


if __name__ == "__main__":
    main ()   
