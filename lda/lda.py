# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 17:25:38 2020

@author: tom9m
"""

import scipy.special
from sklearn import preprocessing
import numpy as np
import sys
import os
import pickle
import random

def main ():
    """
    if not os.path.isdir('C:/Users/tom/src/f_lda/n_kv'):
        os.makedirs('C:/Users/tom9m/prog/f_lda/n_kv')
    if not os.path.isdir('C:/Users/tom9m/prog/f_lda/n_k'):
        os.makedirs('C:/Users/tom9m/prog/f_lda/n_k')
    if not os.path.isdir('C:/Users/tom9m/prog/f_lda/n_dk'):
        os.makedirs('C:/Users/tom9m/prog/f_lda/n_dk')        
    if not os.path.isdir('C:/Users/tom9m/prog/f_lda/perplexity'):
        os.makedirs('C:/Users/tom9m/prog/f_lda/perplexity') 
    if not os.path.isdir('C:/Users/tom9m/prog/f_lda/beta'):
        os.makedirs('C:/Users/tom9m/prog/f_lda/beta') 
    if not os.path.isdir('C:/Users/tom9m/prog/f_lda/train_num'):
        os.makedirs('C:/Users/tom9m/prog/f_lda/train_num') 
        
    if not os.path.isdir('C:/Users/tom/src/f_lda/alpha'):
        os.makedirs('C:/Users/tom/src/f_lda/alpha')     
    if not os.path.isdir('C:/Users/tom/src/f_lda/beta'):
        os.makedirs('C:/Users//tom/src/f_lda/beta') 
    """
    perpw_list=[]
    digamma=scipy.special.digamma
    #print('Please, input document data.')
    file1="C:/Users/tom/src/lda/temp/documents/sakubunpastmeishi_documents_data.pickle"#input()
    #print('Please, input vocabulary data.')
    file2="C:/Users/tom/src/lda/temp/vocabulary/sakubunpastmeishi_vocabulary_data.txt"#input()
    #print('Please, the name of output')
    outname="pastmeishi10"#input()
    with open(file1,"rb")as f0:
        documents_stem_id_ori=pickle.load(f0)
        lines=len(documents_stem_id_ori)
    with open(file2,"rb")as f1:
        data=f1.readlines()
        V=len(data)
    #print('Please,input the number of topics')
    K=10#int(input()) #トピック数の指定
    #print('Please,input the number of epoch')
    epoch=1000#int(input()) #トピック数の指定
    
    train_num=random.sample(range(lines),int(lines*1))
    #train_num=np.loadtxt('C:/Users/tom9m/prog/f_lda/train_num/train_conductnosample10.txt',dtype='int')
    #np.savetxt('train_num/train_{}.txt'.format(outname),train_num)
    documents_stem_id=documents_stem_id_ori#[]
    """
    for i in train_num:
        documents_stem_id.append(documents_stem_id_ori[i])
    """
    D=len(documents_stem_id)#文書数の指定
    
    #V=int(sys.argv[3])#語彙数の指定
    N_dk = np.zeros([D,K]) #文書dでトピックkが割り振られた単語数
    N_kv = np.zeros([K,V]) #文書集合全体で語彙vにトピックkが割り振られた単語数        
    N_k  = np.zeros([K,1]) #文書集合全体でトピックkが割り振られた単語数
    N_d=np.zeros([D,1])#各ドキュメントの長さ
    for d in range(D):
        N_d[d]=len(documents_stem_id[d])
        #文書dのn番目の単語に付与されたトピック
        theta_k_=np.zeros([K])
        phi_kv_=np.zeros([K,V])
    
    z_dn=[]
    for d in range(D):
        z_dn.append(np.random.randint(0, K,len(documents_stem_id[d])) )
        #N_dkとN_kについて
        for i in range(len(z_dn[d])):
            N_dk[d,z_dn[d][i]]+=1
            N_k[z_dn[d][i]]+=1
            #N_kvについて    
        for v,k in zip(documents_stem_id[d],z_dn[d]):
            N_kv[k,v]+=1
    
    alpha=np.ones([K],dtype='float')*50/K
    beta=np.ones([1],dtype='float')*0.1

    for i in range(epoch):

        print("Epoch: {}\n".format(i+1))

        numerator_p = 0
        #denominator_p = 0
        
        for d in range(D):
            sys.stdout.write("\r%d / %d" % (d+1, D))
            sys.stdout.flush()
            loglikelihood=0
            if(sum(N_dk[d],0)==0):  
                continue
            for n in np.random.permutation(len(documents_stem_id[d])):#単語をバラバラに見る
                current_topic = z_dn[d][n]
                v=documents_stem_id[d][n]
             
                #if(current_topic>0):#自身のカウントを引く
                N_dk[d, current_topic] -= 1
                N_kv[current_topic, v] -= 1
                N_k[current_topic] -= 1
                theta_phi=0
                if (N_kv[current_topic, v]<0):
                    print(N_kv[current_topic, v])
            
                    
                    #サンプリング確率と尤度を計算-----------------------------------------------------------
                p_z_dn = np.zeros(K)
                theta_phi=0
                
                for k in range(K):
                            
                    A = N_dk[d,k] + alpha[k]
                    B = (N_kv[k,v] + beta)/(N_k[k] + beta*V)
                            
                    p = A * B 
                    if(p  < 0):
                        break
                    p_z_dn[k] = p
                            
                    theta_k = (N_dk[d,k]+alpha[k]) / (N_d[d]-1+np.sum(alpha)) # 
                          
                    theta_k_[k]=theta_k[0]
                    
                    phi_kv = (N_kv[k,v]+beta) /(N_k[k]+beta*V) #
                    phi_kv_[k,v]=phi_kv[0]
                  
                for k in range(K):
                    phi_kv_[k,:]=preprocessing.normalize(phi_kv_[k,:].reshape(1, -1), norm="l1")[0]
                    theta_k_=preprocessing.normalize(theta_k_.reshape(1, -1), norm="l1")[0]
                    theta_phi +=theta_k_[k]*phi_kv_[k,v]
                #print(sum(np.dot(theta_k_,phi_kv_)))            
                loglikelihood += np.log(theta_phi)
                p_z_dn = preprocessing.normalize(p_z_dn.reshape(1, -1), norm="l1")[0] # 正規化
            
                    #-------------------------------------------------------------------------------
                        
                    #カテゴリカル分布を使って文書dのn番目の単語のトピックをサンプリング   
                new_topic=np.argmax(np.random.multinomial(1, p_z_dn, size=1))#最大となるインデックスを返す
                z_dn[d][n]=new_topic
                                      
                N_dk[d, new_topic] += 1
                N_kv[new_topic, v] += 1
                N_k[new_topic] += 1
            numerator_p += loglikelihood
            

            #denominator_p += N_d[d]
            
            
                        #  パラメータ更新
                        #α トピック分布用のパラメータ
        for k in range(K):
            numerator=0
            denominator=0
            for d in range(D):
                numerator +=digamma(N_dk[d][k]+alpha[k])- digamma(alpha[k])
                denominator += digamma(N_d[d]+np.sum(alpha))- digamma(np.sum(alpha))
            alpha[k] = alpha[k]*(numerator / denominator)
            if(alpha[k]<=0):
                alpha[k]=0.000001

  
                        
                        #β 単語分布用のパラメータ
        numerator = np.sum(digamma(N_kv+beta)) - K*V*digamma(beta)
        denominator = V*(np.sum(digamma(N_k+beta*V)) - K*digamma(beta*V))
        beta = beta*(numerator / denominator)
        
        perplexityw=np.exp(-1*numerator_p/np.sum(N_d))
        print('\n'+'w_perplexity:{}'.format(perplexityw))
        perpw_list.append(perplexityw)            
    
                    #パラメータ出力
        print("\nparameters")
        print("alpha :{}".format(alpha))
        print("beta :{}".format(beta))
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(perpw_list)
    plt.show()
      

    
    #np.savetxt('perplexity/{}_w_perplexity.txt'.format(outname),perpw_list)     
    np.savetxt('C:/Users/tom/src/lda/temp/n_kv/n_kv_{}.txt'.format(outname),N_kv)   
    np.savetxt('C:/Users/tom/src/lda/temp/n_k/n_k_{}.txt'.format(outname),N_k)
    np.savetxt('C:/Users/tom/src/lda/temp/n_dk/n_dk_{}.txt'.format(outname),N_dk)
    np.savetxt('C:/Users/tom/src/lda/temp/theta_k/theta_k_{}.txt'.format(outname),theta_k_)
    np.savetxt('C:/Users/tom/src/lda/temp/phi_kv/phi_kv_{}.txt'.format(outname),phi_kv_)
    np.savetxt('C:/Users/tom/src/lda/temp/alpha/alpha_{}.txt'.format(outname),alpha)
    
   

    #np.savetxt('perplexity/{}_w_perplexity.txt'.format(outname),perpw_list)
    #np.savetxt('beta/beta_{}.txt'.format(outname),beta)
if __name__ == "__main__":
    main ()   
