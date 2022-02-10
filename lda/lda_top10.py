# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 18:44:21 2020

@author: tom9m
"""

import numpy as np
from sklearn import preprocessing
np.set_printoptions(precision=3,suppress=True)
#print(np.get_printoptions())



def main():
    print('Please, input vocabulary data.')
    file2="vocabulary/pastbi_vocabulary_data.txt"#input()
    print('Please, input the name of output')
    name='pastbi5'#input()
    print('Please, input top[-]')
    top=7#int(input())

    
    id_word={}
    with open(file2,encoding='utf-8') as f1:
        lines=f1.readlines()
            
        for i in range(len(lines)):
            id_word.update([(i,lines[i])])
    N_kv=np.loadtxt('n_kv/n_kv_{}.txt'.format(name))
    sum_yoko=np.sum(N_kv,axis=1)
    sum_tate=np.sum(N_kv,axis=0)
    print(sum_yoko )
    
    sum_w=np.sum(N_kv)
    p_w=sum_tate/sum_w
    p_wk=np.zeros([len(sum_yoko),len(sum_tate)])
    for k in range(len(sum_yoko)):
        if(sum_yoko[k]==0):
            continue
        print('-----topic{}-----'.format(k+1))
        for v in range(len(sum_tate)):


            
            p_wk[k,v]=N_kv[k,v]/sum_yoko[k]/p_w[v]
        p_wk[k]=preprocessing.normalize(p_wk[k].reshape(1, -1), norm="l1")[0]
        for i in range(top):
            print('{}:\t{:.5f}'.format(id_word[np.argmax(p_wk[k])].replace('\n',''),np.max(p_wk[k])))
            p_wk[k,np.argmax(p_wk[k])]=0
"""                


if __name__ == "__main__":

    main ()   
