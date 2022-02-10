# -*- coding: utf-8 -*-
"""
Created on Sat May  9 03:08:20 2020

@author: tom9m

"""
import numpy as np
#import sys
#import re
print('Please, input vocabulary data.')
file2="C:/Users/tom/src/lda/temp/vocabulary/futuretri_vocabulary_data.txt"#input()
print('Please, input the name of output')
name='futuretri10'#input()
print('Please, input top[-]')
top=10#int(input())

def logpw(data):
    numerator=np.sum(data, axis=0)
    denominator=np.sum(data)
    
    pw=numerator/denominator
    new_pw=np.zeros([K,V])
    for i in range(K):    
        new_pw[i]=np.log(pw)
    return (new_pw)

def logpk(data):
    new_pk=np.zeros([V,K])
    for i in range(V):
       new_pk[i]=np.log(data)
    return(new_pk.reshape(K,V))

with open('C:/Users/tom/src/lda/temp/n_kv/n_kv_{}.txt'.format(name))as f1:
    data1=np.loadtxt(f1)

with open('C:/Users/tom/src/lda/temp/theta_k/theta_k_{}.txt'.format(name))as f2:
    data2=np.loadtxt(f2)

with open('C:/Users/tom/src/lda/temp/phi_kv/phi_kv_{}.txt'.format(name))as f3:
    data3=np.loadtxt(f3)

K=len(data3)
V=len(data3[0])
print(data1.shape)
print(data2.shape)
print(data3.shape)
logpk=logpk(data2)
logpw_k=np.log(data3+1)
logpw=logpw(data1+1)

numerator=logpw_k-logpw
denominator=-logpw_k-logpk
npmi=numerator/denominator

id_word={}
#name=re.findall('(.*)_[0-9]*',name)[0]
with open(file2,encoding='utf-8') as f1:
    lines=f1.readlines()
word=[]            
for i in range(len(lines)):
    id_word.update([(i,lines[i])])
for i in range(len(npmi)):
    print('-----topic{}-----'.format(i+1))
    tmp=[]
    for j in range(top):
        tmp.append(id_word[np.argmax(npmi[i])].replace('\n',''))
        print('{}'.format(id_word[np.argmax(npmi[i])].replace('\n','')))
        npmi[i,np.argmax(npmi[i])]=-2
    word.append(tmp)
print(len(word))
for j in range(top):
    print(word[0][j]+"&"+word[1][j]+"&"+word[2][j]+"&"+word[3][j]+"&"+word[4][j]+"&"+word[5][j]+"&"+word[6][j]+"&"+word[7][j]+"&"+word[8][j]+"&"+word[9][j]+'\\'+'\\')

"""
for i in range(len(npmi)):
    print('-----topic{}-----'.format(i+1))
    for j in range(top):
        if(len(id_word[np.argmax(npmi[i])].replace('\n',''))>6):
            print('{}:\t{:.3f}'.format(id_word[np.argmax(npmi[i])].replace('\n',''),npmi[i,np.argmax(npmi[i])]))
        else:
            print('{}:\t\t{:.3f}'.format(id_word[np.argmax(npmi[i])].replace('\n',''),npmi[i,np.argmax(npmi[i])]))
        npmi[i,np.argmax(npmi[i])]=-2    
"""
 
       
    

    