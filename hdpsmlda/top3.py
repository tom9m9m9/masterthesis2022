# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 18:31:57 2021

@author: tom9m
"""

with open('C:/Users/tom/src/hdpsmlda/output/summary_aist4.txt','r',encoding='utf-8') as f:
    lines=f.readlines()

flag=0

for i in range(len(lines)):
    
    if('-- topic:' in lines[i]):

        print(lines[i].replace('\n','')[3:])
        for j in range(1,31,2):
            if(lines[i+j]=='\n'):
                flag=1
                print('')
            elif(flag==1):
                print('')
            else: print(lines[i+j].replace('\n',''))
        flag=0

word=[]     
for i in range(len(lines)):
    
    if('-- topic:' in lines[i]):
        tmp=[]
        for j in range(1,11,2):
            if(lines[i+j]=='\n'):
                flag=1
                tmp.append('')
            elif(flag==1):
                tmp.append('')
            else: tmp.append(lines[i+j].replace('\n',''))
        word.append(tmp)
        flag=0

print(word)

print(len(tmp))

for j in range(len(tmp)):
    a=''
    for i in range(16,18):
        a+=word[i][j]+"&"
    
    print(a[:-1]+'\\'+'\\')
