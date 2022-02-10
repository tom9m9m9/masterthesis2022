import pickle
import collections
import numpy as np
file1="C:/Users/tom/src/lda/temp/documents/futuretri_documents_data.pickle"#input()
file2="C:/Users/tom/src/lda/temp/vocabulary/futuretri_vocabulary_data.txt"#input()
with open(file1,"rb")as f0:
    documents=pickle.load(f0)
    lines=len(documents)
with open(file2,encoding='utf-8') as f1:
    voc=f1.readlines()
alldoc=[]
mean=0
bunbo=0
for i in documents:
    if(i==[]):
        continue
    mean+=len(i)
    bunbo+=1
    for j in i:
        alldoc.append(j)
print(mean/bunbo)
V=len(voc)
c=collections.Counter(alldoc)
def perplexity(w_p,documents):

    log_likelihood = 0
    N = 0
    for x_ji in documents:
        for v in x_ji:

            word_prob = w_p[v]
            log_likelihood -= np.log(word_prob)
        N += len(x_ji)
    p=np.exp(log_likelihood / N)
    return p
print(len(c))
print(V)
w_p=np.zeros(V)
de=0
for i in range(V):
    de+=c[i]
for i in range(V):
    w_p[i]=c[i]/de

print(perplexity(w_p,documents))

