import pickle
import collections    
import matplotlib.pyplot as plt


file1="C:/Users/tom/src/lda/temp/documents/walkover3_documents_data.pickle"#input()
file2="C:/Users/tom/src/lda/temp/vocabulary/walkover3_vocabulary_data.txt"#input()
with open(file1,"rb")as f0:
    documents=pickle.load(f0)
    lines=len(documents)
with open(file2,"r",encoding='utf-8')as f1:
    voca=f1.readlines()
    V=len(voca)  
alldoc=[]
for i in documents:
    for j in i:
        alldoc.append(j)
c=collections.Counter(alldoc)
print(c)
count=[]
adv=[]
for k,v in c.items():
    adv.append(voca[k].replace('\n',''))
    count.append(v)

for i in range(int(len(adv)/100)):
    if(i==int(len(adv)/100)):
        print('a')
        adv_=adv[i*100:]
        count_=count[i*100:]  
    else:      
        adv_=adv[i*100:(i+1)*100]
        count_=count[i*100:(i+1)*100]
    fig, ax = plt.subplots()
    ax.bar(adv_, count_)
    plt.xticks(rotation=90, fontname="MS Gothic",fontsize=7)
    plt.show()
