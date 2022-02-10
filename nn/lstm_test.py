import torch
import torch.nn as nn
import numpy as np
import pickle

#make data
frame=np.loadtxt('walk_framedata.txt')
gp=np.loadtxt('walk_data.txt')
min_=int(min(frame))
tmp=np.zeros([int(min_*len(frame)),int(gp.shape[1])])
frame_=0
for i in range(len(frame)):
    frame_=int(frame_)
    tmp[i*min_:(i+1)*min_,:]=gp[frame_:(frame_+min_),:]
    frame_+=int(frame[i])
data=np.zeros([int(min_),int(gp.shape[1]),len(frame)])

for i in range(len(frame)):
    data[:,:,i]=tmp[i*min_:(i+1)*min_,:]
name="walkori"

#make label
file1="walkover3_documents_data.pickle"#input()
file2="walkover3_vocabulary_data.txt"#input()

with open(file1,"rb")as f0:
    doc=pickle.load(f0)
    lines=len(doc)
with open(file2,"r",encoding='utf-8')as f1:
    voca=f1.readlines()
    V=len(voca) 

advall=[]
for d in range(lines):
    advset=set(doc[d])
    adv=[]
    for i in advset:
        tmp=np.zeros([V],dtype='int')
        tmp[i]+=1
        adv.append(tmp)
    advall.append(adv)

advall=[]
for d in range(lines):
    advset=set(doc[d])
    adv=[]
    for i in advset:
        adv.append(i)
    advall.append(adv)

train_num=np.loadtxt('train_walk10.txt')
test_num=list(set(range(lines))-set(train_num))
train_adv=[]
test_adv=[]
train_x=np.zeros([int(min_),int(gp.shape[1]),len(train_num)])
test_x=np.zeros([int(min_),int(gp.shape[1]),len(test_num)])
tr=0
te=0
train_advset=[]
for i in range(len(frame)):
    if(i in train_num):
        train_x[:,:,tr]=data[:,:,i]
        train_adv.append(advall[i])
        tr+=1
        for j in doc[i]:
            train_advset.append(j)
    else:
        test_x[:,:,te]=data[:,:,i]
        test_adv.append(advall[i])
        te+=1
train_advset=set(train_advset)
class LSTMClassifier(nn.Module):
    # モデルで使う各ネットワークをコンストラクタで定義
    def __init__(self, embedding_dim, hidden_dim,  tagset_size):
        # 親クラスのコンストラクタ。決まり文句
        super(LSTMClassifier, self).__init__()
        # 隠れ層の次元数。これは好きな値に設定しても行列計算の過程で出力には出てこないので。
        self.hidden_dim = hidden_dim
        # インプットの単語をベクトル化するために使う
        #self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # LSTMの隠れ層。これ１つでOK。超便利。
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        # LSTMの出力を受け取って全結合してsoftmaxに食わせるための１層のネットワーク
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        # softmaxのLog版。dim=0で列、dim=1で行方向を確率変換。
        self.softmax = nn.LogSoftmax(dim=1)

    # 順伝播処理はforward関数に記載
    def forward(self, sentence):
        # 文章内の各単語をベクトル化して出力。2次元のテンソル
        embeds = sentence
        # 2次元テンソルをLSTMに食わせられる様にviewで３次元テンソルにした上でLSTMへ流す。
        # 上記で説明した様にmany to oneのタスクを解きたいので、第二戻り値だけ使う。
        
        _, lstm_out = self.lstm(embeds.view(len(sentence), 1, -1))
        # lstm_out[0]は３次元テンソルになってしまっているので2次元に調整して全結合。
        tag_space = self.hidden2tag(lstm_out[0].view(-1, self.hidden_dim))
        # softmaxに食わせて、確率として表現
        tag_scores = self.softmax(tag_space)
        return tag_scores
# 単語のベクトル次元数
EMBEDDING_DIM = int(gp.shape[1])
# 隠れ層の次元数
HIDDEN_DIM = 128
# 分類先のカテゴリの数
TAG_SIZE = V
# モデル宣言
model = LSTMClassifier(EMBEDDING_DIM, HIDDEN_DIM,  TAG_SIZE)
model.load_state_dict(torch.load("model_{}.pth".format(name)))
# テストデータの母数計算

# 正解の件数
a = 0
# 勾配自動計算OFF
print(test_num)
with torch.no_grad():
    perp=0
    for i in range(len(test_num)):
        
        title=torch.from_numpy(test_x[:,:,i].astype(np.float32)).clone()
        #cat= torch.tensor(test_adv[i]).view(-1,1)
        # テストデータの予測
        #inputs = sentence2index(title)
        out = model(title)
        prob=torch.exp(out)[0,:]
        
        pe=0
        for j in doc[test_num[i]]:
            if(j not in train_advset):
                print(j)
                continue
            pe+=torch.log(prob[j])
        perp+=torch.exp(-1*(pe)/(len(doc[test_num[i]])))
    print(perp/len(test_num))
            
"""
        # outの一番大きい要素を予測結果をする
        _, predict = torch.max(out, 1)

        answer = category2tensor(category)
        if predict == answer:
            a += 1
print("predict : ", a / test_num)
"""