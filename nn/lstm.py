import torch
import torch.nn as nn
import numpy as np
import pickle
#make data
frame=np.loadtxt('aist_framedata_fps3.txt')
gp=np.loadtxt('aist_data_fps3_gplvm.txt')
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
name="aistgp"

#make label
file1="aist_documents_data.pickle"#input()
file2="aist_vocabulary_data.txt"#input()

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

train_num=np.loadtxt('train_aist10.txt')
test_num=list(set(range(lines))-set(train_num))
train_adv=[]
test_adv=[]
train_x=np.zeros([int(min_),int(gp.shape[1]),len(train_num)])
test_x=np.zeros([int(min_),int(gp.shape[1]),len(test_num)])
tr=0
te=0
for i in range(len(frame)):
    if(i in train_num):
        train_x[:,:,tr]=data[:,:,i]
        train_adv.append(advall[i])
        tr+=1
    else:
        test_x[:,:,te]=data[:,:,i]
        test_adv.append(advall[i])
        te+=1

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

import torch.optim as optim
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 単語のベクトル次元数
EMBEDDING_DIM = int(gp.shape[1])
# 隠れ層の次元数
HIDDEN_DIM = 128
# 分類先のカテゴリの数
TAG_SIZE = V
# モデル宣言
model = LSTMClassifier(EMBEDDING_DIM, HIDDEN_DIM,  TAG_SIZE).to(device)
# 損失関数はNLLLoss()を使う。LogSoftmaxを使う時はこれを使うらしい。
loss_function = nn.NLLLoss()
# 最適化の手法はSGDで。lossの減りに時間かかるけど、一旦はこれを使う。
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 各エポックの合計loss値を格納する
losses = []
# 100ループ回してみる。（バッチ化とかGPU使ってないので結構時間かかる...）

for epoch in range(1000):
    all_loss = 0
    for i in range(len(train_num)):
        
        title=torch.from_numpy(train_x[:,:,i].astype(np.float32)).clone().to(device)
        cat= torch.tensor(train_adv[i]).view(-1,1)
        #print(cat[i].shape)
        # モデルが持ってる勾配の情報をリセット
        model.zero_grad()
        # 文章を単語IDの系列に変換（modelに食わせられる形に変換）

        #inputs = sentence2index(title)
        # 順伝播の結果を受け取る
        out = model(title)
        # 正解カテゴリをテンソル化
        #answer = category2tensor(cat)
        # 正解とのlossを計算
        loss=0
        for j in range(len(cat)):
            ans=cat[j].to(device)
            #loss=nn.CrossEntropyLoss(out, ans)
            loss += loss_function(out, ans)
        loss=loss/len(cat)    
        # 勾配をセット
        loss.backward()
        # 逆伝播でパラメータ更新
        optimizer.step()
        # lossを集計
        all_loss += loss.item()
    losses.append(all_loss)
    print("epoch", epoch, "\t" , "loss", all_loss)
print("done.")
np.savetxt('loss_{}.txt'.format(name),losses)
torch.save(model.state_dict(), "model_{}.pth".format(name))
