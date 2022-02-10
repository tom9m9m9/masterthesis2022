import torch
import torch.nn as nn
import numpy as np
import pickle
import torch.nn.functional as F
import torch.optim as optim
name="walkmlp4"

#make label
file1="walkover3_documents_data.pickle"#input()
file2="walkover3_vocabulary_data.txt"#input()
kernel=4
dim=3
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

tmp=np.zeros([lines,kernel,dim]) 
wtmp=np.zeros([lines,kernel,dim])
for i in range(dim):
    a=np.loadtxt("smk/walk4_{}d.txt".format(i+1))
    w=np.loadtxt("smk/walk4_w{}d.txt".format(i+1))       
    tmp[:,:,i]=a
    wtmp[:,:,i]=w
fre=np.zeros([lines,kernel*dim])
for d in range(lines):
    for i in range(dim):
        for k in range(kernel):
            sample_x=wtmp[d,:,i]/sum(wtmp[d,:,i])
            idx=np.argmax(np.random.multinomial(1, sample_x, size=1))
            fre[d,i*kernel+k]=tmp[d,idx,i]

train_num=np.loadtxt('train_walk10.txt')
test_num=list(set(range(lines))-set(train_num))
train_adv=[]
test_adv=[]
train_x=np.zeros([len(train_num),kernel*dim])
test_x=np.zeros([len(test_num),kernel*dim])
tr=0
te=0
for i in range(lines):
    if(i in train_num):
        train_x[tr,:]=fre[i,:]
        train_adv.append(advall[i])
        tr+=1
    else:
        test_x[te,:]=fre[i,:]
        test_adv
        te+=1


class MLPNet (nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(kernel*dim, 512)   
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, V)
        self.dropout1 = nn.Dropout2d(0.2)
        self.dropout2 = nn.Dropout2d(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return F.relu(self.fc3(x))
 
# select device
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = MLPNet()#.to(device)
 
# optimizing
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
print ('training start ...')
 
losses = [] 
# initialize list for plot graph after training
 
for epoch in range(1000):
    all_loss = 0
    # ======== train_mode ======
    net.train()
    for i in range(len(train_num)):
        
        title=torch.from_numpy(train_x[i,:].astype(np.float32)).clone()#.to(device)
        cat= torch.tensor(train_adv[i]).view(-1,1)
        optimizer.zero_grad()  # 勾配リセット
 
        out = net(title.view(1,title.shape[0]))  # 順伝播の計算

        loss=0
        for j in range(len(cat)):
            ans=cat[j]#.to(device)
            #loss=nn.CrossEntropyLoss(out, ans)
            loss += criterion(out, ans)
        loss=loss/len(cat)
        loss.backward()  # 逆伝播の計算        
        optimizer.step()  # 重みの更新
        all_loss += loss.item()
    losses.append(all_loss)
    
    
    # print log
    print ('Epoch [{}], Loss: {}'.format(epoch+1, all_loss))
np.savetxt('loss_mlp_{}.txt'.format(name),losses)
torch.save(net.state_dict(), "model_mlp_{}.pth".format(name))

