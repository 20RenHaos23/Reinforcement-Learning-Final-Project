import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from env_2048 import Game2048Env
import matplotlib.pyplot as plt

torch.manual_seed(456)  #let's make things repeatable! (only affects PyTorch neural-network param initialization in this demo)
#random.seed(789)
np.random.seed(789)
torch.cuda.manual_seed_all(456)

class SumTree(object): #葉子節點有10000 則整棵樹的節點數為2*10000-1=19999，因為程式皆從0開始 所以是0~19998 共19999個
    data_pointer = 0
    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values  #10000
        self.tree = np.zeros(2 * capacity - 1)   #shape(19999,)     
        self.data = np.zeros(capacity, dtype=object)      

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1   # 在樹的葉子節點的位置  9999~19998(從0開始算的話)
        self.data[self.data_pointer] = data  # 將資訊放入self.data中，self.data為長度10000的容器
        self.update(tree_idx, p)  # update tree_frame
        self.data_pointer += 1
        
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity  #self.capacity=10000
            self.data_pointer = 0

    def update(self, tree_idx, p): #這裡再將葉子節點的p往上傳遞
        change = p - self.tree[tree_idx] #新的p先計算跟原本位置的p的差距 然後再往上傳遞
        self.tree[tree_idx] = p    
        
        while tree_idx != 0:    
            tree_idx = (tree_idx - 1) // 2 #找父節點的公式
            self.tree[tree_idx] += change

    def get_leaf(self, v):      
        parent_idx = 0      
        while True:     
            cl_idx = 2 * parent_idx + 1          #找左子節點的公式
            cr_idx = cl_idx + 1 #用左子節點找右子節點的公式
            
            if cl_idx >= len(self.tree):       #如果找到最下層了 就會發生這個情況 #len(self.tree)為總節點數19999
                leaf_idx = parent_idx
                break
            else:    
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1 #用節點idx推對應的data存放位置
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  
    
class Memory(object): 
    epsilon = 0.01 
    alpha = 0.6 
    beta = 0.4 
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])  #初始self.tree.tree為shape(19999,)的零numpy 
        #當新的經驗（transition）被存儲到記憶庫（Memory）中時，它的優先級（或 p 值）被設定為目前存儲在樹中的所有經驗的最大優先級。這是為了確保新的經驗在最初會被取樣和重播，因為它具有最高的優先級。

        #-self.tree.capacity=-10000 
        #從self.tree.tree這個np選取從後面數來的10000個資料
        if max_p == 0:#防樹中目前的最大 p 值是 0（這可能發生在記憶庫剛初始化時）。在這種情況下，新的經驗的 p 值會被設定為self.abs_err_upper（在這段程式碼中是 1），這也是一個合理的高值，以確保新的經驗在最初會被取樣和學習。
            max_p = self.abs_err_upper #如果是第一次儲存此筆資料 令他的|TD error|為1 
        self.tree.add(max_p, transition) 

    def sample(self, n): #IS importance sampling 重要性抽樣
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty((n, 1)) #b_idx(128,) #b_memory(128,34) #ISWeights(128,1) #34為16+1+1+16
        pri_seg = self.tree.total_p / n    #做10000/128   #將tree的最下面
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # beta慢慢增加，最後最大為1 
        #beta是隨著時間逐步增加的（由self.beta_increment_per_sampling控制）。
        #初始值是0.4，每次抽取時增加0.001，直到達到最大值1。這種設置意味著，在訓練的初始階段，會對優先級抽樣給予較少的重視（即緩和了優先級抽樣的影響），而隨著時間的推移，將逐步增加對優先級抽樣的依賴（即增強了優先級抽樣的影響）。
        #隨著beta的增加，網絡將更多地專注於具有較高TD誤差的經驗，因此對這些經驗的學習將更加積極。 
        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p    #計算記憶緩衝區中所有樣本的最小機率
        
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1) #在每次迴圈中，計算當前區段的範圍。
            v = np.random.uniform(a, b) #在當前區段中隨機抽取一個數字。
            idx, p, data = self.tree.get_leaf(v) #回傳leaf_idx, self.tree[leaf_idx], self.data[data_idx] #使用抽取到的數字，從樹狀數據結構中獲取對應的節點資料。這個方法回傳節點索引（idx），節點的優先級（p），和對應的樣本數據（data）。
            prob = p / self.tree.total_p #計算選中當前節點的概率
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta) #根據該概率和最小概率min_prob以及beta參數，計算重要性抽樣（Importance Sampling, IS）權重。
            b_idx[i], b_memory[i, :] = idx, data #將節點索引和樣本數據儲存到對應的數組中，以便後續使用。
            
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  #這一行為每一個絕對誤差添加一個小的常數self.epsilon，以避免優先級為零。
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper) #p最大為1 #這一行將絕對誤差裁剪到一個上限self.abs_err_upper，通常這樣做是為了防止優先級變得過大。
        ps = np.power(clipped_errors, self.alpha) #這行計算每個裁剪後的誤差的self.alpha次方。
        
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)
#設定
class Setting():
    def __init__(self, capacity=1e4):
        self.memory_size = capacity #10000
        self.memory_counter = 0
        self.memory = Memory(capacity=capacity)      

    def store(self, transition):
        self.memory_counter += 1   
        self.memory.store(transition)        

    def sample(self, batch_size):
        info = None              
        tree_idx, batch_memory, ISWeights = self.memory.sample(batch_size) #tree_idx(128,) #batch_memory(128,34) #ISWeights(128,1)
        info = (tree_idx, ISWeights)      
        
        return batch_memory, info
    
    def update(self, tree_idx, td_errors):
        self.memory.batch_update(tree_idx, td_errors)

class Model(nn.Module):
    def __init__(self, input_len, output_num, conv_size=(32, 64), fc_size=(512, 128)):
        super(Model, self).__init__()
        self.input_len = input_len
        self.output_num = output_num
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, conv_size[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(conv_size[0], conv_size[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )       
        self.fc1 = nn.Linear(conv_size[1] * self.input_len * self.input_len, fc_size[0])
        self.fc2 = nn.Linear(fc_size[0], fc_size[1])
        self.head = nn.Linear(fc_size[1], self.output_num)

    def forward(self, x):
        x = x.reshape(-1,1,self.input_len, self.input_len)#從(1,4,4)變(1,1,4,4)代表(一份,厚度為1,長寬為4*4)
        x = self.conv1(x) #從(1,1,4,4)變(1,32,4,4)代表(一份,厚度為32,長寬為4*4)
        x = self.conv2(x) #從(1,32,4,4)變(1,64,4,4)代表(一份,厚度為32,長寬為4*4)
        x = x.view(x.size(0), -1) #從(1,64,4,4)變(1,1024)代表(一份,4*4*64)
        x = F.relu(self.fc1(x)) #從(1,1024)變(1,512)
        x = F.relu(self.fc2(x)) #從(1,512)變(1,128)
        output = self.head(x) #從(1,128)變(1,4)
       
        return output
    
    
class PRDQN():
    batch_size = 128
    lr = 1e-4
    epsilon = 0.15   
    memory_capacity =  int(10000)
    gamma = 0.99
    target_update_freq = 200
    save_path = "./"
    soft_update_theta = 0.01 #可以更改
    clip_norm_max = 1
    train_interval = 5 #5的倍數更新一次權重

    def __init__(self, num_state, num_action):
        super(PRDQN, self).__init__()
        self.num_state = num_state #16
        self.num_action = num_action #4
        self.state_len = int(np.sqrt(self.num_state)) #4

        self.eval_net = Model(self.state_len,  self.num_action) #預測網路
        self.target_net = Model(self.state_len, self.num_action) #目標網路
        self.target_net.load_state_dict(self.eval_net.state_dict()) #讓兩個網路權種一樣

        self.learn_step_counter = 0
        self.buffer = Setting(self.memory_capacity) #設定 
        self.initial_epsilon = self.epsilon   #0.15
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)

    def Select_Action(self, state, random=False, test=False):
        state = torch.unsqueeze(torch.FloatTensor(state), 0) #將numpy[4,4]變tensor[1,4,4]
        if not random and np.random.random() > self.epsilon or test:  # greedy policy
            action_value = self.eval_net.forward(state)
            action = torch.max(action_value.reshape(-1,4), 1)[1].data.numpy()[0]
            
        else: # random policy #前面一萬次都隨機選擇動作
            action = np.random.randint(0,self.num_action)
        return action

    def Store(self, state, action, reward, next_state, done):
        state = state.reshape(-1) #從(4,4)變成(16,)
        next_state = next_state.reshape(-1) #從(4,4)變成(16,) 
        
        transition = np.hstack((state, [action, reward], next_state,done)) #將state action reward next_state疊加起來shape變成(34,)      
        
        self.buffer.store(transition)
        
    def Update(self):
        #soft update the parameters #判斷方式為updata執行次數的200的倍數
        if self.learn_step_counter % self.target_update_freq ==0 and self.learn_step_counter > 0:#看目標網路要不要更新權重 #用soft更新方式
            for p_e, p_t in zip(self.eval_net.parameters(), self.target_net.parameters()):
                p_t.data = self.soft_update_theta * p_e.data + (1 - self.soft_update_theta) * p_t.data
                
        self.learn_step_counter+=1

        #sample batch from memory       
        batch_memory, (tree_idx, ISWeights) = self.buffer.sample(self.batch_size)      

        batch_state = torch.FloatTensor(batch_memory[:, :self.num_state]) #(128,16)
        batch_action = torch.LongTensor(batch_memory[:, self.num_state: self.num_state+1].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, self.num_state+1: self.num_state+2])
        batch_next_state = torch.FloatTensor(batch_memory[:,self.num_state+2:-1])
        batch_done = torch.FloatTensor(batch_memory[:,-1:])

        
        q_eval = self.eval_net(batch_state).gather(1, batch_action) #做Q(s,指定a,w)   (128,1)     
        q_eval_next = self.eval_net(batch_next_state).detach() #做Q(s',all a,w) (128,4)
        q_target_next = self.target_net(batch_next_state).detach() #做Q'(s',all a,w-) (128,4)
        
        #Q'(s',argmax(a')Q(s',a',w),w-)
        q_eval_argmax = q_eval_next.max(1)[1].view(self.batch_size, 1) #找q_eval_next最大的a的index
        q_max = q_target_next.gather(1, q_eval_argmax).view(self.batch_size, 1) #找相對應index裡面的值
        
        #q_target = batch_reward + self.gamma * q_max
        q_target = batch_reward + (1 - batch_done) * self.gamma * q_max #改成這個才對
        
        with torch.no_grad():    
            abs_errors = (q_target - q_eval.data).abs()
            self.buffer.update(tree_idx, abs_errors)
        loss = (torch.FloatTensor(ISWeights) * (q_target - q_eval).pow(2)).mean()                   
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.eval_net.parameters(), self.clip_norm_max)
        self.optimizer.step()

        return loss
    def Epsilon_Decay(self, episode, total_episode):
        self.epsilon = self.initial_epsilon * (1 - episode / total_episode)
        
    def Save(self, path=None, name='dqn_weights.pkl'):
        path = self.save_path if not path else path
        torch.save(self.eval_net.state_dict(), path + name)

    def Load(self, path=None, name='dqn_weights.pkl'):
        path = self.save_path if not path else path
        self.eval_net.load_state_dict(torch.load(path + name))
    
def regularization(s, divide=16):    
    s = np.log2(1 + s) / divide   
    return s

def test(episodes=20, agent=None, load_path=None, name=None, test_model=False):
    if agent is None:
        agent = PRDQN(num_state=16, num_action=4)       
        agent.Load(path=load_path,name=name)

    env = Game2048Env()
    
    test_score_list = []
    test_steps_list = []
    test_highest_list = []
    
    for i in range(episodes):
        state, reward, done, info = env.reset()       
        state = regularization(state)
     
        while True:
            action = agent.Select_Action(state, test=True)
            next_state, reward, done, info = env.step(action)
            next_state = regularization(next_state)
            state = next_state

            if done:
                if test_model:       
                    print('Test Episodes {}'.format(i+1))
                    env.render()
                test_score_list.append(info['score'])
                test_steps_list.append(info['steps'])
                test_highest_list.append(info['highest'])
                break          
    
    return test_score_list,  test_steps_list, test_highest_list       
        
def train(train_episodes,epsilon_decay_interval,test_episodes,test_interval,path,Show=False):
    env = Game2048Env()  
    agent = PRDQN(num_state=16, num_action=4)
    eval_max_score = 0
    
    train_score_list = []
    train_steps_list = []
    train_highest_list = []
    test_score_list = []
    test_steps_list = []
    test_highest_list = []
    loss = None
    loss_list = []
    for i in range(train_episodes):

        state, reward, done, info = env.reset()#重置遊戲
        state = regularization(state)
        #開始玩遊戲
        while True:
            if agent.buffer.memory_counter <= agent.memory_capacity: #前面10000次先都隨機選
                action = agent.Select_Action(state, random=True)
                
            else:
                action = agent.Select_Action(state) #執行第10001次開始跑這裡
                

            next_state, reward, done, info = env.step(action)
            
            next_state = regularization(next_state)
            
            reward = regularization(reward, divide=1)
            
            agent.Store(state, action, reward, next_state, done)
            state = next_state

            if agent.buffer.memory_counter % agent.train_interval == 0 and agent.buffer.memory_counter > agent.memory_capacity:  # 填滿buffer後 每執行五次才更新一次網路 
                loss = agent.Update()                #agent.train_interval = 5 #第一次執行為當agent.buffer.memory_counter = 10005
                loss_list.append(loss.item())  # 將loss儲存到list中
            if done:
                if Show:                
                    print('Train Episodes {}'.format(i+1))
                    env.render()
                train_score_list.append(info['score'])
                train_steps_list.append(info['steps'])
                train_highest_list.append(info['highest'])
                if i % epsilon_decay_interval == 0:   # episilon 衰減 間格
                    agent.Epsilon_Decay(i, train_episodes)
                break 
       
        # test and save better model weights
        if i % test_interval == 0 and i > 0 and loss!=None:
            test_score_list,  test_steps_list, test_highest_list = test(episodes=test_episodes, agent=agent)

            if int(np.mean(test_score_list)) > eval_max_score:
                eval_max_score = int(np.mean(test_score_list))
                name = 'dqn_{}.pkl'.format(int(eval_max_score))
                agent.Save(path=path,name=name)
            
        if i % 100 == 0 and i > 0:
            # 創建第一張圖形並添加子圖
            fig1 = plt.figure()
            ax1_1 = fig1.add_subplot(311)
            ax1_1.plot(train_score_list)
            ax1_1.set_xlabel('Episode')
            ax1_1.set_ylabel('train_score')
            ax1_1.set_title("Train {} episodes".format(i))
            
            ax1_2 = fig1.add_subplot(312)
            ax1_2.plot(train_steps_list)
            ax1_2.set_xlabel('Episode')
            ax1_2.set_ylabel('train_steps')
            
            ax1_3 = fig1.add_subplot(313)
            ax1_3.plot(train_highest_list)
            ax1_3.set_xlabel('Episode')
            ax1_3.set_ylabel('train_highest')
            
            # 顯示第一張圖形
            plt.show()
    
    # 創建並設定第二張圖形
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(loss_list)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss Curve')
    
    # 顯示第二張圖形
    plt.show()
        
    return train_score_list,train_steps_list,train_highest_list,test_score_list,  test_steps_list, test_highest_list


if __name__ == '__main__':
    #測試用
    #parameter setting
    train_episodes = 10000 #train times
    epsilon_decay_interval = 100 #epsilon update freq
    test_episodes = 50 #one test times
    test_interval = 25 #eval freq

    path = "./model_weights/" #model weights save path
    #train model
    train_score_list,train_steps_list,train_highest_list,test_score_list,  test_steps_list, test_highest_list = train(train_episodes,epsilon_decay_interval,test_episodes,test_interval,path,Show=False)