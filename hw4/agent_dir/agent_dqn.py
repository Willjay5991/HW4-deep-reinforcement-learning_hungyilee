from agent_dir.agent import Agent
import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np

class Net(nn.Module):
    """
    Initialize a deep Q-learning network as described in
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    Arguments:
        in_channels: number of channel of input.
            i.e The number of most recent frames stacked together as describe in the paper
        out_num: number of action-value to output, one-to-one correspondence to action in game.
    """
    def __init__(self, in_channel=4, out_num=6):
        super(Net,self).__init__()
        # layer setting paramters
        layers = [32,64,64]
        # output_size = (input_size-kernel_size)/stride + 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel,layers[0],kernel_size=8,stride=4), # [batch,4,84,84]=>[batch,layers[0],20,20] 
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(layers[0],layers[1],kernel_size=4,stride=2), # [batch,layers[0],20,20]=>[batch,layers[1],9,9]
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(layers[1],layers[2],kernel_size=3,stride=1), # [batch,layers[1],9,9]=>[batch,layers[2],7,7]
            nn.ReLU()
        )
        self.linear1 = nn.Sequential(
            nn.Linear(7*7*layers[2], 512),
            nn.ReLU()
        )
        self.linear2 = nn.Linear(512, out_num)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # flatten
        x = x.reshape(x.size(0), -1)  # [batch, layers[2],7,7]=>[batch,layers[2]*7*7]
        x = self.linear1(x)
        y = self.linear2(x)
        # y = F.softmax(y)  # turn to probability
        return y


# preprocess
def prepro(obs):
    """
    preprocess observation 
    [84,84,4]=>[4,84,84]
    """
    obs = obs.transpose(2,0,1)/255  # 0-255 => 0-1, [84,84,4] => [4,84,84]
    return obs

class Memory_buffer():
    def __init__(self, capacity, device):
        self.device = device
        self.capacity = capacity
        self.memory_state = np.zeros((self.capacity, 4, 84, 84))
        self.memory_state_next = np.zeros((self.capacity, 4, 84, 84))
        self.memory_a_r_done = np.zeros((self.capacity,3))
        self.memory_count = 0
    
    def store_record(self, state,state_next,a,r,done):
        idx = self.memory_count%self.capacity
        self.memory_state[idx,:] = state
        self.memory_state_next[idx,:] = state_next
        a_r_done = np.hstack((a,r,done))
        self.memory_a_r_done = a_r_done
    
    def sample(self, batchsz):
        if batchsz > self.memory_count:
            return None
        memory_count = self.capacity if self.memory_count>self.capacity else self.memory_count
        idxs = np.random.choice(memory_count,batchsz,replace=False)
        
        states = self.memory_state[idx,:,:,:]
        state_nexts = self.memory_state_next[idx,:,:,:]
        a_s = self.memory_a_r_done[idx,0]
        r_s = self.memory_a_r_done[idx,1]
        done_s = self.memory_a_r_done[idx,2]

        # function from numpy to tensor
        numpy2tensor = lambda data: torch.from_numpy(data).to(self.device)
        states, state_nexts = numpy2tensor(states), numpy2tensor(state_nexts)
        a_s, r_s, done_s = numpy2tensor(a_s), numpy2tensor(r_s), numpy2tensor(done_s)

        return states, state_nexts, a_s, r_s, done_s

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_DQN,self).__init__(env)
        # cuda
        cuda = False
        self.device = torch.device('cuda' if torch.cuda.is_available() and cuda else 'cpu')
        # env
        self.env = env
        self.actions_n = self.env.action_space.n 
        print(f'self.actions_n:{self.actions_n}')
        self.state_dim = self.env.observation_space.shape

        # hyperparams
        self.lr = 1e-3
        self.batchsz = 32
        self.file_save = 'model/dqn.mdl'
        self.episodes = 1000
        self.learning_start = 1000
        self.print_interval = 1
        self.save_interval = 20
        self.update_tar_max = 1000
        self.update_tar_iter = 0
        self.gamma = 0.99
        # greed probability
        self.epsilon_max = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 30000

        # bulid model
        self.eval_net = Net(in_channel=self.state_dim[2], out_num=self.actions_n).to(self.device)
        self.tar_net = Net(in_channel=self.state_dim[2], out_num=self.actions_n).to(self.device)
        print(self.eval_net)
        if args.test_dqn:
            #you can load your model here
            self.eval_net.load_state_dict(torch.load(self.file_save))
            print('loading trained model')
        self.tar_net.load_state_dict(self.eval_net.state_dict())  # copy params of eval to tar_net
        # optimizer
        self.optimizer = optim.RMSprop(self.eval_net.parameters(),lr=self.lr,eps=0.001,alpha=0.95)

        # build memory buffer
        self.capacity = 1000 # memory buffer size
        self.memory_buffer = Memory_buffer(self.capacity, self.device)


    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        # load model
        self.eval_net.load_state_dict(torch.load(self.file_save))
        self.tar_net.load_state_dict(self.eval_net.state_dict())
        pass
    

    def train(self):
        """
        Implement your training algorithm here
        """
        # load model
        load_model = False
        if load_model:
            self.eval_net.load_state_dict(torch.load(self.file_save))
        self.tar_net.load_state_dict(self.eval_net.state_dict())

        # epsilon decay function
        epsilon_by_step = lambda step_idx: self.epsilon_min \
            + (self.epsilon_max-self.epsilon_min)*np.exp(-1*step_idx/self.epsilon_decay)
        global_step = 0
        rewards = []
        losses = []
        for epis in range(self.episodes):
            state = self.env.reset()
            # state = prepro(state) # [4,84,84]
            r_episode = 0
            loss = []
            while True:
                epsilon = epsilon_by_step(global_step)
                global_step += 1 
                act = self.make_action(state, epsilon)
                # state = prepro(state)
                # print(type(act), act)
                state_next, r, done, _ = self.env.step(act)
                # state_next = prepro(state_next)
                # state_next = self.stack4obs(state,obs_aft)
                # store record
                self.memory_buffer.store_record(prepro(state),prepro(state_next),act,r,done)

                if done:
                    rewards.append(r_episode)
                    losses.append(np.mean(loss))
                    break
                else:
                    state = state_next
                    r_episode += r
                
                if self.memory_buffer.memory_count > self.learning_start:
                    loss_=self.learn()
                    loss.append(loss_)
                else:
                    loss.append(0)
                
            if epis%self.print_interval==0 and epis>0:
                print('global step:{}'.format(global_step-1),
                      'episode/episodes:{}/{}'.format(epis, self.episodes),
                      'aver loss:{:.5}'.format(np.mean(losses[-10:])),
                      'aver reward:{:.5}'.format(np.mean(rewards[-10:])),
                      'epsilon:{:.5}'.format(epsilon)
                      )
            if epis% self.save_interval==0 and epis>0:
                # save model
                torch.save(self.eval_net.state_dict(), self.file_save)
        # plot reward and losses curve
        self.plot_r_loss(rewards, losses)
        pass

    # def stack4obs(state, obs_aft):
    #     """
    #     stack 3 successive obs as 1 state with 3 channel
    #     input:
    #         state:[3,80,80] with 3 channel
    #         obs_aft:[1,80,80]
    #     """
    #     state_tmp = np.vstack((state[1,:,:], state[2,:,:],state[3,:,:], obs_aft)
    #     return state_tmp
    
    def learn(self):
        states, state_nexts, a_s, r_s, done_s = self.Memory_buffer.sample(self.batchsz)
        # compute evalQ
        eval_q = self.eval_net(states.float()).gather(1, a_s.long().unsqueeze(1))
        eval_q = eval_q.squeeze(1) # [batch,1] => [batch]
        # compute targetQ
        tar_qs = self.tar_net(state_nexts.float()).detach()  # detach no use for update model
        tar_max_q  = tar_qs.max(1)[0]  # torch.max return a tuple (max_data, max_idx)
        gamma = torch.tensor(self.gamma).float().to(self.device)
        target_q = r_s.float() + gamma*tar_max_q

        # target_q = reward when done is true
        target_q = torch.where(done_s>0, r_s.float(), target_q)

        # update eval_net
        # compute loss
        loss = F.smooth_l1_loss(eval_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        # clip gradient
        for param in self.eval_model.parameters():
            param.grad.data.clamp_(-1,1)
        # update
        self.optimizer.step()

        # update tar_net
        self.update_tar_iter += 1
        if self.update_tar_iter% self.update_tar_max==0:
            self.tar_net.load_state_dict(self.eval_net.state_dict())
        
        return loss.cpu().item()
        

    def plot_r_loss(self, rewards,losses):
        import matplotlib.pyplot as plt
        plt.figure()
        plt.title('rewards')
        plt.plot(list(range(len(rewards))), rewards)
    
        plt.figure()
        plt.title('loss')
        plt.plot(list(range(len(losses))),losses)
        plt.show()    


    def make_action(self, observation, test=True, epsilon=None):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)

        Return:
            action: int
                the predicted action from trained model
        """
        observation = prepro(observation)
        state = torch.from_numpy(observation).to(self.device)
        pred_qs = self.eval_net(state.float().unsqueeze(0)).detach()
        epsilon = self.epsilon_min if epsilon==None else epsilon 
        if np.random.uniform()< epsilon:
            action = np.random.choice(self.actions_n) # random choose action
        else:
            action = torch.argmax(pred_qs)
            action = action.item()
        return action

