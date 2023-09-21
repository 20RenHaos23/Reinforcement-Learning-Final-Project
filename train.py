from main import train
import matplotlib.pyplot as plt
#parameter setting
train_episodes = 10000 #train times
epsilon_decay_interval = 100 #epsilon update freq
test_episodes = 50 #one test times
test_interval = 25 #eval freq

path = "./model_weights/" #model weights save path

#train model
train_score_list,train_steps_list,train_highest_list,test_score_list,  test_steps_list, test_highest_list = train(train_episodes,epsilon_decay_interval,test_episodes,test_interval,path,Show=False)

#plot
plt.subplot(311)
plt.title("Train {} episodes".format(train_episodes))
plt.plot(train_score_list)
plt.xlabel('Episode')
plt.ylabel('train_score')
plt.subplot(312)
plt.plot(train_steps_list)
plt.xlabel('Episode')
plt.ylabel('train_steps')
plt.subplot(313)
plt.plot(train_highest_list)
plt.xlabel('Episode')
plt.ylabel('train_highest')
plt.show()
#cleanup plots
plt.cla()
plt.close('all')


