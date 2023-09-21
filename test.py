from main import test
import matplotlib.pyplot as plt
#parameter setting
test_episodes = 50
#load model weights
path = "./model_weights/"
name = 'dqn_10128.pkl'
#test model
test_score_list,  test_steps_list, test_highest_list  = test(episodes=test_episodes,load_path=path,name=name,test_model=True)

#plot
plt.subplot(311)
plt.title("Test")
plt.plot(test_score_list)
plt.xlabel('Episode')
plt.ylabel('score')
plt.subplot(312)
plt.plot(test_steps_list)
plt.xlabel('Episode')
plt.ylabel('steps')
plt.subplot(313)
plt.plot(test_highest_list)
plt.xlabel('Episode')
plt.ylabel('highest')
plt.show()
#cleanup plots
plt.cla()
plt.close('all')

