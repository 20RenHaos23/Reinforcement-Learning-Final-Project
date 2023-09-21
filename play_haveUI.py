import tkinter as tk
from env_2048 import Game2048Env
from main import PRDQN,regularization
import tkinter.messagebox as messagebox
#load model weights
load_path = "./model_weights/"
#name = 'dqn_10128.pkl'
name = 'dqn_7710.pkl'

#load model
agent = PRDQN(num_state=16, num_action=4)
agent.Load(path=load_path,name=name)
#agent.save(path=load_path)
env = Game2048Env()
#env.seed(789)

state, reward, done, info = env.reset()
    
game_bg_color = "#bbada0" #'#9e948a'
mapcolor = {
    0: ("#bbada0", '#776e65'),
    2: ('#eee4da', '#776e65'),
    4: ('#ede0c8', '#f9f6f2'),
    8: ('#f2b179', '#f9f6f2'),
    16: ('#f59563', '#f9f6f2'),
    32: ('#f67c5f', '#f9f6f2'),
    64: ('#f65e3b', '#f9f6f2'),
    128: ('#edcf72', '#f9f6f2'),
    256: ('#edcc61', '#f9f6f2'),
    512: ('#edc850', '#f9f6f2'),
    1024: ('#edc53f', '#f9f6f2'),
    2048: ('#edc22e', '#f9f6f2'),
    4096: ('#04ba04', '#f9f6f2'),
    8192: ("#008000", '#f9f6f2'),
}


# 2048的介面
root = tk.Tk()
root.title('2048')

frame = tk.Frame(root, width=300, height=300, bg=game_bg_color)
frame.grid(sticky=tk.N + tk.E + tk.W + tk.S)

def test():
    global state
    
    state = regularization(state)
    action = agent.Select_Action(state, test=True)
    
    next_state, reward, done, info = env.step(action)   
    
    state = next_state
    
    return next_state,info,done


def handler(event):
    state,sc,done = test() 
    #a = [2,4,8,16,32,64,128,256,512,1024,2048,4096]
    #state = np.random.choice(a, (4,4),)
    
    env.render()
    
    if done:
        messagebox.showinfo('2048', 'Oops!\n''Game over!')
    if done == False:   
        for r in range(4):
            for c in range(4):
                number = state[r][c]
                text = '' if 0 == number else str(number)
                label = tk.Label(frame, text=text, width=4, height=2,font=("Verdana", 24,'bold'),fg=mapcolor[number][1],bg=mapcolor[number][0])
                label.grid(row=r, column=c, padx=5, pady=5, sticky=tk.N + tk.E + tk.W + tk.S)
        label = tk.Label(frame, text='score', font=("楷體", 24, "bold"),bg="#bbada0", fg="#eee4da")
        label.grid(row=4, column=0, padx=5, pady=5)
        s = sc['score']
        label_score = tk.Label(frame, text=int(s), font=("楷體", 24, "bold"),bg="#bbada0", fg="#ffffff")
        label_score.grid(row=4, columnspan=2, column=1, padx=5, pady=5)              
   
#初始化圖形介面
for r in range(4):   
    for c in range(4):
        value = state[r][c]
        text = '' if 0 == value else str(value)
        label = tk.Label(frame, text=text, width=4, height=2,font=("Verdana", 24,'bold'),fg=mapcolor[value][1],bg=mapcolor[value][0])
        label.grid(row=r, column=c, padx=5, pady=5, sticky=tk.N + tk.E + tk.W + tk.S)        

label = tk.Label(frame, text='score', font=("楷體", 24, "bold"),bg="#bbada0", fg="#eee4da")
label.grid(row=4, column=0, padx=5, pady=5)

label_score = tk.Label(frame, text=int(0), font=("楷體", 24, "bold"),bg="#bbada0", fg="#ffffff")
label_score.grid(row=4, columnspan=2, column=1, padx=5, pady=5)

root.bind('<Key>',handler) #按下鍵盤隨意的鍵 則會開始自動執行下一步 一直按著則會自動一直執行下一步

root.mainloop()





    
    
