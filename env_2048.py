import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import itertools
from six import StringIO
import sys

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

class IllegalMove(Exception):
    pass


class Game2048Env(gym.Env):   # directions 0, 1, 2, 3 are up, right, down, left
    metadata = {'render.modes': ['human', 'ansi']}
    max_steps = 100000

    def __init__(self):
        # Definitions for game. Board must be square.
        self.size = 4
        self.w = self.size
        self.h = self.size
        self.squares = self.size * self.size
        self.score = 0
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0, 1, (self.w, self.h, self.squares), dtype=np.int32)
        self.set_illegal_move_reward(0.) #設置非法移動的獎勵。
        self.set_max_tile(None) # 這裡可以設定遊戲最高想要到幾分
        self.max_illegal = 1   # max number of illegal actions 最大的非法移動
        self.num_illegal = 0
        # Initialise seed
        self.seed()
    
    def _get_info(self, info=None):
        if not info:
            info = {}
        else:
            assert type(info) == dict, 'info should be of type dict!'

        info['highest'] = self.highest() #Report the highest tile on the board.
        info['score'] = self.score
        info['steps'] = self.steps
        return info
    #設定隨機種子
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    #設置非法移動的獎勵
    def set_illegal_move_reward(self, reward):
        """Define the reward/penalty for performing an illegal move定義執行非法移動的獎勵/懲罰。. Also need
            to update the reward range for this."""
        # Guess that the maximum reward is also 2**squares though you'll probably never get that.
        # (assume that illegal move reward is the lowest value that can be returned
        self.illegal_move_reward = reward
        self.reward_range = (self.illegal_move_reward, float(2**self.squares)) #這行沒有用
    #設定遊戲的最大磚塊（即遊戲目標分數）
    def set_max_tile(self, max_tile):
        """Define the maximum tile that will end the game (e.g. 2048). None means no limit.
           This does not affect the state returned."""
        assert max_tile is None or isinstance(max_tile, int)
        self.max_tile = max_tile
    #定義了遊戲每一步的動作，包括移動和添加新磚塊，以及判定遊戲是否結束。
    # Implement gym interface
    def step(self, action):
        """Perform one step of the game. This involves moving and adding a new tile.這涉及移動和添加新瓷磚。"""
        self.steps += 1 #步數加一
        score = 0
        done = None
        info = {
            'illegal_move': False,
        }
        try:
            score = float(self.move(action)) #這行程式會看這個move是不是非法移動 如果是的話 會直接跳except IllegalMove as e:這行
            self.score += score #計算分數
            assert score <= 2**(self.w*self.h) #2**(self.w*self.h) --> 65536
            self.add_tile()
            done = self.isend() #判斷還能玩遊戲嗎
            reward = float(score)
            self.num_illegal = 0 #非法移動的次數
            
        except IllegalMove as e: #看有沒有非法移動
            
            info['illegal_move'] = True
            if self.steps > self.max_steps: #先看此次遊戲有沒有超過設定的步數
                done = 1
            else:
                done = 0
            reward = self.illegal_move_reward
            self.num_illegal += 1
            if self.num_illegal >= self.max_illegal:   # exceed the maximum number of illegal actions超過最大非法行為次數
                done = 1

        info = self._get_info(info)

        # Return observation (board state), reward, done and info dict
        return self.Matrix, reward, done, info
    #重置遊戲，清零分數，並添加兩個新的磚塊到遊戲板上
    def reset(self):
        self.Matrix = np.zeros((self.h, self.w), np.int32)
        self.score = 0
        self.steps = 0
        self.num_illegal = 0

        
        self.add_tile() #在4*4遊戲中 空的位置隨機加2或4
        self.add_tile() #在4*4遊戲中 空的位置隨機加2或4

        return self.Matrix, 0, False, self._get_info() #第二個回傳值為weward 第三個為done 第四個為資訊
    # 用來將當前的遊戲狀態渲染到控制台或其他輸出流中
    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        s = 'Score: {}\n'.format(self.score)
        s += 'Highest: {}\n'.format(self.highest())
        npa = np.array(self.Matrix)
        grid = npa.reshape((self.size, self.size))
        s += "{}\n\n".format(grid)
        outfile.write(s)
        return outfile
    #在遊戲板的一個空位置隨機添加一個新的磚塊（2或4）
    # Implement 2048 game
    def add_tile(self):
        """Add a tile, probably a 2 but maybe a 4"""
        possible_tiles = np.array([2, 4])
        tile_probabilities = np.array([0.9, 0.1])
        val = self.np_random.choice(possible_tiles, 1, p=tile_probabilities)[0] #隨機選2或4
        empties = self.empties() #empties為在4*4的遊戲中 找為空的格子的位置 ->shape為(?,2)
        assert empties.shape[0]   #assert 用于判断一个表达式 在表达式条件为 false 的时候触发异常
        empty_idx = self.np_random.choice(empties.shape[0]) #隨機從empties挑選一個index ->type為int
        empty = empties[empty_idx] #shape為(2,)
        
        self.set(empty[0], empty[1], val) #將位置跟值放進去
    #用來獲取和設置遊戲板上某個位置的值
    def get(self, x, y):
        """Return the value of one square."""
        return self.Matrix[x, y]
    #用來獲取和設置遊戲板上某個位置的值
    def set(self, x, y, val):
        """Set the value of one square."""
        self.Matrix[x, y] = val
    #返回遊戲板上所有空位置的列表
    def empties(self):
        """Return a 2d numpy array with the location of empty squares."""
        return np.argwhere(self.Matrix == 0)
    #返回遊戲板上的最大磚塊值
    def highest(self):
        """Report the highest tile on the board."""
        return np.max(self.Matrix)
    #這是一個核心函數，用來處理遊戲的一次移動操作，並返回該移動的得分
    def move(self, direction, trial=False): #trial審判
        """Perform one move of the game. Shift things to one side then,
        combine. directions 0, 1, 2, 3 are up, right, down, left.
        Returns the score that [would have] got."""
        #trial 是一個布爾值，指示是否僅執行模擬移動而不實際更新遊戲狀態。當trial=True時，只計算分數而不更新遊戲板。

        changed = False #這個變數用於跟踪遊戲板是否因動作而改變。如果板沒有改變，則這個動作被認為是非法的。
        move_score = 0 #儲存此次移動所獲得的分數 會return回去
        #dir_div_two 和 dir_mod_two：用於計算移動方向
        dir_div_two = int(direction / 2) #direction==action # up 0->0   ,Right 1->0   ,Down 2->1   ,Left 3->1
        dir_mod_two = int(direction % 2)                    # up 0->0   ,Right 1->1   ,Down 2->0   ,Left 3->1
        #確定是向左/上還是向右/下移動和合併方塊
        shift_direction = dir_mod_two ^ dir_div_two # 0 for towards up left, 1 for towards bottom right
        # 0 ^ 0 ->0   ,   0 ^ 1 ->1   ,   1 ^ 0 ->1   ,   1 ^ 1 ->0
        # Construct a range for extracting row/column into a list
        #分別是範圍[0,1,2,3]的列表，用於索引遊戲板
        rx = list(range(self.w)) #[0,1,2,3]
        ry = list(range(self.h)) #[0,1,2,3]
        
        '''
        1.首先，根據選擇的方向，確定是否是垂直（上/下）或水平（左/右）移動。
        2.然後，對於垂直或水平的每一列/行，首先獲取當前的狀態（old），然後使用shift方法來確定移動後的新狀態（new）。
        3.如果新舊狀態不同，且不僅僅是模擬（即trial為False），那麼遊戲板將被更新。
        4.如果所有列/行都沒有變化，則動作被認為是非法的，並且會引發IllegalMove異常。
        '''
        if dir_mod_two == 0: #up or down
            # Up or down, split into columns
            for y in range(self.h):  #看直的 #self.h為高
                old = [self.get(x, y) for x in rx]   #self.get(x, y) return self.Matrix[x, y]   #看直的
                (new, ms) = self.shift(old, shift_direction)
                move_score += ms
                if old != new:
                    changed = True
                    if not trial:
                        for x in rx:
                            self.set(x, y, new[x])
        else:
            # Left or right, split into rows
            for x in range(self.w): #self.w為寬
                old = [self.get(x, y) for y in ry] #self.get(x, y) return self.Matrix[x, y]   #看橫的
                (new, ms) = self.shift(old, shift_direction)
                move_score += ms
                if old != new:
                    changed = True
                    if not trial:
                        for y in ry:
                            self.set(x, y, new[y])
        if changed != True: #如果所有old跟new都一樣 則changed為False 則會看是不是非法移動
            raise IllegalMove

        return move_score
    #用來處理磚塊的合併操作
    def combine(self, shifted_row):
        """Combine same tiles when moving to one side. This function always
           shifts towards the left. Also count the score of combined tiles."""
        move_score = 0
        combined_row = [0] * self.size
        skip = False
        output_index = 0
        for p in pairwise(shifted_row):#好像長度要大於2才會執行  ex:如果shifted_row為[2,2] 則p為tuple(2,2)
            if skip:                                          #ex:如果shifted_row為[4,2,4,2] 則p為tuple(4,2) (2,4) (4,2) (2,4)
                skip = False
                continue
            combined_row[output_index] = p[0]
            if p[0] == p[1]:
                combined_row[output_index] += p[1]
                move_score += p[0] + p[1]
                # Skip the next thing in the list.跳過列表中的下一件事。?
                skip = True
            output_index += 1
        if shifted_row and not skip: #感覺觸發條件為shifted_row只有一個數??  #如果裡面都是0 則shifted_row為[] 則不會執行此程式
            combined_row[output_index] = shifted_row[-1] #當shifted_row為[4,2,4,2] 此時的combined_row為[4,2,4] 則會直接把shifted_row的最後一個放到combined_row後面

        return (combined_row, move_score)
    # 用來處理磚塊的移動操作
    def shift(self, row, direction):
        """Shift one row left (direction == 0) or right (direction == 1), combining if required."""
        length = len(row)
        assert length == self.size
        assert direction == 0 or direction == 1

        # Shift all non-zero digits up
        shifted_row = [i for i in row if i != 0]#從那個list裡面看除了0以外的數

        # Reverse list to handle shifting to the right
        if direction: #只有執行下或著右才會觸發
            shifted_row.reverse() #反向列表中元素

        (combined_row, move_score) = self.combine(shifted_row)

        # Reverse list to handle shifting to the right
        if direction:
            combined_row.reverse()

        assert len(combined_row) == self.size
        return (combined_row, move_score)
    #用來判斷遊戲是否已經結束。
    def isend(self):
        """Has the game ended. Game ends if there is a tile equal to the limit
           or there are no legal moves. If there are empty spaces then there
           must be legal moves."""

        if self.max_tile is not None and self.highest() == self.max_tile:
            return 1 #如果遊戲格子到設定的最高分後 遊戲就結束了
        
        if self.steps >= self.max_steps:
            return 1 #如果遊玩步數到設定的最大步數後 遊戲就結束了

        for direction in range(4): #看往四個方向走 皆還能不能走
            try:
                self.move(direction, trial=True)
                # Not the end if we can do any move
                return 0
            except IllegalMove: #如果四個方向都不能移動了 則回傳True
                pass
        return 1

    def get_board(self):
        """Retrieve the whole board, useful for testing."""
        return self.Matrix

    def set_board(self, new_board):
        """Retrieve the whole board, useful for testing."""
        self.Matrix = new_board
    


if __name__ == '__main__':    
    env = Game2048Env()
    state, reward, done, info = env.reset()
    env.render()
    done = False
    moves = 0
    while not done:
        action = env.np_random.choice(range(4), 1).item() #隨機選擇一個動作
        next_state, reward, done, info = env.step(action)
        moves += 1
        print(info)
        env.render()
    print('\nTotal Moves: {}'.format(moves))            
    

    