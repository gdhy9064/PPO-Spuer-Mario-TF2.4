import cv2
from IPython.display import clear_output
from gym import Wrapper
from gym.spaces import Box
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
import multiprocessing as mp
from nes_py.wrappers import JoypadSpace
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model, Sequential
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Layer
tf.compat.v1.disable_eager_execution() # 关闭动态图机制

def process_frame(frame, height, width):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (width, height))[:, :, None] / 255.
    return frame


class CustomEnvironment(Wrapper):
    def __init__(self, env, height, width):
        super().__init__(env)
        self.observation_space = Box(low=0, high=1, shape=(height, width, 1))
        self.height = height
        self.width = width

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        state = process_frame(state, self.height, self.width)
        if done:
            if info["flag_get"]:
                reward += 50
            else:
                reward -= 50
        return state, reward / 10., done, info

    def reset(self):
        return process_frame(self.env.reset(), self.height, self.width)


class SkipFrame(Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.observation_space = Box(
            low=0, 
            high=1, 
            shape=(*self.env.observation_space.shape[:-1], skip)
        )
        self.skip = skip

    def step(self, action):
        total_reward = 0
        states = []
        state, reward, done, info = self.env.step(action)
        for i in range(self.skip):
            if not done:
                state, reward, done, info = self.env.step(action)
                total_reward += reward
                states.append(state)
            else:
                states.append(state)
        states = np.concatenate(states, axis=-1)
        return states.astype(np.float32), total_reward, done, info

    def reset(self):
        state = self.env.reset()
        states = np.concatenate([state for _ in range(self.skip)], axis=-1)
        return states.astype(np.float32)

    
class AutoReset(Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        if done:
            state = self.env.reset()
        return state, reward, done, info
    
    def reset(self):
        return self.env.reset()


def create_env(world, stage, action_type, height, width):
    env = gym_super_mario_bros.make(f'SuperMarioBros-{world}-{stage}-v0')
    env = JoypadSpace(env, action_type)
    env = CustomEnvironment(env, height, width)
    env = SkipFrame(env)
    env = AutoReset(env)
    return env


class MultipleEnvironments():
    def __init__(self, num_envs, create_env, *args):
        assert num_envs > 0
        self.agent_conns, self.env_conns = zip(*[mp.Pipe() for _ in range(num_envs)])
        for conn in self.env_conns:
            process = mp.Process(target=self.run, args=(conn, create_env, *args))
            process.start()

    @staticmethod
    def run(conn, create_env, *args):
        env = create_env(*args)
        while True:
            request, action = conn.recv()
            if request == 'step':
                conn.send(env.step(action))
            elif request == 'reset':
                conn.send(env.reset())
            elif request == 'render':
                env.render()
            elif request == 'close':
                env.close()
                break
            elif hasattr(env, request):
                conn.send(getattr(env, request))
            else:
                raise NotImplementedError
    
    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        assert not self.agent_conns[0].closed, 'Environment closed.'
        self.agent_conns[0].send([name, None])
        return self.agent_conns[0].recv()
                
    def step(self, actions):
        assert not self.agent_conns[0].closed, 'Environment closed.'
        for conn, action in zip(self.agent_conns, actions):
            conn.send(['step', action.item()])
        return tuple(zip(*[conn.recv() for conn in self.agent_conns]))
    
    def reset(self):
        assert not self.agent_conns[0].closed, 'Environment closed.'
        for conn in self.agent_conns:
            conn.send(['reset', None])
        return tuple(conn.recv() for conn in self.agent_conns)
    
    def render(self):
        assert not self.agent_conns[0].closed, 'Environment closed.'
        for conn in self.agent_conns:
            conn.send(['render', None])
                
    def close(self):
        assert not self.agent_conns[0].closed, 'Environment closed.'
        for conn in self.agent_conns:
            conn.send(['close', None])
            conn.close()
        for conn in self.env_conns:
            conn.close()


class Memory():
    def __init__(self):
        self.reset()
    
    def store(self, state, action, prob, reward, next_state, done):
        ### 逆序插入方便之后计算GAE
        self.states.insert(0, state) 
        self.actions.insert(0, action)
        self.probs.insert(0, prob)
        self.rewards.insert(0, reward)
        self.next_states.insert(0, next_state)
        self.dones.insert(0, done)
        
    def reset(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.rewards = []
        self.next_states = []
        self.dones = []

        
class Feature(Layer):
    def __init__(self):
        super().__init__()
        self.model = Sequential([
            Conv2D(32, 3, strides=2, activation='relu', padding='same'),
            Conv2D(32, 3, strides=2, activation='relu', padding='same'),
            Conv2D(32, 3, strides=2, activation='relu', padding='same'),
            Conv2D(32, 3, strides=2, activation='relu', padding='same'),
            Flatten(),
            Dense(512, activation='relu'),
        ])
        
    def call(self, x):
        return self.model(x)

    
class PPOTrainer():
    def __init__(
        self, 
        obs_shape, 
        act_n, 
        lmbda=0.97, 
        gamma=0.99, 
        lr=2e-4, 
        eps_clip=0.2,
        train_step=10,
        entropy_coef=0.05,
        checkpoint_path='mario',
    ):
        self.memory = Memory()
        self.lmbda = lmbda
        self.gamma = gamma 
        self.lr = lr
        self.obs_shape = obs_shape
        self.act_n = act_n
        self.eps_clip = eps_clip
        self.train_step = train_step
        self.entropy_coef = entropy_coef
        self.policy, self.value, self.train_model = self.build_model()
        print(1)
        ckpt = tf.train.Checkpoint(
            train_model=self.train_model,
            optimizer=self.train_model.optimizer,
        )
        self.ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1) 
        
    def build_model(self):
        s_input = Input(self.obs_shape)
        prob_old_input = Input([])
        action_old_input = Input([], dtype='int32')
        gae_input = Input([])
        v_target_input = Input([])
        
        feature = Feature()
        x = feature(s_input)
        policy_dense = Dense(self.act_n, activation='softmax')
        value_dense = Dense(1)
        prob = policy_dense(x)
        v = value_dense(x)
        policy = Model(inputs=s_input, outputs=prob)
        value = Model(inputs=s_input, outputs=v)

        prob_cur = tf.gather(prob, action_old_input, batch_dims=1)
        ratio = prob_cur / (prob_old_input + 1e-3)
        surr1 = ratio * gae_input
        surr2 = K.clip(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * gae_input
        
        # 第二项为熵值计算，由于已经按照动作概率采样，因此计算时不再乘上概率，并且只需要计算当前动作概率的对数
        policy_loss = -K.mean(K.minimum(surr1, surr2)) + K.mean(K.log(prob_cur + 1e-3)) * self.entropy_coef 
        
        value_loss = K.mean((v[:, 0] - v_target_input) ** 2)
        loss = policy_loss + value_loss
        train_model = Model(inputs=[s_input, prob_old_input, action_old_input, gae_input, v_target_input], outputs=loss)
        train_model.add_loss(loss)
        train_model.compile(tf.keras.optimizers.Adam(self.lr))
        return policy, value, train_model
    
    def choose_action(self, states):
        # states.shape: (env_num, height, width, skip_frames) 
        probs = self.policy.predict(states) # shape: (env_num, act_n)
        actions = [np.random.choice(range(self.act_n), p=prob) for prob in probs] # shape: (env_num)
        return actions, probs[np.arange(len(probs)), actions]

    def store(self, states, actions, probs, rewards, next_states, dones):
        self.memory.store(states, actions, probs, rewards, next_states, dones)
           
    def update_model(self, batch_size=128):
        states = np.array(self.memory.states) # shape: (-1, env_num, height, width, skip_frames)
        actions = np.array(self.memory.actions) # shape: (-1, env_num)
        probs = np.array(self.memory.probs) # shape: (-1, env_num)
        rewards = np.array(self.memory.rewards) # shape: (-1, env_num)
        next_states = np.array(self.memory.next_states) # shape: (-1, env_num, height, width, skip_frames)
        dones = np.array(self.memory.dones) # shape: (-1, env_num)
        
        env_num = states.shape[1]
        states = states.reshape([-1, *states.shape[2:]])
        next_states = next_states.reshape([-1, *next_states.shape[2:]])
        actions = actions.flatten()
        probs = probs.flatten()
        
        for step in range(self.train_step):
            v = self.value.predict(states, batch_size=batch_size)
            v_next = self.value.predict(next_states, batch_size=batch_size)
            v = v.reshape([v.shape[0] // env_num, env_num])
            v_next = v_next.reshape([v_next.shape[0] // env_num, env_num])
            
            v_target = rewards + self.gamma * v_next * ~dones
            td_errors = v_target - v
            gae_lst = []
            adv = 0
            for delta in td_errors:
                adv = self.gamma * self.lmbda * adv + delta
                gae_lst.append(adv)
            
            gaes = np.array(gae_lst)
            gaes = gaes.flatten()
            v_target = v_target.flatten()
            self.train_model.fit([states, probs, actions, gaes, v_target], batch_size=batch_size)

        self.memory.reset()
        
    def save(self):
        self.ckpt_manager.save()
        
    def load(self):     
        if self.ckpt_manager.latest_checkpoint:
            status = agent.ckpt_manager.checkpoint.restore(self.ckpt_manager.latest_checkpoint)
            status.run_restore_ops() # 关闭动态图后需要添加这句执行restore操作



if __name__ == '__main__': # multiprocessing 创建的进程会导入该文件而不会执行以下操作
    session = tf.compat.v1.InteractiveSession() # 关闭动态图后，ckpt_manager.save()需要有默认的session

    max_step = 512
    num_envs = 8
    height = 84
    width = 84
    world = 1
    stage = 1
    action_type = SIMPLE_MOVEMENT
    try:
        env = MultipleEnvironments(num_envs, create_env, world, stage, action_type, height, width)
        agent = PPOTrainer(
            env.observation_space.shape, 
            env.action_space.n, 
            train_step=10, 
            lr=1e-4, 
            entropy_coef=0.05, 
            checkpoint_path=f'mario_{world}_{stage}'
        )
        agent.load()
        states = env.reset()
        for epoch in range(1, 201):
            max_pos = 0
            min_pos = np.inf
            for step in range(max_step):
                actions, probs = agent.choose_action(np.stack(states, axis=0))
                next_states, rewards, dones, infos = env.step(actions)
                agent.store(states, actions, probs, rewards, next_states, dones)
                states = next_states
                max_pos = max(max_pos, max([info['x_pos'] for info in infos]))
                min_pos = min(min_pos, min([info['x_pos'] if done else np.inf for info, done in zip(infos, dones)]))
            clear_output() # jupyter notebook 清屏
            print(f'epoch: {epoch} | max position: {max_pos} | min position: {min_pos}')
            agent.update_model(batch_size=256)
            if epoch % 10 == 0:
                agent.save()
    finally:
        env.close()
        session.close()