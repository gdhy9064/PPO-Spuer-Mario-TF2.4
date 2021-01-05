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
from tensorflow.keras.layers import *
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
        assert not env.agent_conns[0].closed, 'Environment closed.'
        self.agent_conns[0].send([name, None])
        return self.agent_conns[0].recv()
                
    def step(self, actions):
        assert not env.agent_conns[0].closed, 'Environment closed.'
        for conn, action in zip(self.agent_conns, actions):
            conn.send(['step', action.item()])
        return tuple(zip(*[conn.recv() for conn in self.agent_conns]))
    
    def reset(self):
        assert not env.agent_conns[0].closed, 'Environment closed.'
        for conn in self.agent_conns:
            conn.send(['reset', None])
        return tuple(conn.recv() for conn in self.agent_conns)
    
    def render(self):
        assert not env.agent_conns[0].closed, 'Environment closed.'
        for conn in self.agent_conns:
            conn.send(['render', None])
                
    def close(self):
        assert not env.agent_conns[0].closed, 'Environment closed.'
        for conn in self.agent_conns:
            conn.send(['close', None])
            conn.close()
        for conn in self.env_conns:
            conn.close()


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