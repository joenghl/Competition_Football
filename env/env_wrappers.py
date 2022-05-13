import numpy as np
import gym
from gym import spaces
from abc import ABC,abstractmethod
from multiprocessing import Process, Pipe
class ShareVecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    Used to batch data from multiple copies of an environment, so that
    each observation becomes an batch of observations, and expected action is a batch of actions to
    be applied per-environment.
    """
    def __init__(self, num_envs, observation_space, share_observation_space, action_space,joint_action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.share_observation_space = share_observation_space
        self.action_space = action_space
        self.joint_action_space = joint_action_space
    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    def step(self, actions):
        """
        Step the environments synchronously.

        This is available for backwards compatibility.
        """
        # self.step_async(actions)
        # return self.step_wait()
        pass
    def decode(self, joint_action):
        if isinstance(joint_action, np.ndarray):
            joint_action = joint_action.tolist()
        joint_action_decode = []
        for action in joint_action:
            joint_action_decode.append(action[0].index(1))
        return joint_action_decode
        
    # def render(self, mode='human'):
    #     imgs = self.get_images()
    #     bigimg = tile_images(imgs)
    #     if mode == 'human':
    #         self.get_viewer().imshow(bigimg)
    #         return self.get_viewer().isopen
    #     elif mode == 'rgb_array':
    #         return bigimg
    #     else:
    #         raise NotImplementedError


# single env
class DummyVecEnv(ShareVecEnv):
    def __init__(self, env_list):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.env_list = env_list
        self.envs = [fn() for fn in self.env_list]
        env =self.envs[0]
        ShareVecEnv.__init__(self, len(
            env_list), env.observation_space, env.share_observation_space, env.action_space, env.joint_action_space)
        self.actions = None
    def reset(self):
        obs = [env.reset() for env in self.envs]
        return np.stack(obs)        #(env_num(rollout_thread),agent_num,oberservation_dim)  (1,8,133)

    def step(self,actions):
        self.actions = actions
        results = [env.step(a) for(a,env) in zip(self.actions,self.envs)]
        obs, rews, dones, info_bef, info_aft = map(np.array,zip(*results))
        for(i,done) in enumerate(dones):
            if 'bool' in done.__class__.__name__:
                if done:
                    obs[i] = self.envs[i].reset()
            else:
                if np.all(done):
                    obs[i] = self.envs[i].reset()
        self.actions = None
        return obs, rews, dones, info_bef, info_aft

    def get_current_states(self):
        current_states = [env.current_state for env in self.envs]    
        return current_states

    def close(self):
        for env in self.envs:
            env.close()

    # def render(self, mode="rgb_array"):
    #     pass


class SubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, share_observation_space, action_space ,joint_action_space= self.remotes[0].recv()
        ShareVecEnv.__init__(self, len(env_fns), observation_space,
                             share_observation_space, action_space, joint_action_space)
        print("yes")

    def step(self,actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True
        self.actions = actions
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos_bef, info_aft = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), np.stack(infos_bef), np.stack(info_aft)


    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):

            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def get_current_states(self):
        for remote in self.remotes:
            remote.send(('get_current_state',None))
        current_states = [remote.recv() for remote in self.remotes]    
        return current_states

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self.remotes]
        return np.stack(obs)


    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    # def render(self, mode="rgb_array"):
    #     for remote in self.remotes:
    #         remote.send(('render', mode))
    #     if mode == "rgb_array":   
    #         frame = [remote.recv() for remote in self.remotes]
    #         return np.stack(frame) 


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            obs, rews, done, info_bef, info_aft = env.step(data)
            if 'bool' in done.__class__.__name__:
                if done:
                    obs = env.reset()
            else:
                if np.all(done):
                    obs = env.reset()
            remote.send((obs, rews, done, info_bef,info_aft))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'get_current_state':
            cur_state = env.current_state
            remote.send(cur_state)
        elif cmd == 'render':
            if data == "rgb_array":
                fr = env.render(mode=data)
                remote.send(fr)
            elif data == "human":
                env.render(mode=data)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            env.close()
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.share_observation_space, env.action_space, env.joint_action_space))
        else:
            raise NotImplementedError