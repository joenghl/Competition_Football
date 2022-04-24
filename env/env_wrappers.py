import numpy as np
import gym
from gym import spaces
from abc import ABC,abstractmethod
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
        return np.stack(obs)        #(env_num(rollout_thread),agent_num,oberservation_dim)  (1,4,133)

    def step(self,actions):
        self.actions = actions
        results = [env.step(a) for(a,env) in zip(self.actions,self.envs)]
<<<<<<< HEAD
        obs, rews, dones, info_bef, info_aft = map(np.array,zip(*results))
=======
        obs,rews,dones,info_bef,info_aft = map(np.array,zip(*results))
>>>>>>> 7d8224aeb79fa1e032994b185678e8b7d8b3b56c
        for(i,done) in enumerate(dones):
            if 'bool' in done.__class__.__name__:
                if done:
                    obs[i] = self.envs[i].reset()
            else:
                if np.all(done):
                    obs[i] = self.envs[i].reset()
        self.actions = None
        return obs, rews, dones, info_bef, info_aft
        
    def close(self):
        for env in self.envs:
            env.close()

    # def render(self, mode="rgb_array"):
    #     pass
