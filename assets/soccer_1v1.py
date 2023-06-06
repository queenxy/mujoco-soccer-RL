import numpy as np

from gym import utils
from gym.envs.mujoco import MujocoEnv
from gym.spaces import Box

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class SoccerEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }
    
    def __init__(
        self,
        xml_file="soccer_shooting.xml",
        reset_noise_scale = 0.2,
        kp = 10,
        **kwargs
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            reset_noise_scale,
            **kwargs       
        )

        self._reset_noise_scale = reset_noise_scale
        self.kp = kp
        
        obs_shape = 12
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float64
        )

        MujocoEnv.__init__(self, xml_file, 5, observation_space=observation_space, **kwargs)


    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        ball_pos = self.get_body_com("ball")[:2].copy()
        agentB_pos= self.get_body_com("B1")[:2].copy()
        agentY_pos= self.get_body_com("Y1")[:2].copy()

        if ball_pos[0] > 0.76:
            score1 = 1
            score2 = 0
        elif ball_pos[0] < -0.76 :
            score1 = 0
            score2 = 1
        else:
            score1 = 0
            score2 = 0

        goalY = np.array([0.75,0])
        goalB = np.array([-0.75,0])
        dis = agentB_pos - ball_pos
        dis_rew = -0.05 * np.linalg.norm(dis)
        kickY = ball_pos - goalY
        kickB = ball_pos - goalB
        kick_rew = -0.15 * np.linalg.norm(kickY)
        ang_rewY = 0.01 * np.dot(dis,kickY)/(np.linalg.norm(dis)+1e-5)/(np.linalg.norm(kickY)+1e-5)
        ang_rewB = -0.01 * np.dot(dis,kickB)/(np.linalg.norm(dis)+1e-5)/(np.linalg.norm(kickB)+1e-5)

        reward = dis_rew + kick_rew + ang_rewY + ang_rewB + 100 * score1 + 100 * score2

        done = score1 or score2

        observation = self._get_obs()
        info = {
            "reward": reward,
        }

        return observation, reward, done, False, info

    def _get_obs(self):
        ball_pos = self.get_body_com("ball")[:2].copy()
        ball_vel = self.get_body_vel("ball")[3:5].copy()
        agentB_pos= self.get_body_com("B1")[:2].copy()
        agentB_vel= self.get_body_vel("B1")[3:5].copy()
        agentY_pos= self.get_body_com("Y1")[:2].copy()
        agentY_vel= self.get_body_vel("Y1")[3:5].copy()

        observations = np.concatenate((ball_pos,ball_vel,agentB_pos ,agentB_vel, agentY_pos, agentY_vel))
        # print(observations)
        
        return observations

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel
        self.set_state(qpos, qvel)

        observation = self._get_obs()

        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
