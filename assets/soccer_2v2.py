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
    
    body_id = {
        "ball": 2,
        "B1": 3,
        "B2": 4,
        "Y1": 5,
        "Y2": 6,
    }
    
    def __init__(
        self,
        xml_file="soccer_2v2.xml",
        reset_noise_scale = 0.1,
        kp = 10,
        policy_ = None,
        **kwargs
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            reset_noise_scale,
            kp = 10,
            policy_ = None,
            **kwargs       
        )

        self._reset_noise_scale = reset_noise_scale
        self.kp = kp
        
        obs_shape = 20
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float64
        )

        MujocoEnv.__init__(self, xml_file, 5, observation_space=observation_space, **kwargs)


    def step(self, action):
        # act_Y = - self.policy_.predict(obs)[0]
        # joint_action = np.concatenate([action,act_Y])
        self.do_simulation(action, self.frame_skip)
        ball_pos = self.get_body_com("ball")[:2].copy()
        agentB1_pos= self.get_body_com("B1")[:2].copy()
        agentB2_pos= self.get_body_com("B2")[:2].copy()

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
        dis1 = agentB1_pos - ball_pos
        dis2 = agentB2_pos - ball_pos
        dis = min(np.linalg.norm(dis1), np.linalg.norm(dis2))
        dis_rew = -0.05 * np.linalg.norm(dis)
        kickY = ball_pos - goalY
        kickB = ball_pos - goalB
        kick_rew = -0.15 * np.linalg.norm(kickY)
        ang_rewY_1 = 0.01 * np.dot(dis1,kickY)/(np.linalg.norm(dis1)+1e-5)/(np.linalg.norm(kickY)+1e-5)
        ang_rewB_1 = -0.01 * np.dot(dis1,kickB)/(np.linalg.norm(dis1)+1e-5)/(np.linalg.norm(kickB)+1e-5)
        ang_rewY_2 = 0.01 * np.dot(dis2,kickY)/(np.linalg.norm(dis2)+1e-5)/(np.linalg.norm(kickY)+1e-5)
        ang_rewB_2 = -0.01 * np.dot(dis2,kickB)/(np.linalg.norm(dis2)+1e-5)/(np.linalg.norm(kickB)+1e-5)

        reward = dis_rew + kick_rew + ang_rewY_1 + ang_rewB_1 + ang_rewY_2 + ang_rewB_2 + 100 * score1 - 100 * score2

        done = score1 or score2

        observation = self._get_obs()
        obs_Y = self.get_obsY()
        info = {
            "reward": reward,
            "obs_Y": obs_Y,
        }
        
        return observation, reward, done, False, info

    def _get_obs(self):
        ball_pos = self.get_body_com("ball")[:2].copy()
        ball_vel = self.get_body_vel("ball")[3:5].copy()
        agentB1_pos= self.get_body_com("B1")[:2].copy()
        agentB1_vel= self.get_body_vel("B1")[3:5].copy()
        agentB2_pos= self.get_body_com("B2")[:2].copy()
        agentB2_vel= self.get_body_vel("B2")[3:5].copy()
        agentY1_pos= self.get_body_com("Y1")[:2].copy()
        agentY1_vel= self.get_body_vel("Y1")[3:5].copy()
        agentY2_pos= self.get_body_com("Y2")[:2].copy()
        agentY2_vel= self.get_body_vel("Y2")[3:5].copy()

        observations = np.concatenate((ball_pos,ball_vel,agentB1_pos ,agentB1_vel, 
                                       agentB2_pos ,agentB2_vel, agentY1_pos, agentY1_vel,
                                       agentY2_pos, agentY2_vel))
        # print(observations)
        
        return observations
    
    def get_obsY(self):
        ball_pos = self.get_body_com("ball")[:2].copy()
        ball_vel = self.get_body_vel("ball")[3:5].copy()
        agentB1_pos= self.get_body_com("B1")[:2].copy()
        agentB1_vel= self.get_body_vel("B1")[3:5].copy()
        agentB2_pos= self.get_body_com("B2")[:2].copy()
        agentB2_vel= self.get_body_vel("B2")[3:5].copy()
        agentY1_pos= self.get_body_com("Y1")[:2].copy()
        agentY1_vel= self.get_body_vel("Y1")[3:5].copy()
        agentY2_pos= self.get_body_com("Y2")[:2].copy()
        agentY2_vel= self.get_body_vel("Y2")[3:5].copy()

        observations = np.concatenate((ball_pos,ball_vel,agentY1_pos ,agentY1_vel, 
                                       agentY2_pos ,agentY2_vel, agentB1_pos, agentB1_vel,
                                       agentB2_pos, agentB2_vel))
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

    def get_body_vel(self, body_name):
        return self.data.cvel[self.body_id[body_name]]
    