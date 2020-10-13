#!/usr/bin/env python
# coding: utf-8
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
#gym
import gym
import gym.spaces
#ros
import rospy
from std_srvs.srv import Empty
#メッセージ型をインポート
from std_msgs.msg import Float64
from std_msgs.msg import String
from op3_controller.msg import Command
from sensor_msgs.msg import JointState
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import ModelState


from motion import Motion

NUM_ACTION = 13
NUM_STATE1 = 13
NUM_STATE2 = 18
NUM_STATE = NUM_STATE1  + NUM_STATE2
RESET_TIME = 0.5
# RESET_TIME2 = 4
# RESET_DIS = 0.7
RATE = 16

class OP3Env(gym.Env):
    def __init__(self):
        super().__init__()
        #ノード名を宣言
        rospy.init_node('OP3Env', anonymous=True)
        self.rate = rospy.Rate(RATE)

        action = np.array([-1.5, 1.5, -1.7, 0.00, 1.7, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00])
        # OP3Env.state = np.array([0, 0, 1.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        OP3Env.state = np.zeros(NUM_STATE)
        # state更新
        sub1 = rospy.Subscriber('gazebo/model_states', ModelStates, callback_get_state)
        sub2 = rospy.Subscriber('robotis_op3/joint_states', JointState, callback_get_joint_state)

        reward = 0

        self.action_space = gym.spaces.Discrete(NUM_ACTION)
        self.observation_space = gym.spaces.Box(
            low = -np.pi,
            high = np.pi,
            shape = (NUM_STATE,)
        )
        self.Com = Motion(action)
        self.reset()

    def reset(self):
        #初期値に戻す
        #終了判定・報酬初期化
        self.done = False
        OP3Env.reward = 0
        self.down_flag = 0
        #状態の初期化
        rospy.wait_for_service('/gazebo/unpause_physics')
        unpose = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        unpose()
        time.sleep(1)
        #体勢の初期化
        start_state = np.array([0, 0, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        state_pub(start_state)
        OP3Env.action = np.array([-1.5, 1.5, -1.7, 0.00, 1.7, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00])
        action_pub(OP3Env.action)
        time.sleep(1)
        #世界を初期化
        reset_world = rospy.ServiceProxy('/gazebo/reset_world',Empty)
        reset_world()
        #位置セット
        set_state = np.array([0, 0, 0.2, 0, 1, 0, 0.6, 0, 0, 0, 0, 0, 0])
        state_pub(set_state)
        time.sleep(1)
        rospy.wait_for_service('/gazebo/pause_physics')
        pose = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        pose()

        self.start_time = rospy.get_time()
        self.tmp_time = self.start_time
        self.tmp_state = OP3Env.state

        return OP3Env.state

    def step(self, action_com):
        #ステータスの更新
        now_time = rospy.get_time()
            #アクションのパブリッシュ
        rospy.wait_for_service('/gazebo/unpause_physics')
        unpose = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        unpose()

        print(action_com)
        OP3Env.action = self.Com.motion(action_com)
        action_pub(OP3Env.action)

        self.rate.sleep()
        rospy.wait_for_service('/gazebo/pause_physics')
        pose = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        pose()
            #状態を取得
        next_state = OP3Env.state

        # #報酬
        #     #状態を取得して報酬を得る
        # sub = rospy.Subscriber('gazebo/model_states', ModelStates, callback_get_reward)

        #エピソード終了の判定処理
        #倒れてるかどうか判断して終了判定
        if (now_time - self.tmp_time) >= RESET_TIME:
            self.tmp_time = now_time
            if next_state[4]*next_state[6] < 0:
                self.down_flag += 1
            else:
                self.down_flag = 0
            if self.down_flag == 2:
                self.done = True
                print("******倒れてるやん******")
        # #RESET_TIME2ごとに進み具合を見て終了判定
        # if (now_time - self.tmp_time) > RESET2_TIME:
        #     self.tmp_time = now_time
        #     tmp_dis = calc_dis(self.tmp_state[0], self.tmp_state[1])
        #     now_dis = calc_dis(OP3Env.state[0], OP3Env.state[1])
        #     if (abs(tmp_dis - now_dis)) <= RESET_DIS:
        #         self.done = True
        #         print("******倒れてるやん******")
        #         print("tmp:", tmp_dis, "\nnow:", now_dis)
        #     else:
        #         print("OK")
        #         print("tmp:", tmp_dis, "\nnow:", now_dis)
        #     self.tmp_state = OP3Env.state
        #通常の終了判定
        if now_time >= (self.start_time+20):
            self.done = True
            print("======時間です======")
        #info

        return next_state, OP3Env.reward, self.done, {}

    def render(self):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        pass


def action_pub(action):
    #左肘
    l_el = rospy.Publisher('robotis_op3/l_el_position/command', Float64, queue_size=1)
    #右肘
    r_el = rospy.Publisher('robotis_op3/r_el_position/command', Float64, queue_size=1)
    #左肩
    l_sho_pitch = rospy.Publisher('robotis_op3/l_sho_pitch_position/command', Float64, queue_size=1)
    l_sho_roll = rospy.Publisher('robotis_op3/l_sho_roll_position/command', Float64, queue_size=1)
    #右肩
    r_sho_pitch = rospy.Publisher('robotis_op3/r_sho_pitch_position/command', Float64, queue_size=1)
    r_sho_roll = rospy.Publisher('robotis_op3/r_sho_roll_position/command', Float64, queue_size=1)
    #左腰
    l_hip_pitch = rospy.Publisher('robotis_op3/l_hip_pitch_position/command', Float64, queue_size=1)
    l_hip_roll = rospy.Publisher('robotis_op3/l_hip_roll_position/command', Float64, queue_size=1)
    l_hip_yaw = rospy.Publisher('robotis_op3/l_hip_yaw_position/command', Float64, queue_size=1)
    #右腰
    r_hip_pitch = rospy.Publisher('robotis_op3/r_hip_pitch_position/command', Float64, queue_size=1)
    r_hip_roll = rospy.Publisher('robotis_op3/r_hip_roll_position/command', Float64, queue_size=1)
    r_hip_yaw = rospy.Publisher('robotis_op3/r_hip_yaw_position/command', Float64, queue_size=1)
    #左ひざ
    l_knee = rospy.Publisher('robotis_op3/l_knee_position/command', Float64, queue_size=1)
    #右ひざ
    r_knee = rospy.Publisher('robotis_op3/r_knee_position/command', Float64, queue_size=1)
    # #左足首
    l_ank_pitch = rospy.Publisher('robotis_op3/l_ank_pitch_position/command', Float64, queue_size=1)
    l_ank_roll = rospy.Publisher('robotis_op3/l_ank_roll_position/command', Float64, queue_size=1)
    # #右足首
    r_ank_pitch = rospy.Publisher('robotis_op3/r_ank_pitch_position/command', Float64, queue_size=1)
    r_ank_roll = rospy.Publisher('robotis_op3/r_ank_roll_position/command', Float64, queue_size=1)

    #データをpublish
    l_el.publish(action[0])
    r_el.publish(action[1])
    l_sho_pitch.publish(action[2])
    l_sho_roll.publish(action[3])
    r_sho_pitch.publish(action[4])
    r_sho_roll.publish(action[5])
    l_hip_pitch.publish(action[6])
    l_hip_roll.publish(action[7])
    l_hip_yaw.publish(action[8])
    r_hip_pitch.publish(action[9])
    r_hip_roll.publish(action[10])
    r_hip_yaw.publish(action[11])
    l_knee.publish(action[12])
    r_knee.publish(action[13])
    l_ank_pitch.publish(action[14])
    l_ank_roll.publish(action[15])
    r_ank_pitch.publish(action[16])
    r_ank_roll.publish(action[17])

    # pub = rospy.Publisher('command_pub', Command, queue_size=1)
    # action_msg = Command()
    # # 肘
    # action_msg.l_el = action[0]
    # action_msg.r_el = action[1]
    # #左肩
    # action_msg.l_sho_pitch = action[2]
    # action_msg.l_sho_roll =  action[3]
    # #右肩
    # action_msg.r_sho_pitch =  action[4]
    # action_msg.r_sho_roll =  action[5]
    # #左腰
    # action_msg.l_hip_pitch = action[6]
    # action_msg.l_hip_roll = action[7]
    # action_msg.l_hip_yaw = action[8]
    # #右腰
    # action_msg.r_hip_pitch = action[9]
    # action_msg.r_hip_roll = action[10]
    # action_msg.r_hip_yaw = action[11]
    # #左ひざ
    # action_msg.l_knee = action[12]
    # #右ひざ
    # action_msg.r_knee = action[13]
    # #左足首
    # action_msg.l_ank_pitch = action[14]
    # action_msg.l_ank_roll = action[15]
    # #右足首
    # action_msg.r_ank_pitch = action[16]
    # action_msg.r_ank_roll  = action[17]
    # pub.publish(action_msg)


def state_pub(state):
    pub = rospy.Publisher('gazebo/set_model_state',ModelState, queue_size=1)
    state_msg = ModelState()
    state_msg.model_name = 'robotis_op3'
    state_msg.pose.position.x = state[0]
    state_msg.pose.position.y = state[1]
    state_msg.pose.position.z = state[2]
    state_msg.pose.orientation.x = state[3]
    state_msg.pose.orientation.y = state[4]
    state_msg.pose.orientation.z = state[5]
    state_msg.pose.orientation.w = state[6]
    state_msg.twist.linear.x = state[7]
    state_msg.twist.linear.y = state[8]
    state_msg.twist.linear.z = state[9]
    state_msg.twist.angular.x = state[10]
    state_msg.twist.angular.y = state[11]
    state_msg.twist.angular.z = state[12]
    pub.publish(state_msg)


def callback_get_state(data):
    #状態取得
    state = [[] for _ in range(NUM_STATE1)]
    state[0] = data.pose[1].position.x
    state[1] = data.pose[1].position.y
    state[2] = data.pose[1].position.z
    state[3] = data.pose[1].orientation.x
    state[4] = data.pose[1].orientation.y
    state[5] = data.pose[1].orientation.z
    state[6] = data.pose[1].orientation.w
    state[7] = data.twist[1].linear.x
    state[8] = data.twist[1].linear.y
    state[9] = data.twist[1].linear.z
    state[10] = data.twist[1].angular.x
    state[11] = data.twist[1].angular.y
    state[12] = data.twist[1].angular.z
    #numpyに変換
    # OP3Env.state1 = np.array(state)
    for i in range(NUM_STATE1):
        OP3Env.state[i] = state[i]
    # OP3Env.state[0:NUM_STATE1] = np.array(state)


def callback_get_joint_state(data):
    joint_state = [[] for _ in range(NUM_STATE2)]
    for i in range(NUM_STATE2):
        k = i+2
        joint_state[i] = data.position[k]
    # joint_state[0] = data.head_pan.position
    # joint_state[1] = data.head_tilt.position
    # joint_state[0] = data.position[0] #l_ank_pitch
    # joint_state[1] = data.l_ank_roll.position
    # joint_state[2] = data.l_el.position
    # joint_state[3] = data.l_hip_pitch.position
    # joint_state[4] = data.l_hip_roll.position
    # joint_state[5] = data.l_hip_yaw.position
    # joint_state[6] = data.l_knee.position
    # joint_state[7] = data.l_sho_pitch.position
    # joint_state[8] = data.l_sho_roll.position
    # joint_state[9] = data.r_ank_pitch.position
    # joint_state[10] = data.r_ank_roll.position
    # joint_state[11] = data.r_el.position
    # joint_state[12] = data.r_hip_pitch.position
    # joint_state[13] = data.r_hip_roll.position
    # joint_state[14] = data.r_hip_yaw.position
    # joint_state[15] = data.r_knee.position
    # joint_state[16] = data.r_sho_pitch.position
    # joint_state[17] = data.r_sho_roll.position
    for i in range(NUM_STATE2):
        k = i + NUM_STATE1
        OP3Env.state[k] = joint_state[i]


# def callback_get_reward(data):
#     #進んだ距離を取得
#     dis_x = data.pose[1].position.x
#     #横にずれた幅
#     dis_y = abs(data.pose[1].position.y)
#     #報酬計算
#     dis = np.sqrt(dis_x*dis_x - dis_y*dis_y)
#     if dis_x >0:
#         OP3Env.reward = torch.FloatTensor([dis])
#     else:
#         OP3Env.reward = torch.FloatTensor([dis_x])

def calc_dis(dis_x, dis_y):
    dis = np.sqrt(dis_x**2 + dis_y**2)
    return dis
