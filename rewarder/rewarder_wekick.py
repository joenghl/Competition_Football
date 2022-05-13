import numpy as np

def reward_warpper(rew, prev_obs, obs, act, opp_act):
    for threads in range(rew.shape[0]):
        for(agents) in range(rew.shape[1]):
            rew[threads][agents] = get_reward(rew[threads][agents], prev_obs[threads][agents], obs[threads][agents], act[threads][agents], opp_act[threads])
    return rew

def get_reward(reward, obs_pre, obs_now, action, opponent_action): #action是我方一个agent的动作，opponent_action是对方所有agent的动作
    active = obs_pre['active']#我方球员编号
    ball_owned_player_pre = obs_pre['ball_owned_player']#持球队员编号
    ball_owned_player_now = obs_now['ball_owned_player']
    ball_owned_team_pre = obs_pre['ball_owned_team']#持球队伍,0本队，1对手
    ball_owned_team_now = obs_now['ball_owned_team']
    #判断是否控球
    if ball_owned_team_pre==0 and ball_owned_player_pre==active:
        control_ball_pre = True
    else:
        control_ball_pre = False
    if ball_owned_team_now==0 and ball_owned_player_now==active:
        control_ball_now = True
    else:
        control_ball_now = False
    #控球奖励
    if not control_ball_pre and control_ball_now: #获得控球
        reward += 0.2
    if control_ball_pre and not control_ball_now: #失去控球
        reward -= 0.2
    #铲球奖励
    if action == 16 and ball_owned_team_pre == 1 and ball_owned_team_now != 1: #我方铲球成功
        reward += 0.2
    if 16 in opponent_action and control_ball_pre and not control_ball_now: #对方铲球成功
        reward -= 0.2 
    #带球奖励
    if ball_owned_team_now == 0:
        reward += 0.001 
    if ball_owned_team_now == 1:#对方带球
        reward -= 0.001 
    return reward