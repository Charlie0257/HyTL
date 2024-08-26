"""
This class defines the environments that we are going to use.
Note that this is the place to include the right LTL-Wrapper for each environment.
"""


import gym
from LTL.ltl_wrappers import NoLTLWrapper, LTLEnv

def make_ltlenv(env, progression_mode, seed=None, intrinsic=0, useLTL=True, use_Waypoint=False, bounds=None, task_name=None, way_weight=1., way_dist=0.1):
    env.seed(seed)

    # Adding LTL wrappers
    if (useLTL):
        if use_Waypoint:
            return LTLEnv(env, progression_mode, intrinsic, use_Waypoint, bounds=bounds, task_name=task_name, way_weight=way_weight, way_dist=way_dist)
        else:
            return LTLEnv(env, progression_mode, intrinsic, use_Waypoint, task_name=task_name)
    else:
        if use_Waypoint:
            return NoLTLWrapper(env, use_Waypoint, bounds=bounds, task_name=task_name, way_weight=way_weight, way_dist=way_dist)
        else:
            return NoLTLWrapper(env, use_Waypoint)
