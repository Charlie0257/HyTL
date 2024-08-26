import robosuite as suite
from robosuite.wrappers.gym_wrapper import GymWrapper
from robosuite import load_controller_config

import maple.torch.pytorch_util as ptu
from maple.data_management.env_replay_buffer import EnvReplayBuffer
from maple.samplers.data_collector import MdpPathCollector
from maple.torch.sac.policies import (
    TanhGaussianPolicy,
    PAMDPPolicy,
    MakeDeterministic
)
from maple.torch.sac.sac import SACTrainer
from maple.torch.sac.sac_hybrid import SACHybridTrainer
from maple.torch.networks import ConcatMlp
from maple.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

from LTL.env import make_ltlenv
from LTL.format import get_obss_preprocessor

from plan.plan_layer import LearnablePlanLayer

import numpy as np
import torch

from gnns.graphs.GCN import *
from gnns.graphs.GNN import GNNMaker

from Transformer.TransSyn import TransfomerSyn

def prYellow(prt): print("\033[93m {}\033[00m".format(prt))

def experiment(variant):
    def make_env(mode):
        assert mode in ['expl', 'eval']
        torch.set_num_threads(1)

        env_variant = variant['env_variant']

        controller_config = load_controller_config(default_controller=env_variant['controller_type'])
        controller_config_update = env_variant.get('controller_config_update', {})
        controller_config.update(controller_config_update)

        robot_type = env_variant.get('robot_type', 'Jaco')

        obs_keys = env_variant['robot_keys'] + env_variant['obj_keys']

        env = suite.make(
            env_name=env_variant['env_type'],
            robots=robot_type,
            has_renderer=False,
            has_offscreen_renderer=True,  # default is True
            use_camera_obs=True,
            controller_configs=controller_config,

            **env_variant['env_kwargs']
        )

        env = GymWrapper(env, keys=obs_keys)

        return env

    prYellow(variant)

    if variant['map_goal']:
        prYellow(variant['env_variant']['controller_config_update']['position_limits'])
        expl_env = make_ltlenv(make_env(mode='expl'), progression_mode='full', seed=0, intrinsic=0, useLTL=variant['use_LTL'], use_Waypoint=variant['ues_waypoint'],
                               bounds=variant['env_variant']['controller_config_update']['position_limits'], task_name=variant['env_variant']['env_type'],
                               way_weight=variant['way_weight'])
        eval_env = make_ltlenv(make_env(mode='eval'), progression_mode='full', seed=0, intrinsic=0, useLTL=variant['use_LTL'], use_Waypoint=variant['ues_waypoint'],
                               bounds=variant['env_variant']['controller_config_update']['position_limits'], task_name=variant['env_variant']['env_type'],
                               way_weight=variant['way_weight'])
    else:
        expl_env = make_ltlenv(make_env(mode='expl'), progression_mode='full', seed=0, intrinsic=0,
                               useLTL=variant['use_LTL'], use_Waypoint=variant['ues_waypoint'], task_name=variant['env_variant']['env_type'],
                               way_weight=variant['way_weight'])
        eval_env = make_ltlenv(make_env(mode='eval'), progression_mode='full', seed=0, intrinsic=0,
                               useLTL=variant['use_LTL'], use_Waypoint=variant['ues_waypoint'], task_name=variant['env_variant']['env_type'],
                               way_weight=variant['way_weight'])
    obs_space, preprocess_obss = get_obss_preprocessor(expl_env, gnn=False, progression_mode='full')

    if variant['use_LTL']:
        obs_dim = expl_env.observation_space['features'].low.size
    else:
        obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    if variant['use_LTL']:
        text_embedding_size = variant['d_out']
        gnn_type = 'RGCN_8x32_ROOT_SHARED'
        gnn = TransfomerSyn(obs_size=22, d_model=64, d_out=text_embedding_size,
                            nhead=8, num_encoder_layers=4,
                            pool='mean', dim_ff=256, dropout=0.0, layer_norm_eps=1e-5)
        print("Transformer Number of parameters:", sum(p.numel() for p in gnn.parameters() if p.requires_grad))
        gnn.init_by_TFixup()
        print(f"Transformar Module")
    else:
        text_embedding_size = 0
        gnn = None

    if not variant['use_LTL']:
        text_embedding_size = 0

    if variant['ues_waypoint']:
        goal_dim = 3
        path_max_len = variant['path_max_len']
        device = "cuda:0"
        plan_layer = LearnablePlanLayer(
                                        goal_dim,
                                        path_max_len,
                                        device=device
                                        )
    else:
        plan_layer = None
        goal_dim = 0

    M = variant['layer_size']
    qf1 = ConcatMlp(
        input_size=obs_dim + text_embedding_size + action_dim + goal_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + text_embedding_size + action_dim + goal_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    prYellow(f"qf1 is :\n {qf1}")
    target_qf1 = ConcatMlp(
        input_size=obs_dim + text_embedding_size + action_dim + goal_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + text_embedding_size + action_dim + goal_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    # clean env: action_skill_dim=5 (reach, grasp, lift, hover, bin)
    action_dim_s = getattr(expl_env, "action_skill_dim", 0)
    action_dim_p = action_dim - action_dim_s
    if action_dim_s == 0:
        trainer_class = SACTrainer
        policy = TanhGaussianPolicy(
            obs_dim=obs_dim + goal_dim,
            action_dim=action_dim,
            hidden_sizes=[M, M],
        )

        target_entropy_config = variant['ll_sac_variant'].get('target_entropy_config', {})

        if variant['ll_sac_variant'].get('high_init_ent'):
            target_entropy_config.update(dict(
                init_epochs=200,
                init=0,
            ))

        variant['trainer_kwargs']['target_entropy_config'] = target_entropy_config

    else:
        trainer_class = SACHybridTrainer
        policy_kwargs = {}
        policy_class = PAMDPPolicy

        pamdp_variant = variant.get('pamdp_variant', {})

        for k in [
            'one_hot_s',
        ]:
            policy_kwargs[k] = pamdp_variant[k]

        for k in [
            'target_entropy_s',
            'target_entropy_p',
        ]:
            variant['trainer_kwargs'][k] = pamdp_variant.get(k, None)

        target_entropy_config = pamdp_variant.get('target_entropy_config', {})

        if pamdp_variant.get('high_init_ent'):
            assert pamdp_variant['one_hot_s']
            target_entropy_config['init_epochs'] = 200
            target_entropy_config['init_s'] = 0.97 * np.log(action_dim_s)
            if not pamdp_variant.get('disable_high_init_ent_p', False):
                target_entropy_config['init_p'] = 0

        if pamdp_variant.get('one_hot_factor'):
            target_entropy_config['one_hot_factor'] = pamdp_variant['one_hot_factor']

        variant['trainer_kwargs']['target_entropy_config'] = target_entropy_config

        policy = policy_class(
            obs_dim=obs_dim + text_embedding_size + goal_dim,
            action_dim_s=action_dim_s,
            action_dim_p=action_dim_p,
            hidden_sizes=[M, M],
            **policy_kwargs
        )
        prYellow(f"policy is :\n {policy}")

    eval_policy = MakeDeterministic(policy)

    rollout_fn_kwargs = variant.get('rollout_fn_kwargs', {})

    eval_path_collector = MdpPathCollector(
        gnn,
        preprocess_obss,
        variant['use_LTL'],
        variant['ues_waypoint'],
        plan_layer,
        eval_env,
        eval_policy,
        save_env_in_snapshot=False,
        rollout_fn_kwargs=rollout_fn_kwargs,
    )
    expl_path_collector = MdpPathCollector(
        gnn,
        preprocess_obss,
        variant['use_LTL'],
        variant['ues_waypoint'],
        plan_layer,
        expl_env,
        policy,
        save_env_in_snapshot=False,
        rollout_fn_kwargs=rollout_fn_kwargs,
    )
    replay_buffer = EnvReplayBuffer(
        variant['use_LTL'],
        variant['ues_waypoint'],
        text_embedding_size,
        goal_dim,
        variant['replay_buffer_size'],
        expl_env,
    )
    trainer = trainer_class(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        gnn=gnn,
        freeze=variant['freeze'],
        plan_policy=plan_layer,
        **variant['trainer_kwargs']
    )

    if 'ckpt_epoch' in variant:
        variant['algorithm_kwargs']['num_epochs'] = variant['ckpt_epoch']
        variant['algorithm_kwargs']['eval_epoch_freq'] = 1

    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )

    if variant.get("save_video", True):
        variant['dump_video_kwargs']['imsize'] = eval_env.camera_heights[0]
        variant['dump_video_kwargs']['rollout_fn_kwargs'] = rollout_fn_kwargs
        video_save_func = get_video_save_func(variant)
        algorithm.post_epoch_funcs.append(video_save_func)

    if 'ckpt_path' in variant:
        ckpt_update_func = get_ckpt_update_func(variant)
        algorithm.pre_epoch_funcs.insert(0, ckpt_update_func)

    algorithm.to(ptu.device)
    algorithm.train(gnn, preprocess_obss, variant['use_LTL'], variant['ues_waypoint'], plan_layer, start_epoch=variant.get('ckpt_epoch', 0))

def get_ckpt_update_func(variant):
    import os.path as osp
    import torch
    from maple.launchers.conf import LOCAL_LOG_DIR

    def ckpt_update_func(algo, epoch):
        if epoch == variant.get('ckpt_epoch', None) or epoch % algo._eval_epoch_freq == 0:
            filename = osp.join(LOCAL_LOG_DIR, variant['ckpt_path'], 'itr_%d.pkl' % epoch)
            try:
                print("Loading ckpt from", filename)
                if ptu.gpu_enabled():
                    data = torch.load(filename, map_location='cuda:0')
                else:
                    data = torch.load(filename, map_location='cpu')
                print("checkpoint loaded.")
            except FileNotFoundError:
                print('Could not locate checkpoint. Aborting.')
                exit()

            eval_policy = data['evaluation/policy']
            eval_policy.to(ptu.device)
            algo.eval_data_collector._policy = eval_policy

            expl_policy = data['exploration/policy']
            expl_policy.to(ptu.device)
            algo.expl_data_collector._policy = expl_policy

    return ckpt_update_func

def get_video_save_func(variant):
    from maple.samplers.rollout_functions import rollout
    from maple.launchers.visualization import dump_video

    save_period = variant.get('save_video_period', 50)
    dump_video_kwargs = variant.get("dump_video_kwargs", dict())
    dump_video_kwargs['horizon'] = variant['algorithm_kwargs']['max_path_length']

    def video_save_func(algo, gnn, preprocess_obss, use_LTL, use_waypoint, plan_layer, epoch):
        if epoch % save_period == 0 or epoch == algo.num_epochs:
            if variant.get('vis_expl', True):
                dump_video(
                    gnn,
                    preprocess_obss,
                    use_LTL,
                    use_waypoint,
                    plan_layer,
                    algo.expl_data_collector._env,
                    algo.expl_data_collector._policy,
                    rollout,
                    mode='expl',
                    epoch=epoch,
                    **dump_video_kwargs
                )

            dump_video(
                gnn,
                preprocess_obss,
                use_LTL,
                use_waypoint,
                plan_layer,
                algo.eval_data_collector._env,
                algo.eval_data_collector._policy,
                rollout,
                mode='eval',
                epoch=epoch,
                **dump_video_kwargs
            )
    return video_save_func