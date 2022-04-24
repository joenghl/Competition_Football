import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
from config import get_config
from env.chooseenv import make
from env.env_wrappers import DummyVecEnv
from env.obs_interfaces.observation import obs_type


DEFAULT_ENV_CONFIG = {
  "football_11_vs_11_stochastic": {
    "class_literal": "Football",
    "n_player": 22,
    "max_step": 3000,
    "game_name": "11_vs_11_stochastic",
    "is_obs_continuous": False,
    "is_act_continuous": False,
    "agent_nums": [11,11],
    "obs_type": ["dict", "dict"],
    "act_box": {"discrete_n": 19}
  },
  "football_5v5_malib": {
    "class_literal": "Football",
    "n_player": 8,
    "max_step": 3000,
    "game_name": "malib_5_vs_5",
    "is_obs_continuous": False,
    "is_act_continuous": False,
    "agent_nums": [4,4],
    "obs_type": ["dict", "dict"],
    "act_box": {"discrete_n": 19}
  }
}

# 
def make_train_env(all_args):
    def get_env_list(rank):
        def init_env():
            env = make(all_args.env_name)
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_list(0)])
    else:
        raise NotImplementedError


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = make(all_args.env_name)
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        raise NotImplementedError



def parse_args(args, parser):
    # parser.add_argument("--env_config", type=dict, default = DEFAULT_ENV_CONFIG)
    parser.add_argument("--number_of_left_players_agent_controls", type=int, default=4)
    parser.add_argument('--number_of_right_players_agent_controls', type=int, default=4)
    parser.add_argument('--env_name', type=str, default="football_5v5_malib",
                        help="football_11_vs_11_stochastic/football_5v5_malib")
    parser.add_argument('--representation', type=str, default="raw")
    parser.add_argument("--my_ai", default="football_5v5_mappo", help="football_5v5_mappo/football_11v11_mappo/random")
    parser.add_argument("--opponent", default="football_5v5_mappo", help="football_5v5_mappo/football_11v11_mappo/random")
<<<<<<< HEAD
    parser.add_argument('--rewards', type=str, default="scoring,checkpoints")
=======
    parser.add_argument('--rewards', type=str, default="scoring")
>>>>>>> 7d8224aeb79fa1e032994b185678e8b7d8b3b56c
    parser.add_argument('--run_name', type=str, default="run")
    all_args = parser.parse_known_args(args)[0]
    return all_args



def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        assert (all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy), ("check recurrent policy!")
    elif all_args.algorithm_name == "mappo":
        assert (all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False), (
            "check recurrent policy!")
    else:
        raise NotImplementedError

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)
    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results") / \
              (all_args.env_name + '_' + all_args.representation) / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    #wandb
    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project="football",
                         entity="atan",
                         notes=socket.gethostname(),
                         group=all_args.env_name + "_" + str(all_args.experiment_name),
                         name=str(all_args.experiment_name) + "_" + str(all_args.run_name),
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                             str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

<<<<<<< HEAD
    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
                              str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(
        all_args.user_name))
=======
    # setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
    #                           str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(
    #     all_args.user_name))
>>>>>>> 7d8224aeb79fa1e032994b185678e8b7d8b3b56c

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)
    
    # env init
    num_agents = DEFAULT_ENV_CONFIG[all_args.env_name]["n_player"]
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    if all_args.share_policy:
        from runner.shared.football_runner import FootballRunner as Runner
    else:
        raise NotImplementedError

    runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])