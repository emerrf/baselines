#!/usr/bin/env python3
from mpi4py import MPI
from baselines.common.cmd_util import make_mujoco_env, env_arg_parser
from baselines.common import tf_util as U
from baselines import logger
from baselines.ppo1 import mlp_policy, pposgd_simple
import os, json

import sys
from datetime import datetime

def train(args):
    U.make_session(num_cpu=1).__enter__()

    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        logger.configure()
    else:
        logger.configure(format_strs=[])
        logger.set_level(logger.DISABLED)
    workerseed = args.seed + 10000 * MPI.COMM_WORLD.Get_rank()
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=args.policy_hid_size, num_hid_layers=args.policy_num_hid_layers)

    env = make_mujoco_env(args.env, workerseed)
    learn_keys = ["timesteps_per_batch", "clip_param", "entcoeff",
                  "optim_epochs", "optim_stepsize", "optim_batchsize", "gamma",
                  "lam", "max_timesteps", "policy_snapshot_filepath",
                  "snapshot_every", "snapshot_filepath"]
    learn_kwargs = dict([kv_tuple for kv_tuple in args._get_kwargs()
                         if kv_tuple[0] in learn_keys])

    if args.snapshot_every > 0:
        learn_kwargs["snapshot_filepath"] = create_path(args.snapshot_filepath,
                                                        args.env)
        params = learn_kwargs.copy()
        params['env'] = args.env
        params['cmd'] = ' '.join(sys.argv)

        os.makedirs(os.path.dirname(params["snapshot_filepath"]), exist_ok=True)
        json_fname = params["snapshot_filepath"] + '.json'
        with open(json_fname, '+w') as fd:
            json.dump(params, fd, indent=4, sort_keys=True)

    pposgd_simple.learn(env, policy_fn, **learn_kwargs)
    env.close()


def create_path(filepath, env_id):
    ts_str = datetime.now().strftime('%Y%m%dT%H%M%S')
    if filepath:
        dirname = os.path.dirname(filepath)
        filename = os.path.basename(filepath)
    else:
        dirname = ''
        filename = ''
    if not dirname:
        dirname = os.path.join(os.curdir, 'models', env_id, ts_str)
    if not filename:
        filename = "tf_sess"

    return os.path.join(dirname, filename)


def main():
    args = env_arg_parser().parse_args()
    train(args)



if __name__ == '__main__':
    main()
