from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

from custom_algorithms.pg_runner import PGRunner

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    train_cfg = env_cfg  # Reuse the task's training config
    pg_runner = PGRunner(env=env, train_cfg=train_cfg)
    pg_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations)

if __name__ == '__main__':
    # Run with python legged_gym/scripts/train-2.py --task=go2 --headless
    args = get_args()
    train(args)
