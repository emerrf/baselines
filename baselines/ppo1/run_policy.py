import argparse
from baselines.common import tf_util as U
from baselines.ppo1 import mlp_policy
import tensorflow as tf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hdf")
    parser.add_argument("--timestep_limit",type=int)
    parser.add_argument("--snapname")
    args = parser.parse_args()

    U.make_session(num_cpu=1).__enter__()

    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    hid_size=64, num_hid_layers=2)


    # checkpoint_path = tf.train.latest_checkpoint('baselines/ppo1/models/CartPole-v1/20180216T173751/tf_sess_0005')
    checkpoint_path = 'models/CartPole-v1/20180216T173751/tf_sess_0005'
    import gym
    env = gym.make('CartPole-v1')

    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_fn("pi", ob_space, ac_space)
    saver = tf.train.Saver()
    saver.restore(tf.get_default_session(), checkpoint_path)

    for _ in range(20):
        ob = env.reset()
        done = False
        stochastic = False
        accrew = 0
        while not done:
            ac, vpred = pi.act(stochastic, ob)
            #ac = env.action_space.sample()
            ob, rew, done, info = env.step(ac)
            accrew += rew
        print(accrew)

    # def animate(env, agent):
    #     infos = defaultdict(list)
    #     ob = env.reset()
    #     done = False
    #     while not done:
    #         ob = agent.obfilt(ob)
    #         a, _info = agent.act(ob)
    #         (ob, rew, done, info) = env.step(a)
    #         infos['ob'].append(ob)
    #         infos['reward'].append(rew)
    #         infos['action'].append(a)
    #     env.render()
    #     return infos

if __name__ == "__main__":
    main()