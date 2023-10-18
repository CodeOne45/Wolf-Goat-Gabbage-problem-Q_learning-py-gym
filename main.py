from WolfGoatCabbageEnv import WolfGoatCabbageEnv
from Q_learning import *
from utils import *

def main():
    ############################## TESTS Q-LEARNING ##############################
    # Create environment
    env = WolfGoatCabbageEnv()

    

    # Train agent
    stats = q_learning(exploratory_policy, target_policy, action_values, episodes=10000, alpha=0.1, gamma=0.99)

    # Print average return
    print("Average return: ", np.mean(stats['Returns']))

    # Render environment with learned policy
    rendering(env, target_policy)

    # Plot performance to see Q-learning wins and losses
    # Analyze : Not a lot wins because of the exploration policy
    # but the target policy is learning to reach the goal with the wins
    plot_stats(stats)

    ################### TESTS ENVIRONNEMEN WITH RANDOM ACTIONS ####################
    env.reset()
    done = True # To start the loop set done to False
    while not done:
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        env.render()
        print("Reward: ", reward)
        print("Done: ", done)
        print("Next State: ", next_state)
        print("Action: ", action)
        print("")


if __name__ == "__main__":
    main()

