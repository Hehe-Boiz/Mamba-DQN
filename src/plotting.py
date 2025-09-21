import matplotlib
import matplotlib.pyplot as plt
import torch

is_ipython = 'inline' in matplotlib.get_backend()
try:
    from IPython import display
    is_ipython = True
except ImportError:
    is_ipython = False

def plot_durations(episode_rewards, episode_durations, show_result=False):
    """Vẽ biểu đồ thời lượng và phần thưởng của các episode."""
    plt.figure(1)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')

    plt.subplot(2, 1, 1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())

    plt.subplot(2, 1, 2)
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(rewards_t.numpy())

    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy(), 'r-', label='100-episode average')
        plt.legend()

    plt.tight_layout()
    plt.pause(0.001)
    if is_ipython and not show_result:
        display.display(plt.gcf())
        display.clear_output(wait=True)

    if show_result:
        plt.ioff()
        plt.show()


def plot_training_progress(evaluation_points, avg_rewards, avg_q_values, num_episodes, show_result=False):
    """
    Vẽ biểu đồ tiến trình huấn luyện với 2 ô:
    1. Phần thưởng trung bình qua các lần đánh giá.
    2. Giá trị Q tối đa trung bình qua các lần đánh giá.
    """
    if is_ipython:
        display.clear_output(wait=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle('Training Progress')

    ax1.cla()
    ax1.plot(evaluation_points, avg_rewards, 'b-')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Reward')
    ax1.grid(True)

    ax2.cla()
    ax2.plot(evaluation_points, avg_q_values, 'g-')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Max Q-Value')
    ax2.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig("./results/Training" + str(num_episodes) + ".png")

    if show_result:
        plt.ioff()
        plt.show()
    else:
        if is_ipython:
            display.display(plt.gcf())
            plt.close(fig)
        else:
            plt.pause(0.001)
            if not show_result:
                plt.close(fig)
