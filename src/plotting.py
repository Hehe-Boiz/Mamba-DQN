import matplotlib
import matplotlib.pyplot as plt
import torch

# Thiết lập matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

def plot_durations(episode_rewards, episode_durations, show_result=False):
    """Vẽ biểu đồ thời lượng và phần thưởng của các episode."""
    plt.figure(1)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')

    # Biểu đồ thời lượng
    plt.subplot(2, 1, 1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())

    # Biểu đồ phần thưởng
    plt.subplot(2, 1, 2)
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(rewards_t.numpy())

    # Vẽ đường trung bình 100 episode gần nhất
    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy(), 'r-', label='100-episode average')
        plt.legend()

    plt.tight_layout()
    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython and not show_result:
        display.display(plt.gcf())
        display.clear_output(wait=True)

    if show_result:
        plt.ioff()
        plt.show()