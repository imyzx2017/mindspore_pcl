import matplotlib.pyplot as plt
import numpy as np

def load_loss_file(loss_file):
    f = open(loss_file, 'r')
    data = f.readlines()
    f.close()
    loss_list = []
    epoch_list = []
    run_time_list = []
    for each_line in data:
        if "loss is " in each_line:
            loss = float(each_line.split('loss is ')[-1])
            loss_list.append(loss)
            epoch = int(each_line.split('epoch: ')[-1].split(' ')[0])
            epoch_list.append(epoch)

        elif "Epoch time: " in each_line:
            run_time = float(each_line.split('Epoch time: ')[-1].split(',')[0])
            run_time_list.append(run_time)

    return epoch_list, loss_list, run_time_list

def get_avg_loss(epoch_list, loss_list):
    max_epoch_num = np.max(epoch_list)
    epochs = np.arange(1, max_epoch_num)
    loss_all = []
    loss_avg = []
    for i in epochs:
        cache_loss = []
        cache_indexes = np.argwhere(epoch_list == i)
        for j in cache_indexes:
            cache_loss.append(loss_list[j[0]])
        loss_all.append(cache_loss)

    # calculate epoch avg loss
    for cache_ls in loss_all:
        # this_acg_loss = np.mean(cache_ls)
        this_acg_loss = cache_ls[0]
        loss_avg.append(this_acg_loss)


    return epochs, loss_avg

def plot_avg_loss(epoch_l, loss_list, run_time_list):
    avg_run_time = np.mean(run_time_list) / 1000.0
    plt.plot(epoch_l, loss_list, label='avg_losses')
    # plt.xticks(epoch_l)
    plt.xlabel('epochs')
    plt.ylabel('losses')
    plt.title('GPT-3 350M Model trained on 4 V100 GPUs, speed: {:.1f}s/epoch'.format(avg_run_time))
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # loss_file = 'D:/yunnao/Jobs/results/GPT3-mindspore/GPT3-miniopenweb-GPU8-Batch2-1117.log'
    # loss_file = 'D:/yunnao/Jobs/results/GPT3-mindspore/GPT3-mindspore-GPU4-Batch2-1118.log'
    # loss_file = 'D:/yunnao/Jobs/results/GPT3-mindspore/GPT3-mindspore-GPU4-Batch2-epochs10-1118.log'

    # loss_file = 'D:/yunnao/Jobs/results/GPT3-mindspore/GPT3-mindspore-GPU1-Batch2-1118.log'
    loss_file = 'D:/yunnao/Jobs/results/GPT3-mindspore/GPT3-mindspore-GPU4-Batch2-epoch_to_35.log'
    epoch_list, loss_list, run_time_list = load_loss_file(loss_file)
    epochs, loss_avg = get_avg_loss(epoch_list, loss_list)
    plot_avg_loss(epochs, loss_avg, run_time_list)