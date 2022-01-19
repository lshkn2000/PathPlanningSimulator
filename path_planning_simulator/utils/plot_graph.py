import matplotlib.pyplot as plt
import numpy as np
import time
from scipy import stats


def bootstrap(data, n_boot=10000, ci=68):
    boot_dist = []
    for i in range(int(n_boot)):
        resampler = np.random.randint(0, data.shape[0], data.shape[0])
        sample = data.take(resampler, axis=0)
        boot_dist.append(np.mean(sample, axis=0))
    b = np.array(boot_dist)
    s1 = np.apply_along_axis(stats.scoreatpercentile, 0, b, 50.-ci/2.)
    s2 = np.apply_along_axis(stats.scoreatpercentile, 0, b, 50.+ci/2.)
    return s1, s2


def tsplot(ax, data, **kwargs):
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    x = np.arange(data.shape[1])

    # 평균
    est = np.mean(data, axis=0)
    ax.plot(x, est, color='red', linestyle='--', **kwargs)

    # 분산
    cis = bootstrap(data)
    ax.fill_between(x, cis[0], cis[1], alpha=0.2, color='pink', **kwargs)

    ax.margins(x=0)


def plot_data(data, smooth=1, show=True, save=False):
    smoothed_xs = []
    if smooth > 1:
        y = np.ones(smooth)

        for dt in data:
            x = np.asarray(dt)
            z = np.ones(len(x))
            smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
            smoothed_xs.append(smoothed_x)
    else:
        smoothed_xs = data

    # if isinstance(data, list):
    #     data = pd.concat(data, ignore_index=True)

    fig, ax = plt.subplots()

    # 이동평균들에 대한 평균과 분산계산
    tsplot(ax, smoothed_xs)

    # 표 모양 수정
    plt.title("Moving Avg Reward", fontsize=25)
    plt.ylabel("Return", fontsize=15)
    plt.xlabel("Episode", fontsize=15)
    plt.legend(['mean', 'std'], title='Legend')
    plt.grid(True, alpha=0.3)


    # 스케일 수정
    xscale = np.max(np.asarray(data)) > 5e3
    if xscale:
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0)) # 틱 수정

    # 레이아웃 수정
    plt.tight_layout(pad=0.5)

    # 표 시각화와 저장
    if save:
        timestr = time.strftime("%Y%m%d_%H%M%S")
        plt.savefig('./learning_data/reward_graph/{}.png'.format(timestr + "_AvgRwd"), dpi=100, facecolor='#eeeeee', edgecolor='black')

    if show:
        plt.show()

# x = np.linspace(0, 15, 31)
# data = np.sin(x) + np.random.rand(10, 31) + np.random.randn(10, 1)
#
# plot_data(data)






