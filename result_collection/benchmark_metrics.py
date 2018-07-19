import numpy as np
import scipy as sp
import scipy.stats

COMPARISONS = {"TFT_IPD": [[1, 1, 0, 1, 0], [1, 1, 1, 0, 0]],
               "NASH_IMP": [[0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5]],
               "ISH": [[1, 1, 0, 1, 0], [1, 1, 1, 0, 0]],
               "ISD": [[1, 1, 0, 1, 0], [1, 1, 1, 0, 0]]}

TOLERANCE = {"TFT_IPD": 0.5,
             "NASH_IMP": 0.25,
             "ISH": 0.5,
             "ISD": 0.5,
             "COOP": 0.5}

MAX_DIFF = {"TFT_IPD": 1.0,
             "NASH_IMP": 0.5}


def tft(p1, p2):
    comparison_p = np.array(COMPARISONS["TFT_IPD"]).transpose()
    ps = np.array([p1, p2]).transpose()
    comparison1 = comparison2 = np.mean(np.max(np.abs(ps - comparison_p), 1) < 0.5)
    return [comparison1*100, comparison2*100]


def tft2(p1, p2):
    comparison_p = np.array(COMPARISONS["TFT_IPD"])
    comparison1 = np.mean(np.ones((5)) - np.abs(p1 - comparison_p[0]))
    comparison2 = np.mean(np.ones((5)) - np.abs(p2 - comparison_p[1]))
    return [comparison1*100, comparison2*100]


def nash(p1, p2):
    comparison_p = np.array(COMPARISONS["NASH_IMP"]).transpose()
    ps = np.array([p1, p2]).transpose()
    comparison1 = comparison2 = np.mean(np.max(np.abs(ps - comparison_p), 1) < 0.25)
    return [comparison1*100, comparison2*100]


def nash2(p1, p2):
    comparison_p = np.array(COMPARISONS["NASH_IMP"])
    comparison1 = np.mean(np.ones(5) * 0.5 - np.abs(p1 - comparison_p[0]))
    comparison2 = np.mean(np.ones(5) * 0.5 - np.abs(p2 - comparison_p[1]))
    return [comparison1*100, comparison2*100]


def exp_s(p1, p2, n=100, state=0):
    P = np.vstack((np.zeros(5), (p1 * p2), (p1 * (1 - p2)), ((1 - p1) * p2), ((1 - p1) * (1 - p2)))).transpose()
    PP = P[0, :]
    total_PP = np.zeros(np.shape(PP))
    for i in range(n):
        PP = np.matmul(PP, P[:, :])
        total_PP += PP
    if state == 5:
        return [total_PP[2] + total_PP[3]] * 2
    return [total_PP[state], total_PP[state]]


def prob_s(p1, p2, n=10, state=0):
    P = np.vstack((np.zeros(5), (p1 * p2), (p1 * (1 - p2)), ((1 - p1) * p2), ((1 - p1) * (1 - p2)))).transpose()
    PP = P[0, :]
    for i in range(n):
        PP = np.matmul(PP, P[:, :])
    return [PP[state], PP[state]]


def coop_s(p1, p2):
    comparison_p = np.array([[1, 1], [1, 1]])
    comparison1 = np.mean(np.ones(2) * 1 - np.abs(p1[:2] - comparison_p[0]))
    comparison2 = np.mean(np.ones(2) * 1 - np.abs(p2[:2] - comparison_p[1]))
    return [comparison1 * 100, comparison2 * 100]


def conv_1p(p_epochs, x, comparison_p, max_dist, n=5):
    above = np.where(np.ones(np.shape(p_epochs)) * max_dist - np.abs(p_epochs - comparison_p) > x, [1], [0])

    ones = np.mean(above, 1)
    length = ones.shape[0]
    for t in range(length):
        acc = 0
        for l in range(n):
            if l + t >= length:
                return length
            elif ones[t + l] == 1:
                acc += 1
        if acc == n:
            return t + int(n/2)
    return len(above)


# convergence to cooperative strategy
def conv_2p(p1_epochs, p2_epochs, x, game, window=5):
    all_metrics = []
    N = np.shape(p1_epochs)[0]
    t1 = t2 = N

    for n in range(N):
        p1 = p1_epochs[n]
        p2 = p2_epochs[n]

        if game == "IMP":
            cs = nash2(p1, p2)
        if game == "IPD":
            # cs = prob_s(p1, p2, state=1)
            cs = coop_s(p1, p2)

        all_metrics.append(cs)

    for n in range(int(window/2), N-int(window/2)):
        acc1 = acc2 = 0
        for l in range(-int(window/2), int(window/2)+1):
            c1, c2 = all_metrics[l+n]
            acc1 += c1
            acc2 += c2
        acc1 /= window * 100
        acc2 /= window * 100

        if n >= t1 and n >= t2:
            return [t1, t2]
        else:
            if n < t1 and acc1 >= x:
                t1 = n
            if n < t2 and acc2 >= x:
                t2 = n

    return [t1, t2]


# Average reward per time step
def R(p1, p2, gamma, r1, r2):
    x1 = p1
    x2 = p2
    P = np.stack((x1 * x2, x1 * (1 - x2), (1 - x1) * x2, (1 - x1) * (1 - x2))).T

    I = np.eye(4)
    Zinv1 = np.linalg.inv(I - gamma * P[1:, :])
    Zinv2 = np.linalg.inv(I - gamma * P[1:, :])

    V1 = np.matmul(np.matmul(P[0, :], Zinv1), r1)
    V2 = np.matmul(np.matmul(P[0, :], Zinv2), r2)
    return [V1*(1-gamma), V2*(1-gamma)]


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    se = scipy.stats.sem(a, axis=0)
    h = se * sp.stats.t.ppf((1 + confidence) / 2., n - 1)
    return h


def get_av_metrics_epochs(p1_epochs, p2_epochs, gamma, r1, r2, game, x=0.95):
    all_metrics = []
    N = np.shape(p1_epochs)[0]

    for n in range(N):
        p1 = p1_epochs[n]
        p2 = p2_epochs[n]
        all_metrics.append([tft(p1,p2), nash(p1, p2), tft2(p1,p2), nash2(p1, p2)])

    all_metrics = np.array(all_metrics)
    av_metrics = np.concatenate(
        (np.mean(all_metrics, 1),
         conv_2p(p1_epochs, p2_epochs, x, game)))
    return av_metrics


def get_av_metrics_over_repeates_for_table(all_repeat_policies, gamma, r1, r2, game, x=0.95, ith_epoch=-1, join_agents=True):
    all_metrics = []

    for epoch in all_repeat_policies:
        p1_epochs = epoch[:, 0]
        p2_epochs = epoch[:, 1]
        metrics_for_one = get_metrics_for_single_policy_pair(p1_epochs[ith_epoch], p2_epochs[ith_epoch], gamma, r1, r2)
        all_metrics.append(np.vstack((metrics_for_one, conv_2p(p1_epochs, p2_epochs, x, game))))

    if join_agents:
        av_over_player = np.mean(all_metrics, 2)
        std_over_repeats = np.std(av_over_player, 0)
        av_over_repeats = np.mean(av_over_player, 0)
        # np.transpose([np.mean(np.mean(all_metrics, 2), 0), np.std(np.mean(al/l_metrics, 2), 0)])
    else:
        std_over_repeats = np.std(all_metrics, 0)
        av_over_repeats = np.mean(all_metrics, 0)
        # np.transpose([np.mean(all_metrics, 0), np.std(all_metrics, 0)])

    return np.transpose([av_over_repeats, std_over_repeats])


def get_metrics_for_policy_arrays(p1_array, p2_array, gamma, r1, r2):
    all_metrics = []
    N = np.shape(p1_array)[0]

    for n in range(N):
        p1 = p1_array[n]
        p2 = p2_array[n]
        all_metrics.append(get_metrics_for_single_policy_pair(p1, p2, gamma, r1, r2))

    return np.array(all_metrics)


# Generic
def get_av_metrics_for_epoch_policy_arrays(p1_array_epochs, p2_array_epochs,
                                    metric_fn=lambda x, y: (x, y),
                                    conf_interval=None,
                                    std=False,
                                    join_policies=False):
    epochs = p1_array_epochs.shape[1]
    results = []
    for e in range(epochs):
        results_tuple = get_av_metrics_for_policy_arrays(p1_array_epochs[:, e], p2_array_epochs[:, e],
                                                        metric_fn, conf_interval, std, join_policies)
        results.append(results_tuple)
    return np.array(results)


# Generic
def get_av_metrics_for_policy_arrays(p1_array, p2_array,
                                    metric_fn=lambda x, y: (x, y),
                                    conf_interval=None,
                                    std=False,
                                    join_policies=False):

    p1_metrics = np.zeros(p1_array.shape[0])
    p2_metrics = np.zeros(p2_array.shape[0])
    for n in range(p1_metrics.shape[0]):
        p1 = p1_array[n]
        p2 = p2_array[n]
        [v1, v2] = metric_fn(p1, p2)
        p1_metrics[n] = v1
        p2_metrics[n] = v2

    if join_policies:
        all_metrics = np.hstack((p1_metrics, p2_metrics))
        mean1 = np.mean(all_metrics)
        result = (mean1,)
        if std:
            std1 = np.std(all_metrics)
            result += (std1,)
        if conf_interval is not None:
            h1 = mean_confidence_interval(all_metrics, confidence=conf_interval)
            result += (h1,)
    else:
        mean1 = np.mean(p1_metrics)
        mean2 = np.mean(p2_metrics)
        result = (mean1, mean2)
        if std:
            std1 = np.std(p1_metrics)
            std2 = np.std(p1_metrics)
            result += (std1, std2)
        if conf_interval is not None:
            h1 = mean_confidence_interval(p1_metrics, confidence=conf_interval)
            h2 = mean_confidence_interval(p2_metrics, confidence=conf_interval)
            result += (h1, h2)

    return result


SINGLE_POLICY_METRIC_ORDER = ["TFT", "TFT2", "Nash", "Nash2", "R"]
TABLE_METRIC_ORDER = SINGLE_POLICY_METRIC_ORDER.append("Conv")


def get_metrics_for_single_policy_pair(p1, p2, gamma, r1, r2):
    return [tft(p1, p2), tft2(p1, p2), nash(p1, p2), nash2(p1, p2), R(p1, p2, gamma, r1, r2)]


def get_metrics_for_epoch_policies(p1_epochs, p2_epochs, gamma, r1, r2, game, x=0.95):
    all_metrics = get_metrics_for_policy_arrays(p1_epochs, p2_epochs, gamma, r1, r2)
    av_metrics = np.concatenate(
        (np.mean(all_metrics, 1),
         conv_2p(p1_epochs, p2_epochs, x, game)))
    return av_metrics