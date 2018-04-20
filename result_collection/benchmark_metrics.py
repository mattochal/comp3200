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
             "ISD": 0.5}

MAX_DIFF = {"TFT_IPD": 1.0,
             "NASH_IMP": 0.5}


def tft(p1, p2):
    comparison_p = np.array(COMPARISONS["TFT_IPD"]).transpose()
    ps = np.array([p1, p2]).transpose()
    comparison1 = comparison2 = np.mean(np.max(np.abs(ps - comparison_p), 1) < 0.5)
    return [comparison1, comparison2]


def tft2(p1, p2):
    comparison_p = np.array(COMPARISONS["TFT_IPD"])
    comparison1 = np.mean(np.ones((5)) - np.abs(p1 - comparison_p[0]))
    comparison2 = np.mean(np.ones((5)) - np.abs(p2 - comparison_p[0]))
    return [comparison1, comparison2]


def nash(p1, p2):
    comparison_p = np.array(COMPARISONS["NASH_IMP"]).transpose()
    ps = np.array([p1, p2]).transpose()
    comparison1 = comparison2 = np.mean(np.max(np.abs(ps - comparison_p), 1) < 0.25)
    return [comparison1, comparison2]


def nash2(p1, p2):
    comparison_p = np.array(COMPARISONS["NASH_IMP"])
    comparison1 = np.mean(np.ones(5) * 0.5 - np.abs(p1 - comparison_p[0]))
    comparison2 = np.mean(np.ones(5) * 0.5 - np.abs(p2 - comparison_p[0]))
    return [comparison1, comparison2]


def exp_s(p1, p2, n=100, state=0):
    P = np.vstack((np.zeros(5), (p1 * p2), (p1 * (1 - p2)), ((1 - p1) * p2), ((1 - p1) * (1 - p2)))).transpose()
    PP = P[0, :]
    total_PP = np.zeros(np.shape(PP))
    for i in range(n):
        PP = np.matmul(PP, P[:, :])
        total_PP += PP
    return [total_PP[state], total_PP[state]]


def conv_1p(p_epochs, x, comparison_p, max_dist):
    above = np.where(np.ones(np.shape(p_epochs)) * max_dist - np.abs(p_epochs - comparison_p) > x, [1], [0])
    for t, a in enumerate(np.mean(above, 1)):
        if a == 1:
            return t
    return len(above)


def conv_2p(p1_epochs, p2_epochs, x, game):
    comparison_p = []
    max_diff = 0
    if game == "IMP":
        comparison_p = COMPARISONS["NASH_IMP"]
        max_diff = MAX_DIFF["NASH_IMP"]

    elif game == "IPD":
        comparison_p = COMPARISONS["TFT_IPD"]
        max_diff = MAX_DIFF["TFT_IPD"]

    conv1 = conv_1p(p1_epochs, x, comparison_p[0], max_diff)
    conv2 = conv_1p(p2_epochs, x, comparison_p[1], max_diff)

    return [conv1, conv2]


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
    for e in epochs:
        results_tuple = get_av_metrics_for_policy_arrays(p1_array_epochs[:, e], p2_array_epochs[:, e],
                                                        metric_fn, conf_interval, std, join_policies)
        results.append(results_tuple)
    return results


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