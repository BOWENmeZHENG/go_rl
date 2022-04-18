from scipy.stats import bernoulli


def pick_net(m, n, h_net, e_net, h_net_optimizer, e_net_optimizer,
             scheduled_lr=None, h_net_scheduler=None, e_net_scheduler=None):
    """
    Pick H_Net or E_Net using a Bernoulli distribution
    :param m: number of hydroxyl groups left
    :param n: number of epoxide groups left
    :param h_net: current H_Net
    :param e_net: current E_Net
    :param h_net_optimizer: current H_Net optimizer
    :param e_net_optimizer: current E_Net optimizer
    :param scheduled_lr: Bool, whether a learning rate scheduler is used
    :param h_net_scheduler: H_Net scheduler
    :param e_net_scheduler: E_Net scheduler
    :return: net_index, m, n, func_net, optimizer, scheduler
    """
    scheduler = None
    net_index = bernoulli(m / (m + n)).rvs()
    if net_index == 1:
        func_net = h_net
        optimizer = h_net_optimizer
        if scheduled_lr:
            scheduler = h_net_scheduler
        m -= 1
    else:
        func_net = e_net
        optimizer = e_net_optimizer
        if scheduled_lr:
            scheduler = e_net_scheduler
        n -= 1
    return net_index, m, n, func_net, optimizer, scheduler


def calc_loss(list_of_action_and_action_dist, reward):
    """
    calculate loss/return
    """
    sum_log_prob = 0
    for action_and_action_dist in list_of_action_and_action_dist:
        sum_log_prob += action_and_action_dist[1].log_prob(action_and_action_dist[0])
    return -sum_log_prob * reward
