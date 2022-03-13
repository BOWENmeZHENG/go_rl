from scipy.stats import bernoulli


def pick_net(m, n, h_net, e_net, h_net_optimizer, e_net_optimizer,
             scheduled_lr=None, h_net_scheduler=None, e_net_scheduler=None):
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
    sum_log_prob = 0
    for action_and_action_dist in list_of_action_and_action_dist:
        sum_log_prob += action_and_action_dist[1].log_prob(action_and_action_dist[0])
    return -sum_log_prob * reward
