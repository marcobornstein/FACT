import numpy as np
import matplotlib.pyplot as plt
import os
from utils.truthfulness import truthfulness_mechanism
from tqdm import tqdm


class Postprocessing:
    def __init__(self):
        self.colors = ['r', 'b', 'g', 'orange', 'pink', 'cyan', 'yellow', 'purple']

    def run_loss_histogram(self, data_path, save_file=None, loss=True, runs=3, h=121):

        # loss or accuracy
        dt = 'loss.log' if loss else 'acc-top1.log'

        # check file existence
        _, dataset, begin_path, end_path = self.path_check(data_path)
        num_data = 3125 if dataset == 'cfiar10' else 3750

        y_mean_local, _, _, y_mean_fed, _, _, epochs, num_agents = self.get_loss_data(begin_path, end_path, runs, dt)
        avg_local_loss = y_mean_local[-1]
        avg_fed_loss = y_mean_fed[-1]
        net_loss = avg_local_loss - avg_fed_loss

        # get average benefit from participating in FACT
        agent_net = np.empty((runs, num_agents))
        other_agent_net = np.empty((runs, num_agents))
        for run in range(1, runs + 1):
            file = begin_path + '-run' + str(run) + end_path
            fact_benefit = self.unpack_data(file, 3, num_agents, datatype='benefits.log')
            agent_net[run - 1, :] = fact_benefit[0, :]
            other_agent_net[run - 1, :] = fact_benefit[1, :]

        avg_agent_net = np.mean(agent_net, axis=0)
        avg_other_agent_net = np.mean(other_agent_net, axis=0)

        mcs = self.unpack_data(file, 1, num_agents, datatype='costs.log').flatten()
        mc = mcs[0]

        fbr = np.empty((num_agents, h))
        fbd = np.empty((num_agents, h))
        fl = np.empty(num_agents)
        for i in tqdm(range(num_agents)):
            fl[i], fbr[i, :], fbd[i, :] = truthfulness_mechanism(mc, num_data, avg_agent_net[i], avg_other_agent_net[i],
                                                                 num_agents, h=h, sandwich=True, normal=True,
                                                                 agents=2000, rounds=100000)

        avg_fbr = np.mean(fbr, axis=0)
        avg_fbd = np.mean(fbd, axis=0)

        fact_loss = np.mean(fl)

        # initialize eps
        epsilons = np.linspace(-0.3, 0.3, h, endpoint=True) * 100

        print(epsilons[np.argmin(avg_fbd)])
        print(epsilons[np.argmin(avg_fbr)])

        # plot truthfulness
        plt.figure(figsize=(8, 6))
        plt.plot(epsilons, net_loss - avg_fbr, self.colors[0], label='Random Mechanism')
        plt.plot(epsilons, net_loss - avg_fbd, self.colors[1], label='Deterministic Mechanism')
        plt.xlabel('Percent (%) Added/Subtracted from True Cost $c_i$')
        plt.ylabel('Net Improvement in Loss')
        # plt.title('Average Agent Gain From FACT Participation Under Cost Manipulation')
        plt.legend(loc='best')
        plt.xlim([-30, 30])
        plt.grid(alpha=0.25)

        # save figure
        if save_file is None:
            plt.show()
        else:
            sf = save_file + '-truthfulness-' + str(num_agents) + 'agents-' + dataset.lower() + '.jpg'
            plt.savefig(sf, dpi=200)

        # add for loop here over all the avg_fact_losses
        # plot results
        plt.figure(figsize=(8, 6))
        plt.bar(['Local Training', 'FACT Training', 'Traditional FL Training'],
                [avg_local_loss, fact_loss+avg_fed_loss, avg_fed_loss],
                color=['tab:red', 'tab:blue', 'tab:green'])
        plt.ylabel('Loss')
        plt.grid(alpha=0.25, axis='y')

        # save figure
        if save_file is None:
            plt.show()
        else:
            sf = save_file + '-loss-histogram-' + str(num_agents) + 'agents-' + dataset.lower() + '.jpg'
            plt.savefig(sf, dpi=200)

    def run_loss_plot(self, data_path, save_file=None, loss=True, runs=3):

        # loss or accuracy
        dt = 'loss.log' if loss else 'acc-top1.log'

        # check file existence
        _, dataset, begin_path, end_path = self.path_check(data_path)

        y_mean_local, y_min_local, y_max_local, y_mean_fed, y_min_fed, y_max_fed, epochs, num_agents = (
            self.get_loss_data(begin_path, end_path, runs, dt))

        # plot results
        plt.figure(figsize=(8, 6))

        # local
        plt.plot(range(epochs), y_mean_local, color='r', label='Local Training')
        plt.fill_between(range(epochs), y_min_local, y_max_local, alpha=0.2, color='r')

        # fed
        plt.plot(range(epochs), y_mean_fed, color='b', label='Federated Training')
        plt.fill_between(range(epochs), y_min_fed, y_max_fed, alpha=0.2, color='b')

        plt.xlabel('Epochs')
        plt.ylabel('Test Loss')
        plt.legend(loc='best')
        if dataset.lower() == 'mnist':
            plt.ylim([0.01, 2.5])
            plt.yscale("log")
        else:
            plt.ylim([0, 1.25 * 10**2])
            plt.yscale("symlog")
        plt.grid(alpha=0.25)

        # save figure
        if save_file is None:
            plt.show()
        else:
            save_file = save_file + '-' + str(num_agents) +'agents-' + dataset.lower() + '.jpg'
            plt.savefig(save_file, dpi=200)

    def get_epoch_data(self, data_path, datatype='fed-epoch-loss.log'):

        # determine number of agents
        string, location = [(i, i.find("devices")) for i in data_path.split("-") if i.find("devices") > 0][0]
        num_agents = int(string[:location])

        # determine the number of epochs
        with open(data_path + '/r0-fed-epoch-loss.log') as f:
            epochs = sum(1 for _ in f)

        return self.unpack_data(data_path, epochs, num_agents, datatype), epochs, num_agents

    def get_benefit_data(self, data_path):

        # determine number of agents
        string, location = [(i, i.find("devices")) for i in data_path.split("-") if i.find("devices") > 0][0]
        num_agents = int(string[:location])

        # load in benefit data
        d = np.load(data_path + '/r0-expected-epsilon-benefit.npy')
        benefit_data = np.empty((num_agents, len(d)))
        benefit_data[0, :] = d
        for r in range(1, num_agents):
            benefit_data[r, :] = np.load(data_path + '/r' + str(r) + '-expected-epsilon-benefit.npy')

        return self.unpack_data(data_path, 3, num_agents, datatype='benefits.log'), benefit_data, num_agents

    def get_loss_data(self, begin_path, end_path, runs, data_type):

        # extract benefit data
        losses_fed = []
        losses_local = []
        for run in range(1, runs + 1):
            file = begin_path + '-run' + str(run) + end_path
            loss_data_local, _, _ = self.get_epoch_data(file, datatype='local-epoch-' + data_type)
            loss_data_fed, epochs, num_agents = self.get_epoch_data(file, datatype='fed-epoch-' + data_type)
            losses_fed.append(loss_data_fed[:, 0])
            losses_local.append(np.mean(loss_data_local, axis=1))

        losses_fed = np.stack(losses_fed, axis=0)
        losses_local = np.stack(losses_local, axis=0)

        # compute error bars over all three runs
        y_mean_local, y_min_local, y_max_local = self.generate_confidence_interval(losses_local)
        y_mean_fed, y_min_fed, y_max_fed = self.generate_confidence_interval(losses_fed)

        return y_mean_local, y_min_local, y_max_local, y_mean_fed, y_min_fed, y_max_fed, epochs, num_agents

    def generate_confidence_interval(self, ys, number_per_g=30, number_of_g=1000, low_percentile=1, high_percentile=99):
        means = []
        mins = []
        maxs = []
        for i, y in enumerate(ys.T):
            y = self.bootstrapping(y, number_per_g, number_of_g)
            means.append(np.mean(y))
            mins.append(np.percentile(y, low_percentile))
            maxs.append(np.percentile(y, high_percentile))
        return np.array(means), np.array(mins), np.array(maxs)

    def plot_ci(self, x, y, num_runs, num_dots, mylegend, ls='-', lw=3, transparency=0.2):
        assert(x.ndim == 1)
        assert(x.size == num_dots)
        assert(y.ndim == 2)
        assert(y.shape == (num_runs, num_dots))
        y_mean, y_min, y_max = self.generate_confidence_interval(y)
        plt.plot(x, y_mean, 'o-', label=mylegend, linestyle=ls, linewidth=lw)  # , label=r'$\alpha$={}'.format(alpha))
        plt.fill_between(x, y_min, y_max, alpha=transparency)
        return

    @staticmethod
    def path_check(data_path):

        # ensure data files exist
        test_file = data_path + '/r0-fed-epoch-loss.log'
        if not os.path.isfile(test_file):
            print(test_file)
            raise Exception(f"Incorrect Path Provided")

        # determine which dataset and which truthfulness method
        method = 'Random Mechanism' if data_path.lower().find('random') > -1 else 'Deterministic Mechanism'
        dataset = 'MNIST' if data_path.lower().find('mnist') > -1 else 'Cifar10'

        # extract all runs
        split_paths = data_path.split("-run")
        begin_path = split_paths[0]
        end_path = split_paths[1][1:]

        return method, dataset, begin_path, end_path

    @staticmethod
    def unpack_data(directory_path, epochs, num_workers, datatype):
        directory = os.path.join(directory_path)
        if not os.path.isdir(directory):
            raise Exception(f"custom no directory {directory}")
        data = np.zeros((epochs, num_workers))
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(datatype):
                    j = int(file.split('-')[0][1:])
                    with open(directory_path + '/' + file, 'r') as f:
                        i = 0
                        for line in f:
                            itms = line.strip().split('\n')[0]
                            data[i, j] = float(itms)
                            i += 1
        return data

    @staticmethod
    def bootstrapping(data, num_per_group, num_of_group):
        new_data = np.array([np.mean(np.random.choice(data, num_per_group, replace=True)) for _ in range(num_of_group)])
        return new_data


if __name__ == '__main__':

    # iid
    cifar10_random_path_iid = 'output/CIFAR10/fact-random-sandwich-uniform-cost-run1-cifar10-16devices'
    cifar10_deterministic_path_iid = 'output/CIFAR10/fact-deterministic-sandwich-uniform-cost-run1-cifar10-16devices'
    mnist_random_path_iid = 'output/MNIST/fact-random-sandwich-uniform-cost-run1-mnist-16devices'
    mnist_deterministic_path_iid = 'output/MNIST/fact-deterministic-sandwich-uniform-cost-run1-mnist-16devices'

    # noniid D-0.3
    cifar10_random_path_noniid3 = 'output/CIFAR10/fact-random-sandwich-uniform-cost-noniid-0.3-run1-cifar10-16devices'
    cifar10_deterministic_path_noniid3 = 'output/CIFAR10/fact-deterministic-sandwich-uniform-cost-noniid-0.3-run1-cifar10-16devices'
    mnist_random_path_noniid3 = 'output/MNIST/fact-random-sandwich-uniform-cost-noniid-0.3-run1-mnist-16devices'
    mnist_deterministic_path_noniid3 = 'output/MNIST/fact-deterministic-sandwich-uniform-cost-noniid-0.3-run1-mnist-16devices'

    # noniid D-0.6
    cifar10_random_path_noniid6 = 'output/CIFAR10/fact-random-sandwich-uniform-cost-noniid-0.6-run1-cifar10-16devices'
    cifar10_deterministic_path_noniid6 = 'output/CIFAR10/fact-deterministic-sandwich-uniform-cost-noniid-0.6-run1-cifar10-16devices'
    mnist_random_path_noniid6 = 'output/MNIST/fact-random-sandwich-uniform-cost-noniid-0.6-run1-mnist-16devices'
    mnist_deterministic_path_noniid6 = 'output/MNIST/fact-deterministic-sandwich-uniform-cost-noniid-0.6-run1-mnist-16devices'

    # multiple paths
    mnist_paths_iid = [mnist_random_path_iid, mnist_deterministic_path_iid]
    cifar_paths_iid = [cifar10_random_path_iid, cifar10_deterministic_path_iid]
    mnist_paths_noniid3 = [mnist_random_path_noniid3, mnist_deterministic_path_noniid3]
    cifar_paths_noniid3 = [cifar10_random_path_noniid3, cifar10_deterministic_path_noniid3]
    mnist_paths_noniid6 = [mnist_random_path_noniid6, mnist_deterministic_path_noniid6]
    cifar_paths_noniid6 = [cifar10_random_path_noniid6, cifar10_deterministic_path_noniid6]

    # initialize postprocessing
    pp = Postprocessing()

    # loss plots
    # pp.run_loss_plot(cifar10_random_path_noniid6, save_file='noniid6-2')

    # loss histogram and truthfulness plots
    # pp.run_loss_histogram(mnist_random_path_iid, save_file='iid')
    # pp.run_loss_histogram(mnist_random_path_noniid6, save_file='noniid6')
    # pp.run_loss_histogram(mnist_random_path_noniid3, save_file='noniid3')

    # penalty for using sub-optimal data contributions

    # initialize lambda -- add into the loss computations
    # compute how penalty increases with a very large lambda


