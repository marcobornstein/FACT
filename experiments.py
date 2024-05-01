import numpy as np
import matplotlib.pyplot as plt
import os


class Postprocessing:
    def __init__(self):
        self.colors = ['r', 'b', 'g', 'orange', 'pink', 'cyan', 'yellow', 'purple']

    def run_benefit_plot(self, data_paths, save_file=None, runs=3):

        plt.figure(figsize=(8, 6))

        for j, data_path in enumerate(data_paths):

            # ensure data files exist
            test_file = data_path + '/r0-fed-epoch-loss.log'
            if not os.path.isfile(test_file):
                raise Exception(f"Incorrect Path Provided")

            # determine which dataset and which truthfulness method
            method = 'Random Mechanism' if data_path.lower().find('random') > -1 else 'Deterministic Mechanism'
            dataset = 'MNIST' if data_path.lower().find('mnist') > -1 else 'Cifar10'

            # extract all runs
            split_paths = data_path.split("-run")
            begin_path = split_paths[0]
            end_path = split_paths[1][1:]

            # extract benefit data
            manb = []
            losses = []
            for run in range(1, runs+1):
                file = begin_path + '-run' + str(run) + end_path
                net_losses, mean_agent_net_benefit, num_agents = self.get_benefit_data(file)
                losses.append(net_losses)
                manb.append(mean_agent_net_benefit)

            manb = np.stack(manb, axis=0)
            manb = np.mean(manb, axis=(0, 1))

            epsilons = np.linspace(-0.2, 0.2, len(manb), endpoint=True) * 100

            plt.plot(epsilons, -manb, self.colors[j], label=method)

        plt.xlabel('Percent (%) Added/Subtracted from True Cost $c_i$')
        plt.ylabel('Net Improvement in Loss')
        plt.title('Average Agent Gain From FACT Participation Under Cost Manipulation')
        plt.legend(loc='best')
        plt.xlim([-20, 20])
        plt.grid(alpha=0.25)
        if save_file is None:
            plt.show()
        else:
            save_file = save_file + '-' + str(num_agents) +'agents-' + dataset.lower() + '.jpg'
            plt.savefig(save_file, dpi=200)

        return manb, losses

    def get_epoch_data(self, data_path, datatype='fed-epoch-loss.log'):

        # determine number of agents
        string, location = [(i, i.find("devices")) for i in data_path.split("-") if i.find("devices") > 0][0]
        num_agents = int(string[:location])

        # determine the number of epochs
        with open(data_path + '/r0-fed-epoch-loss.log') as f:
            epochs = sum(1 for _ in f)

        return self.unpack_data(data_path, epochs, num_agents, datatype)

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

    cifar10_random_path = 'output/CIFAR10/fact-random-sandwich-uniform-cost-run1-cifar10-16devices'
    cifar10_deterministic_path = 'output/CIFAR10/fact-deterministic-sandwich-uniform-cost-run1-cifar10-16devices'
    mnist_random_path = 'output/MNIST/fact-random-sandwich-uniform-cost-run1-mnist-16devices'
    mnist_deterministic_path = 'output/MNIST/fact-deterministic-sandwich-uniform-cost-run1-mnist-16devices'

    # initialize postprocessing
    pp = Postprocessing()

    # plot results
    mnist_paths = [mnist_random_path, mnist_deterministic_path]
    cifar_paths = [cifar10_random_path, cifar10_deterministic_path]
    manb, losses = pp.run_benefit_plot(mnist_paths, save_file='sandwich')
