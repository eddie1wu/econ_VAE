import numpy as np
import matplotlib.pyplot as plt

def plot_loss(loss, recon_loss, KLdiv, beta = 1):
    """
    Plot total loss, reconstruction loss, and KL divergence.
    """
    loss = [x.cpu().detach().numpy() for x in loss]
    recon_loss = [x.cpu().detach().numpy() for x in recon_loss]
    KLdiv = [x.cpu().detach().numpy() for x in KLdiv]

    iterations = list(range(len(loss)))

    plt.plot(iterations, loss, label = "total_loss")
    plt.plot(iterations, recon_loss, label = "recon_loss")
    # plt.axhline(y=0, alpha = 0.4)
    plt.plot(iterations, KLdiv, label = "KL_divergence")

    plt.yscale('log')
    plt.title(f"train, beta = {beta}")
    plt.legend(loc='best')

    plt.show()


def plot_histograms(real_data, gen_data, start, end, step, var_name, bar_width = 0.4):
    """
    Plot the marginal densities of real data vs generated data.
    """
    bins = np.arange(start, end, step)

    real_hist, _ = np.histogram(real_data, bins = bins)
    gen_hist, _ = np.histogram(gen_data, bins = bins)

    # Set bar width and positions for side-by-side comparison
    positions_real = bins[:-1] - bar_width / 2
    positions_gen = bins[:-1] + bar_width / 2

    # Plot
    plt.bar(positions_real, real_hist, width = bar_width, color='blue', label='real')
    plt.bar(positions_gen, gen_hist, width = bar_width, color='red', label='generated')

    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.xticks(bins[:-1])  # Set xticks to match the bins
    plt.title(f'Comparisons for {var_name}')

    plt.legend()
    plt.show()


def plot_all_variables(real_data, gen_data):
    """
    Plot all variables in the LaLonde dataset.
    """
    plot_histograms(real_data['re78'], gen_data['re78'], -1, 9, 0.5, 're78', bar_width = 0.2)
    plot_histograms(real_data['education'], gen_data['education'], -5, 5, 1, 'education', bar_width = 0.4)
    plot_histograms(real_data['age'], gen_data['age'], -1, 9, 0.5, 'age', bar_width = 0.2)
    plot_histograms(real_data['black'], gen_data['black'].round(), 0, 3, 1, 'black', bar_width = 0.2)
    plot_histograms(real_data['hispanic'], gen_data['hispanic'].round(), 0, 3, 1, 'black', bar_width = 0.2)
    plot_histograms(real_data['married'], gen_data['married'].round(), 0, 3, 1, 'black', bar_width = 0.2)
    plot_histograms(real_data['nodegree'], gen_data['nodegree'].round(), 0, 3, 1, 'black', bar_width = 0.2)
