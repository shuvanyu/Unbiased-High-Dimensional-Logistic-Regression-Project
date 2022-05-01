import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.lines as mlines
import numpy as np

def plot_p_vals(p_vals_baseline, p_vals_sloe, title, filename):
    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    
    ax[0].hist(p_vals_baseline, bins=20, density=True)
    ax[0].axhline(y=1.0, color='black', linestyle='--')
    ax[0].set_xlabel('p-values')
    ax[0].set_ylabel('Frequency')
    ax[0].set_title('Standard')

    ax[1].hist(p_vals_sloe, bins=20, density=True)
    ax[1].axhline(y=1.0, color='black', linestyle='--')
    ax[1].set_xlabel('p-values')
    ax[1].set_ylabel('Frequency')
    ax[1].set_title('Corrected')
    
    fig.suptitle('{} p-values'.format(title))
    plt.tight_layout()
    plt.savefig('results/{}_p_vals.jpg'.format(filename))
    plt.clf()

def plot_conf_ints(pred_ints_baseline, pred_ints_sloe, X_test, y_test, title, filename, ci=90):
    figure(figsize=(8, 6))
    #Sample from test set for CI
    num_samples = 8
    sample_idxs = np.random.randint(0, X_test.shape[0], size=(num_samples,))

    for i, row_idx in enumerate(sample_idxs):
        # Observed y values
        observed = y_test[row_idx]
        plt.plot(observed, i, '*', color='black')

        # Baseline CI
        lower_ci_baseline = pred_ints_baseline[row_idx,0]
        upper_ci_baseline = pred_ints_baseline[row_idx,1]
        plt.plot((lower_ci_baseline,upper_ci_baseline),(i,i),'b|-',color='blue')

        # SLOE CI
        lower_ci_sloe = pred_ints_sloe[row_idx,0]
        upper_ci_sloe = pred_ints_sloe[row_idx,1]
        plt.plot((lower_ci_sloe,upper_ci_sloe),(i,i),'r|-',color='orange')
        
    plt.yticks([])
    plt.ylabel('Example')
    plt.xlabel('Probability')
    plt.title('{} Confidence Intervals'.format(title))
    observed_patch = mlines.Line2D([],[],color='black', marker='*', linestyle='None', label='Observed Outcome')
    standard_patch = mlines.Line2D([],[],color='blue', marker='|',label='Standard {}% CI'.format(ci))
    corrected_patch = mlines.Line2D([],[],color='orange', marker='|',label='Corrected {}% CI'.format(ci))
    plt.legend(handles=[observed_patch, standard_patch, corrected_patch])
    plt.tight_layout()
    plt.savefig('results/{}_CIs.jpg'.format(filename))
    plt.clf()