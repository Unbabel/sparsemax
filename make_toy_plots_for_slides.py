import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pdb

#font = {'family' : 'cmr10', 'size': 12}
font = {'size': 14}
plt.rc('font', **font)

#all_num_classes = [10, 20, 30, 40, 50, 60, 70, 80]
all_num_classes = [50] #[10, 50]
#all_lengths = [50, 100, 500, 1000, 1500, 2000, 2500, 3000]
#all_lengths = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
all_lengths = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
all_regularizers = ['0.000000001', '0.00000001', '0.0000001', '0.000001', '0.00001', '0.0001', '0.001', '0.01', '0.1', '1.0']
all_results = {}
all_results['softmax'] = np.zeros((len(all_num_classes), len(all_lengths), 4))
all_results['sparsemax'] = np.zeros((len(all_num_classes), len(all_lengths), 4))
settings = ['-proportions']

fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True)

for s, setting in enumerate(settings):
    for algo in ['softmax', 'sparsemax']:
        for k, num_classes in enumerate(all_num_classes):
            for l, length in enumerate(all_lengths):
                print 'Loss: %s, Num classes: %d, Length: %d' % (algo, num_classes, length)
                V = np.zeros((len(all_regularizers), 4))
                for i, regularizer in enumerate(all_regularizers):
                    #filename = 'log_lbfgs2_toy-%d-classes_norm_jack_dev_%s_epochs-100_regularizer-%s.txt' % (num_classes, algo, regularizer)
                    #filename = 'log_lbfgs2_toy-%d-classes-%d-length_norm_jack_dev_%s_epochs-100_regularizer-%s.txt' % (num_classes, length, algo, regularizer)
                    filename = 'logs_toy_experiments/log_lbfgs2_toy%s-%d-classes-%d-length_norm_jack_dev_%s_epochs-100_regularizer-%s.txt' % (setting, num_classes, length, algo, regularizer)
                    f = open(filename)
                    for line in f:
                        line = line.rstrip()
                        if line.startswith('Sq loss'):
                            # Sq loss: 0.000442 +- 0.000080, JS loss: 0.000409 +- 0.000061
                            res = re.findall('Sq loss: (.*) \+- (.*), JS loss: (.*) \+- (.*)', line)
                            assert len(res[0]) == 4, pdb.set_trace()
                            V[i,:] = np.array([float(val) for val in res[0]])
                    f.close()
                best_sq = np.argmin(V[:, 0])
                best_js = np.argmin(V[:, 2])
                sq_loss_av, sq_loss_std = tuple(V[best_sq, :2])
                js_loss_av, js_loss_std = tuple(V[best_js, 2:])
                print 'Best sq loss: %f +- %f (reg = %s)' % (sq_loss_av, sq_loss_std, all_regularizers[best_sq])
                print 'Best js loss: %f +- %f (reg = %s)' % (js_loss_av, js_loss_std, all_regularizers[best_js])

                all_results[algo][k, l, :] = np.concatenate([V[best_sq, :2], V[best_js, 2:]])
  
    for k, num_classes in enumerate(all_num_classes):
        #plt.plot(np.array(all_num_classes), all_results['softmax'][k, :, 0], 'ro-')
        #plt.plot(np.array(all_num_classes), all_results['sparsemax'][k, :, 0], 'bs-')
        #plt.plot(np.array(all_num_classes), all_results['softmax'][k, :, 2], 'ro:')
        #plt.plot(np.array(all_num_classes), all_results['sparsemax'][k, :, 2], 'bs:')
        #plt.plot(np.array(all_lengths), all_results['softmax'][k, :, 0], 'ro-')
        #plt.plot(np.array(all_lengths), all_results['sparsemax'][k, :, 0], 'bs-')
        #plt.plot(np.array(all_lengths), all_results['softmax'][k, :, 2], 'ro:')
        #plt.plot(np.array(all_lengths), all_results['sparsemax'][k, :, 2], 'bs:')

        #plt.errorbar(np.array(all_lengths), all_results['softmax'][k, :, 2], yerr=all_results['softmax'][k, :, 3], fmt='o-', color='r')
        #plt.errorbar(np.array(all_lengths), all_results['sparsemax'][k, :, 2], yerr=all_results['sparsemax'][k, :, 3], fmt='s-', color='b')
        #plt.plot(np.array(all_lengths), all_results['softmax'][k, :, 0], 'r--')
        #plt.plot(np.array(all_lengths), all_results['sparsemax'][k, :, 0], 'b-.')
        #plt.show()

        #ax = axs[s,k]
        ax = axs #[k]
        (_, caps1, _) = ax.errorbar(np.array(all_lengths), all_results['softmax'][k, :, 2], yerr=all_results['softmax'][k, :, 3], fmt='o-', color='b', linewidth=3.0, markersize=8, capsize=10, markeredgecolor='b')
        (_, caps2, _) = ax.errorbar(np.array(all_lengths), all_results['sparsemax'][k, :, 2], yerr=all_results['sparsemax'][k, :, 3], fmt='s-', color='r', linewidth=3.0, markersize=8, capsize=10, markeredgecolor='r')

        for cap in caps1:
            cap.set_color('blue')
            cap.set_markeredgewidth(3)
        for cap in caps2:
            cap.set_color('red')
            cap.set_markeredgewidth(3)

        #ax.plot(np.array(all_lengths), all_results['softmax'][k, :, 0], 'r--')
        #ax.plot(np.array(all_lengths), all_results['sparsemax'][k, :, 0], 'b-.')
        #ax.set_title('$K=%d$' % num_classes)
        ax.set_title('JS Divergence btwn Predicted and True Label Proportions')
        ax.set_xlim([200, 2000])
        ax.grid(False)

        if k == len(all_num_classes)-1:
            leg = ax.legend(('JSD ($L_\mathbf{softmax}$)', 'JSD ($L_\mathbf{sparsemax}$)'), \
                             #'MSE (softmax)', 'MSE (sparsemax)'),
                             'upper right', shadow=False, fontsize=20)
        ax.set_xlabel('Document Length', fontsize=16)
        plt.setp(plt.gca().get_xticklabels(), fontsize=16)        
        plt.setp(plt.gca().get_yticklabels(), fontsize=16)        


plt.show()
