import matplotlib as mpl; mpl.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import lines as mlines

import numpy as np
from scipy import stats
import pprint
import pandas as pd
experiments = ['learn','devalue','reversal','punish','reversal_prat','punish_prat']


# define the axis now because we are gonna
# fill up the subplot as we iterate over
# out experiments
ax = plt.subplot(211)
numSims=100
for i,e in enumerate(experiments):
    df = pd.read_csv(f'data/prob_{e}.csv')
    pfc,pmc = [ df[ [ f'{ctx}{n}'
                      for n in range(numSims) ] ].iloc[-1].values
                for ctx in ['pfc','pmc'] ]
    med_pfc = np.median(pfc)
    mad_pfc = np.median( np.abs(pfc - med_pfc) )  
    med_pmc = np.median(pmc)
    mad_pmc = np.median( np.abs(pmc - med_pmc) )    
    print(e,'pfc median:',med_pfc)
    print(e,'pmc median:',med_pmc)


    
    # make stacked bar plots
    pos = np.array([-0.17,0.17])
    w = 0.3

    bx = ax.boxplot( [pfc],
                     positions = [i+pos[0]],
                     widths = 0.3,
                     notch = True)

    
    for item in ['boxes', 'whiskers', 'medians', 'caps']:
        plt.setp( bx[item],
                  color=[0.,0.,0.],
                  linewidth = 1.5)
                  

    bx = ax.boxplot( [pmc],
                positions = [i+pos[1]],
                widths = 0.3,
                notch = True)
    for item in ['boxes', 'whiskers', 'medians', 'caps']:
        plt.setp( bx[item],
                  color=[0.5,0.5,0.5],
                  linewidth = 1.5)
    plt.setp( bx['fliers'], markeredgecolor = [0.5,0.5,0.5])

    ax.set_xticks(np.arange(6))
    ax.set_xticklabels(['Learn\n(PFC,PMC)','Devalue','RR','Punish','RR\nPFC+','Punish\nPFC+'])
    ax.set_ylabel('Perform. (%)')
    ax.set_xlim([-0.5,5.5])
h1 = mlines.Line2D([], [], color=[0,0,0], label='PFC')
h2 = mlines.Line2D([], [], color=[0.5,0.5,0.5], label='PMC')
plt.legend(handles = [h1,h2],
	   loc ='upper right',
           frameon=False)

plt.savefig('figs/steady_state_boxplot.pdf')
plt.close()
