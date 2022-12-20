import matplotlib as mpl; mpl.use('Agg')
from matplotlib import pyplot as plt
import pandas as pd

experiments = {'learn':200,
               'reversal':500,
               'reversal_prat':500,
               'punish':500,
               'punish_prat':500,
               'devalue':200}

output = {}

numSims =100

for k in experiments.keys():
    df = pd.read_csv(f'data/prob_{k}.csv')
    df_avg = pd.DataFrame()
    df_std = pd.DataFrame()
    for ctx in ['pfc','pmc']:
        ctx_cols = [ f'{ctx}{n}' for n in range(numSims) ]
        df_avg[ctx] = df[ctx_cols].mean(axis=1)
        df_std[ctx] = df[ctx_cols].std(axis=1)


    ## performance plots
    plt.figure()
    ax = plt.subplot(311)
    avg = df_avg['pfc'][:experiments[k]]
    env1 = avg - df_std['pfc'][:experiments[k]]
    env2 = avg + df_std['pfc'][:experiments[k]]
    ax.plot([0,experiments[k]],[0.5,0.5],'--',
            color = [0.5,0.5,0.5])
    ax.fill_between( range(experiments[k]),
                     env1, env2,
                     color = [0,0,0],
                     alpha = 0.5 )
    ax.plot( range(experiments[k]),
             avg,
             color = [0,0,0] )
    ax.set_yticks([0,0.5,1])
    ax.set_ylim([0,1])
    ax.set_xlim([0,experiments[k]])
    ax.set_ylabel( 'PFC perform. (%)' )
    
    ax = plt.subplot(312)
    avg = df_avg['pmc'][:experiments[k]]
    env1 = avg - df_std['pmc'][:experiments[k]]
    env2 = avg + df_std['pmc'][:experiments[k]]    
    ax.plot([0,experiments[k]],[0.5,0.5],'--',
            color = [0.5,0.5,0.5])
    ax.fill_between( range(experiments[k]),
                     env1, env2,
                     color = [0,0,0],
                     alpha = 0.5 )
    ax.plot( range(experiments[k]),
             avg,
             color = [0,0,0] )
    ax.set_yticks([0,0.5,1])
    ax.set_ylim([0,1])
    ax.set_xlim([0,experiments[k]])
    ax.set_ylabel( 'PMC perform. (%)' )
    ax.set_xlabel('trial #')
    plt.savefig(f'figs/p1_{k}.pdf')
    plt.close()


