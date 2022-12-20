import matplotlib as mpl; mpl.use('Agg')
from matplotlib import pyplot as plt
import pandas as pd

experiments = {'reversal':500,
               'punish':500}

output = {}
numSims = 100
for k in experiments.keys():
    df = pd.read_csv(f'data/prob_{k}.csv')
    df1_avg = pd.DataFrame()
    df1_std = pd.DataFrame()
    for ctx in ['pfc','pmc']:
        ctx_cols = [ f'{ctx}{n}' for n in range(numSims) ]
        df1_avg[ctx] = df[ctx_cols].mean(axis=1)
        df1_std[ctx] = df[ctx_cols].std(axis=1)

    df = pd.read_csv(f'data/prob_{k}_prat.csv')
    df2_avg = pd.DataFrame()
    df2_std = pd.DataFrame()
    for ctx in ['pfc','pmc']:
        ctx_cols = [ f'{ctx}{n}' for n in range(numSims) ]
        df2_avg[ctx] = df[ctx_cols].mean(axis=1)
        df2_std[ctx] = df[ctx_cols].std(axis=1)
        
    ## performance plots
    plt.figure()
    ax = plt.subplot(321)
    avg = df1_avg['pfc'][:experiments[k]]
    env1 = avg - df1_std['pfc'][:experiments[k]]
    env2 = avg + df1_std['pfc'][:experiments[k]]
    ax.plot([0,experiments[k]],[0.5,0.5],'--',
            color = [0.5,0.5,0.5])
    ax.fill_between( range(experiments[k]),
                     env1, env2,
                     color = [0,0,0],
                     alpha = 0.5 )
    ax.plot( range(experiments[k]),
             avg,
             color = [0,0,0] )
    avg = df2_avg['pfc'][:experiments[k]]
    env1 = avg - df2_std['pfc'][:experiments[k]]
    env2 = avg + df2_std['pfc'][:experiments[k]]
    ax.fill_between( range(experiments[k]),
                     env1, env2,
                     color = [1,0,0],
                     alpha = 0.5 )
    ax.plot( range(experiments[k]),
             avg,
             color = [1,0,0] )
    ax.set_yticks([0,0.5,1])
    ax.set_ylim([0,1])
    ax.set_xlim([0,experiments[k]])
    ax.set_ylabel( 'PFC\nperform. (%)' )
    
    ax = plt.subplot(323)
    avg = df1_avg['pmc'][:experiments[k]]
    env1 = avg - df1_std['pmc'][:experiments[k]]
    env2 = avg + df1_std['pmc'][:experiments[k]]    
    ax.plot([0,experiments[k]],[0.5,0.5],'--',
            color = [0.5,0.5,0.5])
    ax.fill_between( range(experiments[k]),
                     env1, env2,
                     color = [0,0,0],
                     alpha = 0.5 )
    ax.plot( range(experiments[k]),
             avg,
             color = [0,0,0] )
    avg = df2_avg['pmc'][:experiments[k]]
    env1 = avg - df2_std['pmc'][:experiments[k]]
    env2 = avg + df2_std['pmc'][:experiments[k]]    
    ax.fill_between( range(experiments[k]),
                     env1, env2,
                     color = [1,0,0],
                     alpha = 0.5 )
    ax.plot( range(experiments[k]),
             avg,
             color = [1,0,0] )
    ax.set_yticks([0,0.5,1])
    ax.set_ylim([0,1])
    ax.set_xlim([0,experiments[k]])
    ax.set_ylabel( 'PMC\nperform. (%)' )
    ax.set_xlabel('trial #')
    plt.savefig(f'figs/compare_{k}.pdf')
    plt.close()
