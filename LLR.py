import matplotlib as mpl; mpl.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import lines as mlines

import numpy as np
import pandas as pd
from scipy import stats

def evidence(h,q,ylast):
    n = ( (1-h) * np.exp(ylast) + h )
    d = ( h * np.exp(ylast) + (1-h) )
    y = np.log( q ) + np.log( n / d )
    return y

if __name__ == '__main__':
    numSims = 100
    h = 1/201
    experiments = ['reversal','reversal_prat','punish','punish_prat']
    prob_learn = pd.read_csv('data/prob_learn.csv')
    LLR = pd.DataFrame()
    ZC0 = pd.DataFrame()
    for experiment in experiments:
        prob_data = pd.read_csv(f'data/prob_{experiment}.csv')
        zlist=[]
        for i in range(numSims):
            prob_pfc, prob_pmc = [
                np.concatenate(
                    [ x[ss].values
                      for x in [ prob_learn, prob_data ] ]
                )
                for ss in [f'pfc{i}',f'pmc{i}']
            ]
            zdict = {}
            for p_ctx,n_ctx in zip([prob_pfc,prob_pmc],['pfc','pmc']):
                np.clip( p_ctx, a_min = 1/500, a_max = 499/500, out = p_ctx )
                yy=np.zeros(p_ctx.size)
                q = p_ctx / (1-p_ctx)
                for n in range( p_ctx.size -1):
                    yy[n+1] = evidence( h, q[n], yy[n] )
                LLR[f'{experiment}_{n_ctx}_{i}'] = yy #save LLR
                zc = np.where( np.diff( np.sign( yy[200:] ) ))[0]+1
                zdict[f'{experiment}_{n_ctx}'] = zc[0]
                print( f'zc {experiment} {i} {n_ctx} ',
                       ' '.join([str(x) for x in zc] ) )
            zlist.append( zdict )
        df=pd.DataFrame(zlist)
        for k in df.keys():
            ZC0[k]=df[k]
    LLR.to_csv('data/LLR.csv',index=False)
    ZC0.to_csv('data/ZC0.csv',index=False)
    

    print('')
    print('stats:')
    print('')    

    print('reversal')
    print(np.median(ZC0['reversal_pfc']))
    print(np.median(ZC0['reversal_pmc']))        

    print('punish')    
    print(np.median(ZC0['punish_pfc']))
    print(np.median(ZC0['punish_pmc']))        

    print('PFC+')    
    print('reversal pfc:')
    print(np.median(ZC0['reversal_pfc']))
    print(np.median(ZC0['reversal_prat_pfc']))    

    print('reversal pmc:')
    print(np.median(ZC0['reversal_pmc']))
    print(np.median(ZC0['reversal_prat_pmc']))    

    print('punish pfc:')
    print(np.median(ZC0['punish_pfc']))
    print(np.median(ZC0['punish_prat_pfc']))    

    print('punish pmc:')    
    print(np.median(ZC0['punish_pmc']))
    print(np.median(ZC0['punish_prat_pmc']))    

    
    w = 0.2
    ax = plt.subplot(211)
    bx = ax.boxplot( [ZC0['reversal_pfc'],ZC0['reversal_pmc']],
                     positions =[1,2],
                     widths = w,
                     notch = True )
    for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(bx[item],color=[0,0,0],linewidth = 1.5)


    bx = ax.boxplot( [ZC0['reversal_prat_pfc'],ZC0['reversal_prat_pmc']],
                     positions =[1.3,2.3],
                     widths = w,                
                     notch = True )
    for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(bx[item], color=[1,0,0],linewidth = 1.5)
    plt.setp( bx['fliers'], markeredgecolor = [1,0,0])
    

    
    bx = ax.boxplot( [ZC0['punish_pfc'],ZC0['punish_pmc']],
                     positions =[3,4],
                     widths = w,
                     notch = True )
    for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(bx[item], color=[0,0,0],linewidth = 1.5)


    bx = ax.boxplot( [ZC0['punish_prat_pfc'],ZC0['punish_prat_pmc']],
                     positions =[3.3,4.3],
                     widths = w,                
                     notch = True )
    for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(bx[item], color=[1,0,0],linewidth = 1.5)
    plt.setp( bx['fliers'], markeredgecolor = [1,0,0])
    
    ax.set_xticks([1.15,2.15,3.15,4.15])
    ax.set_xticklabels(['a','b','c','d'])
    #ax.set_ylabel('Punishment\n LLR Zero Crossing')
    ax.set_yscale('log')

    h1 = mlines.Line2D([], [], color=[0,0,0], label='control')
    h2 = mlines.Line2D([], [], color=[1,0.,0.], label='PFC+')
    plt.legend(handles = [h1,h2],
	   loc ='lower right',
           frameon=False)
    ax.set_xlim([0.65,4.65])
    plt.savefig( 'figs/LLR_boxplots.pdf' )
