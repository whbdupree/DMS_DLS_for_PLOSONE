import matplotlib as mpl; mpl.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import sys

#gp0_devalue_sim11_.npy

def evidence(h,q,ylast):
    n = ( (1-h) * np.exp(ylast) + h )
    d = ( h * np.exp(ylast) + (1-h) )
    y = np.log( q ) + np.log( n / d )
    return y

def get_ctx( epoch1, epoch2 ):
    ctx = np.concatenate([epoch1,epoch2],axis=2)
    pfc = ctx[0]
    pmc = ctx[1]
    return pfc,pmc

if __name__ == '__main__':
    is_short=''
    arg1=''
    try:
        arg1 = sys.argv[1]
        # there is probably a better way to do this
    except:
        pass
    
    if arg1 == 'short':
        is_short='short'
        sim_length=200
    else:
        sim_length=2000
        
    my_length = 200+sim_length
    #h = 1/my_length
    h = 1/201
    experiments = ['reversal','reversal_prat','punish','punish_prat']
    learn = np.load('data/learn.npz')
    prob_learn = pd.read_csv('data/prob_learn.csv')
    for experiment in experiments:
        data = np.load(f'data/{experiment}.npz')
        prob_data = pd.read_csv(f'data/prob_{experiment}.csv')
        for i in np.arange(10,19):
            pfc, pmc = np.concatenate(
                [ learn['ctxv'][i],
                  data['ctxv'][i] ]
                , axis = 2
            )
            prob_pfc, prob_pmc = [
                np.concatenate(
                    [ x[ss].values
                      for x in [ prob_learn, prob_data ] ]
                )
                for ss in [f'pfc{i}',f'pmc{i}']
            ]
            
            # use keyword out to operate on array in place
            np.clip( prob_pfc, a_min = 1/500, a_max = 499/500, out = prob_pfc )
            np.clip( prob_pmc, a_min = 1/500, a_max = 499/500, out = prob_pmc )

            ypfc=np.zeros(my_length)
            ypmc=np.zeros(my_length)

            qpfc = prob_pfc / (1-prob_pfc)
            qpmc = prob_pmc / (1-prob_pmc)
            
            for n in range( my_length -1):
                ypfc[n+1] = evidence( h, qpfc[n], ypfc[n] )
                ypmc[n+1] = evidence( h, qpmc[n], ypmc[n] )
                
            ax = plt.subplot(411)
            ax.plot(pfc[0][:my_length])
            ax.plot(pfc[1][:my_length])
            ax.plot([200,200],[0,1],'-k')
            ax.set_title(f'h=1/201')
            ax.set_ylabel('pfc activity')
            ax.set_xlim([0,my_length])
            
            ax = plt.subplot(412, sharex = ax )
            ax.plot([0,my_length],[0,0],'--k')
            ax.plot(ypfc)
            ax.set_ylabel('pfc LLR')
            ax.set_xlim([0,my_length])
            
            ax = plt.subplot(413, sharex = ax )
            ax.plot(pmc[0][:my_length])
            ax.plot(pmc[1][:my_length])
            ax.plot([200,200],[0,1],'-k')            
            ax.set_ylabel('pmc activity')
            ax.set_xlim([0,my_length])
            
            ax = plt.subplot(414, sharex = ax )
            ax.plot([0,my_length],[0,0],'--k')            
            ax.plot(ypmc)
            ax.set_ylabel('pmc LLR')
            ax.set_xlabel('trial #')
            ax.set_xlim([0,my_length])
            
            plt.savefig(f'figs/LLR{is_short}_{experiment}_{i}.pdf')
            plt.close()
