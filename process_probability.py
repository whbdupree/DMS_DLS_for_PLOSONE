import numpy as np
import pandas as pd

experiments = ['learn','reversal','reversal_prat','punish','punish_prat','devalue',]

numSim = 100


def proc_prob( pXc ):
    pXc_T = pXc[:,:,0] > pXc[:,:,1]
    return np.sum( pXc_T, 1 )/pXc_T.shape[1]

for experiment in experiments:
    df = pd.DataFrame()
    for ns in range(numSim):
        ctx = np.load(f'sample1K/{experiment}_sim{ns}_.npy')
        pfc = ctx[:,:,0,:]
        pmc = ctx[:,:,1,:]
        pfc_p = proc_prob( pfc )
        pmc_p = proc_prob( pmc )        
        df[f'pfc{ns}'] = pfc_p
        df[f'pmc{ns}'] = pmc_p     

    df.to_csv(f'data/prob_{experiment}.csv',index=False)
