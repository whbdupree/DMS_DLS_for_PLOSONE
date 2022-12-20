#import jax
#jax.config.update('jax_platform_name', 'cpu')
import corticostriatal as cs
import time
import sys
from jax import vmap
from jax import numpy as jnp
from jax.ops import index,index_update
from jax.random import split,PRNGKey,uniform
from functools import partial
import numpy as np

arg_dict = {
    'learn':        [ False, 1.,  1., 1. ],
    'reversal':     [ True,  1.,  1., 1. ],
    'reversal_prat':[ True,  1.,  1., 0.9 ],
    'punish':       [ False, -.5, 1., 1. ],
    'punish_prat':  [ False, -.5, 1., 0.9 ],
    'devalue':      [ False, 0.2, 1., 1. ],
    }
            
positional_map_keys = ['reversal_learning_flag','reward_coefficient','gain_pfc','rotate_pfc']

def get_prob( key, w_ctx,
              batchSize,
              reversal_learning_flag,
              reward_coefficient,
              gain_pfc,
              rotate_pfc ):
    
    keys = split( key, batchSize )
    trial = partial(
        cs.do_trial_no_update,
        w_ctx = w_ctx,
        reversal_learning_flag = reversal_learning_flag,
        reward_coefficient = reward_coefficient,
        gain_pfc = gain_pfc,
        rotate_pfc = rotate_pfc
    )
    vtrial = vmap(trial)
    return vtrial( keys )

def organize_prob( simNumber,
                   key,
                   batchSize,
                   experiment,
                   wv ):

    w = wv[simNumber]
    u = jnp.moveaxis(w,-1,0) # need first axis to have length of keys
    keys = split( key, u.shape[0] )
    positional_map = dict( zip( positional_map_keys, arg_dict[experiment] ) )
    pgp = partial( get_prob, batchSize = batchSize, **positional_map )
    vpgp = vmap( pgp )
    ctxv = vpgp( keys, u )
    return ctxv
    
if __name__ == '__main__':
    numSims = 100
    
    # total number of samples is batchSize * numBatches    
    # values suggested for trial evaluation:
    numBatches = 1
    batchSize = 32
    # values used for results in manuscript:
    #numBatches = 10
    #batchSize = 100
    # these quantities can be jointly configured to satisfy memory constraints

    experiment = sys.argv[1]    

    data = jnp.load( f'data/{experiment}.npz' )
    # data has numpy arrays in it
    # this is either a huge performance hit
    # or can trigger an error with jax funcion transformations
    wv = jnp.array(data['wv']) 
    numTrials = (wv.shape)[-1]
    key = PRNGKey( time.time_ns() )

    ctxvv = np.zeros(shape=(numSims,numTrials,numBatches*batchSize,2,2,))
    if sys.argv[2] == 'batch':
        pop = partial( organize_prob,
                       batchSize = batchSize,
                       experiment = experiment,
                       wv = wv )
        for nb in range(numBatches):
            key,skey = split(key)
            kpop = partial(pop, key=skey)
            vkpop = vmap(kpop)
            snv = jnp.arange(numSims)
            ctxvv[:,:,nb*batchSize:(nb+1)*batchSize,:,:] = vkpop( snv )
        print(ctxvv.shape)
        for simNumber,ctxv in enumerate(ctxvv):
            np.save(f'sample1K/{experiment}_sim{simNumber}_.npy',ctxv)
    else:
        simNumber = int(sys.argv[2])
        ctxv = organize_prob( simNumber, experiment, wv )
        np.save(f'sample1K/{experiment}_sim{simNumber}_.npy',ctxv)


