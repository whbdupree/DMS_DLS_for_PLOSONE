import jax
jax.config.update('jax_platform_name', 'cpu')
import matplotlib as mpl;mpl.use('Agg')
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Liberation Sans']
from matplotlib import pyplot as plt
import corticostriatal as cs
import numpy as np
import time
import sys
from jax import vmap
from jax import numpy as jnp
from jax.ops import index,index_update
from jax.random import split,PRNGKey,uniform
from functools import partial

arg_dict = {
    'learn':        [200, False, 1.,  1., 1. ],
    'reversal':     [2000, True,  1.,  1., 1. ], # used in main results
    'reversal02':     [2000, True,  .2,  1., 1. ],
    'reversal05':     [2000, True,  .5,  1., 1. ],
    'reversal20':     [2000, True,  2.,  1., 1. ],
    'reversal_prat':[2000, True,  1.,  1., 0.9 ],
    'punish':       [2000, False, -.5, 1., 1. ], # used in main results
    'punish02':       [2000, False, -.2, 1., 1. ],
    'punish10':       [2000, False, -1., 1., 1. ],
    'punish20':       [2000, False, -2., 1., 1. ],    
    'punish_prat':  [2000, False, -.5, 1., 0.9 ],
    'devalue':      [2000, False, 0.2, 1., 1. ],
    }
            

positional_map_keys = ['n_trial','reversal_learning_flag','reward_coefficient','gain_pfc','rotate_pfc']

def learn( key,numSim ):
    key,skey = split(key)
    w_ctx = 1.+0.001*uniform(skey,(2,4,2))
    w_ctx = index_update(w_ctx, index[0,:2,1], [0.,0.])
    w_ctx = index_update(w_ctx, index[1,:2,0], [0.,0.])    
    
    positional_map = dict( zip( ['w_ctx']+positional_map_keys, [w_ctx]+arg_dict[experiment] ) )
    session = partial(cs.session, **positional_map)
    vs = vmap( session )
    keys = jnp.array( split( key, numSim) )
    kv,wv,rv,ctxv = vs( keys  )
    np.save('data/init.npy',wv[:,:,:,:,-1])
    w0 = wv[0]
    r0=rv[0]
    ctx0=ctxv[0]
    cs.panels( w0,ctx0[0],ctx0[1],r0 )
    plt.savefig('figs/learn.pdf')
    plt.close()
    return wv , ctxv
        
def after( key,numSim,experiment ):
    
    positional_map = dict( zip( positional_map_keys, arg_dict[experiment] ) )
    session = partial(cs.session, **positional_map)
    vs = vmap( session )
    keys = jnp.array( split( key, numSim) )
    wvi = jnp.load('data/init.npy')
    kv,wv,rv,ctxv = vs( keys, wvi  )
    w0 = wv[0]
    r0=rv[0]
    ctx0=ctxv[0]
    cs.panels( w0,ctx0[0],ctx0[1],r0 )
    plt.savefig(f'figs/{experiment}.pdf')

    return wv , ctxv

if __name__ == '__main__':
    numSim = 100
    key = PRNGKey( time.time_ns() )

    experiment = sys.argv[1]
    if experiment == 'learn':
        wv , ctxv = learn(key,numSim)
    else:
        wv , ctxv = after(key,numSim,experiment)

    outName = f'data/{experiment}.npz'
    np.savez_compressed( outName, wv = wv, ctxv = ctxv )
    
