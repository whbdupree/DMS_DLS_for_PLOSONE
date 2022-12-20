import jax
jax.config.update('jax_platform_name', 'cpu')
import matplotlib as mpl;mpl.use('Agg')
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Liberation Sans']
from matplotlib import pyplot as plt
import corticostriatal as cs
import numpy as np
from functools import partial
from jax import numpy as jnp
from jax.random import PRNGKey
import sys

if __name__=='__main__':
    seed = 458
    key = PRNGKey( seed )
    session = partial(
        cs.session,
        n_trial = 2000,
        reversal_learning_flag = False,
        reward_coefficient = -2.,
        gain_pfc = 1.,
        rotate_pfc = 1.
    )

    wvi = jnp.load('data/init.npy')
    k,w,r,ctx = session( key, wvi[0] )
    cs.panels( w,ctx[0],ctx[1],r,
               my_xlims = [1100,1300],
               my_xticks = [1100,1150,1200,1250,1300],
               dms_ylims = [0,3.5],
               dms_yticks = [0,1,2,3],
               r_yticks = ([-2,0,2],['-2','0','2']),
               rpe_ylims = [-2,2]
    )
    plt.savefig(f'figs/large_punish_one_off_{seed}.pdf')
