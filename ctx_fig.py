import corticostriatal as cs
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
from matplotlib import lines as mlines
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
from jax.random import PRNGKey,split,uniform
from jax.ops import index,index_update
from jax import numpy as jnp

width  = 0.2
height = 0.2


def ctx_panel(key, x_offset, y_offset):

    w = [[[1.0005909, 0.       ],
          [1.0009123, 0.       ],
          [1.0002466, 1.0000119],
          [1.0009952, 1.0006101]],
         [[0.       , 1.0002766],
          [0.       , 1.000677 ],
          [1.0002412, 1.0009876],
          [1.0000737, 1.0006125]]]

    w = jnp.array(w)
    
    key, subkey = split(key)
    ctx , bg = cs.do_trial_for_figure( subkey, w )

    ax = plt.axes([x_offset,
                   y_offset,
                   width,height])
    ax.plot(ctx[0][0],ctx[0][1],
            '--k',linewidth=2)
    ax.plot(ctx[1][0],ctx[1][1],
            color=[0.5,0.5,0.5],linewidth=2)
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_xticks([0,0.5,1])
    ax.set_yticks([0,0.5,1])
    h1 = mlines.Line2D([], [], color=[0,0,0],linestyle='--', label='PFC')
    h2 = mlines.Line2D([], [], color=[0.5,0.5,0.5], label='PMC')
    plt.legend(handles = [h1,h2],
               #bbox_to_anchor=(1.01,1),
               loc ='right',
               frameon=False)

    return ax,ctx

def pXc_panel(pXc, x_offset, y_offset,lbl):
    ax = plt.axes([x_offset,
                   y_offset,
                   width,
                   height])
    ax.plot(jnp.arange(500)*1.5, pXc[0],
            linewidth=2)
    ax.plot(jnp.arange(500)*1.5, pXc[1],
            linewidth=2)
    ax.set_ylim([0,1])
    ax.set_xticks([0,250,500,750])
    h1 = mlines.Line2D([], [], color='C0', label=f'{lbl} #1')
    h2 = mlines.Line2D([], [], color='C1', label=f'{lbl} #2')
    plt.legend(handles = [h1,h2],
               #bbox_to_anchor=(1.01,1),
               loc ='right',
               frameon=False)
    return ax
    

if __name__ == '__main__':
    
    plt.figure()
    # configure panels
    y_offset_A = 0.36
    y_offset_B = 0.1
    x_offset_ctx = 0.15
    x_offset_pfc = 0.44
    x_offset_pmc = 0.69
    
    ## panel A1
    #seed = 333
    seed = 340
    key = PRNGKey( seed )
    ax,ctx = ctx_panel( key, x_offset_ctx, y_offset_A )
    #ax.set_xlabel('Population #1')
    ax.set_xticklabels(['','',''])
    ax.set_title('PFC,PMC\nPhase Space')
    ax.set_ylabel('Pop. #2')

    pfc = ctx[0]
    pmc = ctx[1]

    ## panel A2
    ax = pXc_panel(pfc, x_offset_pfc, y_offset_A,'PFC')
    #ax.set_ylabel('PFC\n firing rate')
    ax.set_title('PFC\n Firing Rate')
    ax.set_xticklabels(['','','',''])
    
    ## panel A3
    ax = pXc_panel(pmc, x_offset_pmc, y_offset_A,'PMC')
    ax.set_title('PMC\n Firing Rate')
    ax.set_xticklabels(['','','',''])
    ax.set_yticklabels(['','',''])
    

    
    ## panel B1
    seed = 349
    key = PRNGKey( seed )
    ax,ctx = ctx_panel( key, x_offset_ctx, y_offset_B )
    pfc = ctx[0]
    pmc = ctx[1]
    ax.set_xlabel('Pop. #1')
    ax.set_ylabel('Pop. #2')


    ## panel B2
    ax = pXc_panel(pfc, x_offset_pfc, y_offset_B,'PFC')
    ax.set_xlabel('time (ms)')
    #ax.set_ylabel('PFC\n firing rate')
    #ax.set_xticklabels(['','','',''])
    
    ## panel B3
    ax = pXc_panel(pmc, x_offset_pmc, y_offset_B,'PMC')
    #ax.set_ylabel('PMC\n firing rate')
    ax.set_yticklabels(['','',''])    
    ax.set_xlabel('time (ms)')


    # save close
    plt.savefig('figs/ctx_fig.pdf')
    plt.close()
