from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
from matplotlib import lines as mlines
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple

import numpy as np


def panels( w,pf,pm,r,
            my_xlims = [0,200],
            my_xticks = [0,50,100,150,200],
            dms_ylims = [0,2],
            dms_yticks = [0,1,2],
            r_yticks = ([0,0.5,1],['0','','1']),
            rpe_ylims = [-1,1]):


    ln = pm[0].size
    xlims = my_xlims
    w1=w[:,:,0,:]
    w2=w[:,:,1,:]


    def mk_axes( ii ):
        xw = 0.7
        yw = 0.13
        x0 = 0.1
        y0 = (1-0.15)-ii*0.15
        ax = plt.axes( [x0,y0,xw,yw])
        return ax

    # synaptic weighths plot
    plt.figure()

    ctx_x = np.arange(ln)+1
    
    #ax = plt.subplot(6,1,1)
    ax = mk_axes( 0 )
    # ctx activity at end of each trial
    for x in pf:
        ax.plot(ctx_x,x)
    ax.set_ylim([0,1])
    ax.set_xlim(xlims)
    ax.set_xticks( my_xticks )
    ax.set_xticklabels(['']*5)
    ax.set_yticks([0,0.5,1])
    ax.set_yticklabels(['0','','1'])
    h1 = mlines.Line2D([], [], color='C0', label='PFC #1')
    h2 = mlines.Line2D([], [], color='C1', label='PFC #2')
    plt.legend(handles = [h1,h2],
               bbox_to_anchor=(1.01,1),
               loc ='upper left',
               frameon=False )
    #ax.set_ylabel('PFC\nactivity',rotation=45)

    ax = mk_axes( 1 )
    #ax = plt.subplot(6,1,2)
    # ctx activity at end of each trial
    for x in pm:
        ax.plot(ctx_x,x)
    ax.set_ylim([0,1])
    ax.set_xlim(xlims)
    ax.set_xticks( my_xticks )
    ax.set_xticklabels(['']*5)
    ax.set_yticks([0,0.5,1])
    ax.set_yticklabels(['0','','1'])
    h1 = mlines.Line2D([], [], color='C0', label='PMC #1')
    h2 = mlines.Line2D([], [], color='C1', label='PMC #2')
    plt.legend(handles = [h1,h2],
               bbox_to_anchor=(1.01,1),
               loc ='upper left',
               frameon=False )
    #ax.set_ylabel('PMC\nactivity',rotation=45)
    #plt.legend( ['Action1','Action2'],loc='right')    
    #ax.set_ylabel('PMC\nactivity',rotation=45)

    ylims = [0,2.]

    ax = mk_axes( 2 )
    #ax = plt.subplot(6,1,3)
    x = w1[0,:2,:]
    ax.plot(x[0],color = 'C0')
    ax.plot(x[1],'--',color = 'C0')
    x = w2[1,:2,:]
    ax.plot(x[0],color = 'C1')
    ax.plot(x[1],'--',color = 'C1')
    ax.set_xlim(xlims)
    ax.set_ylim(dms_ylims)
    ax.set_yticks( dms_yticks )
    #ax.set_ylabel('PFC-DMS\nweights',rotation=45)
    ax.set_xticks( my_xticks )
    ax.set_xticklabels(['']*5)
    #ax.plot([0,ln],[1,1],'--k',alpha=0.25)
    h1 = mlines.Line2D([], [], color='C0')
    h2 = mlines.Line2D([], [], color='C1')
    h3 = mlines.Line2D( [0,0], [1,0], color='C0',
                        linestyle='--',
                        dashes = [1.5,1.5]),
    h4 = mlines.Line2D( [], [], color='C1',
                        linestyle='--',
                        dashes = [1.5,1.5]),
    plt.legend([(h1,h3),(h2,h4)],['D1,D2 #1','D1,D2 (#2)'],
               numpoints=1,
               handler_map={ tuple: HandlerTuple(ndivide=None) },
               bbox_to_anchor=(1.01,1),
               loc ='upper left',
               borderaxespad=0.,
               frameon=False)

    ax = mk_axes( 3 )
    #ax = plt.subplot(6,1,4)
    x = w1[0,2:,:]
    ax.plot(x[0],color = 'C0')
    ax.plot(x[1],'--',color = 'C0')
    x = w2[1,2:,:]
    ax.plot(x[0],color = 'C1')
    ax.plot(x[1],'--',color = 'C1')
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)        
    #ax.set_ylabel('PMC-DLS\nweights',rotation=45)
    #ax.plot([0,ln],[1,1],'--k',alpha=0.25)
    ax.set_xticks( my_xticks )
    ax.set_xticklabels(['']*5)
    h1 = mlines.Line2D([], [], color='C0')
    h2 = mlines.Line2D([], [], color='C1')
    h3 = mlines.Line2D( [0,0], [1,0], color='C0',
                        linestyle='--',
                        dashes = [1.5,1.5]),
    h4 = mlines.Line2D( [], [], color='C1',
                        linestyle='--',
                        dashes = [1.5,1.5]),
    plt.legend([(h1,h3),(h2,h4)],['D1,D2 #1','D1,D2 (#2)'],
               numpoints=1,
               handler_map={ tuple: HandlerTuple(ndivide=None) },
               bbox_to_anchor=(1.01,1),
               loc ='upper left',
               borderaxespad=0.,
               frameon=False)
    
    #ax.set_xlabel('trial #')
    
    # reward feebdack plots
    ax = mk_axes( 4 )
    #ax = plt.subplot(6,1,5)
    ax.plot( r[1],color='k' )
    ax.plot( r[3],'b' ) # rectified expectation        
    ax.plot( r[0],color='r' )
    ax.set_xlim(xlims)
    #ax.set_ylabel('Reward and\n Exp. Reward',rotation=45)
    ax.set_xticks( my_xticks )
    ax.set_xticklabels(['']*5)
    ax.set_yticks( r_yticks[0] )
    ax.set_yticklabels( r_yticks[1] )    
    h1 = mlines.Line2D([], [], color='k', label='R')
    h2 = mlines.Line2D([], [], color='r', label='ExpRew')
    h3 = mlines.Line2D([], [], color='b', label='RectExp')
    plt.legend(handles = [h1,h2,h3],
               bbox_to_anchor=(1.01,1),
               loc ='upper left',
               frameon=False )

    ax = mk_axes( 5 )
    #ax = plt.subplot(6,1,6)
    #ax.plot( [0,ln],[0,0],'--',
    #         color = [0,0,0,],
    #         alpha = 0.25)    
    ax.plot( r[2],color='k' ) # reward prediction error
    ax.set_ylim( rpe_ylims )
    ax.set_xlim(xlims)    
    #ax.set_ylabel('RPE',rotation=45)
    ax.set_xlabel('trial #')
    ax.set_xticks( my_xticks )
    h1 = mlines.Line2D([], [], color='k', label='RPE')
    plt.legend(handles = [h1],
               bbox_to_anchor=(1.01,1),
               loc ='upper left',
               frameon=False )
    

def ctx_performance(ctx,numSim,ln,ax, clr):

    x = np.zeros((numSim,ln))
    y = np.zeros(ln)
    z = np.zeros(ln)    

    # training
    for i in range(ln):
        x[:,i] = ctx[:,0,i] > ctx[:,1,i]

    # get stats
    for i in range(ln):
        y[i]=np.mean(x[:,i])
        z[i]=np.std(x[:,i])/np.sqrt(numSim)

    # plot
    ax.plot(y,
            color=clr)
    ax.fill_between(range(ln),y+z,y-z,
                    alpha=0.25, color=clr)
    ax.set_xlim([0,ln])    
    ax.set_ylim([0,1])
    ax.set_yticks([0,.5,1])    
    ax.set_yticklabels([0,50,100])

def performance( session_name, numSim ):
    clr = [0,0,0]
    pfc = []
    pmc = []
    for i in range(numSim):
        u = np.load(f'data/{session_name}_{i}.npz')
        pfc.append( u['name1'])
        pmc.append( u['name2'])    
    pfc = np.array(pfc)
    pmc = np.array(pmc)
    ln = pfc.shape[2]
    
    plt.figure()
    ax1 = plt.subplot(211)
    ctx_performance(pfc,numSim,ln,ax1,clr)
    ax1.plot([0,ln],[0.5,0.5],'--k',alpha=0.5)    
    ax1.set_ylabel('PFC\nperform. (%)')
    ax1.set_title(session_name)
    ax2 = plt.subplot(212,sharex=ax1)
    ctx_performance(pmc,numSim,ln,ax2,clr)
    ax2.set_ylabel('PMC\nperform. (%)')
    ax2.plot([0,ln],[0.5,0.5],'--k',alpha=0.5)
    ax2.set_xlabel('trial #')

def compare_pfc_dysfunction( sname1, sname2, numSim ):
    clr1 = [0,0,0]
    clr2 = [1,0,0]
    pfc1 = []
    pmc1 = []
    pfc2 = []
    pmc2 = []
    for i in range(numSim):
        u1 = np.load(f'data/{sname1}_{i}.npz')
        pfc1.append( u1['name1'])
        pmc1.append( u1['name2'])    
        u2 = np.load(f'data/{sname2}_{i}.npz')
        pfc2.append( u2['name1'])
        pmc2.append( u2['name2'])    
    pfc1 = np.array(pfc1)
    pfc2 = np.array(pfc2)
    pmc1 = np.array(pmc1)
    pmc2 = np.array(pmc2)
    ln = pfc1.shape[2]
    
    plt.figure()
    ax1 = plt.subplot(211)
    ctx_performance(pfc1,numSim,ln,ax1,clr1)
    ctx_performance(pfc2,numSim,ln,ax1,clr2)
    ax1.plot([0,ln],[0.5,0.5],'--k',alpha=0.5)    
    ax1.set_ylabel('PFC\nperform. (%)')
    ax1.set_title(sname1)
    ax2 = plt.subplot(212,sharex=ax1)
    ctx_performance(pmc1,numSim,ln,ax2,clr1)
    ctx_performance(pmc2,numSim,ln,ax2,clr2)
    ax2.set_ylabel('PMC\nperform. (%)')
    ax2.plot([0,ln],[0.5,0.5],'--k',alpha=0.5)
    ax2.set_xlabel('trial #')
    

def plot_dXs( pXc, dXs, ctx_label, loop_name):
    # plots firing rates for bg looops
    # in either dms or dls

    lpA,lpB = dXs # lp for LooP
    tt = np.arange( lpA[0].size )
    lnt = tt[-1]    

    plt.figure()

    ax = plt.subplot(321)
    i = 0
    ax.plot(tt,lpA[i])
    ax.plot(tt,lpB[i])
    ax.set_xlim([0,lnt])
    ax.set_ylabel('d1 msn')
    #ax.set_ylim([0,1])
    plt.title(loop_name)
    
    ax = plt.subplot(322)
    i = 1    
    ax.plot(tt,lpA[i])
    ax.plot(tt,lpB[i])
    ax.set_xlim([0,lnt])
    ax.set_ylabel('d2 msn')    
    #ax.set_ylim([0,1])
    plt.title(loop_name)
    
    ax = plt.subplot(323)
    i = 2
    ax.plot(tt,lpA[i])
    ax.plot(tt,lpB[i])
    ax.set_xlim([0,lnt])
    ax.set_ylabel('gpe')    
    #ax.set_ylim([0,1])
    
    ax = plt.subplot(324)
    i = 3
    ax.plot(tt,lpA[i])
    ax.plot(tt,lpB[i])
    ax.set_xlim([0,lnt])
    ax.set_ylabel('stn')
    #ax.set_ylim([0,1])
    
    ax = plt.subplot(325)
    i = 4
    ax.plot(tt,lpA[i])
    ax.plot(tt,lpB[i])
    ax.set_xlim([0,lnt])
    ax.set_ylabel('gpi')    
    #ax.set_ylim([0,1])
    
    ax = plt.subplot(326)
    ax.plot(tt,pXc[0])
    ax.plot(tt,pXc[1])
    ax.set_xlim([0,lnt])
    ax.set_ylabel(ctx_label)
    #ax.set_ylim([0,1])
