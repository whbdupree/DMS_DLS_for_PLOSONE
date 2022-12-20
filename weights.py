import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import rcParams
from matplotlib.backends.backend_pdf import PdfPages

rcParams.update({
        'font.family':'sans-serif',
        'font.sans-serif':['Liberation Sans'],
        })

fnames = [
    'learn.npz',
    'devalue.npz',
    'reversal02.npz',
    'reversal05.npz',
    'reversal.npz', # used in main results
    'reversal20.npz',
    'punish02.npz',
    'punish.npz', # used in main results
    'punish10.npz',
    'punish20.npz',
]

titles = [
    'A.    Initial Learning (R=1)',
    'B.    Devalutaion (R=0.2)',
    'C.    Reward reversal (R=0.2)',
    'D.    Reward reversal (R=0.5)',
    'E.    Reward reversal (R=1)$^*$',
    'F.    Reward reversal (R=2)',
    'G.    Punished outcome (R=-0.2)',
    'H.    Punished outcome (R=-0.5)$^*$',
    'I.    Punished outcome (R=-1)',
    'J.    Punished outcome (R=-2)',
]

## weights figures
with PdfPages('figs/stackedweights.pdf') as pdf:
    for fname,title in zip(fnames,titles):
        print('stacked weights',fname)
        sname = fname.split('.')[0]        
        data = np.load('data/'+fname)
        wv = data['wv']
        nt = wv.shape[-1]
        trials = np.arange(nt)+1
        fig1, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, sharey='all')
        fig1.suptitle(title, fontsize=18)
        for w in wv:
            w1=w[:,:,0,:]
            w2=w[:,:,1,:]

            #DMS D1,D2 weights for channel #1
            # notice here and below, this is split across two axis objects
            x = w1[0,:2,:]
            ax1.plot(trials, x[0],
                     color = 'C0',
                     alpha=0.1)
            ax1.set_xlim([0,nt])

            ax2.plot(trials, x[1],
                     color = 'C0',
                     alpha=0.1)

            ax2.set_xlim([0,nt])        

            #DMS D1,D2 weights for channel #2        
            x = w2[1,:2,:]
            ax1.plot(trials, x[0],
                     color = 'C1',
                     alpha=0.1)
            ax1.set_ylabel('DMS\nD1')
            ax1.set_xlim([0,nt])
            #ax1.set_xlabel('trial #')
            
            ax2.plot(trials, x[1],
                     color = 'C1',
                     alpha=0.1)
            ax2.set_ylabel('DMS\nD2')
            ax2.set_xlim([0,nt])
            #ax2.set_xlabel('trial #')
            #DLS D1,D2 weights for channel #1
            x = w1[0,2:,:]
            ax3.plot(trials, x[0],
                     color = 'C0',
                     alpha=0.1)
            ax4.set_xlim([0,nt])
            
            ax4.plot(trials, x[1],
                     color = 'C0',
                     alpha=0.1)
            ax4.set_xlim([0,nt])
            
            #DLS D1,D2 weights for channel #2                        
            x = w2[1,2:,:]
            ax3.plot(trials, x[0],
                     color = 'C1',
                     alpha=0.1)
            ax3.set_ylabel('DLS\nD1')
            ax3.set_xlim([0,nt])
            ax3.set_xlabel('trial #')
            
            ax4.plot(trials, x[1],
                     color = 'C1',
                     alpha=0.1)
            ax4.set_ylabel('DLS\nD2')
            ax4.set_xlim([0,nt])
            ax4.set_xlabel('trial #')
        # legend:
        if fname =='learn.npz':
            xposRect = 100
            xposText = 115
            width = 10
        else:
            xposRect = 1000
            xposText = 1150
            width = 100
        ylim = ax2.get_ylim()
        yrange = np.diff(ylim)[0]
        yposRect1 = ylim[0]+yrange*0.85
        yposRect2 = ylim[0]+yrange*0.75
        
        rect1 = Rectangle((xposRect, yposRect1),
                          width,yrange*0.06,
                          color='C0')
        rect2 = Rectangle((xposRect, yposRect2),
                          width,yrange*0.06,
                          color='C1')
        ax2.add_patch(rect1)
        ax2.add_patch(rect2)
        ax2.text(xposText,yposRect1,
                 'Outcome #1',
                 color = 'C0')
        ax2.text(xposText,yposRect2,
                 'Outcome #2',
                 color = 'C1')
        # can't re-use artist in more than one Axes
        rect1 = Rectangle((xposRect, yposRect1),
                          width,yrange*0.06,
                          color='C0')
        rect2 = Rectangle((xposRect, yposRect2),
                          width,yrange*0.06,
                          color='C1')
        ax4.add_patch(rect1)
        ax4.add_patch(rect2)
        ax4.text(xposText,yposRect1,
                 'Action #1',
                 color = 'C0')
        ax4.text(xposText,yposRect2,
                 'Action #2',
                 color = 'C1')
        #plt.tight_layout()
        pdf.savefig()
        plt.savefig(f'figs/stackedweights_{sname}.png')
        plt.close()
    

