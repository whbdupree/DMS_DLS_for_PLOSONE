library(comprehenr)

run_stats <- function( data_name ) {
    df <- read.csv( file = paste('data/',data_name,sep='') )
    pfc <- to_vec( for (ii in 0:99) tail( df[paste('pfc',ii,sep='')], 1 ) )
    pmc <- to_vec( for (ii in 0:99) tail( df[paste('pmc',ii,sep='')], 1 ) )
    w = wilcox.test(pfc, pmc, paired = FALSE, alternative = "two.sided")
}

fnames = c('learn',
           'devalue',
 	   'reversal',
	   'punish',
	   'reversal_prat',
	   'punish_prat')

stat = c()
pval = c()
for (fname in fnames) {
    w = run_stats( paste('prob_',fname,'.csv',sep='' ) )
    stat = append(stat,w$statistic)
    pval = append(pval,w$p.value)    
}

df <- data.frame( fnames,stat, pval )
print(df)



df <- read.csv( file = 'data/prob_devalue.csv' )
devalue_pmc <- to_vec( for (ii in 0:99) tail( df[paste('pmc',ii,sep='')], 1 ) )
df <- read.csv( file = 'data/prob_learn.csv' )
learn_pmc <- to_vec( for (ii in 0:99) tail( df[paste('pmc',ii,sep='')], 1 ) )
w = wilcox.test(learn_pmc, devalue_pmc, paired = FALSE, alternative = "two.sided")

print(w)
