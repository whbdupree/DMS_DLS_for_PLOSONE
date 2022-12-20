library(comprehenr)

df <- read.csv('data/ZC0.csv')

fnames = c('reversal','punish')
cnames = c('pfc','pmc')

stat = c()
pval = c()
rname = c()
for ( fname in fnames ) {
    for (ctx in cnames) {
    	xn = paste(fname,ctx,sep='_')
        x <- df[[xn]]
	yn = paste(fname,'prat',ctx,sep='_')
        y <- df[[yn]]
	w = wilcox.test( x, y, paired = FALSE, alternative = "two.sided")
	stat = append(stat,w$statistic)
	pval = append(pval,w$p.value)
	rname = append(rname,xn)
    }
}

df <- data.frame( rname,stat, pval )
print(df)
