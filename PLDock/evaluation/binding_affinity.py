"""
Source code (MIT-Licensed) inspired by PDBbind (http://www.pdbbind.org.cn/)
"""

import numpy as np
import sys,os
import pandas as pd
import scipy
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from decimal import *

class scoring():
    def __init__(self,out_file):
        self.out_file = out_file

    def dec(self,x,y):
        if y == 2:
            return Decimal(x).quantize(Decimal('0.01'),rounding=ROUND_HALF_UP)
        if y == 3:
            return Decimal(x).quantize(Decimal('0.001'),rounding=ROUND_HALF_UP)
        if y == 4:
            return Decimal(x).quantize(Decimal('0.0001'),rounding=ROUND_HALF_UP)

    def scoring(self,core_data_file,docking_score_file,favorable='positive'):
        aa=pd.read_csv(core_data_file,sep='[,,\t, ]+',engine='python')
        aa=aa.drop_duplicates(subset=['#code'],keep='first')
        bb=pd.read_csv(docking_score_file, sep='[,,\t, ]+',engine='python')

        #Process the data and remove the outliers
        testdf1=pd.merge(aa,bb,on='#code')
        if favorable == 'positive':
            testdf2=testdf1[testdf1.score > 0]
            testdf2.to_csv(self.out_file+'_processed_score',columns=['#code','logKa','score'],sep='\t',index=False)
        elif favorable == 'negative':
            testdf1['score']=testdf1['score'].apply(np.negative) 
            testdf2=testdf1[testdf1.score > 0]
            testdf2.to_csv(self.out_file+'_processed_score',columns=['#code','logKa','score'],sep='\t',index=False)
        else:
            print ('please input negative or positive')
            sys.exit()

        #Calculate the Pearson correlation coefficient
        regr=linear_model.LinearRegression()
        regr.fit(testdf2[['score']],testdf2[['logKa']])
        testpredy=regr.predict(testdf2[['score']])
        testr=scipy.stats.pearsonr(testdf2['logKa'].values,testdf2['score'].values)[0]
        testmse=mean_squared_error(testdf2[['logKa']],testpredy)
        num=testdf2.shape[0]
        testsd=np.sqrt((testmse*num)/(num-1))

        #Print the output of scoring power evluation
        def f(x):
            return x+1
        testdf1.rename(columns={'#code':'code'},inplace=True)
        testdf1.index=testdf1.index.map(f)
        testdf1.style.set_properties(align="right")
        pd.set_option('display.max_columns',None)
        pd.set_option('display.max_rows',None)
        print (testdf1[['code','logKa','score']])
        print ("\nSummary of the scoring power: ===================================")
        print ("The regression equation: logKa = %.2f + %.2f * Score"%(self.dec(float(regr.coef_),2), self.dec(float(regr.intercept_),2)))
        print ("Number of favorable sample (N) = %d"%(num))
        print ("Pearson correlation coefficient (R) = %0.3f"%(self.dec(testr,3)))
        print ("Standard deviation in fitting (SD) = %0.2f"%(self.dec(testsd,2)))
        print ("=================================================================")
        print ("\nTemplate command for running the bootstrap in R program==========\n")
        print ("rm(list=ls());\nrequire(boot);\ndata_all<-read.table(\"%s_processed_score\",header=TRUE);\naa<-c(1:nrow(data_all));\n"%(self.out_file))
        print ("mycore<-function(x,indices)\n{\ndata_1<-matrix(NA,%d,2);\nfor(i in 1:%d);\n    {\n        data_1[i,1]=data_all[x[indices][i],2];\n        data_1[i,2]=data_all[x[indices][i],3];\n    }\n        data_2<-data.frame(data_1);\n        names(data_2)<-c(\"a\",\"b\");\n        cor(data_2$a,data_2$b);\n};\n"%(num,num))
        print ("data.boot<-boot(aa,mycore,R=10000,stype=\"i\",sim=\"ordinary\");\nsink(\"%s-ci.results\");\na<-boot.ci(data.boot,conf=0.9,type=c(\"bca\"));\nprint(a);\nsink();\n"%(self.out_file))
        print ("=================================================================")        


class ranking():
    def __init__(self,out_file):
        self.out_file = out_file
    def cal_PI(self,df):
        dfsorted=df.sort_values(['logKa'],ascending=True)
        W=[]
        WC=[]
        lst=list(dfsorted.index)
        for i in np.arange(0,5):
            xi=lst[i]
            score=float(dfsorted.loc[xi]['score'])
            bindaff=float(dfsorted.loc[xi]['logKa'])
            for j in np.arange(i+1,5):
                xj=lst[j]
                scoretemp=float(dfsorted.loc[xj]['score'])
                bindafftemp=float(dfsorted.loc[xj]['logKa'])
                w_ij=abs(bindaff-bindafftemp)
                W.append(w_ij)
                if score < scoretemp:
                    WC.append(w_ij)
                elif score > scoretemp:
                    WC.append(-w_ij)
                else:
                    WC.append(0)
        pi=float(sum(WC))/float(sum(W))
        return pi


    def dec(self,x,y):
            if y == 2:
                return Decimal(x).quantize(Decimal('0.01'),rounding=ROUND_HALF_UP)
            if y == 3:
                return Decimal(x).quantize(Decimal('0.001'),rounding=ROUND_HALF_UP)
            if y == 4:
                return Decimal(x).quantize(Decimal('0.0001'),rounding=ROUND_HALF_UP)

    def ranking(self,core_data_file, docking_score_file, favorable='positive'):
        aa=pd.read_csv(core_data_file,sep='[,,\t, ]+',engine='python')
        bb=pd.read_csv(docking_score_file,sep='[,,\t, ]+',engine='python')
        aa=aa.drop_duplicates(subset=['#code'],keep='first')
        #Process the data
        testdf1=pd.merge(aa,bb,on='#code')
        if favorable == 'negative':
            testdf1['score']=testdf1['score'].apply(np.negative)
            group=testdf1.groupby('target')            
        elif favorable == 'positive':
            group=testdf1.groupby('target')
        else:
            print('please input negative or positive')
            sys.exit()
        #Get the representative complex in each cluster
        def top(df,n=1,column='logKa'):
            return df.sort_values(by=column)[-n:]
        toptardf=testdf1.groupby('target').apply(top)
        targetlst=toptardf['#code'].tolist()


        #Calculate the Spearman correlation coefficient, Kendall correlation coefficient and Predictive index
        spearman=pd.DataFrame(index=targetlst,columns=['spearman'])
        kendall=pd.DataFrame(index=targetlst,columns=['kendall'])
        PI=pd.DataFrame(index=targetlst,columns=['PI'])
        rankresults=pd.DataFrame(index=range(1,len(targetlst)+1),columns=['Target','Rank1','Rank2','Rank3','Rank4','Rank5'])
        tmp=1
        for i,j in group.__iter__():
            testdf2=group.get_group(i)[['#code','logKa','score']]
            testdf2=testdf2.sort_values('score',ascending=False)
            tartemp=top(testdf2)['#code'].tolist()

            tar=''.join(tartemp)
            if len(testdf2) == 5:
                spearman.loc[tar]['spearman']=testdf2.corr('spearman')['logKa']['score']
                kendall.loc[tar]['kendall']=testdf2.corr('kendall')['logKa']['score']
                PI.loc[tar]['PI']=self.cal_PI(df=testdf2)
                rankresults.loc[tmp]['Rank1']=''.join(testdf2[0:1]['#code'].tolist())
                rankresults.loc[tmp]['Rank2']=''.join(testdf2[1:2]['#code'].tolist())
                rankresults.loc[tmp]['Rank3']=''.join(testdf2[2:3]['#code'].tolist())
                rankresults.loc[tmp]['Rank4']=''.join(testdf2[3:4]['#code'].tolist())
                rankresults.loc[tmp]['Rank5']=''.join(testdf2[4:5]['#code'].tolist())
                rankresults.loc[tmp]['Target']=tar
                tmp+=1
            else:
                spearman.drop(tar,inplace=True)
                kendall.drop(tar,inplace=True)
                PI.drop(tar,inplace=True)
  

        #Print the output of ranking power evluation
        spearmanmean=self.dec(float(spearman['spearman'].sum())/float(spearman.shape[0]),3)
        kendallmean=self.dec(float(kendall['kendall'].sum())/float(kendall.shape[0]),3)
        PImean=self.dec(float(PI['PI'].sum())/float(PI.shape[0]),3)
        tmplen=len(PI)
        spearman.to_csv(self.out_file+'_Spearman.results',sep='\t',index_label='#Target')
        kendall.to_csv(self.out_file+'_Kendall.results',sep='\t',index_label='#Target')
        PI.to_csv(self.out_file+'_PI.results',sep='\t',index_label='#Target')

        #Output results
        rankresults.dropna(axis=0,inplace=True)
        rankresults.style.set_properties(align="right")
        pd.set_option('display.max_columns',None)
        pd.set_option('display.max_rows',None)
        print (rankresults)
        print ("\nSummary of the ranking power: ===========================================")
        print ("The Spearman correlation coefficient (SP) = %0.3f"%(self.dec(spearmanmean,3)))
        print ("The Kendall correlation coefficient (tau) = %0.3f"%(self.dec(kendallmean,3)))
        print ("The Predictive index (PI) = %0.3f"%(self.dec(PImean,3)))
        print ("=========================================================================\n")
        print ("\nTemplate command for running the bootstrap in R program==================\n")
        print ("rm(list=ls());\nrequire(boot);\ndata_all<-read.table(\"%s_Spearman.results\",header=TRUE);\ndata<-as.matrix(data_all[,2]);"%(self.out_file))
        print ("mymean<-function(x,indices) sum(x[indices])/%d;"%(tmplen))
        print ("data.boot<-boot(aa,mymean,R=10000,stype=\"i\",sim=\"ordinary\");\nsink(\"%s_Spearman-ci.results\");\na<-boot.ci(data.boot,conf=0.9,type=c(\"bca\"));\nprint(a);\nsink();\n"%(self.out_file))
        print ("=========================================================================\n") 

if __name__=='__main__':
    core='/root/workshop/docking/PLDock/PLDock/evaluation/criteria/power_docking/CoreSet.dat'
    score='/root/workshop/docking/PLDock/PLDock/evaluation/criteria/power_docking/examples/X-Score.dat'
    dp = scoring('docking_out')
    dp.scoring(core,score)

    core='/root/workshop/docking/PLDock/PLDock/evaluation/criteria/power_docking/CoreSet.dat'
    rscore='/root/workshop/docking/PLDock/PLDock/evaluation/criteria/power_ranking/examples/X-Score.dat'
    dpr = ranking('docking_out')
    dpr.ranking(core,score)