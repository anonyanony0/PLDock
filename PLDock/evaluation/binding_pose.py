"""
Source code (MIT-Licensed) inspired by PDBbind (http://www.pdbbind.org.cn/)
"""

import numpy as np
import sys
import pandas as pd
from decimal import *

class binding_pose():
    def __init__(self,out_file):
        self.out_file = out_file
    def dec(seif,x,y):
            if y == 2:
                    return Decimal(x).quantize(Decimal('0.01'),rounding=ROUND_HALF_UP)
            if y == 3:
                    return Decimal(x).quantize(Decimal('0.001'),rounding=ROUND_HALF_UP)
            if y == 4:
                    return Decimal(x).quantize(Decimal('0.0001'),rounding=ROUND_HALF_UP)

    def calcul(self, core_data_file, core_rmsd_dir, docking_score_dir, favorable='positive', rmsd_cut=2.):
        aa=pd.read_csv(core_data_file,sep='[,,\t, ]+',engine='python')
        pdb=aa['#code']
        aa=aa.drop_duplicates(subset=['#code'],keep='first')
        Top1=pd.DataFrame(index=pdb,columns=['success'])
        Top2=pd.DataFrame(index=pdb,columns=['success'])
        Top3=pd.DataFrame(index=pdb,columns=['success'])
        SP2=pd.DataFrame(index=pdb,columns=['spearman'])
        SP3=pd.DataFrame(index=pdb,columns=['spearman'])
        SP4=pd.DataFrame(index=pdb,columns=['spearman'])
        SP5=pd.DataFrame(index=pdb,columns=['spearman'])
        SP6=pd.DataFrame(index=pdb,columns=['spearman'])
        SP7=pd.DataFrame(index=pdb,columns=['spearman'])
        SP8=pd.DataFrame(index=pdb,columns=['spearman'])
        SP9=pd.DataFrame(index=pdb,columns=['spearman'])
        SP10=pd.DataFrame(index=pdb,columns=['spearman'])
        dockresults=pd.DataFrame(index=range(1,len(pdb)+1),columns=['code','Rank1','RMSD1','Rank2','RMSD2','Rank3','RMSD3'])        
        tmp=1
        if favorable == 'positive':
            for i in pdb:
                rmsddf=pd.read_csv(core_rmsd_dir+'/'+str(i)+'_rmsd.dat',sep='[,, ,\t]+',engine='python')
                scoredf=pd.read_csv(docking_score_dir+'/'+str(i)+'_score.dat',sep='[,, ,\t]+',engine='python')
                testdf=pd.merge(rmsddf,scoredf,on='#code')
                #dfsorted=testdf.sort_values(by=['score','rmsd'],ascending=[False,True])
                dfsorted=testdf.sort_values(by=['score'],ascending=[False])
                dockresults.loc[tmp]['Rank1']=''.join(dfsorted[0:1]['#code'])
                dockresults.loc[tmp]['RMSD1']=float(dfsorted[0:1]['rmsd'])
                dockresults.loc[tmp]['Rank2']=''.join(dfsorted[1:2]['#code'])
                dockresults.loc[tmp]['RMSD2']=float(dfsorted[1:2]['rmsd'])
                dockresults.loc[tmp]['Rank3']=''.join(dfsorted[2:3]['#code'])
                dockresults.loc[tmp]['RMSD3']=float(dfsorted[2:3]['rmsd'])
                dockresults.loc[tmp]['code']=i
                tmp+=1
                for j in np.arange(1,4):
                    minrmsd=dfsorted[0:j]['rmsd'].min()
                    varname='Top'+str(j)
                    top=locals()[varname]
                    if minrmsd <=rmsd_cut:
                        top.loc[i]['success']=1
                    else:
                        top.loc[i]['success']=0
                for s in np.arange(2,11):
                    sptemp=testdf[testdf.rmsd <= s]
                    varname2='SP'+str(s)
                    sp=locals()[varname2]
                    if float(sptemp.shape[0]) >= 5:
                        sp.loc[i]['spearman']=np.negative(sptemp.corr('spearman')['rmsd']['score'])
                    else:
                        continue
        elif favorable == 'negative':
            for i in pdb:
                rmsddf=pd.read_csv(core_rmsd_dir+'/'+str(i)+'_rmsd.dat',sep='[,, ,\t]+',engine='python')
                scoredf=pd.read_csv(docking_score_dir+'/'+str(i)+'_score.dat',sep='[,, ,\t]+',engine='python')
                testdf=pd.merge(rmsddf,scoredf,on='#code')
                #dfsorted=testdf.sort_values(['score','rmsd'],ascending=[True,True])
                dfsorted=testdf.sort_values(by=['score'],ascending=[True])
                dockresults.loc[tmp]['Rank1']=''.join(dfsorted[0:1]['#code'])
                dockresults.loc[tmp]['RMSD1']=float(dfsorted[0:1]['rmsd'])
                dockresults.loc[tmp]['Rank2']=''.join(dfsorted[1:2]['#code'])
                dockresults.loc[tmp]['RMSD2']=float(dfsorted[1:2]['rmsd'])
                dockresults.loc[tmp]['Rank3']=''.join(dfsorted[2:3]['#code'])
                dockresults.loc[tmp]['RMSD3']=float(dfsorted[2:3]['rmsd'])
                dockresults.loc[tmp]['code']=i
                tmp+=1
                for j in np.arange(1,4):
                    minrmsd=dfsorted[0:j]['rmsd'].min()
                    varname='Top'+str(j)
                    top=locals()[varname]
                    if minrmsd <= rmsd_cut:
                            top.loc[i]['success']=1
                    else:
                            top.loc[i]['success']=0
                for s in np.arange(2,11):
                    sptemp=testdf[testdf.rmsd <= s]
                    varname2='SP'+str(s)
                    sp=locals()[varname2]
                    if float(sptemp.shape[0]) >= 5:
                        sp.loc[i]['spearman']=sptemp.corr('spearman')['rmsd']['score']
                    else:
                        continue
                        # sp.drop(sp.index[[i]],inplace=True)
        else:
            print('please input negative or positive')
            sys.exit()

        #Calculate success rates and spearman correlation coefficient
        SP2=SP2.dropna(subset=['spearman'])
        SP3=SP3.dropna(subset=['spearman'])
        SP4=SP4.dropna(subset=['spearman'])
        SP5=SP5.dropna(subset=['spearman'])
        SP6=SP6.dropna(subset=['spearman'])
        SP7=SP7.dropna(subset=['spearman'])
        SP8=SP8.dropna(subset=['spearman'])
        SP9=SP9.dropna(subset=['spearman'])
        SP10=SP10.dropna(subset=['spearman'])
        top1success=self.dec (float(Top1['success'].sum())/float(Top1.shape[0]),3)*100
        top2success=self.dec (float(Top2['success'].sum())/float(Top2.shape[0]),3)*100
        top3success=self.dec (float(Top3['success'].sum())/float(Top3.shape[0]),3)*100
        sp2=self.dec (float(SP2['spearman'].sum())/float(SP2.shape[0]),3)
        sp3=self.dec (float(SP3['spearman'].sum())/float(SP3.shape[0]),3)
        sp4=self.dec (float(SP4['spearman'].sum())/float(SP4.shape[0]),3)
        sp5=self.dec (float(SP5['spearman'].sum())/float(SP5.shape[0]),3)
        sp6=self.dec (float(SP6['spearman'].sum())/float(SP6.shape[0]),3)
        sp7=self.dec (float(SP7['spearman'].sum())/float(SP7.shape[0]),3)
        sp8=self.dec (float(SP8['spearman'].sum())/float(SP8.shape[0]),3)
        sp9=self.dec (float(SP9['spearman'].sum())/float(SP9.shape[0]),3)
        sp10=self.dec (float(SP10['spearman'].sum())/float(SP10.shape[0]),3)
        tmplen=len(Top1)

        #Print the output of docking power evluation
        dockresults['RMSD1']=dockresults['RMSD1'].map(lambda x:('%.2f')%x)
        dockresults['RMSD2']=dockresults['RMSD2'].map(lambda x:('%.2f')%x)
        dockresults['RMSD3']=dockresults['RMSD3'].map(lambda x:('%.2f')%x)
        dockresults.style.set_properties(align="right")
        pd.set_option('display.max_columns',None)
        pd.set_option('display.max_rows',None)
        print (dockresults)
        print ("\nSummary of the docking power: ========================================")
        print ("Among the top1 binding pose ranked by the given scoring function:")
        print ("Number of correct binding poses = %d, success rate = %0.1f%%"%(Top1['success'].sum(),top1success))
        print ("Among the top2 binding pose ranked by the given scoring function:")
        print ("Number of correct binding poses = %d, success rate = %0.1f%%"%(Top2['success'].sum(),top2success))
        print ("Among the top3 binding pose ranked by the given scoring function:")
        print ("Number of correct binding poses = %d, success rate = %0.1f%%"%(Top3['success'].sum(),top3success))
        print ("Spearman correlation coefficient in rmsd range [0-2]: %0.3f"%(self.dec (sp2,3)))
        print ("Spearman correlation coefficient in rmsd range [0-3]: %0.3f"%(self.dec (sp3,3)))
        print ("Spearman correlation coefficient in rmsd range [0-4]: %0.3f"%(self.dec (sp4,3)))
        print ("Spearman correlation coefficient in rmsd range [0-5]: %0.3f"%(self.dec (sp5,3)))
        print ("Spearman correlation coefficient in rmsd range [0-6]: %0.3f"%(self.dec (sp6,3)))
        print ("Spearman correlation coefficient in rmsd range [0-7]: %0.3f"%(self.dec (sp7,3)))
        print ("Spearman correlation coefficient in rmsd range [0-8]: %0.3f"%(self.dec (sp8,3)))
        print ("Spearman correlation coefficient in rmsd range [0-9]: %0.3f"%(self.dec (sp9,3)))
        print ("Spearman correlation coefficient in rmsd range [0-10]: %0.3f"%(self.dec (sp10,3)))
        print ("======================================================================\n")
        print ("\nTemplate command for running the bootstrap in R program===============\n")
        print ("rm(list=ls());\nrequire(boot);\ndata_all<-read.table(\"%s_Top1.results\",header=TRUE);\ndata<-as.matrix(data_all[,2]);"%(self.out_file))
        print ("mymean<-function(x,indices) sum(x[indices])/%d;"%(tmplen))
        print ("data.boot<-boot(aa,mymean,R=10000,stype=\"i\",sim=\"ordinary\");\nsink(\"%s_Top1-ci.results\");\na<-boot.ci(data.boot,conf=0.9,type=c(\"bca\"));\nprint(a);\nsink();\n"%(self.out_file))
        print ("========================================================================\n")

        Top1.to_csv(self.out_file+'_Top1.dat',sep='\t')
        Top2.to_csv(self.out_file+'_Top2.dat',sep='\t')
        Top3.to_csv(self.out_file+'_Top3.dat',sep='\t')

if __name__=='__main__':
    core='/root/workshop/docking/PLDock/PLDock/evaluation/criteria/power_docking/CoreSet.dat'
    score='/root/workshop/docking/PLDock/PLDock/evaluation/criteria/power_docking/examples/X-Score'
    rmsd='/root/workshop/docking/PLDock/PLDock/evaluation/criteria/decoys_docking'
    dp = binding_pose('docking_out')
    dp.calcul(core,rmsd,score)