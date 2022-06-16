"""
Source code (MIT-Licensed) inspired by PDBbind (http://www.pdbbind.org.cn/)
"""

import numpy as np
import sys,os
import pandas as pd
from decimal import *

class screening():
    def __init__(self,out_file):
        self.out_file = out_file

    def top(self,df,n=1,column='logKa'):
            return df.sort_values(by=column)[-n:]

    def dec(self,x,y):
        if y == 2:
                return Decimal(x).quantize(Decimal('0.01'),rounding=ROUND_HALF_UP)
        if y == 3:
                return Decimal(x).quantize(Decimal('0.001'),rounding=ROUND_HALF_UP)
        if y == 0:
                return Decimal(x).quantize(Decimal('0'),rounding=ROUND_HALF_UP)

    def screening(self,core_data_file,docking_score_dir,target_info_file, favorable='positive'):
        aa=pd.read_csv(core_data_file,sep='[,,\t, ]+',engine='python')
        aa=aa.drop_duplicates(subset=['#code'],keep='first')
        bb=pd.read_csv(target_info_file,sep='[,,\t, ]+',skiprows=8,engine='python')
        bb=bb.drop_duplicates(subset=['#T'],keep='first')
        cc=bb.set_index('#T')
        #process data
        toptardf=aa.groupby('target').apply(self.top)
        targetlst2=toptardf['#code'].tolist()
        targetlst=[]
        for i in targetlst2:
            if i in list(set(bb['#T'])):
                targetlst.append(i)
        #define decoy list and cutoff
        decoylst2=[]
        for i in np.arange(1,11):
            decoylst2.extend((cc['L'+str(i)].tolist()))
        decoylst=list(filter(None,list(set(decoylst2))))
        t1=int(self.dec(len(decoylst)*0.01,0))
        t5=int(self.dec(len(decoylst)*0.05,0))
        t10=int(self.dec(len(decoylst)*0.10,0))
        if t1 < 1:
            print("The number of top 1%% ligands is %0.1f,less than 1"%(round(len(decoylst)*0.01)))
            print("In this case, we set the cutoff of top 1%% = 1")
            t1=1
        if t5 < 1:
            print("The number of top 5%% ligands is %0.1f,less than 1"%(round(len(decoylst)*0.01)))
            print("In this case, we set the cutoff of top 5%% = 1")
            t5=1
        if t10 < 1:
            print("The number of top 10%% ligands is %0.1f,less than 1"%(round(len(decoylst)*0.01)))
            print("In this case, we set the cutoff of top 10%% = 1")
            t10=1
        #Build DataFrame to store results
        Top1=pd.DataFrame(index=targetlst,columns=['success'])
        Top5=pd.DataFrame(index=targetlst,columns=['success'])
        Top10=pd.DataFrame(index=targetlst,columns=['success'])
        EF1=pd.DataFrame(index=targetlst,columns=['enrichment'])
        EF5=pd.DataFrame(index=targetlst,columns=['enrichment'])
        EF10=pd.DataFrame(index=targetlst,columns=['enrichment'])
        forwardf=pd.DataFrame(index=range(1,len(targetlst)+1),columns=range(0,t10+1))
        forwardf.rename(columns={0:'Target'},inplace=True)
        tmp=1
        if favorable == 'positive':
            for i in targetlst:
                scoredf=pd.read_csv(docking_score_dir+'/'+str(i)+'_score.dat',sep='[ ,_,\t]+',engine='python')
                scoredf=scoredf[(False^scoredf['#code'].isin(decoylst))]
                group=scoredf.groupby('#code')
                testdf=pd.DataFrame(group['score'].max())
                dfsorted=testdf.sort_values('score',ascending=False)
                for m in range(1,t10+1):
                    forwardf.loc[tmp][m]=''.join(dfsorted[m-1:m].index.tolist())
                forwardf.loc[tmp]['Target']=i
                tmp+=1
                Topligand=cc.loc[i]['L1']
                tartemp=cc.loc[i]
                Allactivelig=list(tartemp.dropna())
                NTBtotal=len(Allactivelig)
                dic={'1':t1,'5':t5,'10':t10}
                for name,j in dic.items():
                    lst=list(dfsorted[0:j].index)
                    varname='Top'+str(name)
                    top=locals()[varname]
                    if Topligand in lst:
                        top.loc[i]['success']=1
                    else:
                        top.loc[i]['success']=0
                    varname2='EF'+str(name)
                    ef=locals()[varname2]
                    ntb=0
                    for lig in Allactivelig:
                        if lig in lst:
                            ntb=ntb+1
                        else:
                            continue
                    efvalue=float(ntb)/(float(NTBtotal)*int(name)*0.01)
                    ef.loc[i]['enrichment']=efvalue        
        elif favorable == 'negative':
            for i in targetlst:
                scoredf=pd.read_csv(docking_score_dir+'/'+str(i)+'_score.dat',sep='[ ,_,\t]+',engine='python')
                scoredf=scoredf[(False^scoredf['#code'].isin(decoylst))]
                group=scoredf.groupby('#code')
                testdf=pd.DataFrame(group['score'].min())
                dfsorted=testdf.sort_values('score',ascending=True)
                for m in range(1,t10+1):
                    forwardf.loc[tmp][m]=''.join(dfsorted[m-1:m].index.tolist())
                forwardf.loc[tmp]['Target']=i
                tmp+=1
                Topligand=cc.loc[i]['L1']
                tartemp=cc.loc[i]
                Allactivelig=list(tartemp.dropna())
                NTBtotal=len(Allactivelig)
                dic={'1':t1,'5':t5,'10':t10}
                for name,j in dic.items():
                    lst=list(dfsorted[0:j].index)
                    varname='Top'+str(name)
                    top=locals()[varname]
                    if Topligand in lst:
                        top.loc[i]['success']=1
                    else:
                        top.loc[i]['success']=0
                    varname2='EF'+str(name)
                    ef=locals()[varname2]
                    ntb=0
                    for lig in Allactivelig:
                        if lig in lst:
                            ntb=ntb+1
                        else:
                            continue
                    efvalue=float(ntb)/(float(NTBtotal)*int(name)*0.01)
                    ef.loc[i]['enrichment']=efvalue
        else:
            print ('please input negative or positive')
            sys.exit()

        #Calculate success rates  and enrichment factors
        top1success=self.dec(float(Top1['success'].sum())/float(Top1.shape[0]),3)*100
        top5success=self.dec(float(Top5['success'].sum())/float(Top5.shape[0]),3)*100
        top10success=self.dec(float(Top10['success'].sum())/float(Top10.shape[0]),3)*100
        ef1factor=self.dec(float(EF1['enrichment'].sum())/float(EF1.shape[0]),2)
        ef5factor=self.dec(float(EF5['enrichment'].sum())/float(EF5.shape[0]),2)
        ef10factor=self.dec(float(EF10['enrichment'].sum())/float(EF10.shape[0]),2)


        #Print the output of forward screening power evluation
        tmplen=len((Top1))
        forwardf.style.set_properties(align="right")
        pd.set_option('display.max_columns',None)
        pd.set_option('display.max_rows',None)
        print (forwardf)
        print ("\nSummary of the forward screening power: =========================================================")
        print ("Average enrichment factor among top 1%% = %0.2f"%(ef1factor))
        print ("Average enrichment factor among top 5%% = %0.2f"%(ef5factor))
        print ("Average enrichment factor among top 10%% = %0.2f"%(ef10factor))
        print ("The best ligand is found among top 1%% candidates for %2d cluster(s); success rate = %0.1f%%"%(Top1['success'].sum(),top1success))
        print ("The best ligand is found among top 5%% candidates for %2d cluster(s); success rate = %0.1f%%"%(Top5['success'].sum(),top5success))
        print ("The best ligand is found among top 10%% candidates for %2d cluster(s); success rate = %0.1f%%"%(Top10['success'].sum(),top10success))
        print ("================================================================================================")
        print ("\nTemplate command for running the bootstrap in R program=========================================\n")
        print ("rm(list=ls());\nrequire(boot);\ndata_all<-read.table(\"%s_Top1.results\",header=TRUE);\ndata<-as.matrix(data_all[,2]);"%(self.out_file))
        print ("mymean<-function(x,indices) sum(x[indices])/%d;"%(tmplen))
        print ("data.boot<-boot(aa,mymean,R=10000,stype=\"i\",sim=\"ordinary\");\nsink(\"%s_Top1-ci.results\");\na<-boot.ci(data.boot,conf=0.9,type=c(\"bca\"));\nprint(a);\nsink();\n"%(self.out_file))
        print ("===============================================================================================\n")

        Top1.to_csv(self.out_file+'_Top1.dat',sep='\t',index_label='#Target')
        Top5.to_csv(self.out_file+'_Top5.dat',sep='\t',index_label='#Target')
        Top10.to_csv(self.out_file+'_Top10.dat',sep='\t',index_label='#Target')
        EF1.to_csv(self.out_file+'_EF1.dat',sep='\t',index_label='#Target')
        EF5.to_csv(self.out_file+'_EF5.dat',sep='\t',index_label='#Target')
        EF10.to_csv(self.out_file+'_EF10.dat',sep='\t',index_label='#Target')                

    def reverse_screening(self,core_data_file,docking_score_dir,ligand_info_file, favorable='positive'):
        aa=pd.read_csv(core_data_file,sep='[,,\t, ]+',engine='python')
        aa=aa.drop_duplicates(subset=['#code'],keep='first')
        bb=pd.read_csv(ligand_info_file,sep='[,,\t, ]+',skiprows=8,engine='python')
        bb=bb.drop_duplicates(subset=['#code'],keep='first')
        cc=bb.set_index('#code')

        #process data
        #pdb=aa['code']
        pdb=[]
        pdbtmp=aa['#code']
        codelst=[]
        targettmp=cc.groupby('group')
        targetlst=[]
        for n,m in targettmp.__iter__():
            if len(targettmp.get_group(n)) == 5:
                targetlst.extend(targettmp.get_group(n)['T1'].tolist())
                codelst.extend(targettmp.get_group(n).index.tolist())
        targetlst=list(set(targetlst))
        for c in pdbtmp:
            if c in codelst:
                pdb.append(c)
        t1=int(self.dec(len(targetlst)*0.01,0))
        t5=int(self.dec(len(targetlst)*0.05,0))
        t10=int(self.dec(len(targetlst)*0.10,0))
        print(len(targetlst))
        if t1 < 1:
            print("The number of top 1%% targets is %0.1f,less than 1"%(round(len(targetlst)*0.01)))
            print("In this case, we set the cutoff of top 1%% = 1")
            t1=1
        if t5 < 1:
            print("The number of top 5%% targets is %0.1f,less than 1"%(round(len(targetlst)*0.05)))
            print("In this case, we set the cutoff of top 5%% = 1")
            t5=1
        if t10 < 1:
            print("The number of top 10%% targets is %0.1f,less than 1"%(round(len(targetlst)*0.10)))
            print("In this case, we set the cutoff of top 10%% = 1")
            t10=1

        #toptardf=aa.groupby('target').apply(top)
        #targetlst=toptardf['code'].tolist()
        Top1=pd.DataFrame(index=pdb,columns=['success'])
        Top5=pd.DataFrame(index=pdb,columns=['success'])
        Top10=pd.DataFrame(index=pdb,columns=['success'])
        Ligdf=pd.DataFrame()
        reversedf=pd.DataFrame(index=range(1,len(pdb)+1),columns=range(0,t10+1))
        reversedf.rename(columns={0:'code'},inplace=True)
        tmp=1
        dic={'1':t1, '5':t5, '10':t10}
        if favorable == 'positive':
            for i in targetlst:
                scoredf=pd.read_csv(docking_score_dir+'/'+str(i)+'_score.dat',sep='[ ,_,\t]+',engine='python')
                #print(scoredf)
                group=scoredf.groupby('#code')
                testdf=pd.DataFrame(group['score'].max())
                testdf['T1']=i
                Ligdf=Ligdf.append(testdf)
            Ligdf['ligname']=list(Ligdf.index)
            #print(Ligdf)
            grouplig=Ligdf.groupby('ligname')
            for l in pdb:
                grouptemp=grouplig.get_group(l)
                #print(grouptemp)
                dfsorted=grouptemp.sort_values('score',ascending=False)
                for m in range(1,t10+1):
                    reversedf.loc[tmp][m]=''.join(dfsorted[m-1:m]['T1'].tolist())
                reversedf.loc[tmp]['code']=l
                #print(reversedf)
                tmp+=1
                Toptar=cc.loc[l]['T1']
                for name,j in dic.items():
                    lst=list(dfsorted[0:j]['T1'])
                    varname='Top'+str(name)
                    top=locals()[varname]
                    if Toptar in lst:
                        top.loc[l]['success']=1
                    else:
                        top.loc[l]['success']=0
        elif favorable == 'negative':
            for i in targetlst:
                scoredf=pd.read_csv(docking_score_dir+'/'+str(i)+'_score.dat',sep='[ ,_,\t]+',engine='python')
                group=scoredf.groupby('#code')
                testdf=pd.DataFrame(group['score'].min())
                testdf['T1']=i
                Ligdf=Ligdf.append(testdf)
                Ligdf['ligname']=list(Ligdf.index)
                grouplig=Ligdf.groupby('ligname')
            for l in pdb:
                grouptemp=grouplig.get_group(l)
                dfsorted=grouptemp.sort_values('score',ascending=True)
                for m in range(1,t10+1):
                    reversedf.loc[tmp][m]=''.join(dfsorted[m-1:m]['T1'].tolist())
                reversedf.loc[tmp]['code']=l
                tmp+=1
                Toptar=cc.loc[l]['T1']
                for name,j in dic.items():
                    lst=list(dfsorted[0:j]['T1'])
                    varname='Top'+str(name)
                    top=locals()[varname]
                    if Toptar in lst:
                        top.loc[l]['success']=1
                    else:
                        top.loc[l]['success']=0
        else:
            print ('please input negative or positive')
            sys.exit()
        #Calculate success rates
        reversedf.style.set_properties(align="right")
        pd.set_option('display.max_columns',None)
        pd.set_option('display.max_rows',None)
        print (reversedf)
        top1success=self.dec(float(Top1['success'].sum())/float(Top1.shape[0]),3)*100
        top5success=self.dec(float(Top5['success'].sum())/float(Top5.shape[0]),3)*100
        top10success=self.dec(float(Top10['success'].sum())/float(Top10.shape[0]),3)*100
        tmplen=len(Top1)
        print ("\nSummary of the reverse screening power: =========================================================")
        print ("The best target is found among top 1%% candidates for %2d ligand(s); success rate = %0.1f%%"%(Top1['success'].sum(),top1success))
        print ("The best target is found among top 5%% candidates for %2d ligand(s); success rate = %0.1f%%"%(Top5['success'].sum(),top5success))
        print ("The best target is found among top 10%% candidates for %2d ligand(s); success rate = %0.1f%%"%(Top10['success'].sum(),top10success))
        print ("=================================================================================================")
        print ("\nTemplate command for running the bootstrap in R program=========================================\n")
        print ("rm(list=ls());\nrequire(boot);\ndata_all<-read.table(\"%s_Top1.results\",header=TRUE);\ndata<-as.matrix(data_all[,2]);"%(self.out_file))
        print ("mymean<-function(x,indices) sum(x[indices])/%d;"%(tmplen))
        print ("data.boot<-boot(aa,mymean,R=10000,stype=\"i\",sim=\"ordinary\");\nsink(\"%s_Top1-ci.results\");\na<-boot.ci(data.boot,conf=0.9,type=c(\"bca\"));\nprint(a);\nsink();\n"%(self.out_file))
        print ("===============================================================================================\n")
        Top1.to_csv(self.out_file+'_Top1.dat',sep='\t',index_label='#code')
        Top5.to_csv(self.out_file+'_Top5.dat',sep='\t',index_label='#code')
        Top10.to_csv(self.out_file+'_Top10.dat',sep='\t',index_label='#code')





if __name__=='__main__':
    core='/root/workshop/docking/PLDock/PLDock/evaluation/criteria/power_docking/CoreSet.dat'
    score='/root/workshop/docking/PLDock/PLDock/evaluation/criteria/power_screening/examples/X-Score'
    target='/root/workshop/docking/PLDock/PLDock/evaluation/criteria/power_screening/TargetInfo.dat'
    ligand='/root/workshop/docking/PLDock/PLDock/evaluation/criteria/power_screening/LigandInfo.dat'
    dp = screening('revdocking_out')
    #dp.screening(core,score,target)
    dp.reverse_screening(core,score,ligand)