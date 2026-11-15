"""Create publication figures from the archived project and Synapse results."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'Final_Paper/figures';OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],'font.size':8,
                     'axes.labelsize':8,'xtick.labelsize':8,'ytick.labelsize':8,
                     'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,
                     'savefig.dpi':350,'axes.linewidth':.6})
data=pd.read_csv(ROOT/'Assignment3_Practical/outputs_benchmark/per_project_metrics.csv')
fixed=data.loc[data.threshold_policy=='fixed_0.5']
lr=fixed.loc[fixed.condition=='LR_CK6'].set_index('test_project')
base=fixed.loc[fixed.condition=='Dummy'].set_index('test_project')
projects=['ant-1.7','camel-1.6','ivy-2.0','jedit-4.3','poi-3.0']
fig,ax=plt.subplots(figsize=(3.23,2.05),layout='constrained')
for i,p in enumerate(projects):ax.plot([base.loc[p,'average_precision'],lr.loc[p,'average_precision']],[i,i],color='#888888',lw=.8,zorder=1)
ax.scatter([base.loc[p,'average_precision'] for p in projects],range(5),s=20,marker='s',color='#555555',label='Constant score',zorder=2)
ax.scatter([lr.loc[p,'average_precision'] for p in projects],range(5),s=22,color='#165a91',label='LR6',zorder=3)
ax.set_yticks(range(5),['Ant','Camel','Ivy','JEdit','POI']);ax.invert_yaxis()
ax.set_xlim(0,1);ax.set_xticks(np.arange(0,1.01,.2));ax.set_xlabel('Average precision');ax.grid(axis='x',lw=.4,alpha=.3)
ax.legend(loc='upper center',bbox_to_anchor=(.5,1.23),ncol=2,fontsize=7.5,frameon=False)
fig.savefig(OUT/'project_average_precision.png');fig.savefig(OUT/'project_average_precision.svg');plt.close(fig)

metrics=pd.read_csv(ROOT/'Assignment3_Practical/outputs_holdout/holdout_metrics.csv')
r=metrics.loc[(metrics.seed==42)&(metrics.condition=='LR_CK6')&(metrics.population=='purged')&(metrics.threshold_policy=='validation_F1')].iloc[0]
matrix=np.array([[r.tn,r.fp],[r.fn,r.tp]],dtype=int)
fig,ax=plt.subplots(figsize=(3.23,2.28),layout='constrained')
ax.imshow(matrix,cmap='Blues',vmin=0,vmax=100)
ax.set_xticks([0,1],['Not flagged','Flagged']);ax.set_yticks([0,1],['No recorded defect','Recorded defect'])
ax.set_xlabel('Model decision');ax.set_ylabel('Published label')
for i in range(2):
 for j in range(2):ax.text(j,i,str(matrix[i,j]),ha='center',va='center',fontsize=17,color='white' if matrix[i,j]>60 else 'black')
for s in ax.spines.values():s.set_visible(False)
fig.savefig(OUT/'synapse_confusion_matrix.png');fig.savefig(OUT/'synapse_confusion_matrix.svg');plt.close(fig)
print('Generated two figures from recorded CSV results.')
