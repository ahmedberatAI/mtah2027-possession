"""Aggregate-only scientific figure; individual seeds and sample standard deviation."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import exp_seed_balance as exp


def main():
    rows=exp.read(exp.REPORT/'summary.json');raw=exp.read(exp.REPORT/'comparison.json')
    panels=(('metrica@v3','ctrl','Metrica · kontrollü'),('metrica@v3','events','Metrica · olay'),
            ('sc@v3','pp','SkillCorner · oyuncu possession'),('sc@v3','phase','SkillCorner · faz'))
    colors=['#8a8f98','#444b55','#2670ad','#238364']
    fig,axes=plt.subplots(2,2,figsize=(11,7.2))
    for ax,(source,conv,title) in zip(axes.ravel(),panels):
        for i,(kind,color) in enumerate(zip(exp.KINDS,colors)):
            r=next(r for r in rows if (r['model'],r['source'],r['conv'],r['dead'])==(kind,source,conv,'learned'))
            vals=[r['scores_by_seed'][str(s)] for s in exp.SEEDS]
            ax.scatter(i+np.array([-.15,0,.15]),vals,color=color,s=26,alpha=.65,zorder=3)
            ax.errorbar(i,r['mean'],yerr=r['sd'],fmt='s',color=color,markersize=7,capsize=5,lw=1.5,zorder=4)
        ref=next(r for r in raw if (r['model'],r['source'],r['conv'],r['dead'])==('v7',source,conv,'learned'))
        ax.axhline(ref['spellF1'],color='#b57736',ls='--',lw=1.2)
        ax.set_xticks(range(4),['LGB\nnormal','LGB\ndengeli','TCN','BiGRU'])
        ax.set_ylabel('spellF1');ax.set_title(title,loc='left',fontsize=12,pad=12)
        ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True);ax.set_xlim(-.5,3.5)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Üç eğitim seed’i · sabit ölü-top modeli ve B2',fontsize=16,x=.07,ha='left',y=.98)
    fig.legend(handles=[Line2D([0],[0],marker='o',ls='',color='#777',label='Tek seed'),
                        Line2D([0],[0],marker='s',color='#333',label='Ortalama ± seed standart sapması'),
                        Line2D([0],[0],ls='--',color='#b57736',label='v7 referansı')],
               loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.015))
    fig.tight_layout(rect=(.02,.075,.99,.94),h_pad=2.2)
    fig.savefig(exp.REPORT/'seed_comparison.png',dpi=180)
    plt.close(fig)
    print(exp.REPORT/'seed_comparison.png')


if __name__=='__main__':main()
