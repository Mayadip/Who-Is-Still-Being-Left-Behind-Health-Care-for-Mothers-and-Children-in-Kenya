"""
Equity in maternal and child health service use in Kenya
Secondary analysis of the 2025-26 Kenya DHS children's recode (KEKR91FL.dta)
Author: Maya Biswas

Run:  python kenya_mch_equity.py KEKR91FL.dta
Needs: pandas, numpy, scipy, matplotlib
Outputs: results.json (all estimates + regression), fig1-fig4.png

Steps
 1. Load KR file, attach weights (v005/1e6), map 47 counties to 8 former provinces
 2. Build indicators following the DHS Guide to Statistics
 3. Weighted proportions with Taylor-linearised 95% CIs (clusters v021, strata v022)
 4. Weighted logistic regression for facility delivery, cluster-robust SEs
 5. Charts
"""
import pandas as pd, numpy as np, json
from scipy import stats

import sys
F = sys.argv[1] if len(sys.argv) > 1 else 'KEKR91FL.dta'   # path to the KDHS children's recode
cols = ['v005','v021','v022','v024','v025','v106','v190','v012','m14','m13','m15','m3a','m3b',
        'm2a','m2b','m17','m62','m63','m66','m67','b19','b5','midx','bord','v467d','v467c',
        'h2','h8','h53','h9']
d = pd.read_stata(F, columns=cols)
d['w'] = d.v005 / 1e6

PROV = {
 'Nairobi':['nairobi'],
 'Central':['nyandarua','nyeri','kirinyaga',"murang'a",'kiambu'],
 'Coast':['mombasa','kwale','kilifi','tana river','lamu','taita taveta'],
 'Eastern':['marsabit','isiolo','meru','tharaka-nithi','embu','kitui','machakos','makueni'],
 'North Eastern':['garissa','wajir','mandera'],
 'Nyanza':['siaya','kisumu','homa bay','migori','kisii','nyamira'],
 'Rift Valley':['turkana','west pokot','samburu','trans nzoia','uasin gishu','elgeyo-marakwet','nandi',
                'baringo','laikipia','nakuru','narok','kajiado','kericho','bomet'],
 'Western':['kakamega','vihiga','bungoma','busia']}
cmap = {c:p for p,cs in PROV.items() for c in cs}
d['county'] = d.v024.astype(str)
d['province'] = d.county.map(cmap)
assert d.province.notna().all()

def num(s):
    return pd.to_numeric(s.astype(object).where(~s.astype(str).isin(["no antenatal visits"]), 0), errors='coerce')

# ---------- indicators ----------
last = d.midx == 1                                   # most recent birth in last 3 yrs
anc_n = num(d.m14)
d['anc4'] = np.where(last, (anc_n >= 4).astype(float), np.nan)
m13 = pd.to_numeric(d.m13.astype(object), errors='coerce')
d['anc_early'] = np.where(last, (m13 <= 3).astype(float), np.nan)
d['anc_skilled'] = np.where(last, ((d.m2a == 'yes') | (d.m2b == 'yes')).astype(float), np.nan)

births = d.m15.notna()                               # all live births in last 3 yrs
home = d.m15.astype(str).isin(["respondent's home", 'other home', 'other'])
d['facility'] = np.where(births, (~home).astype(float), np.nan)
d['sba'] = np.where(births, ((d.m3a == 'yes') | (d.m3b == 'yes')).astype(float), np.nan)
d['csec'] = np.where(births, (d.m17 == 'yes').astype(float), np.nan)

def within2d(v):
    s = str(v)
    if s in ('zero hours',): return True
    if s == 'one day': return True
    try: x = float(s)
    except: return False
    return (100 <= x < 148) or (200 <= x <= 202)
pnc = (((d.m62 == 'yes') & d.m63.astype(object).apply(within2d).astype(bool)) | ((d.m66 == 'yes') & d.m67.astype(object).apply(within2d).astype(bool)))
d['pnc2d'] = np.where(last, pnc.astype(float), np.nan)

vac = lambda s: s.astype(str).isin(['vaccination date on card','reported by mother','vaccination marked on card'])
age1223 = (d.b19 >= 12) & (d.b19 <= 23) & (d.b5 == 'yes')
d['penta3'] = np.where(age1223, vac(d.h53).astype(float), np.nan)
d['mcv1'] = np.where(age1223, vac(d.h9).astype(float), np.nan)
d['dist_problem'] = np.where(last, (d.v467d == 'big problem').astype(float), np.nan)
d['money_problem'] = np.where(last, (d.v467c == 'big problem').astype(float), np.nan)

# ---------- design-based proportion with Taylor linearisation ----------
def est(df, y):
    s = df[df[y].notna()]
    w, v = s.w.values, s[y].values
    W = w.sum(); p = (w*v).sum()/W
    z = w*(v-p)/W
    t = pd.DataFrame({'h':s.v022.astype(str).values,'c':s.v021.values,'z':z}).groupby(['h','c']).z.sum().reset_index()
    var = 0
    for h,g in t.groupby('h'):
        n = len(g)
        if n > 1: var += n/(n-1)*((g.z-g.z.mean())**2).sum()
    se = np.sqrt(var)
    return dict(p=round(100*p,1), lo=round(100*max(0,p-1.96*se),1), hi=round(100*min(1,p+1.96*se),1), n=int(len(s)))

IND = ['anc4','anc_early','anc_skilled','facility','sba','csec','pnc2d','penta3','mcv1','dist_problem','money_problem']
res = {'national': {i: est(d,i) for i in IND}}
for g in ['v025','v190','v106','province']:
    res[g] = {}
    for lv in sorted(d[g].dropna().unique(), key=str):
        res[g][str(lv)] = {i: est(d[d[g]==lv], i) for i in ['anc4','facility','sba','pnc2d','penta3','mcv1']}
res['county_facility'] = {c: est(d[d.county==c],'facility') for c in d.county.unique()}

# ---------- weighted logistic regression, cluster-robust SE ----------
m = d[last & d.facility.notna() & d.anc4.notna()].copy()
m['age_b'] = m.v012 - m.b19/12
m['age_grp'] = pd.cut(m.age_b, [0,20,35,60], right=False, labels=['<20','20-34','35+'])
m['province'] = m.province.replace({'Nairobi':'Nairobi & Central','Central':'Nairobi & Central'})
m['bo'] = pd.cut(m.bord, [0,1,3,5,30], labels=['1','2-3','4-5','6+'])
spec = [('v025','urban'), ('v106','no education'), ('v190','poorest'), ('age_grp','20-34'),
        ('bo','1'), ('province','Rift Valley')]
X = [np.ones(len(m))]; names = ['const']
for v, ref in spec:
    for lv in [l for l in m[v].astype(str).unique() if l != ref]:
        if lv == 'nan': continue
        X.append((m[v].astype(str) == lv).astype(float).values); names.append(f'{v}:{lv}')
X.append(m.anc4.values); names.append('anc4:yes')
X.append(m.dist_problem.values); names.append('distance big problem:yes')
X = np.column_stack(X); y = m.facility.values; w = m.w.values * len(m)/m.w.sum()
order = sorted(range(1,len(names)), key=lambda i: names[i])
beta = np.zeros(X.shape[1])
for _ in range(50):
    mu = 1/(1+np.exp(-X@beta)); Wd = w*mu*(1-mu)
    H = X.T @ (X*Wd[:,None]); g = X.T @ (w*(y-mu))
    step = np.linalg.solve(H, g); beta += step
    if np.abs(step).max() < 1e-9: break
mu = 1/(1+np.exp(-X@beta)); Hinv = np.linalg.inv(H)
sc = X*(w*(y-mu))[:,None]
S = pd.DataFrame(sc); S['c'] = m.v021.values; U = S.groupby('c').sum().values
G = len(U); meat = (U - U.mean(0)).T @ (U - U.mean(0)) * G/(G-1)
V = Hinv @ meat @ Hinv; se = np.sqrt(np.diag(V))
reg = []
for i,nm in enumerate(names):
    if i == 0: continue
    reg.append(dict(var=nm, OR=round(float(np.exp(beta[i])),2), lo=round(float(np.exp(beta[i]-1.96*se[i])),2),
                    hi=round(float(np.exp(beta[i]+1.96*se[i])),2), p=float(2*stats.norm.sf(abs(beta[i]/se[i])))))
res['regression'] = dict(n=int(len(m)), clusters=int(G), rows=reg)
res['meta'] = dict(n_births=int(births.sum()), n_last=int(last.sum()), n_1223=int(age1223.sum()),
                   n_counties=int(d.county.nunique()), n_psu=int(d.v021.nunique()))
json.dump(res, open('results.json','w'), indent=1)
print(json.dumps(res['national'], indent=0)); print(json.dumps(res['meta']))
for r in reg: print(r)


# ================= CHARTS =================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r = res
BLUE, ORANGE, INK, INK2, MUTED, SURF = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#898781', '#fcfcfb'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.edgecolor':MUTED,'axes.labelcolor':INK2,
                     'xtick.color':INK2,'ytick.color':INK2,'figure.facecolor':SURF,'axes.facecolor':SURF})
def clean(ax):
    for s in ['top','right']: ax.spines[s].set_visible(False)
    ax.grid(axis='x', color='#e6e5e1', lw=0.8); ax.set_axisbelow(True)

# 1 continuum
n = r['national']
items = [('ANC: 4+ visits','anc4'),('ANC: first visit in 1st trimester','anc_early'),('Delivered in a health facility','facility'),
         ('Skilled birth attendant','sba'),('Mother checked within 2 days of birth','pnc2d'),
         ('Child 12–23 m: Penta 3','penta3'),('Child 12–23 m: Measles 1','mcv1')]
fig, ax = plt.subplots(figsize=(7.2,3.6))
vals = [n[k]['p'] for _,k in items][::-1]; labs = [l for l,_ in items][::-1]
ax.barh(labs, vals, color=BLUE, height=0.55)
for i,v in enumerate(vals): ax.text(v+1, i, f'{v:.1f}%', va='center', color=INK, fontsize=9)
ax.set_xlim(0,105); ax.set_xlabel('Percent (weighted)'); clean(ax)
ax.set_title('Figure 1. Maternal and child health service coverage, Kenya', loc='left', color=INK, fontsize=11)
fig.tight_layout(); fig.savefig('fig1.png', dpi=200); plt.close()

# 2 wealth
q = ['poorest','poorer','middle','richer','richest']
fig, ax = plt.subplots(figsize=(7.2,3.4))
x = range(len(q)); wdt = 0.36
a = [r['v190'][k]['anc4']['p'] for k in q]; f = [r['v190'][k]['facility']['p'] for k in q]
ax.bar([i-wdt/2-0.01 for i in x], a, wdt, color=BLUE, label='ANC 4+ visits')
ax.bar([i+wdt/2+0.01 for i in x], f, wdt, color=ORANGE, label='Facility delivery')
for i in x:
    ax.text(i-wdt/2-0.01, a[i]+1.5, f'{a[i]:.0f}', ha='center', fontsize=8.5, color=INK)
    ax.text(i+wdt/2+0.01, f[i]+1.5, f'{f[i]:.0f}', ha='center', fontsize=8.5, color=INK)
ax.set_xticks(list(x)); ax.set_xticklabels([s.capitalize() for s in q]); ax.set_ylim(0,122); ax.set_yticks([0,20,40,60,80,100]); ax.set_ylabel('Percent')
ax.legend(frameon=False, loc='upper left', ncol=2);
for s in ['top','right']: ax.spines[s].set_visible(False)
ax.grid(axis='y', color='#e6e5e1', lw=0.8); ax.set_axisbelow(True)
ax.set_title('Figure 2. ANC 4+ and facility delivery by household wealth quintile', loc='left', color=INK, fontsize=11)
fig.tight_layout(); fig.savefig('fig2.png', dpi=200); plt.close()

# 3 province
pv = sorted(r['province'].items(), key=lambda kv: kv[1]['facility']['p'])
fig, ax = plt.subplots(figsize=(7.2,3.4))
vals = [v['facility']['p'] for _,v in pv]; labs = [k for k,_ in pv]
ax.barh(labs, vals, color=BLUE, height=0.55)
for i,v in enumerate(vals): ax.text(v+1, i, f'{v:.1f}%', va='center', fontsize=9, color=INK)
nat = n['facility']['p']; ax.axvline(nat, color=INK2, lw=1, ls='--')
ax.text(nat-1, len(labs)-0.35, f'National {nat:.1f}%', ha='right', fontsize=8.5, color=INK2)
ax.set_xlim(0,110); ax.set_xlabel('Percent of births in a health facility'); clean(ax)
ax.set_title('Figure 3. Facility delivery by former province', loc='left', color=INK, fontsize=11)
fig.tight_layout(); fig.savefig('fig3.png', dpi=200); plt.close()

# 4 forest
pick = [('v106:primary','Education: primary'),('v106:secondary','Education: secondary'),('v106:higher','Education: higher'),
        ('v190:poorer','Wealth: poorer'),('v190:middle','Wealth: middle'),('v190:richer','Wealth: richer'),
        ('v025:rural','Rural residence'),('bo:6+','Birth order 6+'),('anc4:yes','Had 4+ ANC visits'),
        ('distance big problem:yes','Distance a big problem')]
rows = {x['var']:x for x in r['regression']['rows']}
fig, ax = plt.subplots(figsize=(7.2,3.9))
for i,(k,l) in enumerate(pick[::-1]):
    o = rows[k]; ax.plot([o['lo'],o['hi']],[i,i], color=BLUE, lw=2); ax.plot(o['OR'], i, 'o', color=BLUE, ms=6)
    ax.text(60, i, f"{o['OR']:.2f} ({o['lo']:.2f}–{o['hi']:.2f})", va='center', fontsize=8.5, color=INK)
ax.set_yticks(range(len(pick))); ax.set_yticklabels([l for _,l in pick[::-1]])
ax.set_xscale('log'); ax.axvline(1, color=INK2, lw=1, ls='--'); ax.set_xlim(0.2, 300)
ax.set_xticks([0.25,0.5,1,2,5,10,20,50]); ax.set_xticklabels(['0.25','0.5','1','2','5','10','20','50'])
ax.set_xlabel('Adjusted odds ratio (log scale)'); clean(ax)
ax.set_title('Figure 4. Adjusted odds of facility delivery (95% CI)', loc='left', color=INK, fontsize=11)
fig.tight_layout(); fig.savefig('fig4.png', dpi=200); plt.close()
print('ok')
