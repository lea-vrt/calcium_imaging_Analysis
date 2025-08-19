

import os, numpy as np, matplotlib.pyplot as plt
P = r"C:\Users\sfatima\Desktop\Feature_extraction\suite2p_output\CNMJ_calcium_DIV21_10X_01_001\suite2p\plane0"
USE_SPIKES_IF_AVAILABLE = False
NEUROPIL_R = 0.7
SMOOTH_K   = 5
CORR_THR   = 0.75
MAX_EDGES_PER_GROUP = None  

def ensure(fp):
    if not os.path.exists(fp): raise FileNotFoundError(fp)

def smooth_rows(X, k=5):
    if k<=1: return X
    ker = np.ones(k)/k
    return np.apply_along_axis(lambda v: np.convolve(v, ker, mode='same'), 1, X)

def dff_trace(x, win=300, q=10):
    T = x.shape[0]; pad = win//2
    xp = np.pad(x,(pad,pad),mode='edge')
    b = np.array([np.percentile(xp[i:i+win], q) for i in range(T)])
    return (x-b)/np.maximum(b,1e-3)

class DSU:
    def __init__(self, n): self.p=list(range(n)); self.r=[0]*n
    def find(self, x):
        while self.p[x]!=x:
            self.p[x]=self.p[self.p[x]]
            x=self.p[x]
        return x
    def union(self, a,b):
        ra,rb=self.find(a),self.find(b)
        if ra==rb: return False
        if self.r[ra]<self.r[rb]: ra,rb=rb,ra
        self.p[rb]=ra
        if self.r[ra]==self.r[rb]: self.r[ra]+=1
        return True

def connected_components(adj):
    n=adj.shape[0]; seen=np.zeros(n,bool); comps=[]
    for s in range(n):
        if seen[s]: continue
        stack=[s]; seen[s]=True; cur=[s]
        while stack:
            u=stack.pop()
            for v in np.where(adj[u])[0]:
                if not seen[v]:
                    seen[v]=True; stack.append(v); cur.append(v)
        comps.append(cur)
    return comps

def maximum_spanning_tree_local(edges_local, n_local):
    """edges_local: (i_local, j_local, w); nodes are 0..n_local-1"""
    dsu=DSU(n_local); mst=[]
    for i,j,w in sorted(edges_local, key=lambda t: t[2], reverse=True):
        if dsu.union(i,j): mst.append((i,j,w))
    return mst
Fp=os.path.join(P,"F.npy"); Fnup=os.path.join(P,"Fneu.npy")
Isp=os.path.join(P,"iscell.npy"); Statp=os.path.join(P,"stat.npy")
Spksp=os.path.join(P,"spks.npy")
for fp in [Fp,Fnup,Isp,Statp]: ensure(fp)

F=np.load(Fp); Fneu=np.load(Fnup)
iscell=np.load(Isp)[:,0].astype(bool)
stat=np.load(Statp, allow_pickle=True)
iscell_idx=np.where(iscell)[0]
F, Fneu = F[iscell], Fneu[iscell]
centroids=np.array([[s['med'][1], s['med'][0]] for s in stat])[iscell]


if USE_SPIKES_IF_AVAILABLE and os.path.exists(Spksp):
    X=np.load(Spksp)[iscell]; X=smooth_rows(X, k=3)
else:
    Fcorr=F-NEUROPIL_R*Fneu
    D=np.vstack([dff_trace(tr) for tr in Fcorr])
    D=(D-D.mean(1,keepdims=True))/(D.std(1,keepdims=True)+1e-9)
    X=smooth_rows(D, k=SMOOTH_K)

C=np.corrcoef(X); np.fill_diagonal(C,0.0)
A=C>=CORR_THR


groups=[g for g in connected_components(A) if len(g)>=2]

palette=plt.get_cmap('tab20').colors
color=lambda k: palette[k % len(palette)]

edges_by_group=[]
for g_ix, g in enumerate(groups):
    g_sorted=sorted(g)
    local_of = {orig:i for i,orig in enumerate(g_sorted)}
    sub_edges_local=[]
    for ii in range(len(g_sorted)):
        for jj in range(ii+1,len(g_sorted)):
            i_orig, j_orig = g_sorted[ii], g_sorted[jj]
            w=C[i_orig, j_orig]
            if w>=CORR_THR:
                sub_edges_local.append((ii, jj, w))

    mst_local = maximum_spanning_tree_local(sub_edges_local, len(g_sorted))
  
    if MAX_EDGES_PER_GROUP is not None and len(sub_edges_local)>0:
        extra = sorted(sub_edges_local, key=lambda t: t[2], reverse=True)
        take = max(0, MAX_EDGES_PER_GROUP - len(mst_local))
        extra = extra[:take]
        pairset = {(min(a,b),max(a,b)) for a,b,_ in mst_local}
        extra = [(a,b,w) for a,b,w in extra if (min(a,b),max(a,b)) not in pairset]
        mst_local += extra
 
    mst_global = [(g_sorted[a], g_sorted[b], w) for a,b,w in mst_local]
    edges_by_group.append((g_ix, g_sorted, mst_global))


order=[]
for _, g, _ in edges_by_group: order += g
order += [i for i in range(C.shape[0]) if i not in order]


fig = plt.figure(figsize=(12,6))
gs = fig.add_gridspec(1,2, width_ratios=[1.2,1])

ax = fig.add_subplot(gs[0,0])
ax.scatter(centroids[:,0], centroids[:,1], s=18, c='lightgray', zorder=1)
ax.invert_yaxis()
ax.set_title(f"Correlated groups (r ≥ {CORR_THR})")
for g_ix, g, mst in edges_by_group:
    col=color(g_ix)
    ax.scatter(centroids[g,0], centroids[g,1], s=26, c=[col], edgecolor='k', linewidths=0.3, zorder=3,
               label=f'Group {g_ix+1} (n={len(g)})')
    for k in g:
        ax.text(centroids[k,0], centroids[k,1], str(iscell_idx[k]), fontsize=7, ha='center', va='center', color='w', zorder=4)
    for i,j,w in mst:
        lw = 0.8 + 2.5*(w - CORR_THR)/(1 - CORR_THR)
        ax.plot([centroids[i,0], centroids[j,0]],
                [centroids[i,1], centroids[j,1]],
                color=col, alpha=0.85, linewidth=lw, zorder=2)
if len(edges_by_group)>0:
    ax.legend(frameon=False, fontsize=8, loc='upper left')
ax.set_xlabel('x (px)'); ax.set_ylabel('y (px)')

axh = fig.add_subplot(gs[0,1])
im=axh.imshow(C[np.ix_(order,order)], vmin=-1, vmax=1, cmap='coolwarm', interpolation='nearest')
axh.set_title('Correlation (reordered by groups)')
axh.set_xlabel('ROI order'); axh.set_ylabel('ROI order')

pos=0
for g_ix, g, _ in edges_by_group:
    sz=len(g)
    axh.add_patch(plt.Rectangle((pos-0.5,-0.5), sz, 0.8, facecolor=color(g_ix), edgecolor='none', alpha=0.9))
    axh.add_patch(plt.Rectangle((-0.5,pos-0.5), 0.8, sz, facecolor=color(g_ix), edgecolor='none', alpha=0.9))
    pos+=sz

cb=fig.colorbar(im, ax=axh, fraction=0.046, pad=0.04)
cb.set_label('Pearson r')
plt.subplots_adjust(wspace=0.12)
plt.show()
