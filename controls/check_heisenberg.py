"""Small matrix checks, not a dataset or a substitute for the uniform proof."""
from itertools import product
import numpy as np

p = 3
inv2 = pow(2, -1, p)
zeta = np.exp(2j * np.pi / p)
A = list(product(range(p), repeat=3))
zero = (0, 0, 0)
units = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

def add(a, b): return tuple((x + y) % p for x, y in zip(a, b))
def neg(a): return tuple(-x % p for x in a)
def scale(k, a): return tuple(k * x % p for x in a)
def dot(a, b): return sum(x * y for x, y in zip(a, b)) % p
def t(a, b, c): return a[0] * b[1] * c[2] % p
def beta(a, x, y): return (t(a,x,y)+t(x,y,a)-t(x,a,y)) % p
def gamma(x, a, b): return (t(a,b,x)+t(x,a,b)-t(a,x,b)) % p
def B(a, x, y): return (beta(a,x,y)-beta(a,y,x)) % p
def q(a, x): return -inv2 * beta(a,x,x) % p

X = np.roll(np.eye(p), 1, axis=0)
Z = np.diag(zeta ** np.arange(p))
def W(Q, P):
    return zeta ** (inv2*Q*P % p) * np.linalg.matrix_power(X,Q%p) @ np.linalg.matrix_power(Z,P%p)

coord_cache = {}
def coords(a):
    if a not in coord_cache:
        u,v = next((u,v) for u in units for v in units if B(a,u,v))
        lookup = {}
        for r,Q,T in A:
            x=add(add(scale(r,a),scale(Q,u)),scale(T,v))
            lookup[x]=(r,Q,(-B(a,u,v)*T)%p)
        assert len(lookup)==p**3
        coord_cache[a]=lookup
    return coord_cache[a]

def rep(a, label=0):
    if a==zero:
        return {x:np.array([[zeta**dot(label if isinstance(label,tuple) else zero,x)]]) for x in A}
    return {x:zeta**((q(a,x)+label*coords(a)[x][0])%p)*W(*coords(a)[x][1:]) for x in A}

def tensor_action(word):
    reps=[rep(a,l) for a,l in word]
    total=zero
    for a,l in word: total=add(total,a)
    out={}
    for x in A:
        current=np.array([[1.0+0j]])
        pref=zero
        phase=0
        for (a,l),r in zip(word,reps):
            phase+=gamma(x,pref,a)
            pref=add(pref,a)
            current=np.kron(current,r[x])
        out[x]=zeta**(phase%p)*current
    return total,out

def projectors(word):
    s,rho=tensor_action(word)
    if s==zero:
        radical=A
        labels=A
        val=lambda l,r:dot(l,r)
    else:
        radical=[scale(k,s) for k in range(p)]
        labels=list(range(p))
        val=lambda l,r:(l*coords(s)[r][0])%p
    out=[]
    for l in labels:
        P=sum(zeta**((-val(l,r)-q(s,r))%p)*rho[r] for r in radical)/len(radical)
        assert np.allclose(P,P.conj().T,atol=1e-8)
        assert np.allclose(P@P,P,atol=1e-8)
        rank=round(np.trace(P).real)
        assert rank==0 or rank in [p**j for j in range(10)]
        out.append((l,P,rank))
    assert np.allclose(sum(P for l,P,k in out),np.eye(len(next(iter(rho.values())))))
    return out

def swap_braid(a,b,la=0,lb=0):
    ra,rb=rep(a,la),rep(b,lb)
    da,db=len(ra[zero]),len(rb[zero])
    swap=np.zeros((da*db,da*db),complex)
    for i in range(da):
        for j in range(db): swap[j*da+i,i*db+j]=1
    return swap@np.kron(np.eye(da),rb[a])

samples=units+[(1,1,0),(1,1,1),(1,2,1)]
for a in samples:
    for label in [0,1]:
        r=rep(a,label)
        for x in A:
            for y in A:
                assert np.allclose(r[x]@r[y],zeta**beta(a,x,y)*r[add(x,y)],atol=1e-8)

for a,b in [(units[0],units[1]),((1,1,1),(1,2,1)),(units[0],neg(units[0]))]:
    word=[(a,0),(b,0)]
    s,src=tensor_action(word)
    _,dst=tensor_action(word[::-1])
    braid=swap_braid(a,b)
    for x in A: assert np.allclose(braid@src[x],dst[x]@braid,atol=1e-8)
    ps=projectors(word)
    print('pair',a,b,'nonzero isotypic ranks',[k for l,P,k in ps if k])

a,b=units[:2]
for word in [[(a,0),(b,0),(neg(add(a,b)),0)],[(a,0),(neg(a),0),(a,0),(neg(a),0)]]:
    ps=projectors(word)
    print('word length',len(word),'nonzero total-charge ranks',[k for l,P,k in ps if k])

for a in samples:
    r=rep(a,0)
    desired={x:zeta**(-gamma(x,a,neg(a))%p)*r[x].conj() for x in A}
    for ld in range(p):
        rd=rep(neg(a),ld)
        found=None
        for Q,P in product(range(p),repeat=2):
            U=W(Q,P)
            if all(np.allclose(U@rd[x]@U.conj().T,desired[x],atol=1e-8) for x in A):
                found=U
                break
        if found is not None: break
    assert found is not None
    bell=np.eye(p).reshape(-1)/np.sqrt(p)
    cup=np.kron(np.eye(p),found.conj().T)@bell
    s,rhopair=tensor_action([(a,0),(neg(a),ld)])
    for x in A: assert np.allclose(rhopair[x]@cup,cup,atol=1e-8)
print('PASS: projective actions, coproduct/R intertwining, full isotypic ranks, dual cups')
