import json, sys
def ev(n, outdir):
    truth=json.load(open(f"truth{n}.json"))["panels"]
    got=json.load(open(f"{outdir}/{n}/board.json"))["loads"]
    tn=cn=cc=0; msgs=[]
    for p,(t,g) in enumerate(zip(truth,got)):
        for i,r in enumerate(g["rows"]):
            wn,wc=t["names"][i],t["cats"][i]
            tn+=1; cn+=r["name"]==wn; cc+=r["cat"]==wc
            m=("" if r["name"]==wn else f"  NAME {r['name']!r}!={wn!r}")+("" if r["cat"]==wc else f"  CAT {r['cat']!r}!={wc!r} (raw {r['cat_raw']!r})")
            if m: msgs.append(f"  p{p+1} r{i+1}:{m}")
    print(f"snap {n}: names {cn}/{tn} cats {cc}/{tn}")
    for m in msgs: print(m)
for n in (4,5,6,8): ev(n, sys.argv[1] if len(sys.argv)>1 else "out2")
