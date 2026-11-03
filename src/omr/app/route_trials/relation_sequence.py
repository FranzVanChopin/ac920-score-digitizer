"""Small exact-symbol sequence alignment with ambiguity abstention."""


def forced_pairs(left,right,limit=64):
    """Pairs present in every longest common subsequence, in O(n*m) space.

    Repeated chords can admit several equally good correspondences. A pair
    is usable only if its left item cannot be skipped on an optimal path and
    has exactly one possible partner. No timing or harmony is fabricated.
    """
    n,m=len(left),len(right)
    if n>limit or m>limit:return []
    prefix=[[0]*(m+1) for _ in range(n+1)]
    suffix=[[0]*(m+1) for _ in range(n+1)]
    for i in range(n):
        for j in range(m):
            prefix[i+1][j+1]=max(prefix[i][j+1],prefix[i+1][j],prefix[i][j]+(left[i]==right[j]))
    for i in range(n-1,-1,-1):
        for j in range(m-1,-1,-1):
            suffix[i][j]=max(suffix[i+1][j],suffix[i][j+1],suffix[i+1][j+1]+(left[i]==right[j]))
    optimum=prefix[n][m];pairs=[]
    if not optimum:return pairs
    for i in range(n):
        if any(prefix[i][j]+suffix[i+1][j]==optimum for j in range(m+1)):
            continue
        possible=[j for j in range(m) if left[i]==right[j]
                  and prefix[i][j]+1+suffix[i+1][j+1]==optimum]
        if len(possible)==1:pairs.append((i,possible[0]))
    return pairs
