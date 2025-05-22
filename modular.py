def funct(pList,kList,nList,B,a):
    numerator = sum([x + a for x in [(pList/kList)**nList[i] for i in range(len(pList))]]) * B
    denominator = 1+sum[[(pList/kList)**nList[i] for i in range(len(pList))]]
    answer = numerator/denominator
    return answer

def returns_dmfdt(mf,t,pf,kf,B,a,sigma):
    dmfdt = funct(pf,kf,t,B,a)-sigma*mf
    return dmfdt

def returns_dpfdt(kf,mf,sigmaf,pf):
    dpfdt = kf*mf-sigmaf*pf
    return dpfdt

def returns_dpfnt(kf,mf,sigmaf,pf,sigmaI,pI):
    dpfndt = kf*mf+sigmaf*pf-sigmaI*pI
    return dpfndt
    

def returns_dpFdt(kf,sigmaf,pf,pI,pII):
    dpFdt = kf*pI*pII-sigmaf-pf
    return dpFdt

y = pf