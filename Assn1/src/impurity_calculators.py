import math

def gini_index(categories):
    '''
        This takes in a list of the classifications and then computes the gini_index
        using the appropriate formula
    '''
    tot = 0
    for key in categories:
        tot += categories[key]
    
    if tot == 0:                        # to avoid division errors
        return 0

    value = 1.0
    for key in categories:
        frac = categories[key] / tot
        value -= frac*frac              # basically 1 - sum(pi^2)

    return value

def entropy(categories):
    '''
        This takes in a list of the classifications and then computes the entropy
        using the appropriate formula
    '''
    tot = 0
    for key in categories:
        tot += categories[key]

    if tot == 0:                # to avoid division/range errors
        return 0

    value = 0.0
    for key in categories:
        if(categories[key] == 0):   
            continue
        frac = categories[key] / tot
        value -= frac*math.log2(frac)   # basically sum(-pilog(pi))

    return value
