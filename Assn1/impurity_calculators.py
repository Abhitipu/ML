import math

def gini_index(categories):
    tot = 0
    for key in categories:
        tot += categories[key]
    
    if tot == 0:
        return 0

    value = 1.0
    for key in categories:
        frac = categories[key] / tot
        value -= frac*frac

    return value

def entropy(categories):
    tot = 0
    for key in categories:
        tot += categories[key]

    if tot == 0:                # check this... should never hit... maybe an assertion would be good here
        return 0

    value = 0.0
    for key in categories:
        if(categories[key] == 0):
            continue
        frac = categories[key] / tot
        value -= frac*math.log2(frac)

    return value
