'''
    This file describes the content of our dataset
'''

# The headers in the csv
header_list = ['Buying', 'Maint', 'Doors', 'Persons', 'Lug_boot', 'Safety', 'Class_value']

# The attributes that are to be tested
attr_list = ['Buying', 'Maint', 'Doors', 'Persons', 'Lug_boot', 'Safety']

# Possible values of the attributes
possible_values = {
            'Buying': ['vhigh', 'high', 'med', 'low'],
            'Maint':  ['vhigh', 'high', 'med', 'low'],
            'Doors':  ['2','3','4','5more'],
            'Persons':['2','4','more'],
            'Lug_boot':['small', 'med', 'big'],
            'Safety': ['low', 'med', 'high'],
}

# Finally the possible target values
classifications = ['unacc', 'acc', 'good', 'vgood']
