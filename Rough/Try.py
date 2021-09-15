my_dict = {
        'One' : '1',
        'Two' : '2',
        'Three' : '',
        'Four' : '',
}

print(my_dict)

my_dict2 = my_dict
my_dict2['One'] = '3'

print(my_dict)
print(my_dict2)

my_dict3 = my_dict2
my_dict3['One'] = '4'
print(my_dict)
print(my_dict2)
print(my_dict3)
