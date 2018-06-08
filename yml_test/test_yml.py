import yaml

dict1 = {
    'name': 'trans',
    'age': 20,
    'male': True
}

# stream = file('documents.yaml', 'w')
# y = yaml.dump(dict1, stream)

## test
# dict_stand = yaml.load('../cifar/cfgs/q_vgg_sgdr.yml')


# with open('q_vgg_sgdr.yml', 'r') as f:
#     dict_stand = yaml.load(f)
#     print(dict_stand)
# print('#########################################')
# with file('test2.yml', 'w') as stream:
#     sv = yaml.dump(dict_stand, stream)
#     print(sv)
#
# with open('test2.yml', 'r') as f:
#     dict2 = yaml.load(f)
#     print(dict2)

from config_parse import cfg
print(cfg)
with open('testxx.yml', 'w') as stream:
    sv = yaml.dump(cfg, stream)

with open('testxx.yml', 'r') as f:
    d = yaml.load(f)
    print(d)

