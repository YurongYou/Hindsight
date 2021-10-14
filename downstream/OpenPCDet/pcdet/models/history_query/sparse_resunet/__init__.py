# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from . import res16unet
from . import fcn
# import model.pointnet2backbone as pointnet2

MODELS = []


def add_models(module):
  MODELS.extend([getattr(module, a) for a in dir(module) if 'Net' in a])

add_models(res16unet)
add_models(fcn)
# add_models(pointnet2)

def get_models():
  '''Returns a tuple of sample models.'''
  return MODELS

def load_model(name):
  '''Creates and returns an instance of the model given its class name.
  '''
  all_models = get_models()
  mdict = {model.__name__: model for model in all_models}
  if name not in mdict:
    print('Invalid model index. Options are:')
    for model in all_models:
      print('\t* {}'.format(model.__name__))
    return None
  NetClass = mdict[name]
  return NetClass