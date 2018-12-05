#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

from pyomo.environ import *
from flat_e import Flattener
from mods.vg import gen_mod_tank
__author__ = 'David Thierry @2018'


def main():
    m = gen_mod_tank()
    print(type(m))

    fl = Flattener(m, m.fs_obj.time)
    fl._classification_dict()
    for i in fl.state_dict.keys():
        print(fl.state_dict[i], type(fl.state_dict[i]))


if __name__ == '__main__':
    main()
