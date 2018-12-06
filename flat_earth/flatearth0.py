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
    fl = Flattener(m, m.fs_obj.time)

    print(len(m.fs_obj.time))
    fl._classification_dict()
    fl._create_variables()
    fl.n_mod.display()
    # with open('myfile.txt', 'w') as f:
    #     for i in fl.state_dict.keys():
    #         f.write(str(len(fl.state_dict[i])) + '\t' + str(type(fl.state_dict[i])) + '\t')
    #         for k in i:
    #             f.write(str(k))
    #             f.write('\t')
    #         f.write('\n')


if __name__ == '__main__':
    main()
