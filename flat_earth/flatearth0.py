#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

from pyomo.environ import *
from flat_e import Flattener
from mods.vg import gen_mod_tank
from pyomo.environ import *
from pyomo.opt import ProblemFormat, SolverFactory
__author__ = 'David Thierry @2018'


def main():
    m = gen_mod_tank()
    fl = Flattener(m, m.fs_obj.time)

    print(len(m.fs_obj.time))
    fl._classification_dict()
    fl._create_variables()
    fl._replacement_alg()
    # fl.n_mod.pprint(filename="flat.txt")
    m = fl.n_mod
    m.write(filename='yeaboi.nl', format=ProblemFormat.nl)
    ip = SolverFactory('ipopt')
    ip.solve(m, tee=True)



if __name__ == '__main__':
    main()
