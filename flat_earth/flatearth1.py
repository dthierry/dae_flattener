#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from pyomo.opt import ProblemFormat, SolverFactory

from flat_e import Flattener
from mods.vg2 import gen_mod_

__author__ = 'David Thierry @2018'


def main():
    m = gen_mod_()
    fl = Flattener(m, m.fs.time)

    print(len(m.fs.time))
    fl._classification_dict()
    fl._create_variables()
    fl._replacement_alg()
    # fl.n_mod.pprint(filename="flat.txt")
    m = fl.n_mod
    m.write(filename='yeaboi.nl', format=ProblemFormat.nl)
    ip = SolverFactory('ipopt')
    ip.solve(m, tee=True)
    fl.o_mod.display(filename='new.txt')


if __name__ == '__main__':
    main()
