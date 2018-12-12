#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from pyomo.opt import SolverFactory, ProblemFormat

from flat_e import Flattener
from mods.vg import gen_mod_tank

__author__ = 'David Thierry @2018'


def main():
    m = gen_mod_tank()
    m.write(filename='originalnl.nl', format=ProblemFormat.nl)
    fl = Flattener(m, m.fs_obj.time)
    fl._classification_dict()
    fl._create_variables()
    fl._replacement_alg()
    # fl.find_diff_states()
    m = fl.n_mod
    m.pprint(filename='whatnot')
    m.write(filename='flattenednl.nl', format=ProblemFormat.nl)
    # input('wait')
    ip = SolverFactory('ipopt')
    ip.solve(m, tee=True, keepfiles=True)
    return fl


if __name__ == '__main__':
    fl = main()
