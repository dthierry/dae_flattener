#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from flat_e import Flattener
from mods.vg import gen_mod_tank

__author__ = 'David Thierry @2018'


def main():
    m = gen_mod_tank()
    fl = Flattener(m, m.fs_obj.time)
    fl._classification_dict()
    fl._create_variables()
    fl.n_mod.pprint()
    # fl._replacement_alg()
    # fl.find_diff_states()
    # m = fl.n_mod
    # ip = SolverFactory('ipopt')
    # ip.solve(m, tee=True)
    return fl


if __name__ == '__main__':
    fl = main()
