# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division

from pyomo.environ import *
from pyomo.core.base.PyomoModel import ConcreteModel

__author__ = 'David Thierry @2018'



class Flattener(object):
    def __init__(self, mod, time_set, **kwargs):
        self.o_mod = mod
        self.o_time_set = time_set
        self.n_mod = ConcreteModel()
        self.n_mod.time = Set(self.o_time_set)

    def _navigate_structure(self, comp, s_l, time_set):
        """recursively go up the dependency tree
        :param comp:
        :param (list) s_l:
        :param (pyomo.core.base.sets.Set) The set that has to be excluded.
        """
        namel = []
        cpc = comp.parent_component()
        if cpc.is_indexed():
            s_l.append(cpc.index_set())
        cpb = cpc.parent_block()
        if cpb is None:
            return namel
        else:
            namel = self._navigate_structure(cpb, s_l, time_set)
            if isinstance(comp, pyomo.core.base.var._GeneralVarData):
                namel.append(cpc.local_name)
                if cpc.is_indexed():  #: if the parent component is indexed I want the current index.
                    idx0 = comp.index()
                    if cpc.index_set().dimen > 1:
                        if time_set in cpc._implicit_subsets:  #: time is in the set
                            idxl = []
                            jth = 0
                            for si in cpc._implicit_subsets:  #: find which element corresponds to time.
                                if time_set == si:
                                    jth += 1
                                    continue  #: skip this one
                                idxl.append(idx0[jth])
                                jth += 1
                            idx = tuple(idxl)
                        else:
                            idx = idx0  #: all good.
                    else:
                        if cpc.index_set() is time_set:
                            idx = None
                        else:
                            idx = idx0  #: all good.
                    namel.append(idx)
            else:
                namel.append(cpc)
            return namel

    def _current_time_index(self, comp, time_set):
        # type: (pyomo.core.base.component.Component, pyomo.core.base.sets.Set) -> float
        if not isinstance(time_set, pyomo.core.base.sets.Set):
            raise TypeError('wrong type')
        cpc = comp.parent_component()
        cpb = cpc.parent_block()
        if cpb is not None:
            if cpc.is_indexed():  #: if the parent component is indexed I want the current index.
                idx0 = comp.index()
                if cpc.index_set().dimen > 1:  #: Multi-set
                    if time_set in cpc._implicit_subsets:  #: time is in the set
                        jth = 0
                        for si in cpc._implicit_subsets:  #: find which element corresponds to time.
                            if time_set == si:
                                return idx0[jth]
                            jth += 1
                else:  #: Singleton set
                    if cpc.index_set() is time_set:
                        return idx0
            return self._current_time_index(cpb, time_set)  #: we didn't find it here go up one level
        else:  #: root node
            return None

    def _assess_time_set(self, var, time_set):
        # type: (pyomo.core.base.var.Var, pyomo.core.base.sets.Set) -> None
        if var.is_indexed():
            #: if a variable has the same sets and name, then it is part of the same IndexedVar
            #: find indexes
            s = var.index_set()
            print(s, 'indexed set')
            #: the following cannot be none
            if s.dimen > 1:
                print("multiple", end='\t')
                if time_set in s._implicit_subsets:
                    print("contains time")
                else:
                    print("")
            elif s.dimen == 1:
                print("singleton", end='\t')
                if s is time_set:
                    print("it is time")
                else:
                    print("")
        else:
            print('not indexed')



