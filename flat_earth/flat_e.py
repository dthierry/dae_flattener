# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division

from pyomo.environ import *
from pyomo.core.expr import current as EXPR
#: I had to use pyomo from github because expr didn't exist in pip

from pyomo.core.kernel.numvalue import NumericConstant
from pyomo.core.kernel.numvalue import value
from pyomo.opt import SolverFactory


__author__ = 'David Thierry @2018'


class Flattener(object):
    def __init__(self, mod, time_set, **kwargs):
        self.o_mod = mod
        self.o_time_set = time_set
        self.n_mod = ConcreteModel()
        self.n_mod.time = Set(self.o_time_set)
        self.state_dict = dict()
        self.flat_dict = dict()

    def _navigate_structure(self, comp, s_l, time_set):
        # type: (pyomo.core.base.component.Component, list, pyomo.core.base.sets.Set) -> list
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
                pass
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
        """
        :param comp:
        :param time_set:
        :return:
        """
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

    def _assess_time_set(self, var_o, time_set):
        # type: (pyomo.core.base.var.Var, pyomo.core.base.sets.Set) -> None
        if var_o.is_indexed():
            #: if a variable has the same sets and name, then it is part of the same IndexedVar
            #: find indexes
            s = var_o.index_set()
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

    def _replacement_alg(self):
        n = ConcreteModel()  #: create a new model
        count = 0
        var_dict = dict()
        for i in self.o_mod.component_data_objects(Var):
            try:
                c_v = value(i)
            except ValueError:
                c_v = 1e-07
            n.add_component("x" + str(count), Var(initialize=c_v))
            n_v = getattr(n, "x" + str(count))
            if i.is_fixed():
                n_v.fix()
            var_dict[id(i)] = n_v
            count += 1
        count = 0
        for i in self.o_mod.component_data_objects(Param):
            # try:
            c_v = value(i)
            # except ValueError:
            #     c_v = 1e-07
            n.add_component("p" + str(count), Param(initialize=c_v))
            n_v = getattr(n, "p" + str(count))
            # n_v = getattr(n, "x" + str(1))
            var_dict[id(i)] = n_v
            count += 1
        count = 0
        visitor = ReplacementVisitor(var_dict, self.state_dict, self.idx_fun)
        for i in self.o_mod.component_data_objects(Constraint):
            o_e = i.expr
            n_e = visitor.dfs_postorder_stack(o_e)
            n.add_component('c' + str(count), Constraint(expr=n_e))
            count += 1

    def _classification_dict(self):
        self.state_dict = dict()  #: this contains the state variables
        for i in self.o_mod.component_data_objects(Var):
            sl = []
            lo = []
            lo = self._navigate_structure(i, sl, self.o_time_set)
            s0 = str(sl)
            s1 = str(lo)
            time = self._current_time_index(i, self.o_time_set)
            if time is None:  #: not time indexed case
                self.state_dict[s0, s1] = i
                continue
            if not (s0, s1) in self.state_dict.keys():  #: if it doesn't exist, create it.
                self.state_dict[s0, s1] = dict()
            d = self.state_dict[s0, s1]  #: dict to dict
            d[time] = i  #: pyomo var

    def _create_variables(self):
        j = 0
        for i in self.state_dict.keys():
            d = self.state_dict[i]
            n = self.n_mod
            if not isinstance(d, dict):
                n.add_component('x' + str(j), Var())
                entry = d
                nvar = getattr(n, 'x' + str(j))
                nvar.setlb(entry.lb)
                nvar.setub(entry.ub)
                nvar.set_value(value(entry))
                nvar.doc = entry.name
                self.flat_dict['x' + str(j), -1] = entry  #: flat to var dict
                j += 1
                continue
            n.add_component('x' + str(j), Var(n.time))
            nvar = getattr(n, 'x' + str(j))
            d = self.state_dict[i]

            for t in n.time:
                entry = d[t]
                nvar[t].setlb(entry.lb)
                nvar[t].setub(entry.ub)
                nvar[t].set_value(value(entry))
                self.flat_dict['x' + str(j), t] = entry  #: flat to var dict
            j += 1

    def idx_fun(self, comp):
        sl = []
        lo = []
        lo = self._navigate_structure(comp, sl, self.o_time_set)
        s0 = str(sl)
        s1 = str(lo)
        time = self._current_time_index(comp, self.o_time_set)
        return s0, s1, time


class ReplacementVisitor(EXPR.ExpressionReplacementVisitor):
    def __init__(self, s_dict, local_time_set, idxfun):
        super(ReplacementVisitor, self).__init__()
        self.d = s_dict
        self.lts = local_time_set
        self.idxf = idxfun

    def visiting_potential_leaf(self, node):
        #
        # Clone leaf nodes in the expression tree
        #
        if node.__class__ in native_numeric_types:
            # print(node, type(node))
            # print('native_numeric_type\t', node, type(node))
            return True, node

        if node.__class__ is NumericConstant:
            # print('NumericConstant\t', node, type(node))
            return True, node

        if node.is_variable_type():
            s0, s1, time = self.idxf(node)
            if time is None:
                d = self.d[s0, s1]
                return True, d
            else:
                d = self.d[s0, s1]
                return True, d[time]

        if node.is_parameter_type():  #: gotta fix that
            n_val = value(node)
            n = NumericConstant(value(node))  #: this one works :S
            return True, n

        return False, None
