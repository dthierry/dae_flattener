# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import operator

import six
from pyomo.core.expr import current as EXPR
from pyomo.core.expr.numvalue import NumericConstant
from pyomo.dae import *
from pyomo.environ import *

#: I had to use pyomo from github because expr didn't exist in pip
#: also PyUtillib
#: ToDo: indexed params

__author__ = 'David Thierry @2018'

"""Fla'enner by D.T."""


class Flattener(object):
    def __init__(self, mod, time_set, **kwargs):
        self.o_mod = mod
        self.o_time_set = time_set
        self.n_mod = ConcreteModel()
        self.o_di = time_set.get_discretization_info()
        self.n_mod.time = ContinuousSet(initialize=self.o_time_set.get_finite_elements())
        self.n_mod.name = "FlatEarth_" + str(hash(mod))
        self.state_dict = dict()
        self.dstate_dict = dict()
        self.dterm_dict = dict()
        self.deqns_dict = dict()
        self.dstate = []
        self.dterm = []
        self.deqns = []
        self.flat_dict = dict()
        self.n_not_indexed_vars = 0
        self.find_diff_states()

    def _navigate_structure(self, comp, s_l, time_set):
        # type: (pyomo.core.base.component.Component, list, pyomo.core.base.sets.Set) -> list
        """recursively go up the dependency tree
        :param comp:
        :param (list) s_l:
        :param (pyomo.core.base.sets.Set) The set that has to be excluded.
        """
        #: the structure m.block1[setProduct]
        namel = []
        cpc = comp.parent_component()
        if cpc.is_indexed():
            # if isinstance(cpc.index_set(), pyomo.core.base.sets._SetProduct):
            #     print(cpc)
            s_l.append(cpc.index_set())
            if cpc.index_set().dimen > 1:
                print(cpc.index_set(), comp.index())
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

    def _navigate_structure2(self, comp, s_l, time_set):
        # type: (pyomo.core.base.component.Component, list, pyomo.core.base.sets.Set) -> list
        """recursively go up the dependency tree
        :param comp:
        :param (list) s_l:
        :param (pyomo.core.base.sets.Set) The set that has to be excluded.
        """
        #: the structure m.block1[setProduct]
        namel = []
        cpc = comp.parent_component()
        #:
        if cpc.is_indexed():
            tmp_s = None
            tmp_t = None
            tmp_s, tmp_t = self.assess_set(comp, time_set)
            if isinstance(tmp_s, tuple):
                for i in tmp_s:
                    if i is not None:
                        s_l.append(i)
            elif tmp_s is not None:
                s_l.append(tmp_s)
        cpb = cpc.parent_block()
        if cpb is None:
            return namel
        else:
            namel = self._navigate_structure2(cpb, s_l, time_set)
            # if isinstance(comp, pyomo.core.base.var._GeneralVarData) or isinstance(comp, pyomo.core.base.var.SimpleVar):
            if not comp.is_indexed():
                namel.append(cpc.local_name)  #: these are generally not hashable.
            else:
                namel.append(cpc)
            return namel

    def assess_set(self, comp, soi):
        if not isinstance(soi, pyomo.core.base.sets.Set):
            raise TypeError('this not a Set')
        cpc = comp.parent_component()
        if cpc is comp:  #: this doesn't have a set
            return None, None
        if cpc.is_indexed():  #: if the parent component is indexed I want the current index.
            idx0 = comp.index()
            if cpc.index_set().dimen > 1:  #: Multi-set
                if soi in cpc._implicit_subsets:  #: time is in the set
                    jth = 0
                    joi = None
                    ns = []
                    for si in cpc._implicit_subsets:  #: find which element corresponds to time.
                        if soi is si:
                            joi = jth
                            jth += 1
                            continue
                        ns.append(idx0[jth])
                        jth += 1
                    return tuple(ns), joi
                else:
                    return comp.index(), None
            else:  #: Singleton set
                if cpc.index_set() is soi:
                    return None, cpc.index_set()
                else:
                    return comp.index(), None
        else:
            return None, None

    def _current_time_index(self, comp, time_set):
        # type: (pyomo.core.base.component.Component, pyomo.core.base.sets.Set) -> float
        """
        :param comp:
        :param time_set:
        :return:
        """
        if not isinstance(time_set, pyomo.core.base.sets.Set):
            raise TypeError('the time_set is not a Set')
        cpc = comp.parent_component()
        cpb = cpc.parent_block()
        if cpb is None:  #: at the root
            return None
        else:
            if cpc.is_indexed() and (not cpc is comp):  #: if the parent component is indexed I want the current index.
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
        n = self.n_mod
        count = 0
        j = -1
        visitor = ReplacementVisitor(self.rev_flat_dict, self.idx_fun)
        for i in self.o_mod.component_data_objects(Constraint):
            j += 1
            if not isinstance(i, pyomo.core.base.constraint.SimpleConstraint):
                if i.parent_component() in self.deqns:
                    continue
            o_e = i.expr
            n_e = visitor.dfs_postorder_stack(o_e)
            n.add_component('c' + str(count), Constraint(expr=n_e))
            count += 1
        print("[[FLATH_EARTH]] Created model with:\n"
              "\t{} replaced out of {} equations."
              .format(count, j))
        # j = 0
        # for i in self.deqns_dict.keys():
        #     # d = self.deqns_dict[i]
        #     localname = 'dx_disc_eq' + str(j)
        #     n.add_component(localname, ConstraintList())
        #     ncl = getattr(n, localname)
        #     for e in six.itervalues(self.deqns_dict[i]):
        #         o_e = e.expr
        #         n_e = visitor.dfs_postorder_stack(o_e)
        #         ncl.add(n_e)
        #         # self.flat_dict['x' + str(j), t] = entry  #: flat to var dict
        #         # self.rev_flat_dict[i, t] = nvar[t]
        #     j += 1
        # print("[[FLATH_EARTH]] Created model with {} variables out of which {} are not indexed."
        #       .format(j, self.n_not_indexed_vars))


    def _classification_dict(self):
        self.state_dict = dict()  #: this contains the state variables
        for i in self.o_mod.component_data_objects(Var):
            sl = []
            lo = self._navigate_structure2(i, sl, self.o_time_set)
            s0 = str(sl)
            s1 = str(lo)
            time = self._current_time_index(i, self.o_time_set)
            if not isinstance(i, pyomo.core.base.var.SimpleVar):
                if i.parent_component() in self.dterm:  #: skip ddt terms
                    if not (s0, s1) in self.dterm_dict.keys():  #: if it doesn't exist, create it.
                        if time is None:  #: not time indexed case
                            raise Exception("no time index")
                        else:
                            self.dterm_dict[s0, s1] = dict()
                    d = self.dterm_dict[s0, s1]  #: dict to dict
                    d[time] = i  #: pyomo var
                if i.parent_component() in self.dstate:  #: skip diff states
                    if not (s0, s1) in self.dstate_dict.keys():  #: if it doesn't exist, create it.
                        if time is None:  #: not time indexed case
                            raise Exception("no time index")
                        else:
                            self.dstate_dict[s0, s1] = dict()
                    d = self.dstate_dict[s0, s1]  #: dict to dict
                    d[time] = i  #: pyomo var

            if not (s0, s1) in self.state_dict.keys():  #: if it doesn't exist, create it.
                if time is None:  #: not time indexed case
                    self.state_dict[s0, s1] = i
                    print(self.state_dict[s0, s1])
                    continue
                else:
                    self.state_dict[s0, s1] = dict()
            d = self.state_dict[s0, s1]  #: dict to dict
            d[time] = i  #: pyomo var
        for de in self.deqns:
            for i in six.itervalues(de):
                sl = []
                lo = self._navigate_structure2(i, sl, self.o_time_set)
                s0 = str(sl)
                s1 = str(lo)
                time = self._current_time_index(i, self.o_time_set)
                if not (s0, s1) in self.deqns_dict.keys():  #: if it doesn't exist, create it.
                    if time is None:  #: not time indexed case
                        raise Exception("no time index")
                    else:
                        self.deqns_dict[s0, s1] = dict()
                d = self.deqns_dict[s0, s1]  #: dict to dict
                d[time] = i  #: pyomo var

        print("[[FLATH_EARTH]] {} deqn out of which {} are not indexed.".format(len(self.deqns), len(self.deqns_dict)))

    def _create_variables(self):
        self.rev_flat_dict = dict()
        n = self.n_mod
        count = 0
        j = 0
        j0 = 0
        jz = 0
        for i in self.state_dict.keys():
            d = self.state_dict[i]
            if not isinstance(d, dict):
                localname = 'x0_' + str(j0)
                n.add_component(localname, Var())
                entry = d
                nvar = getattr(n, localname)
                nvar.setlb(entry.lb)
                nvar.setub(entry.ub)
                nvar.set_value(value(entry))
                if entry.is_fixed():
                    nvar.fix()
                nvar.doc = entry.name
                self.flat_dict[localname, -1] = entry  #: flat to var dict
                j0 += 1
                self.n_not_indexed_vars += 1
                self.rev_flat_dict[i, -1] = nvar
                continue
            #: need to skip both dterms and dstates
            dummy = None
            for dummy in six.itervalues(d):
                break  #: get one value
            #: theorem: differential state variable has got to be indexed by time (i.e. not SimpleVar)
            if not isinstance(dummy, pyomo.core.base.var.SimpleVar):
                if dummy.parent_component() in self.dterm:  #: skip ddt terms
                    continue
                if dummy.parent_component() in self.dstate:  #: skip diff states
                    localname = 'z' + str(jz)
                    localdzname = 'dz' + str(jz) + 'dt'
                    n.add_component(localname, Var(n.time))
                    dzvar = getattr(n, localname)
                    n.add_component(localdzname, DerivativeVar(dzvar))
                    jz += 1
                    continue
            #: non-differential variables might be indirectly indexed by time.
            localname = 'x' + str(j)
            n.add_component(localname, Var(n.time))
            j += 1

        tfs = None
        scheme = None
        ncp = None
        if self.o_di['scheme'] == 'BACKWARD Difference':  #: I ha'e this
            tfs = 'dae.finite_difference'
            scheme = 'BACKWARD'
        elif self.o_di['scheme'] == 'LAGRANGE-RADAU':
            tfs = 'dae.collocation'
            scheme = 'LAGRANGE-RADAU'
        elif self.o_di['scheme'] == 'LAGRANGE-LEGENDRE':
            tfs = 'dae.collocation'
            scheme = 'LAGRANGE-LEGENDRE'

        d = TransformationFactory(tfs)  #: discretize to generate the disc_eqns

        if 'ncp' in self.o_di.keys():
            ncp = self.o_di['ncp']
        d.apply_to(n, scheme=scheme, ncp=ncp)

        for k in n.time.get_discretization_info().keys():
            if n.time.get_discretization_info()[k] != self.o_di[k]:
                raise Exception(k)
        j = 0
        jz = 0
        for i in self.state_dict.keys():
            d = self.state_dict[i]
            if not isinstance(d, dict):
                continue  #: already done.

            dummy = None
            for dummy in six.itervalues(d):
                break  #: get one value
            #: theorem: differential state variable has got to be indexed by time (i.e. not SimpleVar)
            if not isinstance(dummy, pyomo.core.base.var.SimpleVar):  #: need to do this to keep the linking
                if dummy.parent_component() in self.dterm:  #: skip ddt terms
                    localname = 'z' + str(jz)
                    localdzname = 'dz' + str(jz) + 'dt'
                    dzvar = getattr(n, localname)
                    dztermvar = getattr(n, localdzname)
                    tv = dummy.parent_component().get_state_var()
                    for dtv in six.itervalues(tv):
                        pass
                        break
                    s0, s1, time = self.idx_fun(dtv)  #: this part is tricky :S
                    o_ddz = self.state_dict[i[0], s1]  #: original_diff_state :S
                    for t in n.time:
                        entry = d[t]
                        entrydz = o_ddz[t]
                        dztermvar[t].setlb(entry.lb)
                        dztermvar[t].setub(entry.ub)
                        dztermvar[t].set_value(value(entry))
                        if entry.is_fixed():
                            dztermvar[t].fix()

                        dzvar[t].setlb(entrydz.lb)
                        dzvar[t].setub(entrydz.ub)
                        dzvar[t].set_value(value(entrydz))
                        if entrydz.is_fixed():
                            dzvar[t].fix()

                        self.flat_dict[localname, t] = entry  #: flat to var dict
                        self.flat_dict[localdzname, t] = entrydz  #: flat to var dict
                        self.rev_flat_dict[i, t] = dztermvar[t]
                        self.rev_flat_dict[(i[0], s1), t] = dzvar[t]
                        count += 1
                    jz += 1
                    continue
                if dummy.parent_component() in self.dstate:  #: skip diff states
                    continue
            localname = 'x' + str(j)
            nvar = getattr(n, localname)  #: Give these variables an identity
            for t in n.time:
                entry = d[t]
                nvar[t].setlb(entry.lb)
                nvar[t].setub(entry.ub)
                nvar[t].set_value(value(entry))
                if entry.is_fixed():
                    nvar[t].fix()
                self.flat_dict['x' + str(j), t] = entry  #: flat to var dict
                self.rev_flat_dict[i, t] = nvar[t]
                count += 1
            j += 1
        count += self.n_not_indexed_vars
        print("[[FLATH_EARTH]] Created model with:\n"
              "\t{} algebraic variables\n\t{} differential variables\n\t{} not indexed. {} overall"
              .format(j, jz, self.n_not_indexed_vars, count))

    def idx_fun(self, comp):
        sl = []
        lo = []
        lo = self._navigate_structure2(comp, sl, self.o_time_set)
        s0 = str(sl)
        s1 = str(lo)
        time = self._current_time_index(comp, self.o_time_set)
        return s0, s1, time

    def find_diff_states(self):
        for i in self.o_mod.component_objects(Var):
            if isinstance(i, pyomo.dae.diffvar.DerivativeVar):
                self.dterm.append(i)
                self.dstate.append(i.get_state_var())
                d_eq = operator.attrgetter(i.name + "_disc_eq")(self.o_mod)  #: disc_eq !
                self.deqns.append(d_eq)
        if len(self.dterm) < 1:
            raise Exception("no derivative variables were found")


class ReplacementVisitor(EXPR.ExpressionReplacementVisitor):
    def __init__(self, s_dict, idxfun):
        super(ReplacementVisitor, self).__init__()
        self.d = s_dict
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
                d = self.d[(s0, s1), -1]
                return True, d
            else:
                d = self.d[(s0, s1), time]
                return True, d

        if node.is_parameter_type():  #: gotta fix that
            # n_val = value(node)
            n_val = NumericConstant(value(node))  #: this one works :S
            return True, n_val

        return False, None
