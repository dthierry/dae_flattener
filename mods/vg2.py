# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

__author__ = 'David Thierry @2018'

##############################################################################
# Institute for the Design of Advanced Energy Systems Process Systems
# Engineering Framework (IDAES PSE Framework) Copyright (c) 2018, by the
# software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia
# University Research Corporation, et al. All rights reserved.
#
# Please see the files COPYRIGHT.txt and LICENSE.txt for full copyright and
# license information, respectively. Both files are also available online
# at the URL "https://github.com/IDAES/idaes".
##############################################################################

"""
Demonstration and test flowsheet for a dynamic flowsheet.

"""

# Import Pyomo libraries
from pyomo.environ import ConcreteModel, Constraint, SolverFactory, \
    TransformationFactory, Var, Param, Expression, Reals, Set, PositiveReals

from pyomo.core.expr.numvalue import value
from pyomo.core.expr.current import exp, log

# Import IDAES core
from idaes.core import FlowsheetBlock, Stream

from idaes.core import declare_process_block_class, ProcessBlock, \
    PropertyBlockDataBase, PropertyParameterBase
from idaes.core.util.misc import add_object_ref, solve_indexed_blocks

# Import Unit Model Modules
from idaes.models import CSTR, Mixer
import logging

logger = logging.getLogger('idaes.unit_model.properties')


@declare_process_block_class("PropertyParameterBlock")
class _PropertyParameterBlock(PropertyParameterBase):
    """
    Property Parameter Block Class

    Contains parameters and indexing sets associated with proeprties for
    a demonstration VLSE system.

    """

    def build(self):
        '''
        Callable method for Block construction.
        '''
        super(_PropertyParameterBlock, self).build()

        self._make_params()

    def _make_params(self):
        ''' This section is for parameters needed for the property models.'''
        # List of valid phases in property package
        self.phase_list = Set(initialize=['Liq'])

        # Component list - a list of component identifiers
        self.component_list = Set(initialize=['a', 'b', 'c', 'd', 'e', 'f'])

        # List of components in each phase (optional)
        self.phase_component_list = {"Liq": self.component_list}

        # Reaction indices - a list of identifiers for each reaction
        self.rate_reaction_idx = Set(initialize=[1, 2, 3])

        # Stoichiometric coefficients
        '''Stoichiometric coefficient for each component in each reaction'''
        self.rate_reaction_stoichiometry = {
            (1, 'Liq', 'a'): -1,
            (1, 'Liq', 'b'): -2,
            (1, 'Liq', 'c'): 1,
            (1, 'Liq', 'd'): 1,
            (1, 'Liq', 'e'): 0,
            (1, 'Liq', 'f'): 0,
            (2, 'Liq', 'a'): -1,
            (2, 'Liq', 'b'): 0,
            (2, 'Liq', 'c'): -2,
            (2, 'Liq', 'd'): 0,
            (2, 'Liq', 'e'): 2,
            (2, 'Liq', 'f'): 0,
            (3, 'Liq', 'a'): -1,
            (3, 'Liq', 'b'): -1,
            (3, 'Liq', 'c'): 0,
            (3, 'Liq', 'd'): 0,
            (3, 'Liq', 'e'): 0,
            (3, 'Liq', 'f'): 1}

        # Mixture heat capacity
        self.cp_mol = Param(
            within=PositiveReals,
            mutable=True,
            default=75.4,
            doc="Mixture heat capacity [J/mol.K]")

        # Gas constant
        self.gas_const = Param(within=PositiveReals,
                               mutable=False,
                               default=8.314,
                               doc="Gas constant [J/mol.K]")

        # Thermodynamic reference state
        self.temperature_ref = Param(within=PositiveReals,
                                     mutable=True,
                                     default=303.15,
                                     doc='Reference temperature [K]')

    def get_supported_properties(self):
        return {'flow_mol': {'method': None, 'units': 'mol/s'},
                'flow_mol_comp': {'method': None, 'units': 'mol/s'},
                'enth_mol': {'method': None, 'units': 'J/mol'},
                'pressure': {'method': None, 'units': 'Pa'},
                'temperature': {'method': None, 'units': 'K'},
                'dens_mol_phase': {'method': '_dens_mol_phase',
                                   'units': 'mol/m^3'},
                'dh_rxn_mol': {'method': '_dh_rxn_mol', 'units': 'J/mol'},
                'diffus': {'method': '_diffus', 'units': 'm^2/s'},
                'flow_vol': {'method': '_flow_vol', 'units': 'm^3/s'},
                'mole_frac': {'method': None, 'units': None},
                'reaction_rate': {'method': '_reaction_rate',
                                  'units': 'mol/m^3.s'},
                'k_rxn_for': {'method': '_k_rxn_for', 'units': None},
                'k_rxn_back': {'method': '_k_rxn_back', 'units': None},
                'k_eq': {'method': '_k_eq', 'units': None},
                'therm_cond': {'method': '_therm_cond', 'units': 'W/m.K'}}

    def get_package_units(self):
        return {'time': 's',
                'length': 'm',
                'mass': 'g',
                'amount': 'mol',
                'temperature': 'K',
                'energy': 'J',
                'holdup': 'mol'}


class _PropertyBlock(ProcessBlock):
    """
    This Class contains methods which should be applied to Property Blocks as a
    whole, rather than individual elements of indexed Property Blocks.
    """

    def initialize(blk, flow_mol_comp=None, pressure=None, temperature=None,
                   hold_state=False, outlvl=0,
                   solver='ipopt', optarg={'tol': 1e-8}):
        '''
        Initialisation routine for property package.

        Keyword Arguments:
            flow_mol_comp : value at which to initialize component flows
                             (default=None)
            pressure : value at which to initialize pressure (default=None)
            temperature : value at which to initialize temperature
                            (default=None)
            outlvl : sets output level of initialisation routine

                     * 0 = no output (default)
                     * 1 = return solver state for each step in routine
                     * 2 = include solver output infomation (tee=True)

            optarg : solver options dictionary object (default=None)
            solver : str indicating whcih solver to use during
                     initialization (default = 'ipopt')
            hold_state : flag indicating whether the initialization routine
                         should unfix any state variables fixed during
                         initialization (default=False).
                         - True - states varaibles are not unfixed, and
                                 a dict of returned containing flags for
                                 which states were fixed during
                                 initialization.
                        - False - state variables are unfixed after
                                 initialization by calling the
                                 relase_state method

        Returns:
            If hold_states is True, returns a dict containing flags for
            which states were fixed during initialization.
        '''
        # Fix state variables if not already fixed
        Fcflag = {}
        Pflag = {}
        Tflag = {}

        for k in blk.keys():
            for j in blk[k].component_list:
                if blk[k].flow_mol_comp[j].fixed is True:
                    Fcflag[k, j] = True
                else:
                    Fcflag[k, j] = False
                    if flow_mol_comp is None:
                        blk[k].flow_mol_comp[j].fix(1.0)
                    else:
                        blk[k].flow_mol_comp[j].fix(flow_mol_comp[j])

            if blk[k].pressure.fixed is True:
                Pflag[k] = True
            else:
                Pflag[k] = False
                if pressure is None:
                    blk[k].pressure.fix(101325.0)
                else:
                    blk[k].pressure.fix(pressure)

            if blk[k].temperature.fixed is True:
                Tflag[k] = True
            else:
                Tflag[k] = False
                if temperature is None:
                    blk[k].temperature.fix(303.15)
                else:
                    blk[k].temperature.fix(temperature)

        # Set solver options
        if outlvl > 1:
            stee = True
        else:
            stee = False

        opt = SolverFactory(solver)
        opt.options = optarg

        # ---------------------------------------------------------------------
        # Initialise values
        for k in blk.keys():
            for j in blk[k].component_list:
                blk[k].mole_frac[j] = (value(blk[k].flow_mol_comp[j]) /
                                       sum(value(blk[k].flow_mol_comp[i])
                                           for i in blk[k].component_list))

            blk[k].enth_mol == value(
                blk[k].cp_mol *
                (blk[k].temperature - blk[k].temperature_ref) -
                blk[k].mole_frac['d'] * blk[k].dh_rxn_mol[1] -
                0.5 * blk[k].mole_frac['e'] *
                blk[k].dh_rxn_mol[2] -
                blk[k].mole_frac['f'] * blk[k].dh_rxn_mol[3])

            if hasattr(blk, "eq_reaction_rate") is True:
                for i in blk[k].rate_reaction_idx:
                    if i == 1:
                        blk[k].dh_rxn_mol[i] = 60000
                        blk[k].k_rxn_for[i] = 141.54
                        blk[k].k_eq[i] = 20
                    elif i == 2:
                        blk[k].dh_rxn_mol[i] = 50000
                        blk[k].k_rxn_for[i] = 88.46
                        blk[k].k_eq[i] = 5
                    else:
                        blk[k].dh_rxn_mol[i] = 80000
                        blk[k].k_rxn_for[i] = 139.45
                        blk[k].k_eq[i] = 10
                    blk[k].k_rxn_back[i] = (
                            value(blk[k].k_rxn_for[i]) /
                            value(blk[k].k_eq[i]))

                for j in blk[k].rate_reaction_idx:
                    if j == 1:
                        blk[k].reaction_rate[j] = (
                                value(blk[k].k_rxn_for[j]) *
                                (value(blk[k].mole_frac['a'])) *
                                (value(blk[k].mole_frac['b']) ** 2) -
                                value(blk[k].k_rxn_back[j]) *
                                value(blk[k].mole_frac['c']) *
                                value(blk[k].mole_frac['d']))
                    elif j == 2:
                        blk[k].reaction_rate[j] = (
                                value(blk[k].k_rxn_for[j]) *
                                (value(blk[k].mole_frac['a'])) *
                                (value(blk[k].mole_frac['c']) ** 2) -
                                value(blk[k].k_rxn_back[j]) *
                                value(blk[k].mole_frac['e']))
                    else:
                        blk[k].reaction_rate[j] = (
                                value(blk[k].k_rxn_for[j]) *
                                (value(blk[k].mole_frac['a'])) *
                                (value(blk[k].mole_frac['b'])) -
                                value(blk[k].k_rxn_back[j]) *
                                value(blk[k].mole_frac['f']))

        results = solve_indexed_blocks(opt, blk, tee=stee)

        if outlvl > 0:
            if results.solver.termination_condition \
                    == TerminationCondition.optimal:
                logger.info('{} Initialisation Step 1 Complete.'
                            .format(blk.name))
            else:
                logger.warning('{} Initialisation Step 1 Failed.'
                               .format(blk.name))

        # ---------------------------------------------------------------------
        # If input block, return flags, else release state
        flags = {"Fcflag": Fcflag, "Pflag": Pflag, "Tflag": Tflag}

        if outlvl > 0:
            if outlvl > 0:
                logger.info('{} Initialisation Complete.'.format(blk.name))

        if hold_state is True:
            return flags
        else:
            blk.release_state(flags)

    def release_state(blk, flags, outlvl=0):
        '''
        Method to relase state variables fixed during initialisation.

        Keyword Arguments:
            flags : dict containing information of which state variables
                    were fixed during initialization, and should now be
                    unfixed. This dict is returned by initialize if
                    hold_state=True.
            outlvl : sets output level of of logging
        '''
        # Unfix state variables
        for k in blk.keys():
            for j in blk[k].component_list:
                if flags['Fcflag'][k, j] is False:
                    blk[k].flow_mol_comp[j].unfix()
            if flags['Pflag'][k] is False:
                blk[k].pressure.unfix()
            if flags['Tflag'][k] is False:
                blk[k].temperature.unfix()

        if outlvl > 0:
            if outlvl > 0:
                logger.info('{} State Released.'.format(blk.name))


@declare_process_block_class("PropertyBlock",
                             block_class=_PropertyBlock)
class PropertyBlockData(PropertyBlockDataBase):
    """
    Example property package for reactions

    This package contains the necessary property calculations to
    demonstrte the basic unit reactor models.

    System involeves six components (a,b,c,d,e and f) involved in
    three reactions (labled 1, 2 and 3). Reactions equations are:

    a + 2b <-> c + d
    a + 2c <-> 2e
    a + b  <-> f

    Reactions are assumed to be aqueous and only a liquid phase is
    considered.

    Properties supported:
        - stoichiometric coefficients
        - rate of reaction
            - rate coefficients (forward and reverse)
            - equilibrium coefficients
        - heats of reaction
        - specific enthalpy of the fluid mixture

    """

    def build(self):
        """
        Callable method for Block construction
        """
        super(PropertyBlockData, self).build()
        self._make_params()
        self._make_vars()
        self._make_constraints()
        self._make_balance_terms()

    def _make_params(self):
        '''
        This section makes references to the necessary parameters contained
        within the Property Parameter Block provided.
        '''
        # List of valid phases in property package
        add_object_ref(self, "phase_list", self.config.parameters.phase_list)

        # Component list - a list of component identifiers
        add_object_ref(self, "component_list",
                       self.config.parameters.component_list)

        # Reaction indices - a list of identifiers for each reaction
        add_object_ref(self, "rate_reaction_idx",
                       self.config.parameters.rate_reaction_idx)

        # Mixture heat capacity
        add_object_ref(self, "cp_mol",
                       self.config.parameters.cp_mol)

        # Stoichiometric coefficients
        add_object_ref(self, "rate_reaction_stoichiometry",
                       self.config.parameters.rate_reaction_stoichiometry)

        # Gas constant
        add_object_ref(self, "gas_const",
                       self.config.parameters.gas_const)

        # Thermodynamic reference state
        add_object_ref(self, "temperature_ref",
                       self.config.parameters.temperature_ref)

    def _make_vars(self):
        # Create state variables
        self.flow_mol_comp = Var(self.component_list,
                                 domain=Reals,
                                 initialize=0.0,
                                 bounds=(0, 1e3),
                                 doc='Component molar flowrate [mol/s]')
        self.flow_mol = Var(domain=Reals,
                            initialize=0.0,
                            bounds=(1e-2, 1e3),
                            doc='Total molar flowrate [mol/s]')
        self.pressure = Var(domain=Reals,
                            initialize=101325.0,
                            doc='State pressure [Pa]')
        self.temperature = Var(domain=Reals,
                               initialize=303.15,
                               doc='State temperature [K]')
        self.mole_frac = Var(self.component_list,
                             domain=Reals,
                             initialize=0.0,
                             bounds=(0.0, 1.0),
                             doc='State component mole fractions [-]')
        self.enth_mol = Var(domain=Reals,
                            initialize=0.0,
                            doc='Mixture specific entahlpy [J/mol]')

    def _make_constraints(self):
        # Calcuate total flow
        self.sum_comp_flows = Constraint(expr=self.flow_mol ==
                                              sum(self.flow_mol_comp[k]
                                                  for k in self.component_list))

        # Calculate mole fractions
        def mole_fraction_calculation(b, j):
            return b.flow_mol_comp[j] == b.mole_frac[j] * b.flow_mol

        self.mole_fraction_calculation = Constraint(
            self.component_list,
            doc="Mole fraction calculation",
            rule=mole_fraction_calculation)

        # Mixture enthalpy
        ''' The mixture enthalpy is assumed to be equal to that of pure
            water in the liquid state, with a constant heat capacity'''
        self.enth_mol_correlation = Constraint(
            expr=self.enth_mol == self.cp_mol *
                 (self.temperature - self.temperature_ref) -
                 self.mole_frac['d'] * self.dh_rxn_mol[1] -
                 0.5 * self.mole_frac['e'] * self.dh_rxn_mol[2] -
                 self.mole_frac['f'] * self.dh_rxn_mol[3])

    def _dens_mol_phase(self):
        # Molar density
        self.dens_mol_phase = Var(self.phase_list, doc="Molar density")

        def dens_mol_phase_correlation(b, p):
            return b.dens_mol_phase[p] == 55555.0

        self.dens_mol_phase_correlation = Constraint(
            self.phase_list,
            doc="Molar density correlation",
            rule=dens_mol_phase_correlation)

    def _dh_rxn_mol(self):
        # Heat of reaction
        self.dh_rxn_mol = Var(self.rate_reaction_idx,
                              domain=Reals,
                              initialize=0.0,
                              doc='Heats of Reaction [J/mol]')

        def dh_rxn_mol_constraint(b, i):
            if i == 1:
                return b.dh_rxn_mol[i] == 18000
            elif i == 2:
                return b.dh_rxn_mol[i] == 15000
            else:
                return b.dh_rxn_mol[i] == 24000

        self.dh_rxn_mol_constraint = Constraint(
            self.rate_reaction_idx,
            doc="Heat of reaction constraint",
            rule=dh_rxn_mol_constraint)

    def _reaction_rate(self):
        # Reaction rate
        self.reaction_rate = Var(self.rate_reaction_idx,
                                 domain=Reals,
                                 initialize=0.0,
                                 doc='Normalised Rate of Reaction [mol/m^3.s]')

        def rate_expressions(b, j):
            if j == 1:
                return b.reaction_rate[j] == (b.k_rxn_for[j] *
                                              (b.mole_frac['a']) *
                                              (b.mole_frac['b'] ** 2) -
                                              b.k_rxn_back[j] *
                                              b.mole_frac['c'] *
                                              b.mole_frac['d'])
            elif j == 2:
                return b.reaction_rate[j] == (b.k_rxn_for[j] *
                                              (b.mole_frac['a']) *
                                              (b.mole_frac['c'] ** 2) -
                                              b.k_rxn_back[j] *
                                              b.mole_frac['e'])
            else:
                return b.reaction_rate[j] == (b.k_rxn_for[j] *
                                              (b.mole_frac['a']) *
                                              (b.mole_frac['b']) -
                                              b.k_rxn_back[j] *
                                              b.mole_frac['f'])

        try:
            # Try to build constraint
            self.rate_expressions = Constraint(
                self.rate_reaction_idx,
                doc="Rate of reaction expressions",
                rule=rate_expressions)
        except AttributeError:
            # If constraint fails, clean up so that DAE can try again later
            self.del_component(self.reaction_rate)
            self.del_component(self.rate_expressions)
            raise

    def _k_rxn_for(self):
        # Forward rate constants
        self.k_rxn_for = Var(self.rate_reaction_idx,
                             doc='Rate coefficient for forward reaction')

        # Arhenius expression for rate coefficients
        def arrhenius_expression(b, i):
            if i == 1:
                return b.k_rxn_for[i] == (17.7 * exp(-12000 /
                                                     (b.gas_const *
                                                      b.temperature)))
            elif i == 2:
                return b.k_rxn_for[i] == (1.49 * exp(-7000 /
                                                     (b.gas_const *
                                                      b.temperature)))
            else:
                return b.k_rxn_for[i] == (26.5 * exp(-13000 /
                                                     (b.gas_const *
                                                      b.temperature)))

        try:
            # Try to build constraint
            self.arrhenius_expression = Constraint(
                self.rate_reaction_idx,
                doc="Arrhenius expression for forward rate constant",
                rule=arrhenius_expression)
        except AttributeError:
            # If constraint fails, clean up so that DAE can try again later
            self.del_component(self.k_rxn_for)
            self.del_component(self.arrhenius_expression)
            raise

    def _k_rxn_back(self):
        # Reverse rate constants
        self.k_rxn_back = Var(self.rate_reaction_idx,
                              doc='Rate coefficient for reverse reaction')

        # Reverse reaction rates coefficients in terms of forward
        # coefficients and equilibrium coefficient
        def rule_k_rxn_rev(b, i):
            return b.k_rxn_for[i] == (b.k_rxn_back[i] *
                                      b.k_eq[i])

        try:
            # Try to build constraint
            self.k_rxn_relationship = Constraint(
                self.rate_reaction_idx,
                doc="Relationship between forward and reverse rate constants",
                rule=rule_k_rxn_rev)
        except AttributeError:
            # If constraint fails, clean up so that DAE can try again later
            self.del_component(self.k_rxn_back)
            self.del_component(self.k_rxn_relationship)
            raise

    def _k_eq(self):
        # Equilibrium coefficients
        self.k_eq = Var(self.rate_reaction_idx,
                        initialize=1.0,
                        doc='Equilibrium coefficient')

        # Equilibrium coefficients as a function of temperature using the
        # van't Hoff equation'''
        def vant_hoff(b, i):
            if i == 1:
                return log(b.k_eq[i]) - log(20) == (
                        -(b.dh_rxn_mol[i] / b.gas_const) *
                        (b.temperature ** -1 - 1 / 298.15))
            elif i == 2:
                return log(b.k_eq[i]) - log(5) == (
                        -(b.dh_rxn_mol[i] / b.gas_const) *
                        (b.temperature ** -1 - 1 / 298.15))
            else:
                return log(b.k_eq[i]) - log(10) == (
                        -(b.dh_rxn_mol[i] / b.gas_const) *
                        (b.temperature ** -1 - 1 / 298.15))

        try:
            # Try to build constraint
            self.vant_hoff = Constraint(
                self.rate_reaction_idx,
                doc="Van't Hoff equation for equilibrium",
                rule=vant_hoff)
        except AttributeError:
            # If constraint fails, clean up so that DAE can try again later
            self.del_component(self.k_eq)
            self.del_component(self.vant_hoff)
            raise

    def _make_balance_terms(self):
        def material_balance_term(b, i, j):
            return b.flow_mol_comp[j]

        self.material_balance_term = Expression(
            self.phase_list,
            self.component_list,
            rule=material_balance_term)

        def energy_balance_term(b, i):
            return b.enth_mol * b.flow_mol

        self.energy_balance_term = Expression(self.phase_list,
                                              rule=energy_balance_term)

        def material_density_term(b, p, j):
            return b.dens_mol_phase[p] * b.mole_frac[j]

        self.material_density_term = Expression(self.phase_list,
                                                self.component_list,
                                                rule=material_density_term)

        def energy_density_term(b, p):
            return b.dens_mol_phase[p] * b.enth_mol

        self.energy_density_term = Expression(self.phase_list,
                                              rule=energy_density_term)

    def declare_port_members(b):
        members = {"flow_mol_comp": b.flow_mol_comp,
                   "temperature": b.temperature,
                   "pressure": b.pressure}
        return members

    def model_check(blk):
        '''Method containing presovle checks for package'''
        pass


def gen_mod_():
    """
    Make the flowsheet object, fix some variables, and solve the problem
    """
    # Create a Concrete Model as the top level object
    m = ConcreteModel()

    # Add a flowsheet object to the model
    m.fs = FlowsheetBlock(dynamic=True,
                          time_set=[0, 1, 500000])

    # Add property packages to flowsheet library
    m.fs.properties_rxn = PropertyParameterBlock()
    m.fs.config.default_property_package = m.fs.properties_rxn

    # Create unit models
    m.fs.Mix = Mixer(dynamic=False)
    m.fs.Tank1 = CSTR()
    m.fs.Tank2 = CSTR()

    # Add constraints to Tank1
    m.fs.Tank1.height = Var(m.fs.Tank1.time,
                            initialize=1.0,
                            doc="Depth of fluid in tank [m]")
    m.fs.Tank1.area = Var(initialize=1.0,
                          doc="Cross-sectional area of tank [m^2]")
    m.fs.Tank1.volume_flow = Var(m.fs.Tank1.time,
                                 initialize=4.2e5,
                                 doc="Volumetric flow leaving tank")
    m.fs.Tank1.flow_coeff = Var(m.fs.Tank1.time,
                                initialize=5e-5,
                                doc="Tank outlet flow coefficient")

    def geometry(b, t):
        return b.volume[t] == b.area * b.height[t]

    m.fs.Tank1.geometry = Constraint(m.fs.Tank1.time, rule=geometry)

    def volume_flow_calculation(b, t):
        return b.volume_flow[t] == (
                b.holdup.properties_out[t].flow_mol /
                b.holdup.properties_out[t].dens_mol_phase['Liq'])

    m.fs.Tank1.volume_flow_calculation = Constraint(
        m.fs.Tank1.time,
        rule=volume_flow_calculation)

    def outlet_flowrate(b, t):
        return b.volume_flow[t] == b.flow_coeff[t] * b.height[t]

    m.fs.Tank1.outlet_flowrate = Constraint(m.fs.Tank1.time,
                                            rule=outlet_flowrate)

    # Add constraints to Tank2
    m.fs.Tank2.height = Var(m.fs.Tank2.time,
                            initialize=1.0,
                            doc="Depth of fluid in tank [m]")
    m.fs.Tank2.area = Var(initialize=1.0,
                          doc="Cross-sectional area of tank [m^2]")
    m.fs.Tank2.volume_flow = Var(m.fs.Tank2.time,
                                 initialize=4.2e5,
                                 doc="Volumetric flow leaving tank")
    m.fs.Tank2.flow_coeff = Var(m.fs.Tank2.time,
                                initialize=5e-5,
                                doc="Tank outlet flow coefficient")

    m.fs.Tank2.geometry = Constraint(m.fs.Tank2.time, rule=geometry)
    m.fs.Tank2.volume_flow_calculation = Constraint(
        m.fs.Tank2.time,
        rule=volume_flow_calculation)
    m.fs.Tank2.outlet_flowrate = Constraint(m.fs.Tank2.time,
                                            rule=outlet_flowrate)

    # Apply DAE Transformation
    discretizer = TransformationFactory('dae.finite_difference')
    discretizer.apply_to(m.fs,
                         nfe=200,
                         wrt=m.fs.time,
                         scheme='BACKWARD')

    # Post transform build
    m.fs.post_transform_build()

    # Make Streams
    m.fs.stream_1 = Stream(source=m.fs.Mix.outlet,
                           destination=m.fs.Tank1.inlet)

    m.fs.stream_2 = Stream(source=m.fs.Tank1.outlet,
                           destination=m.fs.Tank2.inlet)

    # Set inlet and operating conditions, and some initial conditions.
    m.fs.Mix.inlet[:, "1"].vars["flow_mol_comp"]["a"].fix(10.0)
    m.fs.Mix.inlet[:, "1"].vars["flow_mol_comp"]["b"].fix(0.0)
    m.fs.Mix.inlet[:, "1"].vars["flow_mol_comp"]["c"].fix(1.0)
    m.fs.Mix.inlet[:, "1"].vars["flow_mol_comp"]["d"].fix(0.0)
    m.fs.Mix.inlet[:, "1"].vars["flow_mol_comp"]["e"].fix(0.0)
    m.fs.Mix.inlet[:, "1"].vars["flow_mol_comp"]["f"].fix(0.0)
    m.fs.Mix.inlet[:, "1"].vars["temperature"].fix(303.15)
    m.fs.Mix.inlet[:, "1"].vars["pressure"].fix(101325.0)

    m.fs.Mix.inlet[:, "2"].vars["flow_mol_comp"]["a"].fix(0.0)
    m.fs.Mix.inlet[:, "2"].vars["flow_mol_comp"]["b"].fix(20.0)
    m.fs.Mix.inlet[:, "2"].vars["flow_mol_comp"]["c"].fix(0.0)
    m.fs.Mix.inlet[:, "2"].vars["flow_mol_comp"]["d"].fix(0.0)
    m.fs.Mix.inlet[:, "2"].vars["flow_mol_comp"]["e"].fix(0.0)
    m.fs.Mix.inlet[:, "2"].vars["flow_mol_comp"]["f"].fix(0.0)
    m.fs.Mix.inlet[:, "2"].vars["temperature"].fix(303.15)
    m.fs.Mix.inlet[:, "2"].vars["pressure"].fix(101325.0)

    m.fs.Tank1.area.fix(0.5)
    m.fs.Tank1.flow_coeff.fix(5e-6)
    m.fs.Tank1.heat.fix(0.0)

    m.fs.Tank2.area.fix(0.5)
    m.fs.Tank2.flow_coeff.fix(5e-6)
    m.fs.Tank2.heat.fix(0.0)

    # Set Initial Conditions
    m.fs.fix_initial_conditions('steady-state')

    # Initialize Units
    m.fs.Mix.initialize()
    m.fs.Tank1.initialize(state_args={
        "flow_mol_comp": {
            "a": m.fs.Mix.outlet[0].vars["flow_mol_comp"]["a"].value,
            "b": m.fs.Mix.outlet[0].vars["flow_mol_comp"]["b"].value,
            "c": m.fs.Mix.outlet[0].vars["flow_mol_comp"]["c"].value,
            "d": m.fs.Mix.outlet[0].vars["flow_mol_comp"]["d"].value,
            "e": m.fs.Mix.outlet[0].vars["flow_mol_comp"]["e"].value,
            "f": m.fs.Mix.outlet[0].vars["flow_mol_comp"]["f"].value},
        "pressure": m.fs.Tank1.outlet[0].vars["pressure"].value,
        "temperature": m.fs.Tank1.outlet[0].vars["temperature"].value})
    m.fs.Tank2.initialize(state_args={
        "flow_mol_comp": {
            "a": m.fs.Tank1.outlet[0].vars["flow_mol_comp"]["a"].value,
            "b": m.fs.Tank1.outlet[0].vars["flow_mol_comp"]["b"].value,
            "c": m.fs.Tank1.outlet[0].vars["flow_mol_comp"]["c"].value,
            "d": m.fs.Tank1.outlet[0].vars["flow_mol_comp"]["d"].value,
            "e": m.fs.Tank1.outlet[0].vars["flow_mol_comp"]["e"].value,
            "f": m.fs.Tank1.outlet[0].vars["flow_mol_comp"]["f"].value},
        "pressure": m.fs.Tank1.outlet[0].vars["pressure"].value,
        "temperature": m.fs.Tank1.outlet[0].vars["temperature"].value})

    # Create a solver
    solver = SolverFactory('ipopt')
    results = solver.solve(m, tee=True)

    # Create a disturbance
    for t in m.fs.time:
        if t >= 1.0:
            m.fs.Mix.inlet[t, "2"].vars["flow_mol_comp"]["b"].fix(10.0)

    results = solver.solve(m, tee=True)

    # For testing purposes
    return m
