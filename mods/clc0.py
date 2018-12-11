# -*- coding: utf-8 -*-

"""
Bubbling Fluidised Bed - Chemical Looping Reactor Model

Add documentation here

Change Log:
    Domain renamed to l_n (from l), and normalised to between 0 and 1
    ll renamed to l
    All DAEs rewritten in true differential form, scaled by Lb
    Components lists changed to Pyomo Sets (instead of Python lists)
    Added gas flow through emulsion region
    Changed correlations for vb, ve and ed to better behaved versions
    Switched to using sum of mole fractions instead of sum of concentrations
        for gas phase
    Reworked solid balances to remove dependence on inert species
        - no total solid balance
        - sum of mass fractions for both regions
    Solid species amounts now based on mass fractions, and reaction rate
        expected in kg/m^3.s
    Added a number of intermediate terms
    Added scaling factors on enthalpy flows/fluxes to improve convergence

To Do:
Clean up imports of properties - strange behaviour when package not in same
        folder as model
fc_max limit - done (eq_h10 and eq_h10b)
Switch to two step bulk gas transfer
Bubble diameter constraint - done (eq_h6 and eq_h7)
Slugging
Solids momentum balances
Reversible gas flow in emulsion region
Add back old hydrodynamic correlations as options

"""

from __future__ import division
from __future__ import print_function

__author__ = "Chinedu Okoli and Andrew Lee"
__version__ = "7.0.0 v4"

from pyomo.environ import *
from pyomo.dae import *

import time

# Import IDAES cores
from idaes.core import UnitModel, ProcBlock
import idaes.core.util.misc as model_util

__all__ = ['BFB_CLC Reactor']


@ProcBlock("BFB_CLC")
class _BFB_CLC(UnitModel):
    def __init__(self, *args, **kwargs):
        """
        Args:
        dae_method = method to use for calcuating derivatives (default = OCLR)
                    - BFD1 - 1st order backwards finite difference
                    - OCLR - Orthogonal collocation, Lagrange-Radau
                    - OCLL - Orthogonal collocation, Lagrange-Legendre
        fe_set = set of normalised finite element locations
        nfe = number of finite elements for bed discretization (default = 15)
                    (not used if fe_set specified)
        ncp = number of collocation points (OCLR or OCLL only, default = 3)
        s_inlet = type of solids feed (Top or Bottom)
        s_outlet = type of solids outlet (Overflow or Underflow)
        hx_type = arrangement of heat exchanger tubes (default = SPD)
                    - None
                    - SPD - Single pass tubes with downward flow
        gas_prop_lib = library file for gas phase properties
        sol_prop_lib = library file for solid phase properties
        hx_prop_lib = library file for heat ecxhanger fluid properties
        vb_method = correlation to use to calculate bubble velocity
                    (default = Davidson)
                    - Davidson
                    - Werther A
                    - Werther B
        ve_method = correlation to use to calculate emulsion region gas
                        velocity (default = Davidson)
                    - Davidson   - ve = vmf
                    - Abrahamson - Abrahamson and Geldart (A and AB classes)
                    - Hilligardt - Hilligardt and Werther ( B and D classes)
        ed_method = correlation to use to calculate emulsion region voidage
                        (default = Davidson)
                    - Davidson   - ed = emf
        """
        self.dae_method = kwargs.pop("dae_method", "OCLR")
        self.fe_set = kwargs.pop("fe_set", [])
        if self.fe_set == []:
            # Get nfe and populate fe_set
            self.nfe = kwargs.pop("nfe", 15)
            for i in range(0, self.nfe + 1):
                if i == 0:
                    self.fe_set.append(0.0)
                elif i < self.nfe:
                    self.fe_set.append(float(i / self.nfe))
                else:
                    self.fe_set.append(1.0)
        else:
            # Set nfe from len(fe_set)
            self.nfe = len(self.fe_set) - 1
            kwargs.pop("nfe", 15)

        print(self.fe_set)
        self.ncp = kwargs.pop("ncp", 3)

        self.s_inlet = kwargs.pop("s_inlet", "Bottom")
        if not ((self.s_inlet == 'Top') or (self.s_inlet == 'Bottom')):
            raise Exception('Solid feed type not recognised. ' \
                            'Must be either Top or Bottom')

        self.s_outlet = kwargs.pop("s_outlet", "Overflow")
        if not ((self.s_outlet == 'Overflow') or (self.s_outlet == 'Underflow')):
            raise Exception('Solid outlet type not recognised. ' \
                            'Must be either Overflow or Underflow')

        self.hx_type = kwargs.pop("hx_type", "SPD")

        self.gas_prop_lib = kwargs.pop("gas_prop_lib")
        self.sol_prop_lib = kwargs.pop("sol_prop_lib")
        self.hx_prop_lib = kwargs.pop("hx_prop_lib")

        self.vb_method = kwargs.pop("vb_method", "Davidson")
        self.ve_method = kwargs.pop("ve_method", "Davidson")
        self.ed_method = kwargs.pop("ed_method", "Davidson")

        UnitModel.__init__(self, *args, **kwargs)

    def build(self, *args, **kwargs):

        self._import_props()

        self._make_params()
        self._make_domain()

        self._make_vars()
        self._make_hydro_vars()
        self._make_HX_vars()

        self._dae_transform()

        self._make_props()

        self._make_bed_model()
        self._make_bdry_conds()
        self._make_hydro_model()

        self._make_HX_props()
        self._make_HX_model()

    def _import_props(self):
        # Import property packages
        self.gas_prop_lib = __import__(self.gas_prop_lib)
        self.sol_prop_lib = __import__(self.sol_prop_lib)

        # Declare Component Lists
        self.GasList = Set(initialize=self.gas_prop_lib._PropPack.comp)
        self.SolidList = Set(initialize=self.sol_prop_lib._PropPack.scomp)

        # If HX Tubes present, import HX fluid properties and component list
        if self.hx_type != None or self.hx_type != "None":
            self.hx_prop_lib = __import__(self.hx_prop_lib)
            self.HXList = Set(initialize=self.hx_prop_lib._PropPack.comp)

    def _make_params(self):
        """
        Create the paramters and sets for the Pyomo block
        """
        # Declare Imutable Parameters
        self.pi = Param(default=2 * acos(0), doc='pi')
        self.R = Param(default=8.314472,
                       doc='Universal Gas Constant [J/mol.K]')
        self.gc = Param(default=9.81,
                        doc='Gravitational Acceleration Constant [m^2/s]')
        self.eps = Param(default=1e-6,
                         doc='Smoothing Factor for Smooth IF Statements')

        # Declare Mutable Parameters
        self.ah = Param(within=PositiveReals, mutable=True, default=0.8,
                        doc='Emprical Factor for HX Tube Model')
        self.Cr = Param(within=PositiveReals, mutable=True, default=1,
                        doc='Average Correction Factor for HX Tube Model')
        self.fc_max = Param(within=PositiveReals, mutable=True, default=0.5,
                            doc='Maximum Cloud to Bubble Volume Ratio')
        self.fw = Param(within=PositiveReals, mutable=True, default=0.2,
                        doc='Bubble to Wake Volume Ratio')
        self.hw = Param(within=PositiveReals, mutable=True, default=1500,
                        doc='Heat Transfer Coefficient of HX Tube Walls [W/m^2.K]')
        self.Kd = Param(within=NonNegativeReals, mutable=True, default=0.1,
                        doc='Bulk Gas Permeation Coefficient [m/s]')

        # Declare Scaling Parameters
        self.sf_gc = Param(within=PositiveReals, mutable=True, default=1e6,
                           doc='Scaling factor for gas mole flow')
        self.sf_sc = Param(within=PositiveReals, mutable=True, default=1e6,
                           doc='Scaling factor for solids mass flux')
        self.sf_gh = Param(within=PositiveReals, mutable=True, default=1e-6,
                           doc='Scaling factor for gas enthalpy flow')
        self.sf_sh = Param(within=PositiveReals, mutable=True, default=1e-6,
                           doc='Scaling factor for solids enthalpy flux')
        if self.hx_type != None and self.hx_type != "None":
            self.sf_hh = Param(within=PositiveReals, mutable=True,
                               default=1e-3,
                               doc='Scaling factor for HX fluid enthalpy flow')

    def _make_domain(self):
        """
        Create the axial domains for the Pyomo block
        """
        # Declare Distribution Domain
        self.l_n = ContinuousSet(bounds=(0.0, 1.0),
                                 initialize=self.fe_set,
                                 doc='Set of Discrete Elements')

    def _make_vars(self):
        """
        Create the variables for the Pyomo block
        """
        # Discterisation point location and sizeSize
        self.l = Var(self.l_n, domain=Reals,
                     doc='Location of Discrete Elements [m]')

        # Vessel Dimensions
        self.Areact = Var(domain=Reals,
                          doc='Cross-Sectional Area of Reactor Vessel [m^2]')
        self.Ax = Var(domain=Reals,
                      doc='Cross-Sectional Area of Bed [m^2]')
        self.Dt = Var(domain=Reals,
                      doc='Diameter of Reactor Vessel [m]')
        self.Dte = Var(domain=NonNegativeReals,
                       doc='Hydraulic Diameter of Bed [m]')
        self.Lb = Var(domain=Reals,
                      doc='Depth of Bed [m]')

        # Distributor Design
        self.Ao = Var(domain=Reals,
                      doc='Distributor Plate Area per Orifice [m^2/orifice]')
        self.nor = Var(domain=Reals,
                       doc='Distributor Plate Orifices per Area [orifices/m^2]')

        # Gas Inlet Conditions
        self.Gas_In_F = Var(domain=Reals,
                            doc='Molar Flowrate of Gas at Gas Inlet [mol/s]')
        self.Gas_In_P = Var(domain=Reals,
                            doc='Pressure of Gas at Gas Inlet [Pa]')
        self.Gas_In_T = Var(domain=Reals,
                            doc='Temperature of Gas at Gas Inlet [K]')
        self.Gas_In_y = Var(self.GasList, domain=Reals,
                            doc='Mole Fractions of Gas Species at Gas Inlet [mol/mol]')

        # Solids Inlet Conditions
        self.Solid_In_F = Var(domain=Reals,
                              doc='Mass Flowrate of Solids at Solid Inlet [kg/s]')
        self.Solid_In_T = Var(domain=Reals,
                              doc='Temperature of Solids at Solids Inlet [k]')
        self.Solid_In_x = Var(self.SolidList, domain=Reals,
                              doc=' Mass fraction of Solids at Solids Inlet [-]')

        # Material Flows
        self.Gb = Var(self.l_n, domain=Reals,
                      doc='Bubble Region Molar Gas FLowrate [mol/s]')
        self.Ge = Var(self.l_n, domain=Reals,
                      doc='Emulsion Region Molar Gas FLowrate [mol/s]')
        self.Jc = Var(self.l_n, domain=Reals,
                      doc='Cloud-Wake Region Solids Flux [kg/m^2.s]')
        self.Je = Var(self.l_n, domain=Reals,
                      doc='Emulsion Region Solids Flux (Downward) [kg/m^2.s]')

        # Component Material Flows
        self.Gbc = Var(self.GasList, self.l_n, domain=Reals,
                       doc='Bubble Region Component Molar Gas FLowrate [mol/s]')
        self.Gec = Var(self.GasList, self.l_n, domain=Reals,
                       doc='Emulsion Region Component Molar Gas FLowrate [mol/s]')
        self.Jcc = Var(self.SolidList, self.l_n, domain=Reals,
                       doc='Cloud-Wake Region Solid Species Flux [kg/m^2.s]')
        self.Jec = Var(self.SolidList, self.l_n, domain=Reals,
                       doc='Emulsion Region Solid Species Flux (Downward)' \
                           '[kg/m^2.s]')

        # Enthalpy Flows
        self.Gbh = Var(self.l_n, domain=Reals,
                       doc='Bubble Region Scaled Gas Enthalpy Flowrate ' \
                           '[1/sf_gh * J/s]')
        self.Geh = Var(self.l_n, domain=Reals,
                       doc='Emulsion Region Scaled Gas Enthalpy Flowrate ' \
                           '[1/sf_gh * J/s]')
        self.Jch = Var(self.l_n, domain=Reals,
                       doc='Cloud-Wake Region Scaled Solids Enthalpy Flux ' \
                           '[1/sf_sh * J/m^2.s]')
        self.Jeh = Var(self.l_n, domain=Reals,
                       doc='Emulsion Region Scaled Solids Enthalpy Flux ' \
                           '[1/sf_sh * J/m^2.s]')

        # Temperatures and Pressures
        self.P = Var(self.l_n, domain=Reals, doc='Pressure [Pa]')
        self.Tgb = Var(self.l_n, domain=NonNegativeReals,
                       doc='Bubble Region Gas Temperature [K]')
        self.Tgc = Var(self.l_n, domain=NonNegativeReals,
                       doc='Cloud-Wake Region Gas Temperature [K]')
        self.Tge = Var(self.l_n, domain=NonNegativeReals,
                       doc='Emulsion Region Gas Temperature [K]')
        self.Tsc = Var(self.l_n, domain=NonNegativeReals,
                       doc='Cloud-Wake Region Solids Temperature [K]')
        self.Tse = Var(self.l_n, domain=NonNegativeReals,
                       doc='Emulsion Region Solids Temperature [K]')

        # Gas and Solid Compositions
        self.yb = Var(self.GasList, self.l_n, domain=NonNegativeReals,
                      doc='Bubble Region Gas Mole Fractions [-]')
        self.yc = Var(self.GasList, self.l_n, domain=NonNegativeReals,
                      doc='Cloud-Wake Region Gas Mole Fractions [-]')
        self.ye = Var(self.GasList, self.l_n, domain=NonNegativeReals,
                      doc='Emulsion Region Gas Mole Fractions [-]')
        self.xc = Var(self.SolidList, self.l_n, domain=NonNegativeReals,
                      doc='Cloud-Wake Region Solids Mass Fractions [-]')
        self.xe = Var(self.SolidList, self.l_n, domain=NonNegativeReals,
                      doc='Emulsion Region Solids Mass Fractions [-]')

        # Gas Phase Concentrations
        self.cb = Var(self.GasList, self.l_n, domain=NonNegativeReals,
                      doc='Bubble Region Gas Component Concentrations [mol/m^3]')
        self.cc = Var(self.GasList, self.l_n, domain=NonNegativeReals,
                      doc='Cloud-Wake Region Gas Component Concentrations [mol/m^3]')
        self.ce = Var(self.GasList, self.l_n, domain=NonNegativeReals,
                      doc='Emulsion Region Gas Component Concentrations [mol/m^3]')
        self.cct = Var(self.l_n, domain=NonNegativeReals,
                       doc='Cloud-Wake Region Total Gas Concentration [mol/m^3]')
        self.cet = Var(self.l_n, domain=NonNegativeReals,
                       doc='Emulsion Region Total Gas Concentration [mol/m^3]')

        # Velocities
        self.us = Var(self.l_n,
                      doc='Emulsion Region Solids Velocity (downwards) [m/s]')
        self.vb = Var(self.l_n, domain=NonNegativeReals,
                      doc='Bubble Velocity [m/s]')
        self.ve = Var(self.l_n, domain=NonNegativeReals,
                      doc='Emulsion Region Gas Velocity [m/s]')
        self.vg = Var(self.l_n, domain=NonNegativeReals,
                      doc='Superficial Gas Velocity [m/s]')

        # Bubble Dimensions and Hydrodynamics
        self.Ar = Var(self.l_n, domain=NonNegativeReals,
                      doc='Archimedes Number [-]')
        self.db = Var(self.l_n, domain=Reals,
                      doc='Average Bubble Diameter [m]', bounds=(2e-3, 100))  # bounds can be removed if not needed
        self.delta = Var(self.l_n, domain=Reals,
                         doc='Bubble Region Volume Fraction [m^3/m^3]')
        self.delta_e = Var(self.l_n, domain=Reals,
                           doc='Emulsion Region Volume Fraction [m^3/m^3]')
        self.e = Var(self.l_n, domain=Reals,
                     doc='Cross-Sectional Average Voidage Fraction [m^3/m^3]')
        self.ed = Var(self.l_n, domain=Reals,
                      doc='Emulsion Region Voidage Fraction [m^3/m^3]')
        self.fc = Var(self.l_n, domain=NonNegativeReals,
                      doc='Cloud to Bubble Region Volume Ratio [m^3/m^3]')
        self.fc_temp = Var(self.l_n, domain=Reals,
                           doc='Cloud to Bubble Region Volume Ratio [m^3/m^3], \
                            temporary value used with fc_max to bound fc')
        self.fcw = Var(self.l_n, domain=Reals,
                       doc='Cloud-Wake to Bubble Region Volume Ratio [m^3/m^3]')

        # Heat and Mass Transfer Coefficients
        self.Kbc = Var(self.GasList, self.l_n, domain=Reals,
                       doc='Bubble to Cloud-Wake Gas Mass Transfer Coefficient ' \
                           '[1/s]')
        self.Kce = Var(self.GasList, self.l_n, domain=NonNegativeReals,
                       doc='Cloud-Wake to Emulsion Gas Mass Transfer ' \
                           'Coefficient [1/s]')
        self.Kcebs = Var(self.l_n,
                         doc='Cloud-Wake to Emulsion Solid Mass Transfer ' \
                             'Coefficient [1/s]')
        self.Hbc = Var(self.l_n, domain=Reals,
                       doc='Bubble to Cloud-Wake Gas Energy Transfer ' \
                           'Coefficient [J/m^3.K.s]')
        self.Hce = Var(self.l_n, domain=NonNegativeReals,
                       doc='Cloud-Wake to Emulsion Gas Energy Transfer ' \
                           'Coefficient [J/m^3.K.s]')
        self.hp = Var(self.l_n, domain=Reals,
                      doc='Gas to Solid Energy Convective Heat Transfer ' \
                          'Coefficient [J/m^2.K.s]')

        # Heat and Mass Transfer Terms
        self.Kcebs_c = Var(self.SolidList, self.l_n, domain=Reals,
                           doc='Cloud-Wake to Emulsion Solid Species Mass ' \
                               'Transfer Rate [kg/s]')
        self.Hcebs = Var(self.l_n, domain=Reals,
                         doc='Cloud-Wake to Emulsion Solid Enthalpy Transfer ' \
                             'Rate [J/s]')
        self.hp_c = Var(self.l_n, domain=Reals,
                        doc='Cloud-Wake Region Gas to Solid Enthalpy Transfer ' \
                            'Rate [J/m^3.s]')
        self.hp_e = Var(self.l_n, domain=Reals,
                        doc='Emulsion Region Gas to Solid Enthalpy Transfer ' \
                            'Rate [J/m^3.s]')
        self.hr_c = Var(self.l_n, domain=Reals,
                        doc='Cloud-Wake Region Gas Heat Transfer due ' \
                            'to transfer of reacting mass [J/m^3.s]')
        self.hr_e = Var(self.l_n, domain=Reals,
                        doc='Emulsion Region Gas Heat Transfer due ' \
                            'to transfer of reacting mass [J/m^3.s]')

        # Bulk Heat and Mass Transfer
        self.Kgbulk_c = Var(self.GasList, self.l_n, domain=Reals,
                            doc='Gas Phase Component Bulk Transfer Rate [mol/s]')
        self.Ksbulk = Var(self.l_n, doc='Solid Phase Bulk Transfer Rate [kg/s]')
        self.Ksbulk_c = Var(self.SolidList, self.l_n,
                            doc='Solid Species Bulk Transfer Rate [kg/s]')
        self.Hgbulk = Var(self.l_n, domain=Reals,
                          doc='Gas Phase Bulk Enthalpy Trasnfer Rate [J/s]')
        self.Hsbulk = Var(self.l_n, domain=Reals,
                          doc='Solid Phase Bulk Enthalpy Trasnfer Rate [J/s]')

        # Reaction Rates
        self.rgc = Var(self.GasList, self.l_n, domain=Reals,
                       doc='Cloud-Wake Region Gas Phase Reaction Rates ' \
                           '[mol/m^3.s]')
        self.rge = Var(self.GasList, self.l_n, domain=Reals,
                       doc='Emulsion Region Gas Phase Reaction Rates ' \
                           '[mol/m^3.s]')
        self.rsc = Var(self.SolidList, self.l_n, domain=Reals,
                       doc='Cloud-Wake Region Solid Phase Reaction Rates ' \
                           '[kg/m^3.s]')
        self.rse = Var(self.SolidList, self.l_n, domain=Reals,
                       doc='Cloud-Wake Region Solid Phase Reaction Rates ' \
                           '[kg/m^3.s]')

        # Heat Transfer from Bed
        self.Qhx = Var(self.l_n, domain=Reals,
                       doc='Heat Transfered to from Bed in Discrete Element ' \
                           '[J/s]')

        # Outlet Variables
        # Gas Outlet Conditions
        self.Gas_Out_F = Var(domain=Reals,
                             doc='Gas Flowrate at Gas Outlet [mol/s]')
        self.Gas_Out_P = Var(domain=Reals,
                             doc='Pressure at Gas Outlet [Pa]')
        self.Gas_Out_T = Var(domain=Reals,
                             doc='Gas Temperature at Gas Outlet [K]')
        self.Gas_Out_y = Var(self.GasList, domain=Reals,
                             doc='Gas Mole Fractions at Gas Outlet [mol/mol]')

        # Solids Outlet Conditions
        self.Solid_Out_F = Var(domain=Reals,
                               doc='Solid Flowrate at Solid Outlet [kg/s]')
        self.Solid_Out_T = Var(domain=Reals,
                               doc='Solid Temperature at Solid Outlet [K]')
        self.Solid_Out_x = Var(self.SolidList, domain=Reals,
                               doc='Mass fractions of Solids at Solids Outlet [-]')

        # Derivative Variables
        self.dGbcdx = DerivativeVar(self.Gbc, wrt=self.l_n)
        self.dGecdx = DerivativeVar(self.Gec, wrt=self.l_n)
        self.dGbhdx = DerivativeVar(self.Gbh, wrt=self.l_n)
        self.dGehdx = DerivativeVar(self.Geh, wrt=self.l_n)
        self.dJccdx = DerivativeVar(self.Jcc, wrt=self.l_n)
        self.dJecdx = DerivativeVar(self.Jec, wrt=self.l_n)
        self.dJchdx = DerivativeVar(self.Jch, wrt=self.l_n)
        self.dJehdx = DerivativeVar(self.Jeh, wrt=self.l_n)
        self.dldx = DerivativeVar(self.l, wrt=self.l_n)
        self.dPdx = DerivativeVar(self.P, wrt=self.l_n)

    def _make_bed_model(self):
        """
        Create main body of model.

        Groups: A,B (except boundaries), C, D, E, F, G, L
        """

        # Reactor Sizing and Design
        # a1 - Finite element location
        def rule_eq_a1(b, i):
            if i == b.l_n.first():
                return b.l[i] == 0
            else:
                return b.dldx[i] == b.Lb

        self.eq_a1 = Constraint(self.l_n, rule=rule_eq_a1)

        #        # a2 - Distributor Design
        self.eq_a2 = Constraint(expr=10 == self.nor * self.Ao * 10)

        # a3 - Total Reactor Cross-Sectional Area
        self.eq_a3 = Constraint(expr=self.Areact \
                                     == 0.25 * self.pi * self.Dt ** 2)

        # ---------------------------------------------------------------------
        # Mass and Energy Balances
        # b1 - Bubble Region Gas Component Balances
        def rule_eq_b1(b, i, j):
            if i == b.l_n.first():
                return Constraint.Skip
            else:
                return 0 == -b.dGbcdx[j, i] * b.sf_gc + b.sf_gc * b.Lb * b.Kgbulk_c[j, i] \
                       - b.sf_gc * b.Lb * b.Ax * b.delta[i] * b.Kbc[j, i] \
                       * (b.cb[j, i] - b.cc[j, i])

        self.eq_b1 = Constraint(self.l_n, self.GasList, rule=rule_eq_b1)

        # b2 - Cloud-Wake Region Gas Component Balances
        def rule_eq_b2(b, i, j):
            if i == b.l_n.first():
                return Constraint.Skip
            else:
                return 0 == b.sf_gc * b.Kbc[j, i] * (b.cb[j, i] - b.cc[j, i]) \
                       - b.sf_gc * b.Kce[j, i] * (b.cc[j, i] - b.ce[j, i]) \
                       - b.sf_gc * b.rgc[j, i]

        self.eq_b2 = Constraint(self.l_n, self.GasList, rule=rule_eq_b2)

        # b3 - Emulsion Region Gas Component Balances
        def rule_eq_b3(b, i, j):
            if i == b.l_n.first():
                return Constraint.Skip
            else:
                return 0 == -b.dGecdx[j, i] * b.sf_gc - b.sf_gc * b.Lb * b.Kgbulk_c[j, i] \
                       + b.sf_gc * b.Lb * b.Ax * b.delta[i] * b.Kce[j, i] \
                       * (b.cc[j, i] - b.ce[j, i]) \
                       - b.sf_gc * b.Lb * b.Ax * b.rge[j, i]

        self.eq_b3 = Constraint(self.l_n, self.GasList, rule=rule_eq_b3)

        # b4 - Bubble Region Gas Enthalpy Balance
        def rule_eq_b4(b, i):
            if i == 0:
                return Constraint.Skip
            else:
                return 0 == -b.dGbhdx[i] + b.sf_gh * b.Lb * b.Hgbulk[i] \
                       - b.sf_gh * b.Lb * b.Ax * b.delta[i] * b.Hbc[i] \
                       * (b.Tgb[i] - b.Tgc[i])

        self.eq_b4 = Constraint(self.l_n, rule=rule_eq_b4)

        # b5 - Cloud-Wake Region Gas Enthalpy Balance
        def rule_eq_b5(b, i):
            if i == 0:
                return Constraint.Skip
            else:
                return 0 == b.Hbc[i] * (b.Tgb[i] - b.Tgc[i]) \
                       - b.Hce[i] * (b.Tgc[i] - b.Tge[i]) \
                       - b.hp_c[i] - b.hr_c[i]

        self.eq_b5 = Constraint(self.l_n, rule=rule_eq_b5)

        # b6 - Emulsion Region Gas Enthalpy Balance
        def rule_eq_b6(b, i):
            if i == 0:
                return Constraint.Skip
            else:
                return 0 == -b.dGehdx[i] - b.sf_gh * b.Lb * b.Hgbulk[i] \
                       + b.sf_gh * b.Lb * b.Ax * b.delta[i] \
                       * b.Hce[i] * (b.Tgc[i] - b.Tge[i]) \
                       - b.sf_gh * b.Lb * b.Ax * b.hp_e[i] \
                       - b.sf_gh * b.Lb * b.Ax * b.hr_e[i]

        self.eq_b6 = Constraint(self.l_n, rule=rule_eq_b6)

        # b7 - Cloud-Wake Region Adsorbed Species Balance
        def rule_eq_b7(b, i, j):
            if i == 0:
                return Constraint.Skip
            else:
                return 0 == -b.dJccdx[j, i] * b.sf_sc \
                       + b.sf_sc * b.Lb * b.Ksbulk_c[j, i] - b.sf_sc * b.Lb * b.Kcebs_c[j, i] \
                       + b.sf_sc * b.Lb * b.rsc[j, i]

        self.eq_b7 = Constraint(self.l_n, self.SolidList, rule=rule_eq_b7)

        # b8 - Emulsion Region Adsorbed Species Balance
        def rule_eq_b8(b, i, j):
            if i == 0:
                return Constraint.Skip
            else:
                return 0 == b.dJecdx[j, i] * b.sf_sc \
                       - b.sf_sc * b.Lb * b.Ksbulk_c[j, i] + b.sf_sc * b.Lb * b.Kcebs_c[j, i] \
                       + b.sf_sc * b.Lb * b.rse[j, i]

        self.eq_b8 = Constraint(self.l_n, self.SolidList, rule=rule_eq_b8)

        # b9 - Cloud-Wake Region Solid Enthalpy Balance
        def rule_eq_b9(b, i):
            if i == 0:
                return Constraint.Skip
            else:
                return 0 == -b.dJchdx[i] \
                       + b.sf_sh * b.Lb * b.Hsbulk[i] \
                       - b.sf_sh * b.Lb * b.Hcebs[i] \
                       + b.sf_sh * b.Lb * b.delta[i] * b.hp_c[i] \
                       + b.sf_sh * b.Lb * b.delta[i] * b.hr_c[i]

        self.eq_b9 = Constraint(self.l_n, rule=rule_eq_b9)

        # b10 - Emulsion Region Solid Enthalpy Balance
        def rule_eq_b10(b, i):
            if i == 0:
                return Constraint.Skip
            else:
                return 0 == b.dJehdx[i] * b.Ax \
                       - b.sf_sh * b.Lb * b.Ax * b.Hsbulk[i] \
                       + b.sf_sh * b.Lb * b.Ax * b.Hcebs[i] \
                       + b.sf_sh * b.Lb * b.Ax * b.hp_e[i] \
                       + b.sf_sh * b.Lb * b.Ax * b.hr_e[i] \
                       + b.sf_sh * b.Lb * b.Qhx[i]

        self.eq_b10 = Constraint(self.l_n, rule=rule_eq_b10)

        # ---------------------------------------------------------------------
        # Flowrate and Flux Relationships
        # c1 - Bubble Gas Component Flowrate
        def rule_eq_c1(b, i, j):
            return b.Gbc[j, i] == b.Gb[i] * b.yb[j, i]

        self.eq_c1 = Constraint(self.l_n, self.GasList, rule=rule_eq_c1)

        # c2 - Emulsion Gas Component Flowrate
        def rule_eq_c2(b, i, j):
            return b.Gec[j, i] == b.Ge[i] * b.ye[j, i]

        self.eq_c2 = Constraint(self.l_n, self.GasList, rule=rule_eq_c2)

        # c3 - Bubble Gas Enthalpy Flowrate
        def rule_eq_c3(b, i):
            return b.Gbh[i] == b.sf_gh * b.Gb[i] * b.gas_prop_b[i].h_vap

        self.eq_c3 = Constraint(self.l_n, rule=rule_eq_c3)

        # c4 - Emulsion Gas Enthalpy Flowrate
        def rule_eq_c4(b, i):
            if i == b.l_n.first():
                return b.Geh[i] == b.sf_gh * b.Ge[i] * b.gas_prop_b[i].h_vap
            else:
                return b.Geh[i] == b.sf_gh * b.Ge[i] * b.gas_prop_e[i].h_vap

        self.eq_c4 = Constraint(self.l_n, rule=rule_eq_c4)

        # c5 - Cloud-Wake Region Solids Flux
        def rule_eq_c5(b, i):
            return b.Jc[i] == b.fw * b.delta[i] * b.sol_prop_c[i].rho_sol \
                   * (1 - b.ed[i]) * b.vb[i]

        self.eq_c5 = Constraint(self.l_n, rule=rule_eq_c5)

        # c6 - Emulsion Region Solids Velocity
        def rule_eq_c6(b, i):
            return b.Je[i] == b.delta_e[i] \
                   * b.sol_prop_e[i].rho_sol * (1 - b.ed[i]) * b.us[i]

        self.eq_c6 = Constraint(self.l_n, rule=rule_eq_c6)

        # c7 - CW Solids Species Flux
        def rule_eq_c7(b, i, j):
            return b.Jcc[j, i] == b.Jc[i] * b.xc[j, i]

        self.eq_c7 = Constraint(self.l_n, self.SolidList, rule=rule_eq_c7)

        # c8 - Emulsion Solids Species Flux
        def rule_eq_c8(b, i, j):
            return b.Jec[j, i] == b.Je[i] * b.xe[j, i]

        self.eq_c8 = Constraint(self.l_n, self.SolidList, rule=rule_eq_c8)

        # c9 - CW Solids Enthalpy Flux
        def rule_eq_c9(b, i):
            return b.Jch[i] == b.sf_sh * b.Jc[i] * b.sol_prop_c[i].h_sol

        self.eq_c9 = Constraint(self.l_n, rule=rule_eq_c9)

        # c10 - Emulsion Solids Enthalpy Flux
        def rule_eq_c10(b, i):
            return b.Jeh[i] == b.sf_sh * b.Je[i] * b.sol_prop_e[i].h_sol

        self.eq_c10 = Constraint(self.l_n, rule=rule_eq_c10)

        # c11 - Superficial Gas Velocity
        def rule_eq_c11(b, i):
            if i == b.l_n.first():
                return b.Gas_In_F * b.gas_prop_b[0].V_vap == b.Ax * b.vg[0]
            else:
                return b.vg[i] == b.vb[i] * b.delta[i] + b.ve[i] * b.delta_e[i]

        self.eq_c11 = Constraint(self.l_n, rule=rule_eq_c11)

        # c12 - Bubble Gas Velocity
        def rule_eq_c12(b, i):
            return b.Ax * b.delta[i] * b.vb[i] == b.Gb[i] * b.gas_prop_b[i].V_vap

        self.eq_c12 = Constraint(self.l_n, rule=rule_eq_c12)

        # c13 - Emulsion Gas Flowrate
        def rule_eq_c13(b, i):
            if i == b.l_n.first():
                return b.ve[i] * b.Ax * b.delta_e[i] \
                       == b.Ge[i] * b.gas_prop_b[i].V_vap
            else:
                return b.ve[i] * b.Ax * b.cet[i] * b.delta_e[i] == b.Ge[i]

        self.eq_c13 = Constraint(self.l_n, rule=rule_eq_c13)

        # ---------------------------------------------------------------------
        # Mole Fraction Relationships
        # e1 - Bubble Region Component Concentrations
        def rule_eq_e1(b, i, j):
            if i == b.l_n.first():
                return Constraint.Skip
            else:
                return b.cb[j, i] * b.gas_prop_b[i].V_vap == b.yb[j, i]

        self.eq_e1 = Constraint(self.l_n, self.GasList, rule=rule_eq_e1)

        # e2 - Cloud-Wake Region Component Concentrations
        def rule_eq_e2(b, i, j):
            if i == b.l_n.first():
                return Constraint.Skip
            else:
                return b.cc[j, i] == b.yc[j, i] * b.cct[i]

        self.eq_e2 = Constraint(self.l_n, self.GasList, rule=rule_eq_e2)

        # e3 - Emulsion Region Component Concentrations
        def rule_eq_e3(b, i, j):
            if i == b.l_n.first():
                return Constraint.Skip
            else:
                return b.ce[j, i] == b.ye[j, i] * b.cet[i]

        self.eq_e3 = Constraint(self.l_n, self.GasList, rule=rule_eq_e3)

        # e4 - Bubble Region Sum of Gas Mole Fractions
        def rule_eq_e4(b, i):
            if i == b.l_n.first():
                return Constraint.Skip
            else:
                return 1 == sum(b.yb[j, i] for j in b.GasList)

        self.eq_e4 = Constraint(self.l_n, rule=rule_eq_e4)

        # e5 - Cloud-Wake Region Sum of Gas Mole Fractions
        def rule_eq_e5(b, i):
            if i == b.l_n.first():
                return Constraint.Skip
            else:
                return 1 == sum(b.yc[j, i] for j in b.GasList)

        self.eq_e5 = Constraint(self.l_n, rule=rule_eq_e5)

        # e6 - Emulsion Region Sum of Gas Mole Fractions
        def rule_eq_e6(b, i):
            if i == b.l_n.first():
                return Constraint.Skip
            else:
                return 1 == sum(b.ye[j, i] for j in b.GasList)

        self.eq_e6 = Constraint(self.l_n, rule=rule_eq_e6)

        # e7 - Cloud-Wake Region Sum of Solid Mass Fractions
        def rule_eq_e7(b, i):
            return 1 == sum(b.xc[j, i] for j in b.SolidList)

        self.eq_e7 = Constraint(self.l_n, rule=rule_eq_e7)

        # e8 - Emulsion Region Sum of Solid Mass Fractions
        def rule_eq_e8(b, i):
            return 1 == sum(b.xe[j, i] for j in b.SolidList)

        self.eq_e8 = Constraint(self.l_n, rule=rule_eq_e8)

        # ---------------------------------------------------------------------
        # Bulk Flow and Mixing Relationships
        # f1 - Bulk Gas Mass Transfer
        def rule_eq_f1(b, i, j):
            if i == b.l_n.first():
                return Constraint.Skip
            else:
                return b.Kgbulk_c[j, i] * b.db[i] == (6 * b.Kd * b.delta[i] * b.Ax) \
                       * (0.5 * ((b.cet[i] - (1 / b.gas_prop_b[i].V_vap)) \
                                 + sqrt((b.cet[i] - (1 / b.gas_prop_b[i].V_vap)) ** 2 \
                                        + b.eps ** 2)) * b.ye[j, i] \
                          + 0.5 * ((b.cet[i] - (1 / b.gas_prop_b[i].V_vap)) \
                                   - sqrt((b.cet[i] - (1 / b.gas_prop_b[i].V_vap)) ** 2 \
                                          + b.eps ** 2)) * b.yb[j, i])

        self.eq_f1 = Constraint(self.l_n, self.GasList, rule=rule_eq_f1)

        # f2 - Bulk Gas Enthalpy Transfer
        def rule_eq_f2(b, i):
            if i == b.l_n.first():
                return Constraint.Skip
            else:
                return b.Hgbulk[i] * b.db[i] == (6 * b.Kd * b.delta[i] * b.Ax) \
                       * (0.5 * ((b.cet[i] - (1 / b.gas_prop_b[i].V_vap)) \
                                 + sqrt((b.cet[i] - (1 / b.gas_prop_b[i].V_vap)) ** 2 \
                                        + b.eps ** 2)) * b.gas_prop_e[i].h_vap \
                          - 0.5 * (-(b.cet[i] - (1 / b.gas_prop_b[i].V_vap)) \
                                   + sqrt((b.cet[i] - (1 / b.gas_prop_b[i].V_vap)) ** 2 \
                                          + b.eps ** 2)) * b.gas_prop_b[i].h_vap)

        self.eq_f2 = Constraint(self.l_n, rule=rule_eq_f2)

        # f3 - Bulk Solids Mass Transfer
        def rule_eq_f3(b, i, j):
            if i == 0:
                return Constraint.Skip
            else:
                return b.Ksbulk_c[j, i] == 0.5 * (b.Ksbulk[i] \
                                                  + sqrt(b.Ksbulk[i] ** 2 + b.eps ** 2)) \
                       * b.xe[j, i] \
                       - 0.5 * (-b.Ksbulk[i] \
                                + sqrt(b.Ksbulk[i] ** 2 + b.eps ** 2)) \
                       * b.xc[j, i]

        self.eq_f3 = Constraint(self.l_n, self.SolidList, rule=rule_eq_f3)

        # f4 - Bulk Solids Enthalpy Transfer
        def rule_eq_f4(b, i):
            if i == 0:
                return Constraint.Skip
            else:
                return b.Hsbulk[i] == 0.5 * (b.Ksbulk[i] \
                                             + sqrt(b.Ksbulk[i] ** 2 + b.eps ** 2)) \
                       * b.sol_prop_e[i].h_sol \
                       - 0.5 * (-b.Ksbulk[i] \
                                + sqrt(b.Ksbulk[i] ** 2 + b.eps ** 2)) \
                       * b.sol_prop_c[i].h_sol

        self.eq_f4 = Constraint(self.l_n, rule=rule_eq_f4)

        # f5 - Bulk Solids Component Mixing
        def rule_eq_f5(b, i, j):
            if i == 0:
                return Constraint.Skip
            else:
                return b.Kcebs_c[j, i] == b.delta[i] * b.sol_prop_e[i].rho_sol \
                       * b.Kcebs[i] * (b.xc[j, i] - b.xe[j, i])

        self.eq_f5 = Constraint(self.l_n, self.SolidList, rule=rule_eq_f5)

        # f6 - Bulk Solids Enthalpy Mixing
        def rule_eq_f6(b, i):
            if i == 0:
                return Constraint.Skip
            else:
                return b.Hcebs[i] == b.delta[i] * b.sol_prop_e[i].rho_sol \
                       * b.Kcebs[i] \
                       * (b.sol_prop_c[i].h_sol \
                          - b.sol_prop_e[i].h_sol)

        self.eq_f6 = Constraint(self.l_n, rule=rule_eq_f6)

        # f7 - Cloud-Wake Region Gas-Solids Convective Enthalpy Transfer
        def rule_eq_f7(b, i):
            if i == 0:
                return Constraint.Skip
            else:
                return b.hp_c[i] == b.fcw[i] * (1 - b.ed[i]) \
                       * b.sol_prop_c[i].rho_sol \
                       * b.sol_prop_c[i].ap \
                       * b.hp[i] * (b.Tgc[i] - b.Tsc[i])

        self.eq_f7 = Constraint(self.l_n, rule=rule_eq_f7)

        # f8 - Emulsion Region Gas-Solids Convective Enthalpy Transfer
        def rule_eq_f8(b, i):
            if i == 0:
                return Constraint.Skip
            else:
                return b.hp_e[i] == b.delta_e[i] * (1 - b.ed[i]) \
                       * b.sol_prop_e[i].rho_sol \
                       * b.sol_prop_e[i].ap * b.hp[i] \
                       * (b.Tge[i] - b.Tse[i])

        self.eq_f8 = Constraint(self.l_n, rule=rule_eq_f8)

        def rule_eq_f9(b, i):
            if i == 0:
                return Constraint.Skip
            else:
                return b.hr_c[i] == b.fcw[i] * (1 - b.ed[i]) * b.sol_prop_c[i].ht_rxn

        self.eq_f9 = Constraint(self.l_n, rule=rule_eq_f9)

        def rule_eq_f10(b, i):
            if i == 0:
                return Constraint.Skip
            else:
                return b.hr_e[i] == b.delta_e[i] * (1 - b.ed[i]) \
                       * b.sol_prop_e[i].ht_rxn

        self.eq_f10 = Constraint(self.l_n, rule=rule_eq_f10)

        # ---------------------------------------------------------------------
        # g1 - Pressure Drop
        def rule_eq_g1(b, i):
            if i == b.l_n.first():
                return b.P[i] == b.Gas_In_P - 3400
            else:
                return b.dPdx[i] == -b.Lb * (1 - b.e[i]) \
                       * b.sol_prop_e[i].rho_sol * b.gc

        self.eq_g1 = Constraint(self.l_n, rule=rule_eq_g1)

        # h1 - Archimedes Number
        def rule_eq_h1(b, i):
            if i == 0:
                return Constraint.Skip
            else:
                return b.Ar[i] * b.gas_prop_e[i].mu_vap ** 2 \
                       == (b.sol_prop_e[i].dp ** 3) * b.gas_prop_e[i].rho_vap \
                       * (b.sol_prop_e[i].rho_sol \
                          - b.gas_prop_e[i].rho_vap) * b.gc

        self.eq_h1 = Constraint(self.l_n, rule=rule_eq_h1)

        # h2 - Emulsion Region Volume Fraction
        def rule_eq_h2(b, i):
            return b.delta_e[i] == 1 - (b.fcw[i] + 1) * b.delta[i]

        self.eq_h2 = Constraint(self.l_n, rule=rule_eq_h2)

        # ---------------------------------------------------------------------
        # Reaction rate constraints
        # l1 - Cloud-Wake Gas Phase Reaction Rate
        def rule_l1(b, i, j):
            if i == b.l_n.first():
                return Constraint.Skip
            else:
                return b.rgc[j, i] == b.sol_prop_c[i].rg[j] * b.fcw[i] * (1 - b.ed[i])

        self.eq_l1 = Constraint(self.l_n, self.GasList, rule=rule_l1)

        # l2 - Cloud-Wake Solid Phase Reaction Rate
        def rule_l2(b, i, j):
            if i == b.l_n.first():
                return Constraint.Skip
            else:
                return b.rsc[j, i] == b.sol_prop_c[i].rs[j] * b.delta[i] \
                       * b.fcw[i] * (1 - b.ed[i])

        self.eq_l2 = Constraint(self.l_n, self.SolidList, rule=rule_l2)

        # l3 - Emulsion Gas Phase Reaction Rate
        def rule_l3(b, i, j):
            if i == b.l_n.first():
                return Constraint.Skip
            else:
                return b.rge[j, i] == b.sol_prop_e[i].rg[j] * b.delta_e[i] \
                       * (1 - b.ed[i])

        self.eq_l3 = Constraint(self.l_n, self.GasList, rule=rule_l3)

        # l4 - Emulsion Solid Phase Reaction Rate
        def rule_l4(b, i, j):
            if i == b.l_n.first():
                return Constraint.Skip
            else:
                return b.rse[j, i] == b.sol_prop_e[i].rs[j] * b.delta_e[i] \
                       * (1 - b.ed[i])

        self.eq_l4 = Constraint(self.l_n, self.SolidList, rule=rule_l4)

    def _make_bdry_conds(self):
        """
        Create boundary conditions.
        """
        # Assume excess gas goes to bubble region
        # d1 - Inlet Total Gas Balance
        self.eq_d1 = Constraint(expr=self.Gas_In_F \
                                     == self.Gb[0] + self.Ge[0])

        # d2 - Initial Bubble Region Gas Mole Fractions
        def rule_eq_d2(b, j):
            return b.yb[j, 0] == b.Gas_In_y[j]

        self.eq_d2 = Constraint(self.GasList, rule=rule_eq_d2)

        # d3 - Initial Emulsion Region Gas Mole Fractions
        def rule_eq_d3(b, j):
            return b.ye[j, 0] == b.Gas_In_y[j]

        self.eq_d3 = Constraint(self.GasList, rule=rule_eq_d3)

        # d4 - Bubble Region Gas Inlet Temperature
        self.eq_d4 = Constraint(expr=self.Tgb[0] == self.Gas_In_T)

        # d5 - Emulsion Region Gas Inlet Temperature
        self.eq_d5 = Constraint(expr=self.Tge[0] == self.Gas_In_T)

        # ---------------------------------------------------------------------
        # d6-d9 - Gas outlet conditions
        self.eq_d6 = Constraint(expr=1 == sum(self.Gas_Out_y[j] \
                                              for j in self.GasList))
        self.eq_d7 = Constraint(expr=self.sf_gh * self.Gas_Out_F \
                                     * self.gas_prop_out.h_vap \
                                     == self.Gbh[1] + self.Geh[1])
        self.eq_d8 = Constraint(expr=self.Gas_Out_P == self.P[1])

        def rule_d9(b, j):
            return b.Gas_Out_F * b.Gas_Out_y[j] == b.Gbc[j, 1] + b.Gec[j, 1]

        self.eq_d9 = Constraint(self.GasList, rule=rule_d9)

        # ---------------------------------------------------------------------
        if self.s_inlet == 'Bottom' and self.s_outlet == 'Underflow':
            # Bottom Feed, Underflow Outlet
            # d10 - Solids Recirculation at Bottom of Bed
            def rule_eq_d10(b, j):
                return 0 == b.Jec[j, 0] * b.Ax + b.Solid_In_F * b.Solid_In_x[j] \
                       - b.Jcc[j, 0] * b.Ax - b.Solid_Out_F * b.xe[j, 0]

            self.eq_d10 = Constraint(self.SolidList, rule=rule_eq_d10)

            # d11 - Solids Recirculation at Top of Bed
            def rule_eq_d11(b, j):
                return 0 == b.Jcc[j, 1] - b.Jec[j, 1]

            self.eq_d11 = Constraint(self.SolidList, rule=rule_eq_d11)

            # d12 - Solids Enthalpy Recirculation at Bottom of Bed
            self.eq_d12 = Constraint(expr=0 == self.Jeh[0] * self.Ax \
                                          + self.sf_sh * self.Solid_In_F \
                                          * self.sol_prop_f.h_sol \
                                          - self.Jch[0] * self.Ax \
                                          - self.sf_sh * self.Solid_Out_F \
                                          * self.sol_prop_e[0].h_sol)

            # d13 - Solids Enthalpy Recirculation at Top of Bed
            self.eq_d13 = Constraint(expr=0 == self.Jch[1] - self.Jeh[1])

            # d14 - Solids Outlet Composition
            def rule_eq_d14(b, j):
                return b.Solid_Out_x[j] == b.xe[j, 0]

            self.eq_d14 = Constraint(self.SolidList, rule=rule_eq_d14)

            # d15 - Solids Outlet Temperature
            self.eq_d15 = Constraint(expr=self.Solid_Out_T == self.Tse[0])

        # ---------------------------------------------------------------------
        elif self.s_inlet == 'Bottom' and self.s_outlet == 'Overflow':
            # Bottom Feed, Overflow Outlet
            # d10 - Solids Recirculation at Bottom of Bed
            def rule_eq_d10(b, j):
                return 0 == b.Jec[j, 0] * b.Ax + b.Solid_In_F * b.Solid_In_x[j] \
                       - b.Jcc[j, 0] * b.Ax

            self.eq_d10 = Constraint(self.SolidList, rule=rule_eq_d10)

            # d11 - Solids Recirculation at Top of Bed
            def rule_eq_d11(b, j):
                return 0 == b.Jcc[j, 1] * b.Ax - b.Jec[j, 1] * b.Ax \
                       - b.Solid_Out_F * b.xe[j, 1]

            self.eq_d11 = Constraint(self.SolidList, rule=rule_eq_d11)

            # d12 - Solids Enthalpy Recirculation at Bottom of Bed
            self.eq_d12 = Constraint(expr=0 == self.Jeh[0] * self.Ax \
                                          + self.sf_sh * self.Solid_In_F \
                                          * self.sol_prop_f.h_sol \
                                          - self.Jch[0] * self.Ax)

            # d13 - Solids Enthalpy Recirculation at Top of Bed
            self.eq_d13 = Constraint(expr=0 == self.Jch[1] * self.Ax \
                                          - self.Jeh[1] * self.Ax \
                                          - self.sf_sh * self.Solid_Out_F \
                                          * self.sol_prop_e[1].h_sol)

            # d14 - Solids Outlet Composition
            def rule_eq_d14(b, j):
                return b.Solid_Out_x[j] == b.xe[j, 1]

            self.eq_d14 = Constraint(self.SolidList, rule=rule_eq_d14)

            # d15 - Solids Outlet Temperature
            self.eq_d15 = Constraint(expr=self.Solid_Out_T == self.Tse[1])

        # ---------------------------------------------------------------------
        elif self.s_inlet == 'Top' and self.s_outlet == 'Underflow':
            # Top Feed, Underflow Outlet
            # d10 - Solids Recirculation at Bottom of Bed
            def rule_eq_d10(b, j):
                return 0 == b.Jec[j, 0] * b.Ax \
                       - b.Jcc[j, 0] * b.Ax - b.Solid_Out_F * b.xe[j, 0]

            self.eq_d10 = Constraint(self.SolidList, rule=rule_eq_d10)

            # d11 - Solids Recirculation at Top of Bed
            def rule_eq_d11(b, j):
                return 0 == b.Jcc[j, 1] * b.Ax + b.Solid_In_F * b.Solid_In_x[j] \
                       - b.Jec[j, 1] * b.Ax

            self.eq_d11 = Constraint(self.SolidList, rule=rule_eq_d11)

            # d12 - Solids Enthalpy Recirculation at Bottom of Bed
            self.eq_d12 = Constraint(expr=0 == self.Jeh[0] * self.Ax \
                                          - self.Jch[0] * self.Ax \
                                          - self.sf_sh * self.Solid_Out_F \
                                          * self.sol_prop_e[0].h_sol)

            # d13 - Solids Enthalpy Recirculation at Top of Bed
            self.eq_d13 = Constraint(expr=0 == self.Jch[1] * self.Ax \
                                          + self.sf_sh * self.Solid_In_F \
                                          * self.sol_prop_f.h_sol \
                                          - self.Jeh[1] * self.Ax)

            # d14 - Solids Outlet Composition
            def rule_eq_d14(b, j):
                return b.Solid_Out_x[j] == b.xe[j, 0]

            self.eq_d14 = Constraint(self.SolidList, rule=rule_eq_d14)

            # d15 - Solids Outlet Temperature
            self.eq_d15 = Constraint(expr=self.Solid_Out_T == self.Tse[0])

        # ---------------------------------------------------------------------
        elif self.s_inlet == 'Top' and self.s_outlet == 'Overflow':
            # Top Feed, Overflow Outlet
            # d10 - Solids Recirculation at Bottom of Bed
            def rule_eq_d10(b, j):
                return 0 == b.Jec[j, 0] - b.Jcc[j, 0]

            self.eq_d10 = Constraint(self.SolidList, rule=rule_eq_d10)

            # d11 - Solids Recirculation at Top of Bed
            def rule_eq_d11(b, j):
                return 0 == b.Jcc[j, 1] * b.Ax + b.Solid_In_F * b.Solid_In_x[j] \
                       - b.Jec[j, 1] * b.Ax - b.Solid_Out_F * b.xe[j, 1]

            self.eq_d11 = Constraint(self.SolidList, rule=rule_eq_d11)

            # d12 - Solids Enthalpy Recirculation at Bottom of Bed
            self.eq_d12 = Constraint(expr=0 == self.Jeh[0] - self.Jch[0])

            # d13 - Solids Enthalpy Recirculation at Top of Bed
            self.eq_d13 = Constraint(expr=0 == self.Jch[1] * self.Ax \
                                          + self.sf_sh * self.Solid_In_F \
                                          * self.sol_prop_f.h_sol \
                                          - self.Jeh[1] * self.Ax \
                                          - self.sf_sh * self.Solid_Out_F \
                                          * self.sol_prop_e[1].h_sol)

            # d14 - Solids Outlet Composition
            def rule_eq_d14(b, j):
                return b.Solid_Out_x[j] == b.xe[j, 1]

            self.eq_d14 = Constraint(self.SolidList, rule=rule_eq_d14)

            # d15 - Solids Outlet Temperature
            self.eq_d15 = Constraint(expr=self.Solid_Out_T == self.Tse[1])
        # ---------------------------------------------------------------------
        else:
            raise Exception('Unrecognised solid inlet or outlet')

    def _make_hydro_vars(self):
        """
        Create the variables needed for the hydrodynamic model.

        Group: H, I
        """
        # Create variables internal to hydrodynamic model
        self.dbm = Var(self.l_n, domain=Reals,
                       doc='Maximum Theoretical Bubble Diameter [m]')
        self.g1 = Var(self.l_n, domain=Reals,
                      doc='Bubble Growth Coefficient [-]')
        self.vbr = Var(self.l_n, domain=NonNegativeReals,
                       doc='Bubble Rise Velocity [m/s]')

        self.ddbdx = DerivativeVar(self.db, wrt=self.l_n)

    def _make_hydro_model(self):
        """
        Create the hydrodynamic model for the bed.

        Group: H, I
        """

        # Fluidisation Conditions and Bubble Behaviour
        # h3 - Emulsion Region Gas Velocity
        def rule_eq_h3(b, i):
            if b.ve_method == 'Davidson':
                return b.ve[i] == b.sol_prop_e[i].v_mf
            elif b.ve_method == 'Abrahamson':
                return b.ve[i] ** 0.7 * b.sol_prop_e[i].e_mf ** 3 * (1 - b.ed[i]) \
                       == b.sol_prop_e[i].v_mf ** 0.7 * b.ed[i] ** 3 \
                       * (1 - b.sol_prop_e[i].e_mf)
            elif b.ve_method == 'Hilligardt':
                return 3 * b.ve[i] == b.vg[i] + 2 * b.sol_prop_e[i].v_mf
            else:
                raise Exception('ve_method not recognised.')

        self.eq_h3 = Constraint(self.l_n, rule=rule_eq_h3)

        # h4 - Average Cross-Sectional Voidage
        def rule_eq_h4(b, i):
            return (1 - b.e[i]) == (1 - b.delta[i]) * (1 - b.ed[i])

        self.eq_h4 = Constraint(self.l_n, rule=rule_eq_h4)

        # h5 - Bubble Size Coefficients
        def rule_eq_h5(b, i):
            if i == b.l_n.first():
                return Constraint.Skip
            else:
                return (b.g1[i] * b.sol_prop_e[i].v_mf) ** 2 \
                       == (2.56E-2 ** 2) * (b.Dt / b.gc)

        self.eq_h5 = Constraint(self.l_n, rule=rule_eq_h5)

        # h6 - Maximum Bubble Diameter
        # This eqn from Horio and Nonaka (1987) is same as Mori and Wen (1975),
        # the conv. diff arises from units. dbm is m for Horio, but cm for Mori
        def rule_eq_h6(b, i):
            if i == b.l_n.first():
                return Constraint.Skip
            else:
                return (b.dbm[i] ** 5) * b.gc \
                       == (2.59 ** 5) * ((b.vg[i] - b.ve[i]) * b.Ax) ** 2

        self.eq_h6 = Constraint(self.l_n, rule=rule_eq_h6)

        # h7 - Constrained Bubble Diameter
        def rule_eq_h7(b, i):
            if i == b.l_n.first():
                return (b.db[0] ** 5) == (1.38 ** 5) * (b.gc ** (-1)) \
                       * ((b.vg[0] - b.ve[0]) * b.Ao) ** 2  # Valid at Low gas flow rates
            #                return 1e6*b.db[0]*b.gc == 1e6*3.77*((b.vg[0]-b.ve[0])**2) #Valid at High gas flow rates
            else:
                return b.ddbdx[i] * b.Dt == 0.3 * b.Lb * (b.dbm[i] \
                                                          - b.db[i] - b.g1[i] \
                                                          * (b.Dt * b.db[i]) ** 0.5)

        self.eq_h7 = Constraint(self.l_n, rule=rule_eq_h7)

        # h8 - Bubble Rise Velocity
        def rule_eq_h8(b, i):
            return b.vbr[i] ** 2 == (0.711 ** 2) * (b.gc * b.db[i])

        self.eq_h8 = Constraint(self.l_n, rule=rule_eq_h8)

        # h9 - Bubble Velocity
        def rule_eq_h9(b, i):
            if b.vb_method == 'Davidson':
                return b.vb[i] == b.vg[i] - b.sol_prop_e[i].v_mf + b.vbr[i]
            elif b.vb_method == 'Werther A':
                return b.vb[i] == 1.55 * ((b.vg[i] - b.sol_prop_e[i].v_mf) \
                                          + 14.1 * (b.db[i] + 0.005)) * (b.Dte ** 0.32) + b.vbr[i]
            elif b.vb_method == 'Werther B':
                return b.vb[i] == 1.6 * ((b.vg[i] - b.sol_prop_e[i].v_mf) \
                                         + 1.13 * (b.db[i] ** 0.5)) * (b.Dte ** 1.35) + b.vbr[i]
            else:
                raise Exception('vb_method not recognised.')

        self.eq_h9 = Constraint(self.l_n, rule=rule_eq_h9)

        # h10 - Cloud to Bubble Volume Ratio

        # eq_h10 and eq_h10b are used to calculate the cloud to bubble region ratio (fc),
        # in such a way that it doesn't exceed a maximum value (fc_max).
        def rule_eq_h10(b, i):
            return b.fc_temp[i] * b.vbr[i] - (b.fc_temp[i] + 3) \
                   * b.sol_prop_e[i].v_mf / b.sol_prop_e[i].e_mf == 0

        self.eq_h10 = Constraint(self.l_n, rule=rule_eq_h10)

        # h10b - Cloud to Bubble Volume Ratio
        def rule_eq_h10b(b, i):
            return b.fc[i] == 0.5 * (b.fc_max + b.fc_temp[i] \
                                     - sqrt((b.fc_max - b.fc_temp[i]) ** 2 + b.eps ** 2))

        self.eq_h10b = Constraint(self.l_n, rule=rule_eq_h10b)

        # h11 - Cloud-Wake to Bubble Volume Ratio
        def rule_eq_h11(b, i):
            return b.fcw[i] == b.fc[i] + b.fw

        self.eq_h11 = Constraint(self.l_n, rule=rule_eq_h11)

        # h12 - Emulsion Region Voidage
        def rule_eq_h12(b, i):
            if b.ed_method == 'Davidson':
                return b.ed[i] == b.sol_prop_e[i].e_mf
            else:
                raise Exception('ed_method not recognised.')

        self.eq_h12 = Constraint(self.l_n, rule=rule_eq_h12)

        # ---------------------------------------------------------------------
        # K-L Model Heat and Mass Transfer Coefficients
        # i1 - Bubble to Cloud-Wake Gas Mass Transfer Coefficient
        #        def rule_eq_i1(b, i, j):
        #            if i == b.l_n.first():
        #                return Constraint.Skip
        #            else:
        #                return b.Kbc[j,i]*b.db[i]**(5/4) \
        #                    == 1.32*4.5*b.sol_prop_c[i].v_mf*b.db[i]**(1/4) \
        #                        + 5.85*b.gas_prop_e[i].D_vap[j]**0.5*b.gc**(1/4)
        #        self.eq_i1 = Constraint(self.l_n, self.GasList, rule=rule_eq_i1)

        # eq_i1 is reformulated to avoid scaling issues related to D_vap.
        self.d = Var(self.GasList, self.l_n, domain=NonNegativeReals, initialize=1)

        def rule_eq_i1a(b, i, j):
            if i == b.l_n.first():
                return Constraint.Skip
            else:
                return 1e5 * b.d[j, i] ** 2 == 1e5 * 34.2225 * b.gas_prop_e[i].D_vap[j] * b.gc ** 0.5

        self.eq_i1a = Constraint(self.l_n, self.GasList, rule=rule_eq_i1a)

        def rule_eq_i1(b, i, j):
            if i == b.l_n.first():
                return Constraint.Skip
            else:
                return b.Kbc[j, i] * b.db[i] ** (5 / 4) \
                       == 1.32 * 4.5 * b.sol_prop_c[i].v_mf * b.db[i] ** (1 / 4) \
                       + b.d[j, i]

        self.eq_i1 = Constraint(self.l_n, self.GasList, rule=rule_eq_i1)

        # i2 - Cloud-Wake to Emuslion Gas Mass Transfer Coefficient
        def rule_eq_i2(b, i, j):
            if i == b.l_n.first():
                return Constraint.Skip
            else:
                return b.Kce[j, i] ** 2 * (b.db[i] ** 3) == (6.77 ** 2) * b.ed[i] \
                       * b.gas_prop_e[i].D_vap[j] \
                       * b.vb[i]

        self.eq_i2 = Constraint(self.l_n, self.GasList, rule=rule_eq_i2)

        # i3 - Cloud-Wake to Emulsion Solid Mass Transfer Coefficient
        def rule_eq_i3(b, i):
            if i == b.l_n.first():
                return Constraint.Skip
            else:
                return b.Kcebs[i] * ((1 - b.delta[i]) * b.ed[i] * b.db[i]) \
                       == 3 * (1 - b.ed[i]) * b.ve[i]

        self.eq_i3 = Constraint(self.l_n, rule=rule_eq_i3)

        # i4 - Bubble to Cloud-Wake Gas Heat Transfer Coefficient
        def rule_eq_i4(b, i):
            if i == b.l_n.first():
                return Constraint.Skip
            else:
                return b.Hbc[i] * b.db[i] ** (5 / 4) == 1.32 * 4.5 \
                       * b.sol_prop_c[i].v_mf \
                       * b.gas_prop_b[i].cp_vap * b.db[i] ** (1 / 4) \
                       / b.gas_prop_b[i].V_vap \
                       + 5.85 * (b.gas_prop_b[i].k_vap \
                                 * b.gas_prop_b[i].cp_vap \
                                 / b.gas_prop_b[i].V_vap) ** 0.5 * (b.gc ** 0.25)

        self.eq_i4 = Constraint(self.l_n, rule=rule_eq_i4)

        # i5 - Cloud-Wake to Emulsion Gas Heat Transfer Coefficient
        def rule_eq_i5(b, i):
            if i == b.l_n.first():
                return Constraint.Skip
            else:
                return b.Hce[i] ** 2 * b.db[i] ** 3 == (6.78 ** 2) * b.ed[i] * b.vb[i] \
                       * b.gas_prop_e[i].k_vap * b.cct[i] \
                       * b.gas_prop_e[i].cp_vap

        self.eq_i5 = Constraint(self.l_n, rule=rule_eq_i5)

        # i6 - Convective Heat Transfer Coefficient
        def rule_eq_i6(b, i):
            if i == b.l_n.first():
                return Constraint.Skip
            else:
                return b.hp[i] * b.sol_prop_e[i].dp == 0.03 \
                       * b.gas_prop_e[i].k_vap \
                       * (b.ve[i] * b.sol_prop_e[i].dp \
                          * b.gas_prop_e[i].rho_vap \
                          / b.gas_prop_e[i].mu_vap) ** 1.3

        self.eq_i6 = Constraint(self.l_n, rule=rule_eq_i6)

    #        # eq_i6 is reformulated to avoid scaling issues related to mu_vap
    #        self.hp_term = Var(self.l_n, domain = NonNegativeReals, initialize = 1)
    #        def rule_eq_i6a(b, i):
    #            if i == b.l_n.first():
    #                return Constraint.Skip
    #            else:
    #                return 1e6*b.hp_term[i] == 1e6*(b.ve[i]*b.sol_prop_e[i].dp \
    #                                    * b.gas_prop_e[i].rho_vap \
    #                                        / b.gas_prop_e[i].mu_vap)
    #        self.eq_i6a = Constraint(self.l_n, rule=rule_eq_i6a)
    #
    #        def rule_eq_i6(b, i):
    #            if i == b.l_n.first():
    #                return Constraint.Skip
    #            else:
    #                return 1e12*b.hp[i]*b.sol_prop_e[i].dp == 1e12*0.03 \
    #                                    * b.gas_prop_e[i].k_vap \
    #                                    * (b.hp_term[i])**1.3
    #        self.eq_i6 = Constraint(self.l_n, rule=rule_eq_i6)

    def _make_HX_vars(self):
        """
        Create the variables needed for the heat exchanger.

        Group: J, K
        """
        if self.hx_type == "SPD":
            # Heat Exchanger Dimensions
            self.Ahx = Var(domain=Reals,
                           doc='Total Area of Heat Exchanger Surfaces [m^2]')
            self.dx = Var(domain=Reals,
                          doc='Diameter of Heat Exchanger Tubes [m]')
            self.lhx = Var(domain=Reals,
                           doc='Heat Exchanger Tube Spacing (Pitch-Diameter) [m]')
            self.lp = Var(domain=Reals,
                          doc='Heat Exchanger Tube Pitch [m]')
            self.Nx = Var(domain=Reals,
                          doc='Number of Heat Exchanger Tubes [-]')

            # HX Fluid Inlet Conditions
            self.HX_In_F = Var(domain=Reals,
                               doc='Heat Exchanger Fluid Flowrate at Inlet [mol/s]')
            self.HX_In_T = Var(domain=Reals,
                               doc='Heat Exchanger Fluid Temperature at Inlet [K]')
            self.HX_In_P = Var(domain=Reals,
                               doc='Heat Exchanger Fluid Pressure at Inlet [Pa]')
            self.HX_In_y = Var(self.HXList, domain=Reals,
                               doc='Heat Exchanger Fluid Mole Fractions [-]')

            # HX Fluid State
            self.Hhx = Var(self.l_n, domain=Reals,
                           doc='Heat Exchanger Fluid Scaled Enthalpy Flow ' \
                               '[1/sf_hh * J/s]')
            self.Phx = Var(self.l_n, domain=Reals,
                           doc='Heat Exchanger Fluid Pressure [Pa]')
            self.Thx = Var(self.l_n, domain=Reals,
                           doc='Heat Exchanger Fluid Temperature [K]')
            self.Ttube = Var(self.l_n, domain=Reals,
                             doc='Heat Exchanger Tube Temperature [K]')
            self.dThx = Var(self.l_n, domain=Reals,
                            doc='Temperature Difference between HX Tubes and Bed [K]')

            # HX Tube Heat Transfer Coefficients
            self.fb = Var(self.l_n, domain=Reals,
                          doc='Fraction of Time HX Tubes Contact Dense Packets [-]')
            self.fn = Var(self.l_n, domain=Reals,
                          doc='Fluidisation Number [-]')
            self.hd = Var(self.l_n, domain=Reals,
                          doc='Convective Heat Transfer Coefficient of Dense ' \
                              'Packets [J/m^2.K.s]')
            self.hl = Var(self.l_n, domain=Reals,
                          doc='Convective Heat Transfer Coefficient of Gas Bubbles ' \
                              '[J/m^2.K.s]')
            self.ht = Var(self.l_n, domain=Reals,
                          doc='Overall Convective Heat Transfer Coefficient ' \
                              '[J/m^2.K.s]')
            self.kpa = Var(self.l_n, domain=Reals, initialize=0.1,
                           doc='Thermal Conductivity of Bed at Minimum Fluidisation ' \
                               '[J/b.K.s]')
            self.Pr = Var(self.l_n, domain=Reals, initialize=0.7,
                          doc='Prandlt Number [-]')
            self.tau = Var(self.l_n, domain=Reals, initialize=0.1,
                           doc='Average Residence Time of Dense Packets at ' \
                               'HX Surface [s]')

            # HX Fluid Inlet Conditions
            self.HX_Out_F = Var(domain=Reals,
                                doc='HX Fluid Flowrate at Outlet [mol/s]')
            self.HX_Out_T = Var(domain=Reals,
                                doc='HX Fluid Temperature at Outlet [K]')
            self.HX_Out_P = Var(domain=Reals,
                                doc='HX Fluid Pressure at Outlet [Pa]')
            self.HX_Out_y = Var(self.HXList, domain=Reals,
                                doc='HX Fluid Mole fractions at Outlet [mol/mol]')

            # Derivative Variables
            self.dHhxdx = DerivativeVar(self.Hhx, wrt=self.l_n)
            self.dPhxdx = DerivativeVar(self.Phx, wrt=self.l_n)

    def _make_HX_props(self):
        """
        Create HX fluid property constraints.
        """
        if self.hx_type != None and self.hx_type != "None":
            # Create HX fluid property blocks
            def rule_hx_prop_blk(b, i):
                return self.hx_prop_lib.PropPack(name='hprop',
                                                 parent=self)

            self.hx_prop = Block(self.l_n, rule=rule_hx_prop_blk)

            self.hx_prop_in = self.hx_prop_lib.PropPack(name='hprop',
                                                        parent=self)

            # Link variables between model and properties
            def rule_hprop_T(b, i):
                return b.hx_prop[i].T == b.Thx[i]

            self.eq_hprop_T = Constraint(self.l_n, rule=rule_hprop_T)

            def rule_hprop_P(b, i):
                return b.hx_prop[i].P == b.Phx[i]

            self.eq_hprop_P = Constraint(self.l_n, rule=rule_hprop_P)

            def rule_hprop_y(b, i, j):
                return b.hx_prop[i].y[j] == b.HX_In_y[j]

            self.eq_hprop_y = Constraint(self.l_n, self.HXList,
                                         rule=rule_hprop_y)

            self.eq_gprop0_T = Constraint(expr=self.hx_prop_in.T \
                                               == self.HX_In_T)
            self.eq_gprop0_P = Constraint(expr=self.hx_prop_in.P \
                                               == self.HX_In_P)

            def rule_hprop0_y(b, j):
                return b.hx_prop_in.y[j] == b.HX_In_y[j]

            self.eq_hprop0_y = Constraint(self.HXList, rule=rule_hprop0_y)

    def _make_HX_model(self):
        """
        Create the heat exchanger model for the bed.

        Group: J, K
        """

        # Create HX tube model
        # Total Heat Duty
        def rule_intQhx(b, i):
            return b.Lb * b.Qhx[i]

        self.Q = Integral(self.l_n, wrt=self.l_n, rule=rule_intQhx,
                          doc='Total Heat Transfered from Bed [J/s]')

        if self.hx_type == None or self.hx_type == "None":
            # No HX tubes
            # a4 - Hydraulic Diameter
            self.eq_a4 = Constraint(expr=self.Dt == self.Dte)

            # a5 - Equate Cross Section of Bed to Vessel
            self.eq_a5 = Constraint(expr=self.Ax == self.Areact)

        elif self.hx_type == "SPD":
            # Single pass, downward flowing tubes
            # a4 - Hydraulic Diameter
            self.eq_a4 = Constraint(expr=self.Ax == 0.25 * self.Dte * self.pi \
                                         * (self.Dt + self.dx * self.Nx))

            # a5 - Total Reactor Cross-Sectional Area (incl. HX tubes)
            self.eq_a5 = Constraint(expr=self.Areact == self.Ax \
                                         + (self.pi / 4) * self.dx ** 2 \
                                         * self.Nx)

            # a6, a6 - HX Tube Pitch and Spacing
            self.eq_a6 = Constraint(expr=self.Areact == self.Nx * self.lp ** 2)
            self.eq_a7 = Constraint(expr=self.lhx == self.lp - self.dx)

            # a8 - Surface Area of HX Tubes
            self.eq_a8 = Constraint(expr=self.Ahx \
                                         == self.pi * self.dx * self.Lb \
                                         * self.Nx)

            # -----------------------------------------------------------------
            # Mickley and Fairbanks HX Model
            # j1 - Thermal Conductivity of Bed at Minimum Fluidisation
            def rule_eq_j1(b, i):
                if i == 0:
                    return Constraint.Skip
                else:
                    return b.kpa[i] == (3.58 - 2.5 * b.ed[i]) \
                           * b.gas_prop_e[i].k_vap \
                           * ((b.sol_prop_e[i].k_sol \
                               / b.gas_prop_e[i].k_vap) \
                              ** (0.46 - 0.46 * b.ed[i]))

            self.eq_j1 = Constraint(self.l_n, rule=rule_eq_j1)

            # j2 - Fluidisation Number
            def rule_eq_j2(b, i):
                if i == 0:
                    return Constraint.Skip
                else:
                    return b.fn[i] * b.sol_prop_e[i].v_mf == b.vg[i]

            self.eq_j2 = Constraint(self.l_n, rule=rule_eq_j2)

            # j3 - Residence Time of Emulsion Packets at HX Surface
            def rule_eq_j3(b, i):
                if i == 0:
                    return Constraint.Skip
                else:
                    return b.tau[i] == 0.44 * ((b.sol_prop_e[i].dp * b.gc \
                                                / ((b.sol_prop_e[i].v_mf ** 2) \
                                                   * ((b.fn[i] - b.ah) ** 2))) ** 0.14) \
                           * ((b.sol_prop_e[i].dp / b.dx) ** 0.225)

            self.eq_j3 = Constraint(self.l_n, rule=rule_eq_j3)

            # j4 - Fraction of Time HX Surface is Exposed to Emulsion Packets
            def rule_eq_j4(b, i):
                if i == 0:
                    return Constraint.Skip
                else:
                    return b.fb[i] == 0.33 * (((b.sol_prop_e[i].v_mf ** 2) \
                                               * ((b.fn[i] - b.ah) ** 2) \
                                               / (b.sol_prop_e[i].dp * b.gc)) ** 0.14)

            self.eq_j4 = Constraint(self.l_n, rule=rule_eq_j4)

            # j5 - Dense Region Heat Transfer Coefficient
            def rule_eq_j5(b, i):
                if i == 0:
                    return Constraint.Skip
                else:
                    return b.hd[i] * sqrt(b.pi * b.tau[i]) == 2 * sqrt(b.kpa[i] \
                                                                       * b.sol_prop_e[i].rho_sol \
                                                                       * b.sol_prop_e[i].cp_sol * (1 - b.ed[i]))

            self.eq_j5 = Constraint(self.l_n, rule=rule_eq_j5)

            # j6 - Bubble Region Heat Transfer Coefficient - Prandlt Number
            def rule_eq_j6(b, i):
                if i == 0:
                    return Constraint.Skip
                else:
                    return b.Pr[i] * b.gas_prop_e[i].k_vap \
                           == b.gas_prop_e[i].cp_vap \
                           * b.gas_prop_e[i].mu_vap \
                           / b.gas_prop_e[i].MW_vap

            self.eq_j6 = Constraint(self.l_n, rule=rule_eq_j6)

            # j7 - Bubble Region Heat Transfer Coefficient
            def rule_eq_j7(b, i):
                if i == 0:
                    return Constraint.Skip
                else:
                    return b.hl[i] * b.sol_prop_e[i].dp == 0.009 * (b.Ar[i] ** 0.5) \
                           * (b.Pr[i] ** 0.33) * b.gas_prop_e[i].k_vap

            self.eq_j7 = Constraint(self.l_n, rule=rule_eq_j7)

            # j8 - Total HX Heat Transfer Coefficient
            def rule_eq_j8(b, i):
                if i == 0:
                    return Constraint.Skip
                else:
                    return b.ht[i] == b.fb[i] * b.hd[i] + (1 - b.fb[i]) * b.hl[i]

            self.eq_j8 = Constraint(self.l_n, rule=rule_eq_j8)

            # -----------------------------------------------------------------
            # k2 - HX Tube Heat Transfer
            def rule_eq_k2(b, i):
                if i == 0:
                    return Constraint.Skip
                else:
                    return b.Qhx[i] == b.pi * b.dx * b.ht[i] * b.dThx[i] \
                           * b.Nx * b.Cr

            self.eq_k2 = Constraint(self.l_n, rule=rule_eq_k2)

            # k3 - HX Fluid Pressure Drop
            def rule_eq_k3(b, i):
                if i == 0:
                    return self.Phx[1] == self.HX_In_P
                else:
                    return b.dPhxdx[i] == -b.hx_prop[i].rho_mix * b.gc * b.Lb

            self.eq_k3 = Constraint(self.l_n, rule=rule_eq_k3)

            # k4 - HX Fluid Energy Balance
            def rule_eq_k4(b, i):
                if i == 0:
                    return self.Hhx[1] == self.sf_hh * self.HX_In_F \
                           * self.hx_prop_in.h_mix
                else:
                    return 0 == b.dHhxdx[i] - self.sf_hh * b.Lb * b.Qhx[i]

            self.eq_k4 = Constraint(self.l_n, rule=rule_eq_k4)

            # k5 - Temperature Difference between Tube and Bed
            def rule_eq_k5(b, i):
                if i == 0:
                    return Constraint.Skip
                else:
                    return b.dThx[i] == b.Ttube[i] - b.Tse[i]

            self.eq_k5 = Constraint(self.l_n, rule=rule_eq_k5)

            # k6 - HX Tube Wall Energy Balance
            def rule_eq_k6(b, i):
                if i == 0:
                    return Constraint.Skip
                else:
                    return b.ht[i] * b.dThx[i] * b.Cr \
                           == b.hw * (b.Thx[i] - b.Ttube[i])

            self.eq_k6 = Constraint(self.l_n, rule=rule_eq_k6)

            # k7 - HX Fluid Enthalpy Flow
            def rule_eq_k7(b, i):
                return b.Hhx[i] == self.sf_hh * b.HX_In_F * b.hx_prop[i].h_mix

            self.eq_k7 = Constraint(self.l_n, rule=rule_eq_k7)

            # k8, k9 - HX Fluid Outlet Temperature and Pressure
            self.eq_k8 = Constraint(expr=self.HX_Out_T == self.Thx[0])
            self.eq_k9 = Constraint(expr=self.HX_Out_P == self.Phx[0])

        else:
            raise Exception('HX type not recognised.')

    def _make_props(self):
        """
        Create property constraints.
        """

        # Create gas property blocks
        def rule_gas_prop_self_b(b, i):
            return self.gas_prop_lib.PropPack(name='gprop_b',
                                              parent=self,
                                              prop_list={"V_vap",
                                                         "cp_vap",
                                                         "k_vap",
                                                         "h_vap"})

        self.gas_prop_b = Block(self.l_n, rule=rule_gas_prop_self_b)

        def rule_gas_prop_self_c(b, i):
            if i != 0:
                return self.gas_prop_lib.PropPack(name='gprop_c',
                                                  parent=self,
                                                  prop_list={"h_vap"})

        self.gas_prop_c = Block(self.l_n, rule=rule_gas_prop_self_c)

        def rule_gas_prop_self_e(b, i):
            if i != 0:
                return self.gas_prop_lib.PropPack(name='gprop_e',
                                                  parent=self,
                                                  prop_list={"mu_vap",
                                                             "rho_vap",
                                                             "D_vap",
                                                             "cp_vap",
                                                             "MW_vap",
                                                             "k_vap",
                                                             "h_vap"})

        self.gas_prop_e = Block(self.l_n, rule=rule_gas_prop_self_e)

        self.gas_prop_out = self.gas_prop_lib.PropPack(name='gprop_out',
                                                       parent=self,
                                                       prop_list={"h_vap"})

        # Link variables between model and properties
        def rule_gpropb_T(b, i):
            return b.gas_prop_b[i].T == b.Tgb[i]

        self.eq_gpropb_T = Constraint(self.l_n, rule=rule_gpropb_T)

        def rule_gpropb_P(b, i):
            return b.gas_prop_b[i].P == b.P[i]

        self.eq_gpropb_P = Constraint(self.l_n, rule=rule_gpropb_P)

        def rule_gpropb_y(b, i, j):
            return b.gas_prop_b[i].y[j] == b.yb[j, i]

        self.eq_gpropb_y = Constraint(self.l_n, self.GasList,
                                      rule=rule_gpropb_y)

        def rule_gpropc_T(b, i):
            if i == 0:
                return Constraint.Skip
            else:
                return b.gas_prop_c[i].T == b.Tgc[i]

        self.eq_gpropc_T = Constraint(self.l_n, rule=rule_gpropc_T)

        def rule_gpropc_P(b, i):
            if i == 0:
                return Constraint.Skip
            else:
                return b.gas_prop_c[i].P == b.P[i]

        self.eq_gpropc_P = Constraint(self.l_n, rule=rule_gpropc_P)

        def rule_gpropc_y(b, i, j):
            if i == 0:
                return Constraint.Skip
            else:
                return b.gas_prop_c[i].y[j] == b.yc[j, i]

        self.eq_gpropc_y = Constraint(self.l_n, self.GasList,
                                      rule=rule_gpropc_y)

        def rule_gprope_T(b, i):
            if i == 0:
                return Constraint.Skip
            else:
                return b.gas_prop_e[i].T == b.Tge[i]

        self.eq_gprope_T = Constraint(self.l_n, rule=rule_gprope_T)

        def rule_gprope_P(b, i):
            if i == 0:
                return Constraint.Skip
            else:
                return b.gas_prop_e[i].P == b.P[i]

        self.eq_gprope_P = Constraint(self.l_n, rule=rule_gprope_P)

        def rule_gprope_y(b, i, j):
            if i == 0:
                return Constraint.Skip
            else:
                return b.gas_prop_e[i].y[j] == b.ye[j, i]

        self.eq_gprope_y = Constraint(self.l_n, self.GasList,
                                      rule=rule_gprope_y)

        self.eq_gpropo_T = Constraint(expr=self.Gas_Out_T \
                                           == self.gas_prop_out.T)
        self.eq_gpropo_P = Constraint(expr=self.Gas_Out_P \
                                           == self.gas_prop_out.P)

        def rule_gpropo_y(b, j):
            return b.Gas_Out_y[j] == b.gas_prop_out.y[j]

        self.eq_gpropo_y = Constraint(self.GasList, rule=rule_gpropo_y)

        # ---------------------------------------------------------------------
        # Create solid property blocks
        def rule_sol_prop_self_c(b, i):
            if i == 0:
                return self.sol_prop_lib.PropPack(name='sprop_c',
                                                  parent=self,
                                                  prop_list={"h_sol", "rho_sol"})
            else:
                return self.sol_prop_lib.PropPack(name='sprop_c',
                                                  parent=self,
                                                  prop_list={"v_mf",
                                                             "ap",
                                                             "h_sol",
                                                             "r",
                                                             "rho_sol"})

        self.sol_prop_c = Block(self.l_n, rule=rule_sol_prop_self_c)

        def rule_sol_prop_self_e(b, i):
            return self.sol_prop_lib.PropPack(name='sprop_e',
                                              parent=self)

        self.sol_prop_e = Block(self.l_n, rule=rule_sol_prop_self_e)

        self.sol_prop_f = self.sol_prop_lib.PropPack(name='sprop_in',
                                                     parent=self,
                                                     prop_list={"h_sol"})

        # Link variables between model and properties
        def rule_spropc_T(b, i):
            return b.sol_prop_c[i].T == b.Tsc[i]

        self.eq_spropc_T = Constraint(self.l_n, rule=rule_spropc_T)

        def rule_spropc_P(b, i):
            return b.sol_prop_c[i].P == b.P[i]

        self.eq_spropc_P = Constraint(self.l_n, rule=rule_spropc_P)

        def rule_spropc_y(b, i, j):
            if i == 0:
                return b.sol_prop_c[i].y[j] == b.yb[j, i]
            else:
                return b.sol_prop_c[i].y[j] == b.yc[j, i]

        self.eq_spropc_y = Constraint(self.l_n, self.GasList,
                                      rule=rule_spropc_y)

        def rule_spropc_x(b, i, j):
            return b.sol_prop_c[i].x[j] == b.xc[j, i]

        self.eq_spropc_x = Constraint(self.l_n, self.SolidList,
                                      rule=rule_spropc_x)

        def rule_sprope_T(b, i):
            return b.sol_prop_e[i].T == b.Tse[i]

        self.eq_sprope_T = Constraint(self.l_n, rule=rule_sprope_T)

        def rule_sprope_P(b, i):
            return b.sol_prop_e[i].P == b.P[i]

        self.eq_sprope_P = Constraint(self.l_n, rule=rule_sprope_P)

        def rule_sprope_y(b, i, j):
            return b.sol_prop_e[i].y[j] == b.ye[j, i]

        self.eq_sprope_y = Constraint(self.l_n, self.GasList,
                                      rule=rule_sprope_y)

        def rule_sprope_x(b, i, j):
            return b.sol_prop_e[i].x[j] == b.xe[j, i]

        self.eq_sprope_x = Constraint(self.l_n, self.SolidList,
                                      rule=rule_sprope_x)

        self.eq_spropf_T = Constraint(expr=self.sol_prop_f.T \
                                           == self.Solid_In_T)

        def rule_spropf_x(b, j):
            return b.sol_prop_f.x[j] == b.Solid_In_x[j]

        self.eq_spropf_x = Constraint(self.SolidList, rule=rule_spropf_x)

        # Hack added to allow easier ht_rxn code in prop pack - pass gas temp to solid prop packs
        def rule_spropc_Tg(b, i):
            if i == 0:
                return Constraint.Skip
            else:
                return b.sol_prop_c[i].Tg == b.Tgc[i]

        self.eq_spropc_Tg = Constraint(self.l_n, rule=rule_spropc_Tg)

        def rule_sprope_Tg(b, i):
            return b.sol_prop_e[i].Tg == b.Tge[i]

        self.eq_sprope_Tg = Constraint(self.l_n, rule=rule_sprope_Tg)

        # NEEDED CURRENTLY FOR SQUARE PROBLEM
        if self.s_inlet == "Bottom":
            self.eq_spropf_P = Constraint(expr=self.sol_prop_f.P \
                                               == self.P[0])
        else:
            self.eq_spropf_P = Constraint(expr=self.sol_prop_f.P \
                                               == self.P[1])

        def rule_spropf_y(b, j):
            return b.sol_prop_f.y[j] == b.Gas_In_y[j]

        self.eq_spropf_y = Constraint(self.GasList, rule=rule_spropf_y)

    def _dae_transform(self):
        """
        Apply DAE transformation to domain.
        """
        if self.dae_method == "OCLR":
            discretizer = TransformationFactory('dae.collocation')
            discretizer.apply_to(self, nfe=self.nfe, ncp=self.ncp,
                                 wrt=self.l_n, scheme='LAGRANGE-RADAU')
        elif self.dae_method == "BFD1":
            discretizer = TransformationFactory('dae.finite_difference')
            discretizer.apply_to(self, nfe=self.nfe, wrt=self.l_n,
                                 scheme='BACKWARD')
        elif self.dae_method == "OCLL":
            discretizer = TransformationFactory('dae.collocation')
            discretizer.apply_to(self, nfe=self.nfe, ncp=self.ncp,
                                 wrt=self.l_n, scheme='LAGRANGE-LEGENDRE')
        else:
            raise Exception('DAE method type not recognised.')

    def _initialize(blk, Tg=None, Ts=None, Th=None, P=None, x=None, y=None,
                    outlvl=0, optarg=None):
        # Start time for total elapsed time
        total_time_start = time.time()

        if optarg == None:
            sopt = {"tol": 1e-8,
                    "max_cpu_time": 300,
                    "print_level": 5}
        else:
            sopt = optarg

        if outlvl > 1:
            stee = True
        else:
            stee = False

        if outlvl > 0:
            print("\n")
            print("----------------------------------------------------------")
            print("BFB Reactor Initialisation\n")

        # Set Initial Values of State and Property Variables
        for i in blk.l_n:
            blk.P[i].fix(value(blk.Gas_In_P))
            blk.Tgb[i].fix(value(blk.Gas_In_T))
            blk.Tgc[i].fix(value(blk.Gas_In_T))
            blk.Tge[i].fix(value(blk.Gas_In_T))
            blk.Tsc[i].fix(value(blk.Solid_In_T))
            blk.Tse[i].fix(value(blk.Solid_In_T))
            for j in blk.GasList:
                blk.yb[j, i].fix(value(blk.Gas_In_y[j]))
                blk.yc[j, i].fix(value(blk.Gas_In_y[j]))
                blk.ye[j, i].fix(value(blk.Gas_In_y[j]))
            for j in blk.SolidList:
                blk.xc[j, i].fix(value(blk.Solid_In_x[j]))
                blk.xe[j, i].fix(value(blk.Solid_In_x[j]))

        blk.Gas_Out_F.fix(value(blk.Gas_In_F))
        blk.Gas_Out_T.fix(value(blk.Gas_In_T))
        blk.Gas_Out_P.fix(value(blk.Gas_In_P))
        for j in blk.GasList:
            blk.Gas_Out_y[j].fix(value(blk.Gas_In_y[j]))

        if blk.hx_type != None and blk.hx_type != "None":
            for i in blk.l_n:
                blk.Thx[i].fix(value(blk.HX_In_T))
                blk.Phx[i].fix(value(blk.HX_In_P))
                blk.Hhx[i].fix(0)

        # ---------------------------------------------------------------------
        # Deactivate constraints
        # All property blocks active

        # All a constraints active

        blk.eq_b1.deactivate()
        blk.eq_b2.deactivate()
        blk.eq_b3.deactivate()
        blk.eq_b4.deactivate()
        blk.eq_b5.deactivate()
        blk.eq_b6.deactivate()
        blk.eq_b7.deactivate()
        blk.eq_b8.deactivate()
        blk.eq_b9.deactivate()
        blk.eq_b10.deactivate()

        blk.eq_c1.deactivate()
        blk.eq_c2.deactivate()
        blk.eq_c3.deactivate()
        blk.eq_c4.deactivate()
        blk.eq_c5.deactivate()
        blk.eq_c6.deactivate()
        blk.eq_c7.deactivate()
        blk.eq_c8.deactivate()
        blk.eq_c9.deactivate()
        blk.eq_c10.deactivate()
        blk.eq_c11.deactivate()
        blk.eq_c12.deactivate()
        blk.eq_c11.deactivate()
        blk.eq_c13.deactivate()

        blk.eq_d1.deactivate()
        blk.eq_d2.deactivate()
        blk.eq_d3.deactivate()
        blk.eq_d4.deactivate()
        blk.eq_d5.deactivate()
        blk.eq_d6.deactivate()
        blk.eq_d7.deactivate()
        blk.eq_d8.deactivate()
        blk.eq_d9.deactivate()
        blk.eq_d10.deactivate()
        blk.eq_d11.deactivate()
        blk.eq_d12.deactivate()
        blk.eq_d13.deactivate()
        blk.eq_d14.deactivate()
        blk.eq_d15.deactivate()

        blk.eq_e1.deactivate()
        blk.eq_e2.deactivate()
        blk.eq_e3.deactivate()
        blk.eq_e4.deactivate()
        blk.eq_e5.deactivate()
        blk.eq_e6.deactivate()
        blk.eq_e7.deactivate()
        blk.eq_e8.deactivate()

        blk.eq_f1.deactivate()
        blk.eq_f2.deactivate()
        blk.eq_f3.deactivate()
        blk.eq_f4.deactivate()
        blk.eq_f5.deactivate()
        blk.eq_f6.deactivate()
        blk.eq_f7.deactivate()
        blk.eq_f8.deactivate()
        blk.eq_f9.deactivate()
        blk.eq_f10.deactivate()

        blk.eq_g1.deactivate()

        blk.eq_h1.deactivate()
        blk.eq_h2.deactivate()
        blk.eq_h3.deactivate()
        blk.eq_h4.deactivate()
        blk.eq_h5.deactivate()
        blk.eq_h6.deactivate()
        blk.eq_h7.deactivate()
        blk.eq_h8.deactivate()
        blk.eq_h9.deactivate()
        blk.eq_h10.deactivate()
        blk.eq_h10b.deactivate()
        blk.eq_h11.deactivate()
        blk.eq_h12.deactivate()

        blk.eq_i1.deactivate()
        blk.eq_i2.deactivate()
        blk.eq_i3.deactivate()
        blk.eq_i4.deactivate()
        blk.eq_i5.deactivate()
        blk.eq_i6.deactivate()

        if blk.hx_type == 'SPD':
            blk.eq_j1.deactivate()
            blk.eq_j2.deactivate()
            blk.eq_j3.deactivate()
            blk.eq_j4.deactivate()
            blk.eq_j5.deactivate()
            blk.eq_j6.deactivate()
            blk.eq_j7.deactivate()
            blk.eq_j8.deactivate()

            blk.eq_k2.deactivate()
            blk.eq_k3.deactivate()
            blk.eq_k4.deactivate()
            blk.eq_k5.deactivate()
            blk.eq_k6.deactivate()
            blk.eq_k7.deactivate()
            blk.eq_k8.deactivate()
            blk.eq_k9.deactivate()

        # l1 to l4 constraints are deactivated as only prop pack should be initialized

        blk.eq_l1.deactivate()
        blk.eq_l2.deactivate()
        blk.eq_l3.deactivate()
        blk.eq_l4.deactivate()

        # ---------------------------------------------------------------------
        # Fix distributed variables to ensure square problem
        blk.db.fix(1.0)
        blk.Gbc.fix(1.0)
        blk.Gec.fix(1.0)
        blk.Gbh.fix(1.0)
        blk.Geh.fix(1.0)
        blk.Jcc.fix(1.0)
        blk.Jec.fix(1.0)
        blk.Jch.fix(1.0)
        blk.Jeh.fix(1.0)
        blk.Qhx.fix(0.0)

        # %%        # ---------------------------------------------------------------------
        # 1st Initialisation Step - Properties Initialisation
        # Initialise solids mass fractions at equilibrium

        for i in blk.l_n:
            if i != 0:
                # This block of constraints deactivates the general reaction rate and sets it to zero
                blk.sol_prop_c[i].r_gen.fix(0.0)
                blk.sol_prop_e[i].r_gen.fix(0.0)
                blk.sol_prop_c[i].eq_r4.deactivate()
                blk.sol_prop_e[i].eq_r4.deactivate()
                for j in blk.SolidList:
                    blk.xc[j, i].unfix()
                    blk.xe[j, i].unfix()

        ts = time.time()
        # Solve and print progress
        if outlvl > 0:
            print("Step  1, Initialise Properties,                     Time: ",
                  end="")
        results = blk.solve(options=sopt, tee=stee, symbolic_solver_labels=True)
        if outlvl > 0:
            print("{}, {}".format(model_util.hhmmss(time.time() - ts),
                                  results.solver.message))

        # Set solid mass fractions at bottom of bed
        for j in blk.SolidList:
            blk.xc[j, 0] = value(blk.xc[j, 1])
            blk.xe[j, 0] = value(blk.xe[j, 1])

        # %%        # ---------------------------------------------------------------------
        # 2nd Initialisation Step - Hydrodynamics
        # Initialise variables
        blk.cct.fix(1 / value(blk.gas_prop_b[0].V_vap))
        blk.cet.fix(value(blk.cct[0]))

        for i in blk.l_n:
            for j in blk.GasList:
                blk.cb[j, i].fix(value(blk.Gas_In_y[j]) \
                                 / value(blk.gas_prop_b[i].V_vap))
                blk.cc[j, i].fix(value(blk.cb[j, i]))
                blk.ce[j, i].fix(value(blk.cb[j, i]))
            blk.vg[i].fix(value(blk.Gas_In_F) * value(blk.gas_prop_b[i].V_vap) \
                          / value(blk.Ax))
            blk.ve[i] = (1 / 3) * (value(blk.vg[i]) \
                                   - value(blk.sol_prop_e[i].v_mf)) \
                        + value(blk.sol_prop_e[i].v_mf)
            blk.db[i] = 1.38 * (value(blk.gc) ** (-0.2)) \
                        * ((value(blk.vg[0]) \
                            - value(blk.ve[0])) * value(blk.Ao)) ** 0.4

            if i != 0:
                blk.dbm[i] = 2.59 * (value(blk.gc) ** (-0.2)) \
                             * ((value(blk.vg[i]) - value(blk.ve[i])) \
                                * value(blk.Ax)) ** 0.4
                blk.g1[i] = 2.56E-2 * sqrt(value(blk.Dt) / value(blk.gc)) \
                            / value(blk.sol_prop_e[i].v_mf)
            blk.vbr[i] = 0.711 * sqrt(value(blk.gc) * value(blk.db[i]))
            if blk.vb_method == 'Werther A':
                blk.vb[i] = 1.55 * ((value(blk.vg[i]) \
                                     - value(blk.sol_prop_e[i].v_mf)) \
                                    + 14.1 * (value(blk.db[i]) + 0.005)) \
                            * (value(blk.Dte) ** 0.32) + value(blk.vbr[i])
            elif blk.vb_method == 'Werther B':
                blk.vb[i] = 1.6 * ((value(blk.vg[i]) \
                                    - value(blk.sol_prop_e[i].v_mf)) \
                                   + 1.13 * (value(blk.db[i]) ** 0.5)) \
                            * (value(blk.Dte) ** 1.35) + value(blk.vbr[i])
            else:
                blk.vb[i] = value(blk.vg[i]) + value(blk.vbr[i]) \
                            - value(blk.sol_prop_e[i].v_mf)
            blk.delta[i].fix((value(blk.vg[0]) - value(blk.ve[0])) \
                             / (value(blk.vb[0]) - value(blk.ve[0])))
            blk.fc[i] = -6 * (value(blk.sol_prop_e[i].v_mf) \
                              / value(blk.sol_prop_e[i].e_mf)) \
                        / ((1 - 3 / value(blk.fc_max)) \
                           * (value(blk.sol_prop_e[i].v_mf) \
                              / value(blk.sol_prop_e[i].e_mf)) \
                           - value(blk.vbr[i]) \
                           - (((1 + 3 / value(blk.fc_max)) \
                               * (value(blk.sol_prop_e[i].v_mf) \
                                  / value(blk.sol_prop_e[i].e_mf)) \
                               - value(blk.vbr[i])) ** 2 \
                              + value(blk.eps)) ** 0.5)
            blk.fcw[i] = value(blk.fc[i]) + value(blk.fw)
            blk.ed[i] = value(blk.sol_prop_e[i].e_mf)
            blk.e[i] = 1 - (1 - value(blk.ed[i])) * (1 - value(blk.delta[i]))
            blk.delta_e[i] = 1 - (value(blk.fcw[i]) + 1) * value(blk.delta[i])
            if i != 0:
                blk.Kcebs[i] = 3 * (1 - value(blk.ed[i])) * value(blk.ve[i]) \
                               / ((1 - value(blk.delta[i])) * value(blk.ed[i]) \
                                  * value(blk.db[i]))
                for j in blk.SolidList:
                    blk.Kcebs_c[j, i] = value(blk.delta[i]) \
                                        * value(blk.sol_prop_e[i].rho_sol) \
                                        * value(blk.Kcebs[i]) \
                                        * (value(blk.xc[j, i]) \
                                           - value(blk.xe[j, i]))
                blk.Hcebs[i] = value(blk.delta[i]) \
                               * value(blk.sol_prop_e[i].rho_sol) \
                               * value(blk.Kcebs[i]) \
                               * (value(blk.sol_prop_c[i].h_sol) \
                                  - value(blk.sol_prop_e[i].h_sol))
                blk.Hbc[i] = 1.32 * 4.5 * value(blk.sol_prop_c[i].v_mf) \
                             * value(blk.gas_prop_b[i].cp_vap) \
                             / (value(blk.db[i]) \
                                * value(blk.gas_prop_b[i].V_vap)) \
                             + 5.85 * sqrt(value(blk.gas_prop_b[i].k_vap) \
                                           * value(blk.gas_prop_b[i].cp_vap) \
                                           / value(blk.gas_prop_b[i].V_vap)) \
                             * (value(blk.gc) ** 0.25) / value(blk.db[i]) ** (5 / 4)
                blk.Hce[i] = 6.78 * sqrt(value(blk.ed[i]) * value(blk.vbr[i]) \
                                         * value(blk.gas_prop_e[i].k_vap) * value(blk.cct[i]) \
                                         * value(blk.gas_prop_e[i].cp_vap)) \
                             / value(blk.db[i]) ** (3 / 2)
                for j in blk.GasList:
                    blk.Kbc[j, i] = 1.32 * 4.5 * (value(blk.sol_prop_c[i].v_mf) \
                                                  / value(blk.db[i])) \
                                    + 5.85 * ((value(blk.gc) ** 0.25) \
                                              * (value(blk.gas_prop_e[i].D_vap[j])) ** 0.5 \
                                              / (value(blk.db[i]) ** (5 / 4)))
                    blk.Kce[j, i] = 6.77 * sqrt(value(blk.ed[i]) \
                                                * value(blk.gas_prop_e[i].D_vap[j]) \
                                                * value(blk.vbr[i]) / value(blk.db[i]) ** 3)
                blk.hp[i] = 0.03 * value(blk.gas_prop_e[i].k_vap) \
                            * (value(blk.ve[i]) * value(blk.sol_prop_e[i].dp) \
                               * value(blk.gas_prop_e[i].rho_vap) \
                               / value(blk.gas_prop_e[i].mu_vap)) ** 1.3 \
                            / value(blk.sol_prop_e[i].dp)
                blk.hp_c[i] = value(blk.fcw[i]) * (1 - value(blk.ed[i])) \
                              * value(blk.sol_prop_c[i].rho_sol) \
                              * value(blk.sol_prop_c[i].ap) \
                              * value(blk.hp[i]) \
                              * (value(blk.Tgc[i]) \
                                 - value(blk.Tsc[i]))
                blk.hp_e[i] = value(blk.delta_e[i]) * (1 - value(blk.ed[i])) \
                              * value(blk.sol_prop_e[i].rho_sol) \
                              * value(blk.sol_prop_e[i].ap) \
                              * value(blk.hp[i]) \
                              * (value(blk.Tge[i]) \
                                 - value(blk.Tse[i]))

        # Activate constraints
        blk.eq_f1.activate()
        blk.eq_f2.activate()
        blk.eq_f5.activate()
        blk.eq_f6.activate()
        blk.eq_f7.activate()
        blk.eq_f8.activate()

        blk.eq_g1.activate()

        blk.eq_h1.activate()
        blk.eq_h2.activate()
        blk.eq_h3.activate()
        blk.eq_h4.activate()
        blk.eq_h5.activate()
        blk.eq_h6.activate()
        blk.eq_h7.activate()
        blk.eq_h8.activate()
        blk.eq_h9.activate()
        blk.eq_h10.activate()
        blk.eq_h10b.activate()
        blk.eq_h11.activate()
        blk.eq_h12.activate()

        blk.eq_i1.activate()
        blk.eq_i2.activate()
        blk.eq_i3.activate()
        blk.eq_i4.activate()
        blk.eq_i5.activate()
        blk.eq_i6.activate()

        blk.eq_l1.activate()
        blk.eq_l2.activate()
        blk.eq_l3.activate()
        blk.eq_l4.activate()

        # Unfix variables
        blk.db.unfix()
        blk.P.unfix()

        ts = time.time()
        # Solve and print progress
        if outlvl > 0:
            print("Step  2, Hydrodynamics,                             Time: ",
                  end="")
        results = blk.solve(options=sopt, tee=stee, symbolic_solver_labels=True)
        if outlvl > 0:
            print("{}, {}".format(model_util.hhmmss(time.time() - ts),
                                  results.solver.message))

        # %%        # ---------------------------------------------------------------------
        # 3rd Initialisation Step - Mass Balances
        # Part 1 - Gas Balances
        # Initialise variables
        for i in blk.l_n:
            blk.Ge[i] = value(blk.ve[i]) * value(blk.Ax) * value(blk.cet[i]) \
                        * value(blk.delta_e[i])
            blk.Gb[i] = value(blk.Gas_In_F) - value(blk.Ge[i])
            blk.delta[i] = value(blk.Gb[i]) * value(blk.gas_prop_b[0].V_vap) \
                           / (value(blk.Ax) * value(blk.vb[i]))
            for j in blk.GasList:
                blk.Gbc[j, i] = value(blk.Gb[i]) * value(blk.yb[j, i])
                blk.Gec[j, i] = value(blk.Ge[i]) * value(blk.ye[j, i])

        # Activate constraints
        blk.eq_b1.activate()
        blk.eq_b2.activate()
        blk.eq_b3.activate()

        blk.eq_c1.activate()
        blk.eq_c2.activate()
        blk.eq_c11.activate()
        blk.eq_c12.activate()
        blk.eq_c13.activate()

        blk.eq_d1.activate()
        blk.eq_d2.activate()
        blk.eq_d3.activate()

        blk.eq_e1.activate()
        blk.eq_e2.activate()
        blk.eq_e3.activate()
        blk.eq_e4.activate()
        blk.eq_e5.activate()
        blk.eq_e6.activate()

        # Unfix variables
        blk.cb.unfix()
        blk.cc.unfix()
        blk.ce.unfix()
        blk.cct.unfix()
        blk.cet.unfix()
        blk.delta.unfix()
        blk.Gbc.unfix()
        blk.Gec.unfix()
        blk.vg.unfix()
        blk.yb.unfix()
        blk.yc.unfix()
        blk.ye.unfix()

        # Part 2 - Solids Balances
        for i in blk.l_n:
            blk.Jc[i] = value(blk.fw) * value(blk.delta[i]) \
                        * value(blk.sol_prop_c[i].rho_sol) \
                        * (1 - value(blk.ed[i])) * value(blk.vb[i])
            if blk.s_inlet == 'Top' and blk.s_outlet == 'Underflow':
                blk.Je[i] = value(blk.Jc[i]) + value(blk.Solid_In_F) \
                            / value(blk.Ax)
            elif blk.s_inlet == 'Bottom' and blk.s_outlet == 'Overflow':
                blk.Je[i] = value(blk.Jc[i]) - value(blk.Solid_In_F) \
                            / value(blk.Ax)
            else:
                blk.Je[i] = value(blk.Jc[i])
            blk.us[i] = value(blk.Je[i]) / (value(blk.delta_e[i]) \
                                            * value(blk.sol_prop_e[i].rho_sol) \
                                            * (1 - value(blk.ed[i])))
            blk.Ksbulk[i] = 03
            for j in blk.SolidList:
                blk.Jcc[j, i] = value(blk.Jc[i]) * value(blk.xc[j, i])
                blk.Jec[j, i] = value(blk.Jcc[j, i])
                blk.Ksbulk_c[j, i] = 0

        blk.Solid_Out_F = value(blk.Solid_In_F)
        for j in blk.SolidList:
            blk.Solid_Out_x[j] = value(blk.xe[j, 0])

        # Activate constraints
        blk.eq_b7.activate()
        blk.eq_b8.activate()

        blk.eq_c5.activate()
        blk.eq_c6.activate()
        blk.eq_c7.activate()
        blk.eq_c8.activate()

        blk.eq_d10.activate()
        blk.eq_d11.activate()
        blk.eq_d14.activate()

        blk.eq_e7.activate()
        blk.eq_e8.activate()

        blk.eq_f3.activate()

        # Unfix variables
        blk.Jcc.unfix()
        blk.Jec.unfix()
        blk.xc.unfix()
        blk.xe.unfix()

        # Part 3 - Activate reaction constraints and unfix reaction variables
        for i in blk.l_n:
            if i != 0:
                blk.sol_prop_c[i].r_gen.unfix()
                blk.sol_prop_e[i].r_gen.unfix()
                blk.sol_prop_c[i].eq_r4.activate()
                blk.sol_prop_e[i].eq_r4.activate()

        sopt_mb = {"tol": 1e-7,
                   "max_cpu_time": 300,
                   "print_level": 5,
                   "halt_on_ampl_error": 'yes',
                   "mu_strategy": 'monotone',
                   "bound_push": 1e-8}

        ts = time.time()

        # Solve and print progress
        if outlvl > 0:
            print("Step  3, Mass Balances,                  Time: ",
                  end="")
        results = blk.solve(options=sopt_mb, tee=stee, symbolic_solver_labels=True)
        if outlvl > 0:
            print("{}, {}".format(model_util.hhmmss(time.time() - ts),
                                  results.solver.message))

        # %%        # ---------------------------------------------------------------------
        # 4th Initialisation Step - Energy Balances

        for i in blk.l_n:
            blk.Gbh[i] = value(blk.sf_gh) * value(blk.Gb[i]) \
                         * value(blk.gas_prop_b[i].h_vap)
            if i == 0:
                blk.Geh[i] = value(blk.sf_gh) * value(blk.Ge[i]) \
                             * value(blk.gas_prop_b[i].h_vap)
            else:
                blk.Geh[i] = value(blk.sf_gh) * value(blk.Ge[i]) \
                             * value(blk.gas_prop_e[i].h_vap)

        # Activate constraints
        blk.eq_b4.activate()
        blk.eq_b5.activate()
        blk.eq_b6.activate()

        blk.eq_c3.activate()
        blk.eq_c4.activate()
        blk.eq_d4.activate()
        blk.eq_d5.activate()

        # Unfix variables
        blk.Gbh.unfix()
        blk.Geh.unfix()
        blk.Tgb.unfix()
        blk.Tgc.unfix()
        blk.Tge.unfix()

        # Part 2 - Solids Balances
        for i in blk.l_n:
            blk.Jch[i] = value(blk.sf_sh) * value(blk.Jc[i]) \
                         * value(blk.sol_prop_c[i].h_sol)
            blk.Jeh[i] = value(blk.sf_sh) * value(blk.Je[i]) \
                         * value(blk.sol_prop_e[i].h_sol)
            blk.Hsbulk[i] = 0
        blk.Solid_Out_F = value(blk.Solid_In_F)
        for j in blk.SolidList:
            blk.Solid_Out_x[j] = value(blk.xe[j, 0])

        # Activate constraints
        blk.eq_b9.activate()
        blk.eq_b10.activate()

        blk.eq_c9.activate()
        blk.eq_c10.activate()

        blk.eq_d12.activate()
        blk.eq_d13.activate()
        blk.eq_d15.activate()

        blk.eq_f4.activate()
        blk.eq_f9.activate()
        blk.eq_f10.activate()

        # Unfix variables
        blk.Jch.unfix()
        blk.Jeh.unfix()
        blk.Tsc.unfix()
        blk.Tse.unfix()

        sopt_eb = {"tol": 1e-7,
                   "max_cpu_time": 300,
                   "print_level": 5,
                   "halt_on_ampl_error": 'yes',
                   "mu_strategy": 'monotone',
                   "bound_push": 1e-8}

        ts = time.time()
        # Solve and print progress
        if outlvl > 0:
            print("Step  4, Energy Balances,                  Time: ",
                  end="")
        results = blk.solve(options=sopt_eb, tee=stee, symbolic_solver_labels=True)
        if outlvl > 0:
            print("{}, {}".format(model_util.hhmmss(time.time() - ts),
                                  results.solver.message))

        #        # Initialization step 3c - change initial conditions
        #
        ##        blk.Gas_In_F.fix(298.11) #400 #330 #300 #298.11 #273
        #        blk.Gas_In_T.fix(800)
        ##        blk.Gas_In_P.fix(585000)
        ##        blk.Gas_In_y.unfix()
        #
        #        ts = time.time()
        #        # Solve and print progress
        #        if outlvl > 0:
        #            print("Step  3c, Change initial conditions,                  Time: ",
        #                  end="")
        #        results = blk.solve(options=sopt_eb,tee=stee)
        #        if outlvl > 0:
        #            print("{}, {}".format(model_util.hhmmss(time.time() - ts),
        #                    results.solver.message))

        ##%%        # ---------------------------------------------------------------------
        #        # 5th Initialisation Step - HX Tubes
        #        if blk.hx_type == 'SPD':
        #            for i in blk.l_n:
        #                blk.Hhx[i] = value(blk.sf_hh)*value(blk.HX_In_F) \
        #                                * value(blk.hx_prop_in.h_mix)
        #                if i != 0:
        #                    blk.kpa[i] = (3.58-2.5*value(blk.ed[i])) \
        #                                * value(blk.gas_prop_e[i].k_vap) \
        #                                * ((value(blk.sol_prop_e[i].k_sol) \
        #                                    / value(blk.gas_prop_e[i].k_vap)) \
        #                                ** (0.46-0.46*value(blk.ed[i])))
        #                    blk.fn[i] = value(blk.vg[i])/value(blk.sol_prop_e[i].v_mf)
        #                    blk.tau[i] = 0.44*((value(blk.sol_prop_e[i].dp) \
        #                                * value(blk.gc) \
        #                                / ((value(blk.sol_prop_e[i].v_mf)**2) \
        #                                * ((value(blk.fn[i]) \
        #                                    - value(blk.ah))**2)))**0.14) \
        #                                * ((value(blk.sol_prop_e[i].dp) \
        #                                    / value(blk.dx))**0.225)
        #                    blk.fb[i] = 0.33*(((value(blk.sol_prop_e[i].v_mf)**2) \
        #                                * ((value(blk.fn[i])-value(blk.ah))**2) \
        #                                / (value(blk.sol_prop_e[i].dp) \
        #                                * value(blk.gc)))**0.14)
        #                    blk.hd[i] = 1000
        #                    blk.hl[i] = 10
        #                    blk.ht[i] = 100
        #                    blk.Ttube[i] = value(blk.Thx[i])
        #                    blk.dThx[i] = value(blk.Thx[i]) - value(blk.Tse[i])
        #
        #            # Activate constraints
        #            blk.eq_j1.activate()
        #            blk.eq_j2.activate()
        #            blk.eq_j3.activate()
        #            blk.eq_j4.activate()
        #            blk.eq_j5.activate()
        #            blk.eq_j6.activate()
        #            blk.eq_j7.activate()
        #            blk.eq_j8.activate()
        #
        #            blk.eq_k2.activate()
        #            blk.eq_k3.activate()
        #            blk.eq_k4.activate()
        #            blk.eq_k5.activate()
        #            blk.eq_k6.activate()
        #            blk.eq_k7.activate()
        #            blk.eq_k8.activate()
        #            blk.eq_k9.activate()
        #
        #            # Unfix variables
        #            blk.Hhx.unfix()
        #            blk.Phx.unfix()
        #            blk.Qhx.unfix()
        #            blk.Thx.unfix()
        #
        #        if blk.hx_type == None or blk.hx_type == 'None':
        #            if outlvl > 0:
        #                print("Step  5, HX Tubes: Skipped")
        #        else:
        #            ts = time.time()
        #            # Solve and print progress
        #            if outlvl > 0:
        #                print("Step  5, HX Tubes,                                  "\
        #                      "Time: ",
        #                  end="")
        #            results = blk.solve(options=sopt,tee=stee)
        #            if outlvl > 0:
        #                print("{}, {}".format(model_util.hhmmss(time.time() - ts),
        #                    results.solver.message))
        #
        # %%        # ---------------------------------------------------------------------
        # 6th Initialisation Step - Outlets
        # Activate constraints
        blk.eq_d6.activate()
        blk.eq_d7.activate()
        blk.eq_d8.activate()
        blk.eq_d9.activate()

        # Unfix variables
        blk.Gas_Out_F.unfix()
        blk.Gas_Out_T.unfix()
        blk.Gas_Out_P.unfix()
        blk.Gas_Out_y.unfix()

        sopt_outlet = {"tol": 1e-7,
                       "max_cpu_time": 300,
                       "print_level": 5,
                       "halt_on_ampl_error": 'yes',
                       "mu_strategy": 'monotone',
                       "bound_push": 1e-8}

        ts = time.time()
        # Solve and print progress
        if outlvl > 0:
            print("Step  6, Outlets,                                   Time: ",
                  end="")
        results = blk.solve(options=sopt_outlet, tee=stee, symbolic_solver_labels=True)
        if outlvl > 0:
            print("{}, {}".format(model_util.hhmmss(time.time() - ts),
                                  results.solver.message))

        # ---------------------------------------------------------------------
        if outlvl > 0:
            print("\n\nTotal Initialisation Time: {0}".format(
                model_util.hhmmss(time.time() - total_time_start)))
            print("BFB Initialization Complete")
            print("----------------------------------------------------------")

        # Additional variables used for plotting purposes in flowsheet
        blk.cbt = Var(blk.l_n, domain=NonNegativeReals)
        for i in blk.l_n:
            blk.cbt[i] = sum(blk.cb[j, i].value for j in blk.GasList)
