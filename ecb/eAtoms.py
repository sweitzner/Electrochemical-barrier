#!/usr/bin/env python
"""Generalized Atoms class for the simultaneous optimization of geometry and number of electrons under constant voltage.

Contact: Penghao Xiao (pxiao@dal.ca, pxiao@utexas.edu)
Version: 1.0
Usage: first set the vacuum/solution along z axis and move the slab to the center of the simulation box
Please cite the following reference:
"""

from ase import *
from ase.io import read, write
from ase import units
from .read_LOCPOT import align_vacuum
import numpy as np


class eAtoms(Atoms):
    def __init__(self, atomsx, target_voltage=0.0, mu_e_ref=-4.6, solPoisson=True, weight=1.0, e_only=False, slab_norm='z'):
        """An extended ASE Atoms object for performing relaxation under constant electrochemical potential.

        Electrochemical potential: the work function of the counter electrode under the given target_voltage
        i.e. target_voltage vs. SHE + workfunction of SHE

        Attributes:
            atomsx: ase.Atoms
                A conventional ASE Atoms object.
            target_voltage: real
                Applied voltage with respect to the reference electrode.
            mu_e_ref: real
                Fermi level of the reference electrode. Default: -4.6 for SHE.
            solPoisson: bool, default=True
                True : compensate charge in the solvent, where VASPsol is required with lambda_d_k=3.0.
                False: uniform background charge.
            weight: real, default=1.0
                Weight of the number of electrons vs. atomic positions.
            jacobian: real
                Jacobian of the system
            e_only: bool, default=False
                True: Only optimize the number of electrons, corresponding to weight=infinity.
                False: Optimize both geometry and number of electrons.
            direction: string, default='z'
                Direction that corresponds to the surface normal. Options are ('x', 'y', 'z').
            target_mu_e: real
                Target Fermi level of the working level derived from target_voltage and mu_e_ref.
            n_atom: int
                Number of atoms in the system.
            n_e: np.ndarray, shape=(1, 3)
                Number of electrons in the system. Allowed to be fractional.
            mu_e: np.ndarray, shape=(1, 3)
                Fermi level / electron chemical potential (mu = dE/dn_e).
            vtot: real
                Shift of the electrostatic potential due to the compensating charge in DFT.
            n0: real
                Number of electrons associated with the neutrally charged system (at potential of zero charge).
            vtot0: real
                Potential of zero charge of the system.
        """

        # -- Initialize attributes based on args passed to the constructor
        self.atomsx = atomsx
        self.target_voltage = target_voltage
        self.mu_e_ref = mu_e_ref
        self.solPoisson = solPoisson
        self.weight = weight
        self.jacobian = weight
        self.e_only = e_only
        self.direction = slab_norm

        # -- Initialize remaining attributes
        self.target_mu_e = -target_voltage + mu_e_ref
        self.n_atom = len(atomsx)
        self.n_e = np.zeros((1, 3))  # number of electrons
        self.mu_e = np.zeros((1, 3))  # electron mu, dE/dne
        self.vtot = 0.0  # shift of the electrostatic potential due to the compensating charge in DFT
        self.n0 = None
        self.vtot0 = None

        Atoms.__init__(self, atomsx)

    def get_positions(self):
        """Returns the generalized position.

        The generalized position contains n_atom atomic positions and one 
        additional row containing the electron number jacobian.
        """
        r = self.atomsx.get_positions()
        Rc = np.vstack((r, self.n_e * self.jacobian))
        return Rc

    def set_positions(self, newr):
        """Helper function to update the atomic positions.

        Note: the last position in the extended atoms object is associated with
        the extra degree of freedom associated with the electrons.
        """
        ratom = newr[:-1, :]
        self.n_e[0, 0] = newr[-1, 0] / self.jacobian
        self.atomsx.set_positions(ratom)
        self._calc.set(nelect=self.n_e[0, 0])

    def __len__(self):
        return self.n_atom+1

    def get_forces(self, apply_constraint=True):
        """Returns the forces on the system."""

        f = self.atomsx.get_forces(apply_constraint)
        self.get_mu_e()

        if self.e_only:
            Fc = np.vstack((f*0.0, self.mu_e / self.jacobian))
        else:
            Fc = np.vstack((f, self.mu_e / self.jacobian))

        return Fc

    def get_potential_energy(self, force_consistent=False):
        """
        Returns the electronic grand canonical energy of the system.

        Eq. 1 of Ref 1:  H(r, n) = E(r, n) - (n - n0) * phi
        """

        # -- Get the uncorrected total energy from Vasp
        E_N = self.atomsx.get_potential_energy(force_consistent)

        # -- Get the total number of electrons at neutral.
        # should have been done in __ini__, but self._calc is not accessible there
        if self.n0 is None:
            self.n0 = self._calc.default_nelect_from_ppp()
            print(f"NELECT at PZC, n0: {self.n0}\n")

        # -- Compute the excess number of electrons
        if self.n_e[0, 0] < 0.01:
            self.n_e[0, 0] = self._calc.get_number_of_electrons()
        N_e = self.ne[0][0]
        Delta_N = N_e - self.n0

        # -- Get the unaligned Fermi level and the potential shift
        self.get_mue()
        E_F = self._calc.get_fermi_level()

        # -- Apply corrections due to the potential shift
        # Note: self.vtot = -phi_inner is the potential shift
        E_N_corr = E_N + Delta_N * self.vtot
        mu_e = E_F + self.vtot

        Phi = -mu_e
        Phi_SHE = -self.epotential  # Phi vs SHE

        # -- Compute thermodynamic potentials
        Omega = E_N_corr - Delta_N * mu_e
        H_N = E_N_corr - Delta_N * self.epotential
        # E0 += (self.ne[0][0]-self.n0) * (-self.epotential + self.vtot)

        d = f"@ {E_N:12.6f} {H_N:12.6f} {Omega:12.6f} {N_e:12.6f} {mu_e:12.6f} {self.epotential:12.6f}"
        print(d)

        return H_N

    def get_mu_e(self):
        """
        Updates the electronic chemical potential.

        Updates self.vtot: shift of the electrostatic potential due to the compensating charge in DFT

        Eq. 2 of Ref 1: mu = E_f / e - phi
        """
        # the initial guess for ne is passed through the calculator
        self.n_e[0, 0] = self._calc.get_number_of_electrons()

        # align the vacuum level and get the potential shift due to the compensating charge
        # vacuumE is used when the charge is compensated in solvent by VSAPsol
        # vtot_new is integrated when the charge is compensated by uniform background charge

        # vacuumE is the planar average of the potential at the edge of the cell
        # vtot_new is the average of the aligned potential [np.average(avg_pot_z - vacuumE)]
        vacuumE, vtot_new = align_vacuum(direction=self.direction,
                                         LOCPOTfile='LOCPOT')

        if self.solPoisson:
            self.vtot = -vacuumE
        else:
            if self.vtot0 is None:
                # start from zero charge for absolute reference
                self.vtot0 = vtot_new
            self.vtot = (self.vtot0 + vtot_new) * 0.5

        # Compute the chemical potential
        E_f = self._calc.get_fermi_level()
        self.mu_e[0, 0] = self.target_mu_e - (E_f - vacuumE)

        # Following my convention
        mu_e = E_f - vacuumE  # align E_f to solvent inner potential
        Phi = -mu_e           # Compute electrode potential relative to solvent inner potential
        Phi_ref = -self.mu_e_ref
        energy = self.extract_energy()
        if self.n0 is None:
            self.n0 = self._calc.default_nelect_from_ppp()

        Q = -(self.n_e[0, 0] - self.n0)
        # Omega = energy + (self.n_e[0, 0] - self.n0) * mu_e  # NOTE: Switched - -> +
        Omega = energy - Q * Phi
        # -- Print a summary
        print(f'\nN_elect          = {self.n_e[0, 0]}')
        print(f'mu_e (inner)     = {mu_e}')
        print(f'E(N_elect)       = {energy}')
        print(f'Omega(mu_e)      = {Omega}')
        print(f'Phi (SHE)        = {Phi - Phi_ref}')
        print(f'Target Phi (SHE) = {self.target_voltage} \n')

    def copy(self):
        """Return a copy of the eAtoms object."""
        # import copy
        atomsy = self.atomsx.copy()
        atoms = self.__class__(atomsy, self.target_mu_e)

        atoms.arrays = {}
        for name, a in self.arrays.items():
            atoms.arrays[name] = a.copy()
        # atoms.constraints = copy.deepcopy(self.constraints)
        # atoms.adsorbate_info = copy.deepcopy(self.adsorbate_info)
        return atoms

    def extract_energy(self):
        """Gets the energy from the vasp.out file"""

        energy = None

        p = re.compile(r"F= ([+-]?(?=\.\d|\d)(?:\d+)?(?:\.?\d*)[Ee][+-]?\d+)?",
                       re.IGNORECASE)

        with open('vasp.out', 'r') as f:
            vout = f.read()
            result = re.findall(p, vout)

        if result:
            energy = float(result[0])

        return energy
