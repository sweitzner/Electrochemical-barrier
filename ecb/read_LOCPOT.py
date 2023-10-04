from ase import Atoms
import numpy as np
from typing import Optional

"""The VaspLocpot class is contributed by Nathan Keilbart.
https://gitlab.com/nathankeilbart/ase/-/blob/VaspLocpot/ase/calculators/vasp/vasp_auxiliary.py
"""


class VaspLocpot:
    """Class for reading the Locpot VASP file.

    Filename is normally LOCPOT. Code is borrowed from the VaspChargeDensity
    class and altered to work for LOCPOT.
    """

    def __init__(self,
                 atoms: Atoms,
                 pot: np.ndarray,
                 spin_down_pot: Optional[np.ndarray] = None,
                 magmom: Optional[np.ndarray] = None
                 ) -> None:
        self.atoms = atoms
        self.pot = pot
        self.spin_down_pot = spin_down_pot
        self.magmom = magmom

    @staticmethod
    def _read_pot(fobj, pot):
        """Read potential from file object.

        Utility method for reading the actual potential from a file object. 
        On input, the file object must be at the beginning of the charge block, on
        output the file position will be left at the end of the
        block. The pot array must be of the correct dimensions.
        """
        # VASP writes charge density as
        # WRITE(IU,FORM) (((C(NX,NY,NZ),NX=1,NGXC),NY=1,NGYZ),NZ=1,NGZC)
        # Fortran nested implied do loops; innermost index fastest
        # First, just read it in
        for zz in range(pot.shape[2]):
            for yy in range(pot.shape[1]):
                pot[:, yy, zz] = np.fromfile(fobj, count=pot.shape[0], sep=' ')

    @classmethod
    def from_file(cls, filename='LOCPOT'):
        """Read the local potential from the VASP LOCPOT file.

        Currently will check for a spin-up and spin-down component but has not been
        configured for a noncollinear calculation.
        """

        import ase.io.vasp as aiv

        with open(filename, 'r') as fd:
            try:
                atoms = aiv.read_vasp(fd)
            except (IOError, ValueError, IndexError):
                return print('Error reading in initial atomic structure.')

            fd.readline()
            ngr = fd.readline().split()
            ng = tuple(map(int, ngr))
            pot = np.empty(ng)
            cls._read_pot(fd, pot)

            # Check if the file has a spin-polarized local potential, and
            # if so, read it in.
            fl = fd.tell()

            # Check to see if there is more information
            line1 = fd.readline()
            if line1 == '':
                return cls(atoms, pot)

            # Check to see if the next line equals the previous grid settings
            elif line1.split() == ngr:
                spin_down_pot = np.empty(ng)
                cls._read_pot(fd, spin_down_pot)

            elif line1.split() != ngr:
                fd.seek(fl)
                magmom = np.fromfile(fd, count=len(atoms), sep=' ')
                line1 = fd.readline()
                if line1.split() == ngr:
                    spin_down_pot = np.empty(ng)
                    cls._read_pot(fd, spin_down_pot)

        return cls(atoms, pot, spin_down_pot=spin_down_pot, magmom=magmom)

    def compute_planar_average(self, axis=2, spin_down=False):
        """Returns the planar average potential along an axis (0,1,2).

        axis: Which axis to average long
        spin_down: Whether to use the spin_down_pot instead of pot
        """

        if axis not in (0, 1, 2):
            raise ValueError('axis must be an integer value of 0, 1, or 2.')

        if spin_down:
            pot = self.spin_down_pot
        else:
            pot = self.pot

        indices = ((1, 2), (0, 2), (0, 1))
        return np.average(pot, axis=indices[axis])

    def distance_along_axis(self, axis=2):
        """Returns the scalar distance along axis (from 0 to 1)."""
        if axis not in [0, 1, 2]:
            raise ValueError('axis must be an integer value of 0, 1, or 2.')
        return np.linspace(0, 1, self.pot.shape[axis], endpoint=False)

    def is_spin_polarized(self):
        return (self.spin_down_pot is not None)


def align_vacuum(direction='Z', LOCPOTfile='LOCPOT'):
    """Align the vacuum level to the avg LOCPOT at the edge of the cell.

    (make sure it is vacuum there)
    returns: 
         the vacuum level ()
         the average electrostatic potential (vtot_new)
    """

    # the direction to make average in
    # input should be x y z, or X Y Z. Default is Z.
    allowed = "xyzXYZ"
    if allowed.find(direction) == -1 or len(direction) != 1:
        print("** WARNING: The direction was input incorrectly.")
        print("** Setting to z-direction by default.")

    if direction.islower():
        direction = direction.upper()

    # -- Open geometry and density class objects
    axis_translate = {'X': 0, 'Y': 1, 'Z': 2}
    ax = axis_translate[direction]
    # vasp_charge = VaspChargeDensity(filename = LOCPOTfile)
    vasp_locpot = VaspLocpot.from_file(filename=LOCPOTfile)
    average = vasp_locpot.compute_planar_average(axis=ax)

    # -- Lattice parameters and scale factor
    cell = vasp_locpot.atoms.cell

    # -- Find length of lattice vectors
    latticelength = np.dot(cell, cell.T).diagonal()
    latticelength = latticelength**0.5

    # -- Write the planar average of the potential to a file
    averagefile = f"average_{LOCPOTfile}_{direction}.dat"
    with open(averagefile, "w") as outputfile:
        outputfile.write("#  Distance(Ang)     Potential(eV)\n")
        xdis = vasp_locpot.distance_along_axis(axis=ax) * latticelength[ax]
        for xi, poti in zip(xdis, average):
            outputfile.write("%15.8g %15.8g\n" % (xi, poti))

    del vasp_locpot

    # -- Get the planar averaged potential at the edge of the cell
    vacuumE = average[-1]
    # -- Get the average of the aligned potential
    vtot_new = np.average(average - vacuumE)

    return vacuumE, vtot_new
