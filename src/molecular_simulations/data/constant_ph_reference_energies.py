"""
Reference energies for constant pH simulations.

These reference energies are calibrated to give the correct pKa values
for titratable residues. The convention is:

  ref_energies[residue_type] = [E_deprotonated, E_protonated, ...]

Where E_deprotonated = 0 and E_protonated = kT * ln(10) * pKa

This ensures that at pH = pKa, both protonation states have equal probability.
"""


def get_ref_energies(ff: str = 'amber19'):
    """
    Get reference energies for constant pH simulations.

    Parameters
    ----------
    ff : str
        Force field name (currently only 'amber19' is supported)

    Returns
    -------
    dict
        Maps residue type names to lists of reference energies (kJ/mol).
        Index 0 is the deprotonated state, index 1+ are protonated states.
    """
    match ff.lower():
        case 'amber19':
            # Reference energies based on experimental pKa values at 300K
            # E_protonated = kT * ln(10) * pKa = 2.494 * 2.303 * pKa
            # This gives the correct equilibrium at each residue's pKa
            ref_energies = {
                'CYS': [0.0, 47.68],    # pKa = 8.3
                'ASP': [0.0, 22.40],    # pKa = 3.9
                'GLU': [0.0, 24.70],    # pKa = 4.3
                'LYS': [0.0, 60.32],    # pKa = 10.5
                'HIS': [0.0, 37.34],    # pKa = 6.5 (2-state HID/HIP)
            }
        case _:
            raise ValueError(f'Forcefield {ff} not yet computed!')

    return ref_energies
