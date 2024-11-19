from solovision.motion.cmc.ecc import ECC
from solovision.motion.cmc.orb import ORB
from solovision.motion.cmc.sift import SIFT
from solovision.motion.cmc.sof import SOF


def get_cmc_method(cmc_method):
    if cmc_method == 'ecc':
        return ECC
    elif cmc_method == 'orb':
        return ORB
    elif cmc_method == 'sof':
        return SOF
    elif cmc_method == 'sift':
        return SIFT
    else:
        return None
