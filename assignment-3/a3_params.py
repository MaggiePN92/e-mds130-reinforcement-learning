from dataclasses import dataclass
import multiprocessing as mp

@dataclass
class DA2CParams:
    """Class for keeping track of params used in training of 
    DA2C-models. 
    """
    epochs: int
    h1_size: int
    lr : float
    gamma: float
    workers: int
    assert workers <= mp.cpu_count()

@dataclass
class NDA2CParams(DA2CParams):
    """Class for keeping track of params used in training of 
    NDA2C-models. 
    """
    nstep: int