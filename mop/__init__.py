"""MoP - Analyzes single-cell transcriptomic and epigenomic data"""

__version__ = '0.1.0'
__author__ = 'Wayne Doyle <widoyle@.ucsd.edu>'
__all__ = ['cemba',
           'clustering',
           'counts',
           'decomposition',
           'general_utils',
           'neighbors',
           'io',
           'loom_utils',
           'plot',
           'helpers',
           'recipes',
           'qc',
           'smooth',
           'snmcseq',
           'statistics'
           ]
# Set-up logger
import logging
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO,
                    format=log_fmt,
                    datefmt='%b-%d-%y  %H%M:%S')
logger = logging.getLogger(__name__)

# Import packages
from mop import *
