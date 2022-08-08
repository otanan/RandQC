#!/usr/bin/env python3
"""Unittest for randqc.loader.

**Author: Jonathan Delgado**

"""
from pathlib import Path
import numpy as np
import qutip as qt

import randqc.loader as loader

N = 4
input_state = 'prod'
instructions = 'Clx50'
bipartition = N // 2
renyi_parameter = 1.5

num_states = 3
state_loader = loader.Loader()

states = state_loader.get_states(N, input_state, instructions, num_states=num_states, data_only=False, GUI=True)

entropies, serials = state_loader.get_entropies(N, input_state, instructions, bipartition=bipartition, renyi_parameter=renyi_parameter, num_trials=num_states, serials=True)

#------------- Tests -------------#

assert len(states) == num_states
assert type(states[0]) is qt.Qobj, f'State type is: {type(states[0])}'
assert states[0] != states[num_states - 1]
# Adjust the column size to account for the serial
assert entropies.shape == (num_states, 51)


### Check legacy loading support ###
nrows = 4
folder_path = Path(__file__).parent / f'N({N})-q0({input_state})-inst[{instructions}]'
folder_path.mkdir()

fname = folder_path / 'test_matrix.txt'
test_matrix = np.eye(nrows)
# Save the test matrix in a legacy format
np.savetxt(fname, test_matrix)
# Use legacy loading
# We can silence the verbose flag since we know this matrix will be 
    # "unserialized".
entropies = loader.load_entropies(fname.with_suffix(''), verbose=True)
print(entropies)
# Delete the file and folder before the test is done in case of crash
fname.unlink()
folder_path.rmdir()
assert entropies.shape == (nrows, nrows)