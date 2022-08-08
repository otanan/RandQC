#!/usr/bin/env python3
"""Unittest for randqc.qc.

**Author: Jonathan Delgado**

"""
import qutip as qt

import randqc

N = 5
input_state = 'prod'
instructions = 'Clx50;Tx3;Clx50'
renyi_parameter = 1.5

states, data = randqc.randomqc(N, instructions=instructions, input_state_label=input_state, renyi_parameter=renyi_parameter)

#------------- Tests -------------#

# assert 1 + 1 == 3 # Purposefully incorrect test
assert len(states) == 3 + 1
assert type(states[-1]) is qt.Qobj
# Adjust the column size to account for the serial
assert data.shape == (1, 1 + 104)
# Assert the entropy_data is serialized.
assert randqc.loader.is_serialized(data)