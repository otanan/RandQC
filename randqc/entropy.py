#!/usr/bin/env python3
"""Handles entropy related calculations.

**Author: Jonathan Delgado**

This module can calculate various entropies of interest such as the Renyi entropy of a bipartitioned state.

"""
import numpy as np
import skimage.measure

import randqc.tools.parser as parser
import randqc.imager as imager


def _sing_values_of_bipart_state(state, bipartition=None):
    # Calculates the singular values of a bipartitioned state.
    return np.linalg.svd(
        imager.partition_state(state, bipartition),
        compute_uv=False
    )


def mean_spec_gap_ratio(states, bipartition=None):
    """ Calculates the mean of the ratio of spectral gaps of the bipartitioned
        state or collection of states. Intimately related to the entanglement spectrum statistics of the state. https://arxiv.org/abs/1906.01079.
        
        Args:
            states (list/np.ndarray/qt.Qobj): a list of states or a single state.
    
        Returns:
            (float): the mean of the ratio of the spectral gaps of the state.
    
    """
    if type(states) != list:
        # Use the code for handling multiple states in one
        states = [states]

    if bipartition == None: bipartition = parser.get_N(states[0]) // 2

    eig_vals = np.array([])

    for state in states:
        eig_vals = np.append(eig_vals, _sing_values_of_bipart_state(state, bipartition))

    # Sorts and removes duplicates eigenvalues via np.unique
    eig_vals = np.unique(eig_vals)
    # filtered_vals = np.unique(eig_vals)
    # if len(eig_vals) - len(filtered_vals) > 0:
        # Announce how many duplicate eigenvalues appeared and were filtered.
        # print(f'Number of duplicate eigenvalues removed: {len(eig_vals) - len(filtered_vals)}')
    # Rename the variable for readability
    # eig_vals = filtered_vals

    # svd[i] - svd[i+1]
    gaps = eig_vals[:-1] - eig_vals[1:]

    rs = []

    # This process can probably be vectorized.
    for index, gap in enumerate(gaps[:-1]):
        gap2 = gaps[index + 1]

        r = gap / gap2
        # Use the min gap in the numerator and max in denominator
        if r > 1: r = 1/r

        rs.append(r)

    return np.mean(rs)


def shannon(state, bipartition=None):
    """ Calculates the Shannon entropy of the image representation of a state.
        
        Args:
            state (qt.Qobj/np.ndarray): the state.

        Kwargs:
            bipartition (int/None): the bipartition to use. Defaults to N//2 if None is provided.
    
        Returns:
            (float): the Shannon entropy.
    
    """
    if bipartition == None: bipartition = parser.get_N(state) // 2

    image = imager.state_to_image(state, bipartition, data_only=True)

    return skimage.measure.shannon_entropy(image)


def renyi(state, bipartition=None, renyi_parameter=1):
    """ Bipartitions a state and calculates the Renyi entropy of this 
        bipartitioned state. Returns the Von Neumann entropy of the bipartitioned state if renyi_parameter == 1.
        
        Args:
            state (qt.Qobj/np.ndarray): the state.

        Kwargs:
            bipartition (int/None): the bipartition. Defaults to N//2 if None is provided.
            
            renyi_parameter (float): the renyi_parameter. Defaults to the Von Neumann entropy.
    
        Returns:
            (float): the Renyi entropy of the bipartitioned state.
    
    """
    if bipartition == None: bipartition = parser.get_N(state) // 2

    if renyi_parameter == 1:
        return vn(state, bipartition)

    squared_singular_values = np.square(_sing_values_of_bipart_state(state, bipartition))
    return np.log2((squared_singular_values**renyi_parameter).sum()) / (1 - renyi_parameter)    


def vn(state, bipartition):
    """ Calculates the Von Neumann entropy of a bipartitioned state.
        
        Args:
            state (qt.Qobj/np.ndarray): the state.
            
            bipartition (int): the bipartition to use.
    
        Returns:
            (float): the Von Neumann entropy of the bipartitioned state.
    
    """
    squared_singular_values = np.square(_sing_values_of_bipart_state(state, bipartition))
    # Remove any zeros to avoid issues when applying logarithm
        # fine since we define 0 log(0) to be 0 anyways
    probabilities = squared_singular_values[squared_singular_values != 0]
    return -(probabilities * np.log2(probabilities)).sum()



#------------- Entry code -------------#

def main():
    print('entropy.py')
    

if __name__ == '__main__':
    main()