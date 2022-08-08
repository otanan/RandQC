#!/usr/bin/env python3
"""Module that handles the logic of manipulating states to bipartition them, generate images, and more.

**Author: Jonathan Delgado**

This module can handle converting states to images, manipulate these images, bipartition states, multi-partition states, change bases, and more.

"""
from pathlib import Path
import numpy as np

# Image generation
from PIL import Image
from matplotlib import cm
from matplotlib.colors import ListedColormap

import qutip as qt

import randqc.tools.parser as parser

# Ensures the grayscale colormap is gray linearly.
GRAY_CMAP = ListedColormap(np.array([
    np.linspace(0, 1, 256),
    np.linspace(0, 1, 256),
    np.linspace(0, 1, 256),
    np.full((256,), 1)
]).T)


#------------- State manipulations -------------#

def partition_state(state, partition):
    """ Generalizes the bipartition by converting the state into a 
        multi-dimensional array.
    
        Args:
            state (qutip.Qobj/numpy.ndarray): the state to partition.

            partition (list/int): the list of indices to partition the state into, the list is expected to be in increasing order and its last element cannot be larger than the number of qubits in the system. An example partitioning for an N=8 state would be (2, 4, 6) to equally break the state into a 4-dimensional array each with 4 elements (2 qubits). Note that a partitioning list of length n will result in an (n+1)-dimensional array. If a single integer is taken, it will be taken as a bipartition.
    
        Returns:
            (numpy.ndarray): the partitioned state represented as a numpy matrix.
    
    """
    N = parser.get_N(state)

    # In the case of a bipartition, only a single number is passed
    if type(partition) is int:
        partition = [partition]

    # If the last element of the partition exceeds the number of qubits
    if partition[-1] > N:
        print('Invalid partitioning of state. Partitioning must not exceed number of qubits.')
        return

    # We need to include the number of qubits to the partition
    partition = np.append(partition, N)
    # The partitioning before served as a list of indices determining where
        # to cut off the state, but now we need to use this list to keep track
        # of how big each dimension of the array needs to be
    # Note that diff will make us lose the first element so we need to
        # re-include it
    new_shape = 2**np.append(partition[0], np.diff(partition))

    # In the case of a bipartition we take the state, as a row and wrap it 
        # over as to fit into the dimensions of the matrix, i.e. if the system 
        # has 3 qubits then the state will have 2**3 components, if we choose 
        # a bipartition of 1 then the dimensions of the matrix will be 
        # 2^1x2^(3-1)= 2x4 so q[0] will be mapped to A[0, 0] and q[5] will be 
        # mapped to A[1,1]
    return np.reshape(state, new_shape)


def bipartition_state(state, bipartition):
    """ Bipartitions the state. Serves as a mask for partition_state where 
        there is only one partition.
        
        Args:
            state (qutip.Qobj/numpy.ndarray): the state to partition.

            bipartition (int): the bipartition to use.
    
        Returns:
            (numpy.ndarray): the bipartitioned state represented as a numpy matrix.
    
    """
    return partition_state(state, bipartition)


def state_to_prob_matrix(state, bipartition=None):
    """ Converts a state to a bipartitioned matrix where each element is the 
        square moduli of the amplitude of the corresponding bipartition matrix, where by Born's rule represents the probability.
        
        Args:
            state (qutip.Qobj/numpy.ndarray): the state.
    
        Kwargs:
            bipartition (int): the bipartition to use. Defaults to N//2.
    
        Returns:
            (numpy.ndarray): the probability matrix.
    
    """
    if bipartition is None: bipartition = parser.get_N(state) // 2

    return np.abs(bipartition_state(state, bipartition))**2


#------------- Image Operations -------------#


def state_to_image(state, bipartition=None, cmap=GRAY_CMAP, data_only=False):
    """ Converts a state to a corresponding image based on the given 
        bipartition.
        
        Args:
            state (qutip.qObj/numpy.ndarray): the state to be converted to an image.
    
        Kwargs:
            bipartition (int): the bipartition to be used, related to the dimensions of the image produced.

            cmap (matplotlib.cm): the colormap to be applied to the image.

            data_only (bool): False returns the Image object. True returns the numpy.ndarray of the image object.
    
        Returns:
            (PIL.Image/numpy.ndarray): the image object or image matrix depending on the data_only arg.
    
    """
    image = matrix_to_image(
        state_to_prob_matrix(state, bipartition=bipartition),
        cmap=cmap
    )
    return np.array(image) if data_only else image


def matrix_to_image(matrix, cmap=GRAY_CMAP):
    """ Converts a matrix of nonnegative real numbers to a gray-scale or 
        colored image depending on a colormap. Since each component is only one real number associated to each pixel, the exact colors themselves are for visualization purposes only and are not information from the matrix itself. Each matrix is renormalized to the interval [0, 1] to prevent larger images from being "dimmer".
        
        Args:
            matrix (numpy.ndarray): the matrix.
    
        Kwargs:
            cmap (matplotlib.cm): the colormap to be applied to the image.
    
        Returns:
            (PIL.Image): the image associated to the provided matrix.
    
    """
    # Normalize the matrix to fill the entire gray-scale range
    image_matrix = matrix / matrix.max()
    # Color it using the provided colormap
    image_matrix = cmap(image_matrix)        

    image_matrix *= 255
    return Image.fromarray(image_matrix.astype(np.uint8))


def qubit_change_of_basis(state, basis='x'):
    """ Changes the basis of a state by calculating the change of basis to 
        each qubit, then applies the corresponding transformation to the
        entire state.
        
        Args:
            state (qutip.Qobj/numpy.ndarray): the state.
    
        Kwargs:
            basis (str): the basis to be changed to from the computational basis. Pass in 'rand' in order to use a random qubit basis. Other options include: 'x' for the Pauli-X basis.
    
        Returns:
            (qutip.Qobj): the state as represented in the new basis.
    
    """
    N = parser.get_N(state)

    # The original basis
    computational_basis = np.array([
                            [1, 0],
                            [0, 1]
                        ])

    # Pick the new basis
    if basis == 'x':
        new_basis = np.array( [
                        computational_basis[0] + computational_basis[1],
                        computational_basis[0] - computational_basis[1]
                    ]).T / np.sqrt(2)

    elif basis == 'rand':
        new_basis = qt.rand_unitary_haar(2)

    else:
        print(f'Invalid basis provided: {basis}.')
        print('Returning the state unchanged...')
        return state

    # Construct the change of basis operator for the entire state
    tensored_new_basis = new_basis
    for _ in range(N - 1):
        tensored_new_basis = np.kron(tensored_new_basis, new_basis)
    # Returns the object as a proper qObj
    return parser.qobj_from_array(np.matmul(tensored_new_basis, state))



#------------- Entry code -------------#

def main():
    print('imager.py')
    

if __name__ == '__main__':
    main()