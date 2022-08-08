#!/usr/bin/env python3
"""Module that handles saving of states and other relevant data files.

**Author: Jonathan Delgado**

This module will provides functionality for saving states, entropy data files, and more, in a manner that, by default, will follow the rest of the packages conventions, but otherwise provides optimal functionality for even custom saving.

"""
from pathlib import Path
import numpy as np
import h5py

import randqc.tools.paths as paths
import randqc.imager as imager
from randqc.tools.paths import default_folder
from randqc.imager import GRAY_CMAP
from randqc.tools.parser import get_serial


######################## Prep. ########################


def prepare_saving(path):
    """ Handles folder generation for saving outputs of simulations and the 
        naming of each file.
        
        Args:
            path (pathlib.PosixPath/str): the path to where saving will be done.

        Returns:
            (None): none
    
    """
    if type(path) is str: path = Path(path)

    # We need to make the necessary folders if they don't exist
    path.mkdir(parents=True, exist_ok=True)


def _adjust_fname(fname=default_folder()/'state', extension=
    '.npz', serial=None):
    # Handles serializing the filename and correcting the extension in the 
        # case where one was incorrectly provided. If no serial is provided, 
        # don't serialize (i.e. for entropy files).

    if type(fname) == str:
        fname = Path(fname)
    
    # Remove any existing suffix
    if paths.has_suffix(fname):
        fname = fname.with_suffix('')

    # Serialize it if one has been provided
    if serial != None:
        # Serialize the state
        fname = str(fname) + f'-{serial}'

    return Path(str(fname) + extension)


######################## States ########################


def save_state(state, fname=default_folder()/'state', serial=None):
    """ Saves states to file.
        
        Args:
            state (qutip.Qobj): the state to be saved.
    
        Kwargs:
            fname (pathlib.PosixPath/str): the path where the file is to be saved without the extension.

            serial (str/float): the option to provide a custom serial for this state. Serializes otherwise.
    
        Returns:
            (None): none
    
    """
    # Sets to the default serial if none was provided
    if serial == None:
        serial = get_serial()

    fname = _adjust_fname(fname, extension='.npz', serial=serial)

    # qt.qsave(state, fname)
    # After some elementary testing, .npz was preferred to save the states due
        # to file size
    with fname.open('wb') as f:
        np.savez_compressed(f, array1=state)


def save_state_as_image(state, fname=default_folder()/'stateimage', bipartition=None, cmap=GRAY_CMAP, serial=None):
    """ Converts a state to an image and then saves the image.
        
        Args:
            state (qutip.Qobj/numpy.ndarray): the state.
    
        Kwargs:
            fname (pathlib.PosixPath/str): the path where the file is to be saved without extension.

            bipartition (int): the bipartition.

            cmap (matplotlib.cm): the colormap to be applied to the image. Defaults to a linear grayscaling.

            serial (str/float): the option to provide a custom serial for this state. Serializes otherwise.
    
        Returns:
            (None): none
    
    """
    if serial == None:
        serial = get_serial()

    fname = _adjust_fname(fname, extension='.png', serial=serial)

    # Converts the quantum state to the corresponding image matrix
        # converts the image matrix to the image, then saves the output
        # to the path and filename provided
    # Colormaps: https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
    if colored: cmap = cm.cubehelix # {cm.viridis, cm.plasma}
    imager.state_to_image(state, bipartition, cmap=cmap).save(fname)


######################## Data ########################


def save_matrix(data, fname, overwrite=False):
    """ Saves generic numpy matrix to fname.
        
        Args:
            matrix (numpy.ndarray): the matrix to be saved.
            
            fname (pathlib.PoxixPath/str): the path and filename (without extension) to save the matrix to.

        Kwargs:
            overwrite (bool): whether to overwrite the matrix data.
    
        Returns:
            (None): none
    
    """
    fname = _adjust_fname(fname, extension='.hdf5', serial=None)
    
    if fname.exists() and not overwrite:
        # Append to data since the file already exists.

        with h5py.File(fname, 'a') as f:
            dset = f['default']
            # Add rows for data
            dset.resize(dset.shape[0] + data.shape[0], axis=0)
            # Copy new data set to last rows of file:
            dset[-data.shape[0]:] = data

    else:
        # Create a new dataset since the file doesn't exist or we are 
            # requested to overwrite it.
        with h5py.File(fname, 'w') as f:
            f.create_dataset('default', data=data, maxshape=(None, None))


def save_entropies(data, fname=default_folder()/'entropies', overwrite=False):
    """ Saves entropy data to file.
        
        Args:
            data (numpy.ndarray): the entropy data to be saved.
    
        Kwargs:
            fname (pathlib.PosixPath/str): the path where the file is to be saved without extension.

            serial (str/float): the option to provide a custom serial for this state. Serializes otherwise.

            overwrite (bool): whether to overwrite the entropy data.
    
        Returns:
            (None): none
    
    """
    save_matrix(data, fname, overwrite=overwrite)


#------------- Entry code -------------#

def main():
    print('saver.py')
    

if __name__ == '__main__':
    main()