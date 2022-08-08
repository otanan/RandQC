#!/usr/bin/env python3
"""Handles generating paths for IO of files and data in a RandQC data folder.

**Author: Jonathan Delgado**

Creates a unified system that creates a platform-independent method of storing relevant RandQC files in a way that's convenient for other modules (such as randqc.loader) to access for other kinds of manipulation.

The rule of thumb for separating folder paths is: if the quantity affects the state and output (such as the topology) it belongs in a separate folder. If it only affects calculating a quantity of interest (bipartition or renyi_parameter) it can be placed in the same folder with a separate identifier for the file name.

"""
from pathlib import Path

from randqc.tools.parser import classify_circuit, get_serial


def allowed_chars():
    """ Returns the allowed characters to use in paths and filenames based off 
        of the intersection of allowed characters for MacOS, Windows, and Linux.
    
        Returns:
            (str): the allowed characters as a single string.
    
    """
    symbols = '!@#$%^&*()_-+=[]|/ .,<>?:;\{\}\\\'"'
    # First filter: http://xahlee.info/mswin/allowed_chars_in_file_names.html
    symbols = symbols.replace('/','').replace('\\','').replace(':','').replace('*','').replace('?','').replace('<','').replace('>','').replace('|','')
    # Second filter: https://en.wikipedia.org/wiki/Filename.
    symbols = symbols.replace('%','').replace('"','').replace('.','').replace(',','').replace(';','').replace(' ','').replace('=','')
    # Third filter, for Glob: https://stackoverflow.com/questions/34899214/glob-is-not-working-when-directory-name-with-special-characters-like-square-brac
    symbols = symbols.replace('[', '').replace(']', '')

    return symbols


def has_suffix(fname):
    """ Checks whether the fname provided has a real suffix or just a period, 
        which may be used for something like a Renyi parameter.
        
        Args:
            fname (str/pathlib.PosixPath): the file path to check.
    
        Returns:
            (bool): True if there is a real extension, False otherwise.
    
    """

    # Check if the suffix is really a suffix and not a renyi_parameter
    suffix = fname.suffix
    if suffix == '':
        return False

    # Remove the dot and try to cast it to a float
    suffix = str(suffix)[1:]
    # Remove trailing parentheses if they exist:
    if suffix[-1] == ')':
        suffix = suffix[:-1]

    try:
        float(suffix)
    except ValueError:
        # It was a real suffix
        return True
    # The cast was successful, it was not a real suffix.
    return False

######################## Defaults ########################


def get_fname(N, input_state, instructions, topology='complete', identifier=''):
    """ Fills out an fname template according to the module's conventions.
        
        Args:
            N (int): the number of qubits.

            input_state (str): the input state label.

            instructions (str): the circuit instructions.

        Kwargs:
            topology (str): the topology of the circuit applied to the state.

            identifier (str): optional additional string that may be used to identify and separate this folder path from others.
    
        Returns:
            (str): the formatted fname.
    
    """
    fname = f'N({N})-q0({input_state})-inst[{instructions}]'

    top_label = '' if topology.lower() == 'complete' else f'-top({topology})'
    id_label = '' if identifier == '' else f'-{identifier}'

    # Append additional information regarding the states
    fname += top_label + id_label

    return fname

def default_folder():
    """ The default folder used as the most top-level directory usable for 
        RandQC saving and IO.
    
        Returns:
            (pathlib.PosixPath): the default folder path.
    
    """
    return Path.home() / 'RandQC'


######################## Custom paths ########################


#------------- General paths -------------#


def get_save_path(root_path, N, input_state, instructions, topology='complete', identifier=''):
    """ Gets the absolute path to the folder whose parent is the root_path. 
        This folder is used to save/load any data relevant to the system with the argument's parameters.
        
        Args:
            root_path (pathlib.PosixPath): the most top-level folder for RandQC IO.

            N (int): the number of qubits.

            input_state (str): the input state label.

            instructions (str): the circuit instructions.

        Kwargs:
            topology (str): the topology of the circuit applied to the state.

            identifier (str): optional additional string that may be used to identify and separate this folder path from others.

        Returns:
            (pathlib.PosixPath): the path to the folder for this system.
    
    """
    if type(root_path) is str: root_path = Path(root_path)

    # The base save path will not care about the bipartition, we will
        # distinguish this in the sub folder such as training data paths
        # and entropy file names
    return root_path / get_fname(N, input_state, instructions, topology=topology, identifier=identifier)


#------------- State paths -------------#


def get_save_state_path(root_path, N, input_state, instructions, topology='complete', identifier=''):
    """ Gets the absolute path to the folder whose ancestor is the root_path. 
        This folder is used to save/load states that may exist in a system with the argument's parameters.
        
        Args:
            root_path (pathlib.PosixPath/str): the most top-level folder for RandQC IO.

            N (int): the number of qubits.

            input_state (str): the input state label.

            instructions (str): the circuit instructions.

        Kwargs:
            topology (str): the topology of the circuit applied to the state.

            identifier (str): optional additional string that may be used to identify and separate this folder path from others.

        Returns:
            (pathlib.PosixPath): the path to the folder for this system.
    
    """
    if type(root_path) is str: root_path = Path(root_path)

    save_path = get_save_path(root_path, N, input_state, instructions, topology=topology, identifier=identifier)

    fname = get_fname(N, input_state, instructions, topology=topology, identifier=identifier)

    fname = f'states-{fname}-{classify_circuit(input_state, instructions)}'

    return save_path / fname


def get_save_state_fname(N, input_state, instructions, topology='complete', identifier=''):
    """ Returns the conventional file name for state saving. Distinct from 
        get_save_state_file_path which uses this filename and the path for get_save_state_path and creates the path for saving a file rather than just the filename.
        
        Args:
            N (int): the number of qubits.

            input_state (str): the input state label.

            instructions (str): the circuit instructions.

        Kwargs:
            topology (str): the topology of the circuit applied to the state.

            identifier (str): optional additional string that may be used to identify and separate this folder path from others.

        Returns:
            (str): the state filename.
    
    """
    fname = get_fname(N, input_state, instructions, topology=topology, identifier=identifier)

    return f'state-{fname}'


def get_save_state_file_path(root_path, N, input_state, instructions, topology='complete', identifier=''):
    """ Gets the absolute path to the file whose ancestor is the root_path. 
        This path is to save an individual state file. Distinct from get_save_state_path which returns the path to the folder not the filename.
        
        Args:
            root_path (pathlib.PosixPath): the most top-level folder for RandQC IO.

            N (int): the number of qubits.

            input_state (str): the input state label.

            instructions (str): the circuit instructions.

        Kwargs:
            topology (str): the topology of the circuit applied to the state.

            identifier (str): optional additional string that may be used to identify and separate this folder path from others.

        Returns:
            (pathlib.PosixPath): the path to the state for this system.
    
    """
    save_state_path = get_save_state_path(root_path, N, input_state, instructions, topology=topology, identifier=identifier)
    fname = get_save_state_fname(N, input_state, instructions, topology=topology, identifier=identifier)

    return save_state_path / fname


#------------- Data paths -------------#

def get_entropy_file_path(root_path, N, input_state, instructions, bipartition=None, renyi_parameter=1, topology='complete', identifier=''):
    """ Generates the path to the file whose ancestor is the root_path folder. 
        This path is used to save entropy data based on the system's signature.
        
        Args:
            root_path (pathlib.PosixPath): the most top-level folder for RandQC IO.

            N (int): the number of qubits.

            input_state (str): the input state label.

            instructions (str): the circuit instructions.
    
        Kwargs:
            bipartition (int): the bipartition to use.
            
            renyi_parameter (float): the type of Renyi entropy used.

            topology (str): the topology of the circuit applied to the state.

            identifier (str): optional additional string that may be used to identify and separate this file path from others.
    
        Returns:
            (pathlib.PosixPath): the path to the entropy data file.
    
    """
    save_path = get_save_path(root_path, N, input_state, instructions, topology=topology, identifier=identifier)

    fname = get_fname(N, input_state, instructions, topology=topology, identifier=identifier)

    if bipartition != N//2: fname += f'-bipart({bipartition})'

    # Label it as an entropy file and mark the Renyi_parameter if it's not 
        # unity.
    fname += '-entropies'
    if renyi_parameter != 1:
        fname += f'({renyi_parameter})'

    return save_path / fname


def get_data_file_path(root_path, N, input_state, instructions, map_name, topology='complete', identifier='', ):
    """ Generates the path to the file whose ancestor is the root_path folder. 
        This path is used to save data based on the system's signature.
        
        Args:
            root_path (pathlib.PosixPath): the most top-level folder for RandQC IO.

            N (int): the number of qubits.

            input_state (str): the input state label.

            instructions (str): the circuit instructions.
    
            map_name (str): the name of the function used for this quantity of interest.
    
        Kwargs:
            topology (str): the topology of the circuit applied to the state.

            identifier (str): optional additional string that may be used to identify and separate this file path from others.
    
        Returns:
            (pathlib.PosixPath): the path to the entropy data file.
    
    """
    save_path = get_save_path(root_path, N, input_state, instructions, topology=topology, identifier=identifier)

    fname = get_fname(N, input_state, instructions, topology=topology, identifier=identifier)

    fname += f'-map({map_name})' if map_name != '' else ''

    return save_path / fname

#------------- Entry code -------------#

def main():
    print('paths.py')

    chars = allowed_chars()
    print(chars)
    

if __name__ == '__main__':
    main()