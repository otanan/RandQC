#!/usr/bin/env python3
"""Miscellaneous tools.

**Author: Jonathan Delgado**

Script which holds miscellaneous tools and functions for use through RandQC.

Example:
    An example being:

        has_duplicates(array)

    which will return a boolean depending on whether the array provided has duplicate entries.

"""


######################## Logic ########################


def has_duplicates(array):
    """ Checks for duplicate items in the array without sorting.
        
        Args:
            array (list): the array to be checked
    
        Returns:
            (bool): True if the array has duplicate entries, False otherwise.
    
    """
    return len(array) != len(set(array))


#------------- Entry code -------------#

def main():
    print('misc.py')
    

if __name__ == '__main__':
    main()