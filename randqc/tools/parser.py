#!/usr/bin/env python3
"""Tool to handle parsing of strings and other relevant objects.

**Author: Jonathan Delgado**

Extracts relevant information from strings such as file names that are saved in conventional manners (randqc.saver or randqc.tools.paths), as well as from qutip.Qobj's and numpy.ndarray's.

Example:
    get_N(state)

    which will return the number of qubits from a qutip.Qobj or a numpy.ndarray representing the state of a system with multiple qubits.

"""
from datetime import datetime
from pathlib import Path
import numpy as np
import qutip as qt
import ast


######################## Serialization ########################


def get_serial():
    """ Generates a unique serial code, useful for file saving to prevent 
        overwriting and conflicts. First portion of the serial code is the date and time requested (ymdHMS), with up to millisecond implementation. This makes the serial code sortable. Useful in creating identifiers for states.
    
        Returns:
            (int): the serial.
    
    """
    # Up to millisecond precision included
    return datetime.now().strftime('%y%m%d%H%M%S%f')[:-3]


######################## String parsers ########################


#------------- N -------------#


def parse_N_from_fname(fname):
    """ Parses the number of qubits (N) from a string such as a filename.
        
        Args:
            fname (str/pathlib.PosixPath): the filename or path.
    
        Returns:
            (int): N.
    
    """
    return int(str(fname).split('N(')[1].split(')')[0])


#------------- Instructions -------------#


def parse_instructions_from_fname(fname):
    """ Parses the circuit instructions or label from a string such as a 
        filename.
        
        Args:
            fname (str/pathlib.PosixPath): the filename or path.
    
        Returns:
            (str): the instructions.
    
    """
    return str(fname).split('inst[')[1].split(']')[0]


def classify_circuit(input_state_label, instructions):
    """ Parses the parameters of the system and returns the classification as 
        either Clifford or Universal.
        
        Args:
            input_state_label (str): the input state.

            instructions (str): the instructions.
    
        Returns:
            (str): the classification.
    
    """
    # Only interested in Universal gates represented by a single upper case 
    # letter.
    instructions = instructions.upper()

    if 'rand' in input_state_label.lower() or 'T' in instructions or 'U' in instructions:
        return 'Universal'

    return 'Clifford'


def get_num_gates(instructions):
    """ Gets the total number of gates used in a circuit by its instructions.
        
        Args:
            instructions (str/list): the instructions string or parsed instructions.
    
        Returns:
            (int): the number of gates.
    
    """
    if type(instructions) is str:
        instructions = parse_instructions(instructions)

    return sum([instruction[1] for instruction in instructions])


def get_number_of_blocks(instructions):
    """ Calculates the number of circuit blocks. For example, a circuit with 
        instructions: Clx100;Tx1;Cl100 has 3 blocks. The first one being 100 Clifford gates, the second being 1 T gate and so on.
        
        Args:
            instructions (str/list): the instructions string or parsed instructions.
    
        Returns:
            (int): the number of circuit blocks.
    
    """
    if type(instructions) is str:
        instructions = parse_instructions(instructions)

    return len(instructions)


def parse_instructions(instructions, delimiter=';'):
    """ String parser for circuit instructions. Parses the amount of circuit 
        blocks to use, the number of gates and gate types for each block. An example valid instructions string is: 4*(Clx8,Tx12);Clx40;Ux10. The n* indicates n layers of the circuit blocks in parentheses.
        
        Args:
            instructions (str): the circuit instructions.
    
        Kwargs:
            delimiter (str): how blocks are separated.
    
        Returns:
            (list): the parsed instructions.
    
    """
    # The actual instructions to later be interpreted by the circuit generator
    circuit_instructions = []
    # If no instructions are passed we simply want to run an empty circuit.
        # i.e. an identity operation on our input state
    if instructions == '':
        return circuit_instructions

    # Each piece is separated by a semicolon or comma depending on the
        # nesting. The delimiter command is used for recursively parsing
        # the nested circuit.
    instructions = instructions.split(delimiter)

    for instruction in instructions:
        # Split any repeated blocks into the sub-block and its amount of times to be repeated
        instruction = instruction.split('*')

        # If true, then the instruction is being asked to be repeated
        if len(instruction) == 2:
            # Remove the parentheses, append a comma to the end of the repeated
                # command. Repeat it, then remove the final additional comma
            instruction = (int(instruction[0]) * (instruction[1][1:-1] + ','))[:-1]

            # Recursively call the function to now parse the instructions
                # for this repeated block and then append it to our
                # current list of instructions
            circuit_instructions += parse_instructions(instruction, delimiter=',')

            continue

        # Separates each circuit into (type, gate_amounts)
        pair = instruction[0].split('x')
        # Corrects any capitalization. Important to be done here
            # since this will go into the filename
        pair[0] = pair[0].capitalize()

        try:
            gate_count = int(pair[1])
        except ValueError:
            print(f'Invalid instructions. Expected integer for gate count. Received "{pair[1]}" instead.')
            raise

        circuit_instructions.append((pair[0], gate_count))

    return circuit_instructions


def parse_T_gate_placements(instructions):
    """ Parses the placement of T gates in the instructions provided. In a 
        format usable by randqc.insert_T_gates for regenerating the instructions.
        
        Args:
            instructions (str): the circuit instructions.
    
        Returns:
            (list): the T gate placements.
    
    """
    instructions = parse_instructions(instructions)
    
    t_gate_placements = []
    gate_count = 0

    for gate_type, num_gates in instructions:

        if gate_type != 'T':
            gate_count += num_gates
        else:
            # Handle adjacent T gates.
            for _ in range(num_gates):
                gate_count += 1
                t_gate_placements.append(gate_count)
            
    return t_gate_placements


######################## Object parsers ########################


def get_N(state):
    """ Calculates the number of qubits represented by a qt.Qobj ket or a 
        numpy.ndarray vector.
        
        Args:
            state (qutip.Qobj/numpy.ndarray): the state.
    
        Returns:
            (int): N.
    
    """
    # return len(q.dims[0])
    # More general, works with normal numpy arrays
    return int(np.log2(get_dimension(state)))


def get_dimension(state):
    """ Gets the dimension of the Hilbert space the state lives in.
        
        Args:
            state (np.ndarray/qt.Qobj): the state.
    
        Returns:
            (int): the dimension of the Hilbert space. 2^N, where N is the number of qubits in the system.
    
    """
    return state.shape[0]


def qobj_from_array(data):
    """ Instantiates a proper qutip.Qobj KET not a general object from given 
        numpy.ndarray data.
        
        Args:
            data (numpy.ndarray): the state's amplitudes.
    
        Returns:
            (qutip.Qobj): the state.
    
    """
    N = get_N(data)
    dims = [[2] * N, [1] * N]

    return qt.Qobj(data, dims=dims, shape=data.shape, type='ket', isherm=False, copy=False, isunitary=False)


######################## Info file ########################


def read_info_file(fname):
    """ Reads and parses data from info.txt file found in simulation folder 
        containing additional information such as the total number of trials done.
        
        Args:
            fname (pathlib.PosixPath/str): the location of the info file.
    
        Returns:
            (dict): a dictionary containing any relevant information from the file.
    
    """
    info = {}
    
    try:
        with open(fname) as f:
            info = ast.literal_eval(f.read())
    except IOError:
        # If the file does not exist that is fine, it will be created on
            # update
        print('Info file does not exist. Generating an empty one.')

        if type(fname) is str: fname = Path(fname)
        # Make the folders if they don't exist.
        path.parent.mkdir(exist_ok=True, parents=True)
        # Will handle creating and updating the info file.
        update_info_file(path, info)

    return info


def update_info_file(info, fname):
    """ Updates an information file containing relevant metadata regarding a 
        system or other data such as the number of trials done.
        
        Args:
            info (dict): the new information.
            
            fname (pathlib.PosixPath/str): the file path.
    
        Returns:
            (None): none
    
    """
    # Read the old info file and update its contents with this new information
    old_info = read_info_file(fname)
    # Update the old information with new information.
    # This maintains any information that was not altered.
    old_info.update(info)

    # We overwrite since we've already read and accounted for the previous
        # contents of this info file
    with open(fname, 'w') as f:
        f.write(str(info))



#------------- Entry code -------------#

def main():
    print('parser.py')
    

if __name__ == '__main__':
    main()