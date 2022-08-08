#!/usr/bin/env python3
"""Generate random quantum circuits and random quantum states.

**Authors: Jonathan Delgado, Joseph R. Farah, Valentino Crespi**

Operate on random quantum circuits and random quantum states while tracking various quantities of interest.

"""
import sys
from pathlib import Path

#------------- Math imports -------------#
import numpy as np, scipy
# For usage in cluster_doping
from scipy.stats import norm
import qutip as qt

#------------- RandQC scripts -------------#
from randqc.tools.misc import *
import randqc.tools.gui as gui
import randqc.tools.parser as parser
import randqc.tools.paths as paths
import randqc.imager as imager
import randqc.saver as saver
from randqc.entropy import renyi as renyi_entropy
from randqc.tools.paths import default_folder

# The random number generator meant for use only in this script. Takes over
    # for usage for random and np.random directly.
    # See: https://numpy.org/doc/stable/reference/random/index.html
_RNG = np.random.default_rng()


######################## Functions ########################


def parse_state_string(N, state_string):
    """ Parses the state label and returns a quantum state depending on the 
        string passed in. Example valid state_strings are 'rand' for a random Haar state, or 'prod' for a random product state.
    
        Args:
            N (int): the number of qubits in the system.

            state_string (str): the name of the state itself. Examples include: zero, prod, rand.
    
        Returns:
            (qutip.Qobj): the quantum state
    
    """
    # Make case insensitive
    state_string = state_string.lower()

    # Selects the input state based on the command
    if state_string == 'zero':
        return zero_state(N)
    
    if state_string == 'prod':
        return rand_prod_state(N)
    
    if state_string == 'phase':
        return phase_state(N)

    if state_string == 'rand':
        return rand_haar_state(N)

    if state_string == 'exp_rand':
        return _experimental_rand_haar_state(N)

    if state_string == 'exp_prod':
        return _experimental_rand_prod_state(N)
    
    # No state was found to return
    print('Invalid input state.')
    print('Exiting...')
    sys.exit()


######################## Quantum states ########################

def zero_state(N):
    """ Build N tensor product of |0>
    
        Args:
            N (int): the number of qubits in the system this state will represent.
    
        Returns:
            (qutip.Qobj): the state
    
    """
    return qt.qip.qubits.qubit_states(N, states=[0])

#------------- Random states -------------#

def rand_prod_state(N):
    """ Generates a random Haar product state.
    
        Args:
            N (int): the number of qubits in the system this state will represent.
    
        Returns:
            (qutip.Qobj): the state
    
    """
    return qt.tensor( [qt.rand_ket_haar(2) for _ in range(N)] )


def _experimental_rand_prod_state(N):
    """ Generates a random Haar product state by making use of the 
        experimental random Haar state function.
    
        Args:
            N (int): the number of qubits in the system this state will represent.
    
        Returns:
            (qutip.Qobj): the state
    
    """
    return qt.tensor( [_experimental_rand_haar_state(1) for _ in range(N)] )


def rand_haar_state(N):
    """ Generates a random Haar state, NOT a generic random state.
    
        Args:
            N (int): the number of qubits in the system this state will represent.
    
        Returns:
            (qutip.Qobj): the state    
    """
    # Ensures the dimensions of the random Haar state match the dimensions of
        # other states    
    return qt.rand_ket_haar(2**N, dims=[[2] * N, [1] * N])


def phase_state(N, phases=[]):
    """ Creates a phase state of the form: (cos(𝜃1) |0> + sin(𝜃1) |1>) ⊗ … ⊗ (
        cos(𝜃n) |0> + sin(𝜃n) |1>) with 𝜃 ∈ [0,𝜋].
    
        Args:
            N (int): the number of qubits in the system.

        Kwargs:
            phases (list): a list of the phases to use, leave as default argument to generate a uniformly distributed random phase state.
    
        Returns:
            (qutip.Qobj): the phase state
    
    """
    if len(phases) > 0 and len(phases) < N:
        print('Insufficient phases to generate a phase state. Returning a random phase state.')
        phases = []

    # We need to generate the random phases
    if phases == []:
        phases = np.cos( (np.pi * _RNG.random(N)) )

    return qt.qip.qubits.qubit_states(N, states=phases)


def _experimental_rand_haar_state(N):
    """ Experimental approach to generating a random haar state.
    
        Args:
            N (int): the number of qubits in the system this state will represent.
    
        Returns:
            (qutip.Qobj): the state    
    """
    components = np.array([
        complex( _RNG.normal(), _RNG.normal() )
        for _ in range(2**N)
    ])

    # Normalize the state
    components /= np.sqrt(np.sum(np.abs(components)**2))

    return parser.qobj_from_array(components)


######################## Random gates ########################

def rand_cnot(N, topology='complete'):
    """ Create a random CNOT gate. The CNOT gate takes an input gate (control)
        and depending on the control, changes the value of the target gate. It performs the NOT operation on the target qubit iff the control qubit is |1>, otherwise, it leaves the target qubit untouched.
    
        Args:
            N (int): dimension of your gate, so you can create an unitary matrix of appropriate size.

        Kwargs:
            topology (str): complete: the control and target qubits can be any two qubits. ring: the control and target qubits must beg "adjacent".
    
        Returns:
            (3-tuple): (cnot gate, control, target)
    
    """
    [control, target] = _RNG.choice(np.arange(N), 2, replace=False)
    if topology == 'ring':
        # The topology of the qubits is assumed to be a ring to ensure
            # endpoints have an equal probability of being selected as other
            # qubits. So if the control already picks the last qubit, we move
            # the control to the first qubit.
        target = control + 1 if control + 1 < N else 0
    return (qt.qip.operations.cnot(N, control, target), control, target)


def rand_hadamard(N):
    """ Generates a Hadamard gate that acts on a single qubit. If the input is 
        |0>, it maps  to (|0> + |1>)/sqrt(2). If the input is |1> it maps to (|0> - |1>)/sqrt(2). This is effectively a rotation within the basis |0>, |1>.
    
        Args:
            N (int): dimension of your gate, so you can create an unitary matrix of appropriate size.
    
        Returns:
            (2-tuple): (Hadamard gate, target)
    
    """
    [target] = _RNG.choice(np.arange(N), 1, replace=False)
    return (qt.qip.operations.snot(N, target), target)


def rand_phasegate(N, phase=_RNG.uniform(0, 2*np.pi)):
    """ Generates a phase gate where both the phase can be made randomly or 
        fixed (as in the case of an S or T gate), and its target is random.
    
        Args:
            N (int): number of qubits in the system. Used to scale the phasegate accordingly.

        Kwargs:
            phase (float): the phase for the gate.
    
        Returns:
            (2-tuple): (phasegate, target)
    
    """
    [target] = _RNG.choice(np.arange(N), 1, replace=False)
    return (qt.qip.operations.phasegate(phase, N, target), target)


def rand_S(N):
    """ Generates a phase gate that acts on a single qubit. It changes the 
        phase by pi/2 but does not change the probability of the state being |0> or |1>. This phase change belongs to Clifford gate set. 
    
        Args:
            N (int): dimension of your gate, so you can create an unitary matrix of appropriate size.
    
        Returns:
            (2-tuple): (S, target)
    
    """
    return rand_phasegate(N, phase=np.pi/2)


def rand_T(N):
    """ Generates a phase gate that acts on a single qubit. It changes the 
        phase by pi/4 but does not change the probability of the state being |0> or |1>. This phase change corresponds to Universal gate set. 
    
        Args:
            N (int): dimension of your gate, so you can create an unitary matrix of appropriate size.
    
        Returns:
            (2-tuple): (T, target)
    
    """
    return rand_phasegate(N, phase=np.pi/4)


######################## Circuits ########################


def rand_circuit(N, num_gates, gate_options):
    """ Generates a random circuit represented as a list of gates to be 
        successively applied to some input quantum state.
    
        Args:
            N (int): the number of qubits. Also related to the dimension of the system.

            num_gates (int): the number of gates to add to the circuit.

            gate_options (list): the gates available for applying to the circuit.
    
        Returns:
            (list): the circuit as a list of gates ordered by when they should be applied to the states.
    
    """
    gates = _RNG.choice(gate_options, size=num_gates, replace=True)
    # For each gate, we must take it, expand it to our N-qbit system
        # RandQC then returns (gate, control, target) and we must pull out
        # the gate itself and append it to our circuit
    return [ gate(N)[0] for gate in gates ]


def rand_circuit_from_instructions(N, instructions, topology='complete'):
    """ Runs through parsed instructions to generate a random quantum circuit
        that satisfies the instructions provided.
        
        Args:
            N (int): the number of qubits in the circuit is expected to act on.

            instructions (list/str): the parsed or unparsed instructions.
    
        Kwargs:
            topology (str): complete: the control and target qubits can be any two qubits. ring: the control and target qubits must beg "adjacent".
    
        Returns:
            (list): the circuit blocks satisfying the parsed 
                instructions.
    
        Raises:
            KeyError
    
    """
    # Hadamard: snot(N, target)
    # cnot: cnot(N, control, target)
    # S(\pi/4): phasegate(\pi/2, N, target)
    # T: phasegate(\pi/4, N, target)

    # CNot is the only 2 qubit gate, as a result we need to adjust for the 
        # topology provided
    _rand_cnot = lambda N : rand_cnot(N, topology=topology)

    GATES_OPTIONS = {
        'Cl': [rand_hadamard, _rand_cnot, rand_S],
        'U': [rand_hadamard, _rand_cnot, rand_T],
        'T': [rand_T],
    }

    # Allows for parsing or using pre-parsed instructions
    if type(instructions) is str:
        instructions = parser.parse_instructions(instructions)

    circuits = []

    for i, instruction in enumerate(instructions):
    
        gate_type, num_gates = instruction

        try:
            gates = GATES_OPTIONS[gate_type]
        except KeyError:
            raise KeyError(f'Invalid circuit type instructions, invalid gate provided: {gate_type}.')


        circuits.append(rand_circuit(N, num_gates, gate_options=gates))

    return circuits


def rand_unitary_haar(N):
    """ Generates a random unitary Haar matrix on N qubits.
    
        Args:
            N (int): the number of qubits
    
        Returns:
            (numpy.ndarray): the random unitary Haar matrix.
    
    """
    return qt.rand_unitary_haar(2**N).full()


def rand_unitary_clifford(N, circuit_length=1):
    """ Generates a unitary Clifford matrix, a matrix representation of a 
        Clifford circuit.
    
        Args:
            N (int): the number of qubits input into the circuit. Affects the dimensionality of the matrix.

        Kwargs:
            circuit_length (int): the number of Clifford gates that should be included in the circuit.
    
        Returns:
            (numpy.ndarray): the unitary matrix
    
    """
    # rand_circuit_from_instructions returns a list containing
        # blocks of circuits, since we only wanted Clifford gates we
        # will only have, and will only be interested in the first block
    # Finally, we reverse the circuit, since the gates applied first are the
        # first gates in the circuit, however in matrix multiplication
        # they will be the rightmost unitary operators in the product
        # hence we must multiply them last if we multiply the operators
        # from left to right
    circuit = rand_circuit_from_instructions(
        N=N,
        instructions=parser.parse_instructions(f'Clx{circuit_length}'
    ))[0][::-1]

    # Converts the qutip.Qobj to a proper numpy.ndarray
    if circuit_length == 1:
        return circuit[0].full()

    U = circuit[0]

    for gate in circuit[1:]:
        U = np.matmul(U, gate)

    return U


def eval_circuit(state, circuit, renyi_parameter=None, bipartition=None, state_maps=None):
    """ Evaluates a quantum circuit by taking in an initial quantum state and 
        returning the output of the circuit.
    
        Args:
            state (qutip.Qobj): the initial input state (i.e. |0>).

            circuit (list): the quantum circuit.

        Kwargs:
            renyi_parameter (float/None): which Renyi entropy to calculate, provide None to not calculate any entropy data.
    
            bipartition (int): the bipartition to use for the Renyi entropy if the renyi_parameter is not None. Pass in 0 to return the bipartition average of the entropy.

            state_maps (list/None): list of functions to calculate additional quantities of interest.

        Returns:
            (qutip.Qobj/tuple): the final quantum state resulting from the evaluation of the circuit. Returns additional entropy data and data for any other state maps if provided.
    
    """
    # Whether entropy data should be gathered
    get_entropy = renyi_parameter is not None
    get_data = state_maps != None

    if get_entropy:
        entropies = []
        # We're going to need N for both the bipartition average and the 
            # default bipartition
        N = parser.get_N(state)

        if bipartition == None:
            # Set the default
            bipartition = N // 2
            
    if get_data:
        num_gates = len(circuit)
        num_maps = len(state_maps)
        data = []

    #------------- Circuit loop -------------#

    for gate in circuit:
        # Sequentially apply each gate in the circuit to each state
        state = gate(state)
        
        ### Data gathering ###

        if get_entropy:

            if bipartition == 0:
                # Calculate the bipartition average
                entropy = np.mean([
                    renyi_entropy(state, bipart, renyi_parameter)
                    for bipart in range(1, N)
                ])
            else:
                entropy = renyi_entropy(state, bipartition, renyi_parameter)

            entropies.append(entropy)

        if get_data:
            # This is a column of data
            trial_data = [
                state_map(state)
                for state_map in state_maps
            ]

            data.append(trial_data)

    ### End circuit loop ###
                
    # Convert to numpy matrix and flip to proper row for state_map and col for
        # gate_num
    if get_data:
        data = np.array(data).T

    if get_entropy and get_data:
        return state, entropies, data

    if get_entropy:
        return state, entropies

    if get_data:
        return state, data

    return state


#------------- Instructions generation -------------#

def rearrange_circuit(Cl_gates, T_gates, instructions):
    """ Takes a pre-made collection of gates and rearranges it according to a 
        set of instructions. Useful for seeing how rearranging gates can affect a circuit.
    
        Args:
            Cl_gates (list): the Clifford gates.

            T_gates (list): the T-gates.

            instructions (str): the new circuit's instructions.
    
        Returns:
            (list): the new circuit
    
    """
    instructions = parser.parse_instructions(instructions)
    circuit = []

    Cl_gates_used = T_gates_used = 0

    for (gate_type, num_gates) in instructions:
        if gate_type == 'Cl':
            circuit += [Cl_gates[Cl_gates_used:Cl_gates_used + num_gates]]
            Cl_gates_used += num_gates
        elif gate_type == 'T':
            circuit += [T_gates[T_gates_used:T_gates_used + num_gates]]
            T_gates_used += num_gates
        else:
            print(f'Invalid gate type: {gate_type}')
            print('Returning circuit as-is.')
            break

    return circuit


def insert_T_gates(t_gate_placements, num_gates):
    """ Inserts T gates following t_gate_placements and provides the 
        instructions that satisfies this placement.
    
        Args:
            t_gate_placements (list/numpy.ndarray): the list of indices instructing where to place T gates. The list should be 1-indexed, so to make the first gate a T gate one would provide t_gate_placements = [1, ...].

            num_gates (int): the total number of gates (including T).
    
        Returns:
            (str): the instructions with the inserted T gates.
    
    """
    # Helper function.
    def _parse_consecutive_t_placements(t_gate_placements):
        """ Runs through the T gate placements and checks for consecutive blocks to format it in a more helpful way for the main body. """
        
        # Tracks the number of consecutive T gates
        t_multiplier = 1
        
        # Holds the placement pairs:]
        # (zero-indexed T gate placement, number of T gates)
        placements = []

        for i, t_place in enumerate(t_gate_placements):
                
            # If there is a next element, and it is adjacent to the current T index
            if i < len(t_gate_placements) - 1 and t_gate_placements[i + 1] == t_place + 1:
                # then add to the multiplier since this is a consecutive block
                t_multiplier += 1
                continue

            placements.append( (t_place - t_multiplier, t_multiplier) )
            # Reset the multiplier
            t_multiplier = 1

        return placements


    #------------- Error checking -------------#
    
    # Check for duplicates
    if has_duplicates(t_gate_placements):
        print('Duplicate T gate placements provided. This is not supported.')
        print('Exiting...')
        sys.exit()

    # Sort T gates
    t_gate_placements = np.unique(t_gate_placements)

    # No duplicates beyond this point so length of list is accurate for number
        # of T gates (n_T) requested 
    n_T = len(t_gate_placements)

    if n_T == 0:
        print('No T gate placements provided, returning complete Clifford circuit.')
        return f'Clx{num_gates}'
    if n_T == num_gates:
        print('The number of T gates is equal to the total number of gates. Returning a purely T gate circuit.')
        return f'Tx{num_gates}'
    if n_T > num_gates:
        print('T gate placements exceeds the total number of gates requested.')
        print('Exiting...')
        sys.exit()

    if t_gate_placements[0] < 1:
        print(f'Invalid T gate placement: {t_gate_placements[0]}. T gates are 1-indexed.')
        sys.exit()

    if t_gate_placements[-1] > num_gates:
        # We are trying to add a T gate after the circuit has ended
        print('T gates cannot be inserted in placements longer than the total circuit length.')
        print('Exiting...')
        sys.exit()

    #------------- Main body -------------#

    instructions = ''
    gate_count = 0

    for t_place, t_multiplier in _parse_consecutive_t_placements(t_gate_placements):

        if t_place > gate_count:
            # Clifford gates should be placed here
            gap = t_place - gate_count
            instructions += f'Clx{gap};'
            gate_count += gap

        # Now T gates should be placed here
        instructions += f'Tx{t_multiplier};'
        gate_count += t_multiplier        

    # There may be a tail
    if gate_count < num_gates:
        instructions += f'Clx{num_gates - gate_count}'
        return instructions

    # There is a trailing semi colon
    return instructions[:-1]


def rand_doping(n_T, num_gates):
    """ Generates circuit instructions for a circuit with a random placement 
        of T gates and the remaining being Clifford.
    
        Args:
            n_T (int): the number of T gates.

            num_gates (int): the total number of gates, with (num_gates - n_T) of them being Clifford.
    
        Returns:
            (str): the circuit instructions
    
    """
    if num_gates < n_T:
        print('There cannot be fewer total gates than T gates.')
        num_gates = n_T

    if num_gates == n_T:
        print('Returning a purely T gate circuit.')
        return f'Tx{n_T}'

    t_gate_placements = _RNG.choice(np.arange(1, num_gates), size=n_T, replace=False)

    return insert_T_gates(t_gate_placements, num_gates)


#------------- Cluster doping -------------#


def get_default_cluster_parameters(num_gates):
    """ Provides the default mean and width for the normal distribution
        for use in other functions.
        
        Args:
            num_gates (int): the total number of gates in the circuit.
    
        Returns:
            (2-tuple): the default cluster center and width respectively.
    
    """
    # If no center is provided then just assume it's in the middle
    # If no width is provided then we adjust it based on the number of 
        # gates to ensure we can get a unique set of n_T T gate placements
    # cluster_mean, cluster_width
    return num_gates // 2, 0.25 * num_gates 


def check_cluster_parameters(num_gates, cluster_center, cluster_width):
    """ Handles checking the cluster parameters and sets them accordingly.
    
        Args:
            num_gates (int): the total gates to be used in the circuit.
            
            cluster_center (float): the placement of the center of the distribution.
            
            cluster_width (float): the width of the distribution
    
        Returns:
            (2-tuple): the checked and adjusted cluster center and width respectively.
    
    """
    # Sets defaults if needed
    if cluster_center is None or cluster_width is None:
        default_center, default_width = get_default_cluster_parameters(num_gates)
        if cluster_center is None: cluster_center = default_center
        if cluster_width is None: cluster_width = default_width

    return cluster_center, cluster_width


def cluster_doping(n_T, num_gates, cluster_center=None, cluster_width=None):
    """ Generates instructions to randomly dope circuits with T gates such 
        that they are clustered together following a normal distribution.
        
        Args:
            n_T (int): the number of T gates to dope the circuit with.

            num_gates (int): the total number of gates in the circuit.
    
        Kwargs:
            cluster_center (int): the gate index the distribution should be center around.

            cluster_width (float): the width of the distribution.
    
        Returns:
            (str): the cluster doped circuit instructions.
    
    """
    # Checks values and sets defaults if needed
    cluster_center, cluster_width = check_cluster_parameters(num_gates, cluster_center, cluster_width)

    # Since the normal pdf is continuous, we need to discretize
        # the cdf to avoid integrating it to get the probabilities of ints
    possible_placements = np.arange(1, num_gates + 1)
    # lower and upper since cdf is monotonically increasing
    xL, xU = possible_placements - 0.5, possible_placements + 0.5
    norm_probs = norm.cdf(xU, loc=cluster_center, scale=cluster_width) - \
                norm.cdf(xL, loc=cluster_center, scale=cluster_width)
    # We only have a portion of the discrete distribution so renormalize it
    norm_probs /= norm_probs.sum()

    # Selects the unique (replace=False) integers corresponding to T gate
        # placements
    t_gate_placements = _RNG.choice(possible_placements, size=n_T, replace=False, p=norm_probs)
    return insert_T_gates(t_gate_placements, num_gates)


def equal_gap_doping(n_T, num_gates, dope_start=1, dope_end=None, gap=None, lead_T=False):
    """ Instructions generator for doped instructions with equal gaps between 
        T gates.
    
        Args:
            n_T (int): the number of T gates.

            num_gates (int): the total number of gates (including T gates).

        Kwargs:
            dope_start (int): the start of where the T gates should be spread within.

            dope_end (int): the end of where the T gates should be spread within.

            gap (int): the forced gap between T gates. Ignores dope_end.

            lead_T (bool): whether the T gate should be before the Clifford gate.
    
        Returns:
            (str): the generated instructions.
    
    """
    if dope_end == None: dope_end = num_gates

    # Create a Clifford gap if none was provided
    if gap == None:
        dope_len = dope_end - dope_start + 1
        if dope_end == num_gates:
            # We need a tail
            gap = (dope_len - n_T) // (n_T + 1)
            clif_tail = num_gates - n_T * (gap + 1) - (dope_start - 1)
        else:
            # A custom tail was provided
            gap = (dope_len - n_T) // n_T
            clif_tail = num_gates - n_T * (gap + 1) - (dope_start - 1)

    else: 
        # A fixed width was provided, we just need a tail
        clif_tail = num_gates - n_T * (gap + 1) - (dope_start - 1)

    instructions = ''

    if dope_start > 1:
        instructions = f'Clx{dope_start - 1};'

    if lead_T:
        instructions += f'{n_T}*(Tx1,Clx{gap});Clx{clif_tail}'
        return instructions

    instructions += f'{n_T}*(Clx{gap},Tx1);Clx{clif_tail}'
    return instructions


######################## Main body ########################

def qc(
        N=8, circuit=None, input_state=None,
        renyi_parameter=None, bipartition=None, topology='complete',
        state_maps=None,
        save_state=False, save_data=False,
        circuit_label='Gx100', input_state_label='zero',
        identifier='', save_path=default_folder(),
        GUI=False):
    """ The main body of the RandQC package and randqc.qc module. Takes in the 
        necessary parameters for a quantum circuit and allows the circuit to operate on an input state one gate at a time with the option to gather data such as the entropy of the state versus the gate applied and save these results for future analysis. The entropy data is kept separate from general state maps since it is far more commonly used.
    
        Kwargs:
            N (int): the number of qubits in the system.

            circuit (list): the circuit as a list of blocks containing gates to be successively applied to an input state.

            input_state (qutip.Qobj): the input state.

            renyi_parameter (float/None): the type of Renyi entropy to gather. Leave as None to not gather any entropy data.

            bipartition (int/None): the way to bipartition the state, affecting the entropy gathered. Defaults to N//2 when None is provided.

            topology (str): controls what gates can be applied to which qubit, defaults to the complete topology where gates can be applied to any qubit. Another option is "ring" where gates can only be applied to "adjacent" qubits. For this function it will only affect saving since the circuit provided is expected to conform to the topology provided.

            state_maps (None/dict): dictionary of functions to calculate additional quantities of interest. The key is a string that will be used to save the data file, and the value is a function which takes in a state and outputs a number.

            save_state (bool): the option to save the output state of the entire circuit to a file.

            save_data (bool): the option to save additional data such as the entropy data gathered to a file.

            circuit_label (str): a label for the circuit. Used in file saving. Ideally just the circuit's instructions.

            input_state_label (str): a string containing the type of input state to use, some options are: zero, prod, or rand, for a zero state, product state, or a random Haar state respectively. See randqc.parse_state_string for all options.

            save_path (pathlib.PosixPath/str): the base location for data to be saved to.

            identifier (str): optional additional string that may be used to identify and separate this files and folders from others.

            GUI (bool): whether to show a Tkinter progress bar window
    
        Returns:
            (list/2-tuple): the list of states after each gate was applied. Also returns the entropy data for each state if entropy data was gathered. Note that the entropy data will be prefixed with the serial for this trial. This will be the same serial the state's filename has been saved with.

    """
    #------------- Error checking of arguments -------------#  

    # Check N
    if N < 2:
        print(f'Invalid number of qubits: {N}. Must be at least 2 in order to use cnot gate.')
        print('Exiting...')
        sys.exit()

    # Check bipartition
    if bipartition is None:
        # Split the qubits in half, use integer division to floor it in the 
            # case of an odd number of qubits.
        bipartition = N//2
    elif bipartition < 0 or bipartition >= N:
        print(f'Invalid bipartition: {bipartition}. Must be an integer greater than or equal to 0 and fewer than the number of qubits in the system.')
        print('Exiting...')
        sys.exit()

    # Adjust any capitalization on input state
    input_state_label = input_state_label.lower()

    #------------- Initialization -------------#

    # Serialize this experiment.
    SERIAL = int(parser.get_serial())

    ### Flags ###
    SAVE_FLAG = save_state or save_data
    ENTROPY_FLAG = renyi_parameter is not None
    # We want to collect additional data along with entropy data.
    DATA_FLAG = state_maps is not None

    # List containing the output state of each block of the circuit in the
        # case where the number of samples is only one. If we request more
        # than one sample then only return the output state of all of the
        # circuits
    states = [input_state.copy()]

    # Variable renaming for loop logic
    state = input_state

    #------------- Check flags -------------#

    # Since we're not saving nothing, we want to have a directory ready
        # to hold everything
    if SAVE_FLAG:
        # Updates the save_path as handled by the loader object
        saver.prepare_saving(save_path)

    ### ProgressBar ###
    if GUI:
        max_progress_value = len(circuit_label) + 1
        bar_title = 'QC simulation progress'
        bar = gui.ProgressBar(max_progress_value, width=350, title=bar_title)

    if ENTROPY_FLAG:
        # Vector that holds the current trial's entropy data
        if bipartition == 0:
            initial_entropy = np.mean([
                renyi_entropy(state, bipart, renyi_parameter)
                for bipart in range(1, N)
            ])
        else:
            initial_entropy = renyi_entropy(state, bipartition, renyi_parameter)


        # Entropy data will have first column be the serial of the output 
            # state to identify which state had this entropy.
        # We prepend the serial as a negative number to identify it from the '
            # entropy data.
        entropy_data = np.array([-SERIAL, initial_entropy])

    if DATA_FLAG:
        state_data = {}

        for map_name, state_map in state_maps.items():
            # Go through all of the additional data maps and prepend the serial
                # along with the initial value for the state.
            state_data[map_name] = np.array([-SERIAL, state_map(state)])

    # Now that we've generated the image of the initial state, we are
        # ready to update our progress
    if GUI:
        bar.update()

    #------------- Circuit evaluation -------------#

    for i, block in enumerate(circuit):

        # Handle the return of eval_circuit based on what was requested.
        if ENTROPY_FLAG and DATA_FLAG:
            state, block_entropy_data, block_map_data = eval_circuit(state, block, renyi_parameter=renyi_parameter, bipartition=bipartition, state_maps=state_maps.values())

        elif ENTROPY_FLAG:
            state, block_entropy_data = eval_circuit(state, block, renyi_parameter=renyi_parameter, bipartition=bipartition)

        elif DATA_FLAG:
            state, block_map_data = eval_circuit(state, block, renyi_parameter=None, state_maps=state_maps.values())

        else:
            state = eval_circuit(state, block, renyi_parameter=None)

        # Store the data in our matrices
        if ENTROPY_FLAG:
            entropy_data = np.append(entropy_data, block_entropy_data)

        if DATA_FLAG:
            for map_index, map_name in enumerate(state_maps.keys()):
                state_data[map_name] = np.append(state_data[map_name], block_map_data[map_index])


        states.append(state)

        if GUI:
            bar.next()

    #------------- Post circuit evaluation -------------#

    if GUI and SAVE_FLAG:
        bar.set_title('Saving data...')

    if ENTROPY_FLAG:
        # Convert to the proper row vector format for a single trial
        entropy_data = entropy_data.reshape(1, -1)

    if DATA_FLAG:
        for map_name in state_data.keys():
            state_data[map_name] = state_data[map_name].reshape(1, -1)


    # Saves the output state using the loader's functionality to ensure
        # it is easily loadable again in the future
    if save_state:
        state_fname = paths.get_save_state_file_path(save_path, N, input_state=input_state_label, instructions=circuit_label, topology=topology, identifier=identifier)

        state_fname.parent.mkdir(exist_ok=True, parents=True)

        saver.save_state(state, fname=state_fname, serial=SERIAL)


    if save_data:
        if ENTROPY_FLAG:
            # Save entropy data to a file
            entropy_fname = paths.get_entropy_file_path(save_path, N, input_state_label, circuit_label, topology=topology, bipartition=bipartition, renyi_parameter=renyi_parameter, identifier=identifier)

            # Make the folder
            entropy_fname.parent.mkdir(exist_ok=True, parents=True)

            saver.save_entropies(entropy_data, fname=entropy_fname)

        if DATA_FLAG:
            # Save other data.
            for map_name, map_data in state_data.items():
                data_fname = paths.get_data_file_path(save_path, N, input_state_label, circuit_label, topology=topology, identifier=identifier, map_name=map_name)

                # Make the folder
                data_fname.parent.mkdir(exist_ok=True, parents=True)

                saver.save_matrix(map_data, fname=data_fname)


    #------------- Post simulation -------------#

    if GUI:
        bar.finish()

    if ENTROPY_FLAG:
        return states, entropy_data

    return states


def randomqc(
        N=8,
        instructions='Clx100',
        input_state_label='zero', input_state=None,
        renyi_parameter=None, bipartition=None, topology='complete',
        state_maps=None,
        save_state=False, save_data=False,
        identifier='', save_path=default_folder(),
        GUI=False):
    """ Takes in the necessary parameters for a random quantum circuit and 
        allows the circuit to operate on an input state one gate at a time with the option to gather data such as the entropy of the state versus the gate applied and save these results for future analysis.
    
        Kwargs:
            N (int): the number of qubits in the system.

            instructions (str): a string formatted to encode the structure of the random quantum circuit of interest. See randqc.tools.parser.parse_instructions for examples.

            input_state_label (str): the label for the input_state. If no input_state is provided this label will be parsed to generate an input_state.

            input_state (qutip.Qobj/None): the input state, if None is provided then input_state_label will be parsed to generate an input_state.

            renyi_parameter (float/None): the type of Renyi entropy to gather. Leave as None to not gather any entropy data.

            bipartition (int/None): the way to bipartition the state, affecting the entropy gathered. Defaults to N//2 when None is provided.

            topology (str): controls what gates can be applied to which qubit, defaults to the complete topology where gates can be applied to any qubit. Another option is "ring" where gates can only be applied to "adjacent" qubits.

            state_maps (None/dict): dictionary of functions to calculate additional quantities of interest. The key is a string that will be used to save the data file, and the value is a function which takes in a state and outputs a number.

            save_state (bool): the option to save the output state of the entire circuit to a file.

            save_data (bool): the option to save additional data such as the entropy data gathered to a file.

            identifier (str): optional additional string that may be used to identify and separate this files and folders from others.            

            save_path (pathlib.PosixPath/str): the base location for data to be saved to.

            GUI (bool): whether to show a Tkinter progress bar window

        Returns:
            (list/2-tuple): the list of states after each gate was applied. Also returns the entropy data for each state if entropy data was gathered.

    """
    #------------- Error checking -------------#
    # Check N
    if N < 2:
        print(f'Invalid number of qubits: {N}. Must be at least 2 in order to use cnot gate.')
        print('Exiting...')
        sys.exit()

    #------------- Main body -------------#

    circuit = rand_circuit_from_instructions(N, instructions, topology=topology)
    if input_state == None:
        input_state = parse_state_string(N, input_state_label)

    return qc(
        N=N, input_state=input_state, circuit=circuit,
        renyi_parameter=renyi_parameter, bipartition=bipartition, topology=topology,
        state_maps=state_maps,
        save_state=save_state, save_data=save_data,
        input_state_label=input_state_label, circuit_label=instructions, 
        identifier=identifier, save_path=save_path, GUI=GUI
    )


#------------- Entry code -------------#

def main():
    print('qc.py')


    import randqc.entropy as entropy

    state_maps = {
        'renyi2-b4': lambda state : entropy.renyi(state, 4, 2),
        'renyi4-b4': lambda state : entropy.renyi(state, 4, 4),
        'renyi5-b4': lambda state : entropy.renyi(state, 4, 5),
    }

    states, data, = randomqc(10, input_state_label='zero', instructions='Clx100', state_maps=state_maps, save_data=True)



if __name__ == '__main__':
    main()
