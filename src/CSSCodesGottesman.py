import numpy as np
import sympy as sp
from qiskit.circuit import QuantumCircuit, QuantumRegister, AncillaRegister,ClassicalRegister
from qiskit.quantum_info import Statevector, Operator, partial_trace
from qiskit_aer import AerSimulator
from IPython.display import display, Latex
import networkx as nx
from networkx.algorithms import bipartite



def standard_generator_matrix_form(pauli_strings, pivot_mode=False):
    # pauli strings are entered as a list of tuples [ ,... , (X string , Z string) ,... ]. "X part" and "Z part" are 
    # binary strings describing the corresponding Pauli string.. X string = 010110 corresponds to X_{134} 
    
    f = lambda x: x % 2
    # if convert:
    #     pauliX_block = [ [int(x) for x in s[0]]  for s in pauli_strings]
    #     pauliZ_block  = [ [int(x) for x in s[1]]  for s in pauli_strings]
    #     pauliZ_block_matrix = sp.Matrix(pauliZ_block).applyfunc(f)
    #     pauliX_block_matrix = sp.Matrix(pauliX_block).applyfunc(f) 

        
    pauliX_block_matrix = pauli_strings[0].applyfunc(f)  
    pauliZ_block_matrix = pauli_strings[1].applyfunc(f)
       
    r = pauliX_block_matrix.rank()

    # X_rref , X_rhs = pauliX_block_matrix.rref_rhs( pauliZ_block_matrix )

    
    X_rref,X_pivots  =  pauliX_block_matrix.rref()
    pivots = [list(X_pivots)]
    
    
    pauliX_block_matrix = X_rref.applyfunc(f)
     
    block = pauliZ_block_matrix[: , r:]       
    block_rref, block_pivots = block.rref()
    pivots.append( list(block_pivots) ) 

    pauliZ_block_matrix[: , r: ] = block_rref.applyfunc(f)    
    # generator_matrix = pauliX_block_matrix.row_join(pauliZ_block_matrix)
    
    if pivot_mode == True:
        return pauliX_block_matrix, pauliZ_block_matrix, pivots
        
    return  pauliX_block_matrix, pauliZ_block_matrix




def locate_ones(L, mode='str'):
    ones = []
    for idx in range(len(L)):
        if mode == 'str':
            if L[idx] == '1':
                ones.append(idx)
        if mode == 'list':
            if L[idx] == 1:
                ones.append(idx)            
            
    return ones 

def initialize_code(pauli_strings,quantum_circuit, return_generators = True):
    # given a generator matrix, convert it to standard form and 
    # return a quantum circuit which prepares the logical zero state for the code
    X_block,Z_block, pivots =  standard_generator_matrix_form(pauli_strings,pivot_mode=True)
    r = X_block.rank()
    
    # physicalqubits = QuantumRegister(n, name = 'physical')
    n = len(quantum_circuit.qubits)
     
    for row_idx in range( r ):
        ones = []
        pivot = pivots[0][row_idx]
        quantum_circuit.h(pivot)
        # ones = locate_ones( [ X_block[i,j] for j in range(r,n) ], mode = 'list')
        for j in range(pivot+1,n):
            if X_block[row_idx,j] == 1:
                ones.append(j)
        for idx in ones:
            quantum_circuit.cx(pivot, idx )
            
    if return_generators:
        return (X_block,Z_block)

    # return qc, (X_block,Z_block)
def initialize_groundstate(pauli_strings):
    # given a generator matrix, convert it to standard form and 
    # return a quantum circuit which prepares the logical zero state for the code
    X_block,Z_block, pivots =  standard_generator_matrix_form(pauli_strings,pivot_mode=True)
    r = X_block.rank()
    
    # physicalqubits = QuantumRegister(n, name = 'physical')
    n = X_block.cols
    quantum_circuit = QuantumCircuit(n) 
    for row_idx in range( r ):
        ones = []
        pivot = pivots[0][row_idx]
        quantum_circuit.h(pivot)
        # ones = locate_ones( [ X_block[i,j] for j in range(r,n) ], mode = 'list')
        for j in range(pivot+1,n):
            if X_block[row_idx,j] == 1:
                ones.append(j)
        for idx in ones:
            quantum_circuit.cx(pivot, idx )
    return quantum_circuit
    



def syndrome_measurement(qc, generator_data, syndromes, meas, n,k, pauli_type):
    # syndromes = qc.ancillas[:]
    # meas = qc.clbits[:]
    data = qc.qubits[: n ]
    X_block = generator_data[1]
    r = X_block.rank()
    X_block = X_block[:r, :]   
    Z_block = generator_data[2][r: , :]
    

    if pauli_type == 'X':
        block = X_block 
        # block_shape = block.rows()
        block_shape = r
    if pauli_type == 'Z':
        block = Z_block
        block_shape = block.rows
        # block_shape = n - k - r 
        
    
    for row_idx in range( block_shape ):
        # # syndrome = AncillaRegister(1) 
        # qc.add_register(syndrome) 
        
        qc.h(syndromes[row_idx])
        ones = []
        # ones = locate_ones( [block.row(row_idx)[idx] for idx in range(block.row(row_idx).shape[1])   ]   )
        for idx in range(block.row(row_idx).shape[1]):
            if block.row(row_idx)[idx] == 1:
                ones.append(idx)
                
        for idx in ones:
            if pauli_type == 'X':
                qc.cx( syndromes[row_idx], data[idx] )
            if pauli_type == 'Z':
                qc.cz( syndromes[row_idx], data[idx] )
                
        qc.h(syndromes[row_idx])
        qc.measure(syndromes[row_idx], meas[row_idx])
        
        
        ##reset syndromes
        qc.h(syndromes[row_idx])
        
        for idx in ones:
            if pauli_type == 'X':
                qc.cx( syndromes[row_idx], data[idx] )
            if pauli_type == 'Z':
                qc.cz( syndromes[row_idx], data[idx] )
                
        qc.h(syndromes[row_idx])
        
            
            
def CSS_logical_operator(qc,generator_data,n,k,pauli_type, index):
    r = generator_data[0].rank()
    
    E_matrix = generator_data[1][r:,n-k:]
    C1 = generator_data[1][:r , r:n-k]
    C2 = generator_data[1][:r, n-k: ]
    A2 = generator_data[0][:r,n-k:]
    
    U = sp.zeros(k,n)
    V = sp.zeros(k,n)
    
    if pauli_type == 'X':
        U2 = E_matrix.transpose()
        V1 = U2 * C1.transpose() + C2.transpose()
        V3 = 0 
        U3 = sp.eye(k)
        
        U[:, n-k:] = U3
        U[:, r  : n-k] = U2
        V[:, :r] = V1
        
        UV = U.row_join(V)
        
        ones = locate_ones( [ UV[index,j] for j in range(2*n) ], mode = 'list')
            
        qc.x(ones)
            

    
        
        
    if pauli_type == 'Z':
        V3 = sp.eye(k)
        V1 = A2.transpose()
        
        V[ :, :r ] = V1 
        V[:, n-k:] = V3
    
        UV = U.row_join(V)
      
        ones = locate_ones( [ UV[index,j] for j in range(2*n) ], mode = 'list')
        qc.z(ones)
                
    
    
def ApplyPauliError(quantum_circuit, qubits, p_error):
    for qubit in qubits:
        error_choice = np.random.choice([0,1,2,3],p=[1 - p_error, p_error/3, p_error/3, p_error/3])
        if error_choice == 1:
            quantum_circuit.x(qubit)
        if error_choice == 2: 
            quantum_circuit.z(qubit)
        if error_choice == 3:
            quantum_circuit.y(qubit)
               


   
def generate_tanner_graph( gen_matrix, label):    
    
    tanner_graph = nx.Graph()
    for idx in range(gen_matrix.rows):
        check_node_label =  label + '-' + str(idx) 
        tanner_graph.add_node( check_node_label , bipartite = 0 )
        for i,j in enumerate( gen_matrix.row(idx) ):
            if j == 1:
                tanner_graph.add_node(i, bipartite = 1)
                tanner_graph.add_edge( check_node_label, i)

    return tanner_graph
     
def compute_boundary(tanner_graph):
    _, data = bipartite.sets(tanner_graph)
    boundary = []
    # tanner_graph.add_node('bdry' , bipartite = 0 )
    for node in data:
        if tanner_graph.degree(node) ==  1:
            boundary.append(node)
            # boundary_label = 'b'+str(node)
            
            # tanner_graph.add_node(boundary_label, bipartite = 0)
            # tanner_graph.add_edge(boundary_label, node, weight=0)
            
    return boundary       
            
    
       
        
    # ## Z checks
    # ZGraph = nx.Graph()
    # for idx in range(gen_matrix[1].rows):
    #     check_node_label = 'r' + str(idx)
    #     ZGraph.add_node( check_node_label , bipartite = 0 )
    #     for i,j in enumerate( gen_matrix[1].row(idx) ):
    #         if j == 1:
    #             ZGraph.add_node(i, bipartite = 1)
    #             ZGraph.add_edge( check_node_label, i)    
    # return XGraph, ZGraph
    
                
        
        
    

    
def CSS_code(gen_matrix, n, k, p_error):
    
    ## include decoder as input 
    
    
    CSS_code, generator_data = initialize_code(gen_matrix,n,k)
    CSS_decode = CSS_code.inverse()
    
    physical_qubits = CSS_code.qubits
    rx = generator_data[1].rank()
    rz = generator_data[2].rank()
    
    CSS_X_syndromes = AncillaRegister(rx)
    CSS_Z_syndromes = AncillaRegister(rz)
    
    CSS_X_meas = ClassicalRegister(rx)
    CSS_Z_meas = ClassicalRegister(rz)    
    CSS_code.add_register(CSS_X_syndromes,CSS_Z_syndromes)
    CSS_code.add_register(CSS_X_meas, CSS_Z_meas)
    
    ApplyPauliError(CSS_code, physical_qubits, p_error)
    
    
    ########## measuring X stabilizers ####################
    
    syndrome_measurement(CSS_code, generator_data, CSS_X_syndromes, CSS_X_meas, n,k, 'X')

    for i in range(len(physical_qubits)):
        ones = [str(x) for x in generator_data[1].col(i)]
        ones_matching = int(''.join(ones)[::-1],2) 
    
    with CSS_code.if_test( (CSS_X_meas, ones_matching) ):
        CSS_code.z(i)
    ####################
    
    
     ####### measuring Z stabilizers  ###################
    syndrome_measurement(CSS_code, generator_data, CSS_X_syndromes, CSS_X_meas, n,k, 'X')

    for i in range(len(physical_qubits)):
        ones = [str(x) for x in generator_data[1].col(i)]
        ones_matching = int(''.join(ones)[::-1],2) 
    
    with CSS_code.if_test( (CSS_X_meas, ones_matching) ):
        CSS_code.z(i)
    ####################
    
    CSS_code.compose(CSS_decode, qubits=physical_qubits, inplace=True)
    finalmeasure = ClassicalRegister(len(physical_qubits))
    CSS_code.add_register(finalmeasure)
    CSS_code.measure(physical_qubits,finalmeasure) 
       

    return CSS_code



    
