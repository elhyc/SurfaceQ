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
