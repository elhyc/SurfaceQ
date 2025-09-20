from qiskit.circuit import QuantumCircuit, QuantumRegister, AncillaRegister,ClassicalRegister
from qiskit.quantum_info import Statevector, Operator, partial_trace
from qiskit.circuit import Measure
import itertools
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt
from qiskit.primitives import SamplerResult
from qiskit.providers.basic_provider import BasicProvider
from qiskit import transpile
import numpy as np
import sympy
from CSSCodesGottesman import *
from networkx.algorithms import bipartite
import math 

def erasure_channel( quantum_circuit, qubits, p_error, print_option=False):
    erasure_flags = []

    for qubit in qubits:
        error_choice = np.random.choice([0,1], p =[1-p_error,p_error])
        if error_choice == 1:
            erasure_flags.append(qubit)
            pauli_prob =  np.random.choice([0,1,2,3],p=[1/4, 1/4,1/4, 1/4])
            if pauli_prob == 1:
                if print_option:
                    print( 'X on ' + str(qubit)  )
                quantum_circuit.x(qubit)
            if pauli_prob == 2:
                if print_option:
                    print( 'Y on ' + str(qubit)  )
                quantum_circuit.y(qubit)
            if pauli_prob == 3:
                if print_option:
                    print( 'Z on ' + str(qubit))
                quantum_circuit.z(qubit)
    return erasure_flags 



def depolarizing_channel(quantum_circuit, qubits, p_error, print_option=False):
    for qubit in qubits:
        error_choice = np.random.choice([0,1,2,3],p=[1 - p_error, p_error/3, p_error/3, p_error/3])
        if error_choice == 1:
            if print_option:
                print( 'X on ' + str(qubit)  )
            quantum_circuit.x(qubit)
        if error_choice == 2:
            if print_option:
                print( 'Z on ' + str(qubit)  ) 
            quantum_circuit.z(qubit)
        if error_choice == 3:
            if print_option:
                print( 'Y on ' + str(qubit)  )
            quantum_circuit.y(qubit)
            

class RotatedSurfaceCode:
    def __init__(self, rows,cols):
        self.rows = rows
        self.cols = cols
        self.lattice_grid = nx.grid_2d_graph(rows,cols)
        self.plaquettes = {start_node: face for (start_node,face) in [ self.cycle_at(node) for node in self.lattice_grid ] if len(face.nodes) == 4 }
        self.num_of_nodes = len(self.lattice_grid.nodes )


################
        self.top =  [ [x[0], x[1] ] for x in nx.utils.pairwise( [ (0,x) for x in range(-1,cols + 1) ]  )  ]
        self.top_physical  = self.top[1:-1:2]
        self.top_virtual = self.top[::2] 
        
        self.boundary_plaquettes = { ( cell[0][0]-1, cell[0][1] ): cell   for cell in self.top_physical }
        # self.boundary_plaquettes_virtual =  { ( cell[0][0]-1, cell[0][1] ): cell   for cell in self.top_virtual }

        self.side_plus =  [ [x[0], x[1] ] for x in nx.utils.pairwise( [ (x, cols - 1 ) for x in range(-1,rows + 1)  ]   )  ]
        self.side_plus_physical = self.side_plus[1:-1:2]
        self.side_plus_virtual = self.side_plus[::2]
        
        self.boundary_plaquettes.update( {  (cell[0][0],cell[0][1]) :cell for cell in self.side_plus_physical   } )
        # self.boundary_plaquettes_virtual =  { ( cell[0][0]-1, cell[0][1] ): cell   for cell in self.top_virtual }

        
        self.bottom = [ [x[0], x[1] ] for x in nx.utils.pairwise( [ ( rows - 1, cols - 1 - x) for x in range(-1,cols+1) ]  )  ]
        self.bottom_physical = self.bottom[1:-1:2]
        self.bottom_virtual = self.bottom[::2] 

        self.boundary_plaquettes.update(   {  (cell[1][0],cell[1][1]) :cell[::-1] for cell in self.bottom_physical  }   )
        
        self.side_minus = [ [x[0], x[1] ] for x in nx.utils.pairwise([ (x,0) for x in range(-1,cols +1 ) ][::-1] )  ]
        self.side_minus_physical = self.side_minus[1:-1:2 ]
        self.side_minus_virtual = self.side_minus[::2] 

        self.boundary_plaquettes.update(   {  (cell[0][0] - 1,cell[0][1]-1) : cell[::-1] for cell in self.side_minus_physical   }   )
####################

        self.X_plaquettes = [ (key,list(self.plaquettes[key].nodes)) for key in self.plaquettes if (key[1] - key[0]) % 2 == 0 ] + [ (self.boundary_plaquettes[key][0],self.boundary_plaquettes[key]) for key in self.boundary_plaquettes if (key[1] - key[0]) % 2 == 0 ]  
        self.Z_plaquettes = [ (key,list(self.plaquettes[key].nodes)) for key in self.plaquettes if (key[1] - key[0]) % 2 == 1 ] + [ (self.boundary_plaquettes[key][0],self.boundary_plaquettes[key]) for key in self.boundary_plaquettes if (key[1] - key[0]) % 2 == 1 ]


    #########################
        self.LatticeCircuit = self.initialize_LatticeCircuit()
        self.DataQubits = self.LatticeCircuit.qubits        
        
        self.generator_matrix = self.get_generator_matrix()
        self.X_graph = generate_tanner_graph( self.generator_matrix[0], 'X')

        self.X_checks = [ 'X-' + str(idx) for idx in range(len(self.X_plaquettes) ) ]
        self.Z_checks = [ 'Z-' + str(idx) for idx in range(len(self.Z_plaquettes) ) ]

        self.X_boundary_data = [  node for node in self.X_graph.nodes if self.X_graph.degree[node] == 1  ]
        self.X_virtual_checks = []
        
        
        self.initializer = initialize_groundstate(self.generator_matrix)                
        self.LatticeCircuit.compose( self.initializer, inplace=True )    

    ###### add in virtual check nodes
        for idx, edge in enumerate(self.top_virtual + self.bottom_virtual):

            for i in range(2):
                node = self.flat_idx( edge[i] )
                if node in self.X_graph:
                    self.X_graph.add_edge( 'vX-' + str(idx), node)
       
            self.X_virtual_checks.append( 'vX-' + str(idx)  )
        
        
        self.Z_graph = generate_tanner_graph( self.generator_matrix[1], 'Z')

        self.Z_boundary_data = [  node for node in self.Z_graph.nodes if self.Z_graph.degree[node]  == 1  ]
        self.Z_virtual_checks = []
        
        for idx,edge in enumerate(self.side_plus_virtual + self.side_minus_virtual):
            for i in range(2):
                node = self.flat_idx( edge[i] )
                if node in self.Z_graph:
                    self.Z_graph.add_edge( 'vZ-' + str(idx), node)

            self.Z_virtual_checks.append('vZ-' + str(idx)   )
        
        self.tanner_graphs = {'X': self.X_graph, 'Z': self.Z_graph} 
        

        self.ancillas = self.initialize_ancillias()

        self.Z_parity_ancilla = AncillaRegister(1,name='parity')
        self.Z_parity_cbit = ClassicalRegister(1, name = 'parity-cl')

        self.LatticeCircuit.add_register(self.Z_parity_ancilla)
        self.LatticeCircuit.add_register(self.Z_parity_cbit)

######################

    def draw(self, mode, coords='default'):
        if mode == 'X':
            if coords == 'flat':
                G = self.tanner_graphs['X']
            else:
                G = nx.relabel_nodes(self.tanner_graphs['X'], { node: self.coord(node) if type(node) == int else node for node in self.tanner_graphs['X'] } ) 
            node_list = self.X_checks + self.X_virtual_checks
            data_nodes = [node for node in G if type(node) != 'str' ]
            pos = nx.spring_layout(G, iterations=2000)
            nx.draw_networkx_nodes(G, pos,nodelist = data_nodes,  node_color="tab:blue" , node_size = 90 )
            nx.draw(G, pos, with_labels=True, nodelist = node_list, node_color="tab:green", font_size=6, node_size=90)
        elif mode == 'Z':
            if coords == 'flat':
                G = self.tanner_graphs['Z']
            else:
                G = nx.relabel_nodes(self.tanner_graphs['Z'], { node: self.coord(node) if type(node) == int else node for node in self.tanner_graphs['Z'] } )
            node_list = self.Z_checks + self.Z_virtual_checks
            data_nodes = [node for node in G if type(node) != 'str' ]
            
            pos = nx.spring_layout(G, iterations=2000)
            nx.draw_networkx_nodes(G, pos,nodelist=data_nodes, node_color="tab:blue", node_size = 90 )
            nx.draw(G, pos, with_labels=True, nodelist = node_list, node_color="tab:green", font_size=6, node_size = 90)
            
        elif mode == 'primal':
            if coords == 'flat':
                G = nx.relabel_nodes(self.lattice_grid, { node : self.flat_idx(node)  for node in self.lattice_grid} )
            else:
                G = self.lattice_grid 
                
            nx.draw(G, nx.spring_layout(G, iterations=2000), with_labels=True, font_size=6, node_size=90)
            
        
    def initialize_ancillias(self):
        ancilla_nodes = {'X': [], 'Z': []}
        
        for check_node in self.X_checks:
        
            syndrome = AncillaRegister(1,name=check_node)
            self.LatticeCircuit.add_register(syndrome)
            meas = ClassicalRegister(1, name=check_node + ' meas' )
            self.LatticeCircuit.add_register(meas)

            ancilla_nodes['X'].append((syndrome,meas))

        for check_node in self.Z_checks:
            syndrome = AncillaRegister(1,name=check_node)
            self.LatticeCircuit.add_register(syndrome)

            meas = ClassicalRegister(1, name=check_node + ' meas')
            self.LatticeCircuit.add_register(meas)
    
            ancilla_nodes['Z'].append((syndrome,meas))

        return ancilla_nodes          
        
    
    def syndrome_measurement(self,label):

        ## label is 'X' or 'Z' 
        options = {'X' : self.LatticeCircuit.cx, 'Z': self.LatticeCircuit.cz}
        tanner_graph = self.tanner_graphs[label]
        
        for ancilla_pair in self.ancillas[label]:
            syndrome = ancilla_pair[0]
            meas = ancilla_pair[1]

            self.LatticeCircuit.h(syndrome)
            
            
            for idx in tanner_graph.neighbors(syndrome.name):
                options[label](syndrome, self.LatticeCircuit.qubits[idx])
                
            self.LatticeCircuit.h(syndrome)
            
            self.LatticeCircuit.measure(syndrome,meas)





    def reset(self):
        LatticeCircuit = self.initialize_LatticeCircuit()
        LatticeCircuit.compose( self.initializer, inplace=True )
        self.LatticeCircuit = LatticeCircuit
        self.ancillas = self.initialize_ancillias()

        self.Z_parity_ancilla = AncillaRegister(1,name='parity')
        self.Z_parity_cbit = ClassicalRegister(1, name = 'parity-cl')

        self.LatticeCircuit.add_register(self.Z_parity_ancilla)
        self.LatticeCircuit.add_register(self.Z_parity_cbit)
        


    def initialize_LatticeCircuit(self):
        LatticeCircuit = QuantumCircuit()
        for node in self.lattice_grid.nodes:
            qubit = QuantumRegister(1, name=str(node))  
            LatticeCircuit.add_register(qubit) 

        return LatticeCircuit 

    def coord(self, idx):
        return ( math.floor( idx/ (self.cols) ) , idx - math.floor( idx/ self.cols )*self.cols   )
    
    
    def flat_idx(self, node):

        if node in self.DataQubits:
            return self.DataQubits.index(node)
        
        return node[1] + (self.cols)*node[0]

    def cycle_at(self, start):
        return (start, self.lattice_grid.subgraph([ start, (start[0]+1, start[1]), (start[0], start[1]+1), (start[0]+1,start[1]+1)] ) )
    
    def get_generator_matrix(self):
        
        ZBlock = sympy.Matrix()

        for plaquette in self.Z_plaquettes: 
            Zrow = sympy.zeros(1,self.num_of_nodes)
            for node in plaquette[1]: 
                Zrow[0, self.flat_idx(node) ] = 1
            ZBlock = ZBlock.col_join(Zrow)
            

        XBlock = sympy.Matrix()    
        for plaquette in self.X_plaquettes:
            Xrow = sympy.zeros(1,self.num_of_nodes)
            for node in plaquette[1]:
                Xrow[0, self.flat_idx(node )] = 1
            XBlock = XBlock.col_join(Xrow)  

        return (XBlock,ZBlock)         


    def marked_tanner_graph(self, label, marked_nodes):
        
        tanner_graph = self.tanner_graphs[label]
        marked_graph = nx.Graph()
        marked_graph.add_nodes_from(marked_nodes)
        edge_list = list(itertools.combinations(marked_nodes, 2))
        pair_graph = marked_graph.copy()
        
        for edge in edge_list:
            path = nx.shortest_path(tanner_graph, edge[0], edge[1])
            marked_graph.add_edges_from( list(itertools.pairwise(path) ) )
            pair_graph.add_edge(edge[0],edge[1], weight = len(path) )
            
        boundary_nodes = []
        for node in marked_nodes:
            nearest_bdry = self.nearest_boundary(node, label)
            path = nx.shortest_path(tanner_graph, node, nearest_bdry)
            marked_graph.add_edges_from( list(itertools.pairwise(path)))
                            
            boundary_node = 'b-' + node 
            boundary_nodes.append(boundary_node)
            marked_graph.add_edge( nearest_bdry, boundary_node, weight = 0 )
            pair_graph.add_edge( node, boundary_node, weight = len(path) )

            boundary_edges_internal = list(itertools.combinations(boundary_nodes, 2))
            for edge in boundary_edges_internal:
                pair_graph.add_edge(edge[0], edge[1], weight = 0)
                   
        return marked_graph, pair_graph
    
    
    
    def nearest_boundary(self, node, label):
        # return node closest to given node, on the tanner graph 
        # associated to the label (which is 'X' or 'Z')
        if label == 'Z':
            node_int = int( node.replace('Z-','') )
            node_coords = self.Z_plaquettes[node_int][0]

            if node_coords[1] >= math.floor(self.cols/2):
                return self.flat_idx(  (node_coords[0], self.cols - 1) )
            else:
                return self.flat_idx( (node_coords[0],0) )


        if label == 'X':
            node_int = int( node.replace('X-','') )
            node_coords = self.X_plaquettes[node_int][0]
            
            if node_coords[0] >= math.floor(self.rows/2): 
                return self.flat_idx( (self.rows - 1, node_coords[1]) )
            else:
                return self.flat_idx( (0, node_coords[1]) ) 
    
    def error_channel_cycle(self, error_type, p_error, decoder_option, report_option=False):

        if error_type=='depolarizing':
            depolarizing_channel(self.LatticeCircuit, self.DataQubits, p_error, report_option)


        if error_type=='erasure':
            erasure = erasure_channel(self.LatticeCircuit, self.DataQubits, p_error, report_option)       

    ##### syndrome measurements ##########
    ####################################


    ######### phase flips ###########
    
        self.syndrome_measurement('X')

    # simulator = AerSimulator()
    # transpiled_circuit = transpile(LatticeCircuit, simulator)
    # job = simulator.run(transpiled_circuit, shots=1, memory=True)
    # transpiled_circuit = transpile(LatticeCircuit)
     
        job = AerSimulator().run(self.LatticeCircuit, shots=1, memory=True)   
    # job = AerSimulator().run(transpiled_circuit, shots=1, memory=True)
    
        result = job.result()
    # memory = result.get_memory(transpiled_circuit)
        memory = result.get_memory(self.LatticeCircuit)
        memory_result = memory_result = memory[0].replace(' ','')[len(self.X_plaquettes)+1: ][::-1]
    

        positions = []
        for i in range(len(memory_result)):
            if memory_result[i] == '1':
                positions.append('X-' + str(i) )

        if error_type == 'depolarizing':
            if decoder_option == 'union-find':
                self.UnionFind_decoder('X', positions)
            else:            
                self.MWPM_decoder('X', positions )
        if error_type == 'erasure':
            self.erasure_decoder(erasure, 'X', positions )    
               
    
    ######### bit flips ###########
    
        self.syndrome_measurement('Z')
    
    # transpiled_circuit = transpile(LatticeCircuit)
    # job = AerSimulator().run(transpiled_circuit, shots=1, memory=True)
    
        job = AerSimulator().run(self.LatticeCircuit, shots=1, memory=True)
        result = job.result()
    
    
    # memory = result.get_memory(transpiled_circuit)
        memory = result.get_memory(self.LatticeCircuit)
        memory_result = memory[0].replace(' ','')[1:len(self.Z_plaquettes)+1][::-1]


        positions = []
        for i in range(len(memory_result)):
            if memory_result[i] == '1':
                positions.append('Z-' + str(i) )
            
        if error_type == 'depolarizing':
            if decoder_option == 'union-find':
                self.UnionFind_decoder('Z', positions)
            else:           
                self.MWPM_decoder('Z', positions )
        if error_type == 'erasure':
            self.erasure_decoder(erasure, 'Z', positions )   

    def single_round(self,p_error, error_type, decoder_option=None, report_option=False):
        self.error_channel_cycle(error_type,p_error, decoder_option, report_option)
        self.measure_data()
        

    def measure_data(self):
    # ##### final logical Z-parity measurements ####
    # ##############################################
        ZReadAncilla = self.Z_parity_ancilla
        ZReadout = self.Z_parity_cbit

        # self.LatticeCircuit.add_register(ZReadAncilla)
        # self.LatticeCircuit.add_register(ZReadout)

        self.LatticeCircuit.h(ZReadAncilla[0])
        boundary_edge = nx.shortest_path(self.lattice_grid, (0,0), ( self.rows - 1, 0 ) )

        for node in boundary_edge:
            qubit = self.LatticeCircuit.qubits[ self.flat_idx(node) ]
            self.LatticeCircuit.cz(ZReadAncilla[0], qubit)
        
        self.LatticeCircuit.h(ZReadAncilla[0])  
    
        self.LatticeCircuit.measure(ZReadAncilla[0],ZReadout[0])
    

    def MWPM_decoder(self, label, marked_nodes):

        if label == 'X':
            marked_X_graph, X_subgraph  = self.marked_tanner_graph('X', marked_nodes)
        
        ## address syndrome 
            X_matchings = nx.min_weight_matching(X_subgraph,  weight='weight')
            
            for match in X_matchings:
                if not ( (  match[0][0] == 'b' ) and (match[1][0] == 'b' ) ):
                    path = [ node for node in nx.shortest_path(marked_X_graph, match[0], match[1] ) if type(node) == int  ]
                    self.LatticeCircuit.z(path)
        
        if label == 'Z':
            marked_plaquette_graph, plaquette_pairing_graph = self.marked_tanner_graph('Z', marked_nodes )

        ## address syndrome 
            plaquette_matchings = nx.min_weight_matching( plaquette_pairing_graph,  weight='weight')

            for match in plaquette_matchings:
                if not ( (  match[0][0] == 'b' ) and (match[1][0] == 'b'  ) ):
                    path = [ node for node in nx.shortest_path( marked_plaquette_graph , match[0], match[1] ) if type(node) == int  ]
                    self.LatticeCircuit.x(path)
        



    def erasure_decoder(self, erasure, label, marked_nodes, with_nodes=False ):
        decoder_nodes = []
        (checks,virtual_checks) = (self.X_checks, self.X_virtual_checks) if label == 'X' else (self.Z_checks, self.Z_virtual_checks)         
        if with_nodes == False:
            erasure_graph = self.tanner_graphs[label].subgraph([ self.flat_idx(node) for node in erasure ] + checks + virtual_checks )
        else:
            erasure_graph = self.tanner_graphs[label].subgraph( erasure )
            
        erasure_comps = [erasure_graph.subgraph(c).copy() for c in nx.connected_components(erasure_graph) if len(c) > 1]
        
        component_seed = { next( ( x for x in comp.nodes if x in virtual_checks) , next( ( x for x in comp.nodes if x in checks) , None)  )  : comp for comp in erasure_comps  }  


        forest = {seed :  nx.bfs_tree(component_seed[seed], seed )   for seed in component_seed  }
            

        for tree in forest.items():

            current_tree = tree[1].copy()
            while len(current_tree.edges) > 0:
                pendants = [node for node in current_tree.nodes if current_tree.out_degree(node) == 0  and current_tree.in_degree(node) > 0]
                for pendant in pendants:
                    if type(pendant) == int:
                        current_tree.remove_node(pendant)
                    else:    
                        edge_node = set(current_tree.pred[pendant]).pop()
                        if pendant in marked_nodes:
                            decoder_nodes.append( edge_node )
                            adj_check = set(current_tree.pred[edge_node]).pop()
                    
                            if adj_check in marked_nodes:
                                marked_nodes.remove(adj_check)
                            elif adj_check not in virtual_checks:
                                marked_nodes.append(adj_check)
                        current_tree.remove_nodes_from([pendant, edge_node])

        if label == 'Z' and decoder_nodes:
            self.LatticeCircuit.x(decoder_nodes)

        if label == 'X' and decoder_nodes:
            self.LatticeCircuit.z(decoder_nodes)

        



    def UnionFind_decoder(self, label, marked_nodes):
                

        tanner_graph = self.tanner_graphs[label]

        root_nodes = self.X_checks + self.X_virtual_checks  if label == 'X' else self.Z_checks + self.Z_virtual_checks
        virtual_checks = self.X_virtual_checks if label == 'X' else self.Z_virtual_checks
        Clusters = { node : Cluster(node , tanner_graph, parity = int(node in marked_nodes), active = ( node not in virtual_checks ) ) for node in marked_nodes }  

        node_table = {key: { 'nodes': [], 'visits': 0 }  for key in tanner_graph.nodes }

        for root in Clusters:
            node_table[root]['nodes'].append(root)
            node_table[root]['visits'] = 1

            
        Cluster_forest = nx.DiGraph()
        Cluster_forest.add_nodes_from( marked_nodes )


        active_odd_cluters = [ cluster for cluster in Clusters if Clusters[cluster].parity == 1 and Clusters[cluster].active  ]
        phase = 0 

        

        while active_odd_cluters:

            phase = phase%2
            for root in active_odd_cluters:

                Clusters[root].grow(phase)
                for v in Clusters[root].boundary:
                    if root not in node_table[v]['nodes']:
                        node_table[v]['nodes'].append(root)
                        node_table[v]['visits'] += 1 
                if phase == 1:
                    if Clusters[root].boundary & set(virtual_checks):
                        Clusters[root].active = False  

                Cluster_forest.update(Clusters[root ].graph)
            fusion_table = { key: node_table[key]['nodes'] for key in node_table if len(node_table[key]['nodes']) >= 2 }
            
            while fusion_table:
            ### fusion phase
                for key in list(fusion_table.keys()):

                    try:
                        nx.find_cycle(Cluster_forest)
                        print('cycle found!')
                    except:
                        pass
                    
                    item_1 = node_table[key]['nodes'].pop(0)
                    item_2 = node_table[key]['nodes'].pop(0)
                    root_1 = find_root(Cluster_forest, item_1 )
                    root_2 = find_root(Cluster_forest, item_2 )  
                    
                    ordered = sorted( [ ( Clusters[root_1].size, root_1),  ( Clusters[root_2].size, root_2) ] )[::-1]
                    if root_1 != root_2:
                        Clusters[ordered[0][1] ].union( Clusters[ordered[1][1]], key )
                        Cluster_forest.update( Clusters[ordered[0][1]].graph )

                    node_table[key]['nodes'].append( ordered[0][1] ) 

                fusion_table = { key: node_table[key]['nodes'] for key in node_table if len(node_table[key]['nodes']) >= 2 }



            phase += 1 
            active_odd_cluters = [ cluster for cluster in Clusters if Clusters[cluster].parity == 1 and Clusters[cluster].active  ]

        total_nodes = list(Cluster_forest.nodes) + list(itertools.chain(*[ Clusters[root].edge_nodes for root in Clusters  ]) )

        self.erasure_decoder(total_nodes, label, marked_nodes , with_nodes=True) 
                       
    def fuse(self, QLattice, side='right'):
        pass
    

    
class Cluster:
    def __init__(self, root, parent_graph, parity = 0, active= None ):
        self.edge_nodes = []  
        self.root = root 
        self.support = { n for n in parent_graph.adj[root]  if n != root}
        
        self.boundary =  { root }

        self.active =  active

        self.graph = nx.DiGraph() 
        self.graph.add_node(root)
        self.size = 1   
        self.parity = parity 
        self.children = {}
        self.parent_graph = parent_graph 
        

    def grow(self, phase):
        new_support = set() 
        for node in self.support:
            new_support.update( { n for n in self.parent_graph.neighbors(node) if n not in self.graph.nodes  })

        if phase == 0:
            new_boundary = self.support 


        if phase == 1:
            new_boundary = set()
            for e in self.boundary:
                if type(e) == int: 
                    new_boundary_nodes = {n for n in self.parent_graph.neighbors(e) if n not in self.graph.nodes}

                    new_boundary.update(  new_boundary_nodes  )
                    self.graph.add_edge( self.root, e )
                    self.graph.add_edges_from({(self.root,n) for n in new_boundary_nodes} )
                    self.graph.add_edges_from({  (n,e) for n in new_boundary_nodes  })
                    self.size += len(new_boundary_nodes )

                

        self.support = new_support 
        self.boundary = new_boundary

    def union(self, cluster, fusion_node ):


        self.edge_nodes.append(fusion_node)
        ## update boundary
        combined_boundary = self.boundary | cluster.boundary
        self.boundary =  combined_boundary - { n for n in self.parent_graph.neighbors( fusion_node  ) if n in self.graph.nodes or n in self.edge_nodes}  
        new_support = set()
        for node in self.boundary:
            new_support.update({  n for n in self.parent_graph.neighbors(node) if n not in self.graph.nodes and n not in self.edge_nodes }    )
            
        self.support = new_support

        ## update graph, size, parity 
        self.graph.update( cluster.graph )
        self.graph.add_edge( self.root, cluster.root )
        self.children[cluster.root] = cluster 
        self.size += cluster.size 
        self.parity = (self.parity + cluster.parity)%2

        if not cluster.active:
            self.active = False  

        cluster.active = False       


def find_root(tree, node):
    current = node
    path = []    
    try:
        cycle = nx.find_cycle(tree)
        print('cycle found: ' + str(cycle) )
        print(nx.is_tree(tree) )
        print(node )
        print( tree.in_degree(node) ) 
    except:
        pass
    
    while tree.in_degree(current) > 0:
        if current not in tree:
            path.append(current)
        current = set(tree.pred[current]).pop()    
    
    if current != node:
        tree.add_edge(current,node)    
    
    return current 


