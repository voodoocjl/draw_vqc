import pennylane as qml
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from Arguments import Arguments
from pennylane import numpy as np
import warnings
import matplotlib.cbook

warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

args = Arguments()
symbols = ["H", "H", "H"]
coordinates = np.array([0.028, 0.054, 0.0, 0.986, 1.610, 0.0, 1.855, 0.002, 0.0])

# Building the molecular hamiltonian for the trihydrogen cation
hamiltonian, qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates, charge=1)


def translator(net):
    assert type(net) == type([])
    updated_design = {}

    r = net[0]
    q = net[1:8]
    c = net[8:15]
    p = net[15:]

    # num of layer repetitions
    layer_repe = [2, 3]
    updated_design['layer_repe'] = layer_repe[r]

    # categories of single-qubit parametric gates
    for i in range(args.n_qubits):
        if q[i] == 0:
            category = 'Rx'
        elif q[i] == 1:
            category = 'Ry'
        else:
            category = 'Rz'
        updated_design['rot' + str(i)] = category

    # categories and positions of entangled gates
    for j in range(args.n_qubits):
        if c[j] == 0:
            category = 'IsingXX'
        else:
            category = 'IsingZZ'
        updated_design['enta' + str(j)] = (category, [j, p[j]])

    updated_design['total_gates'] = len(q) + len(c)
    return updated_design


dev = qml.device("lightning.qubit", wires=args.n_qubits)
@qml.qnode(dev, interface="torch", diff_method="adjoint")
def quantum_net(q_input_features_flat, q_weights_flat, design):
    current_design = design
    q_input_features = q_input_features_flat.reshape(args.n_qubits, 3)
    q_weights = q_weights_flat.reshape(current_design['layer_repe'], args.n_qubits, 2)
    for layer in range(current_design['layer_repe']):
        # data reuploading
        for i in range(args.n_qubits):
            qml.Rot(*q_input_features[i], wires=i)
        # single-qubit parametric gates and entangled gates
        for j in range(args.n_qubits):
            if current_design['rot' + str(j)] == 'Rx':
                qml.RX(q_weights[layer][j][0], wires=j)
            elif current_design['rot' + str(j)] == 'Ry':
                qml.RY(q_weights[layer][j][0], wires=j)
            else:
                qml.RZ(q_weights[layer][j][0], wires=j)
            if current_design['enta' + str(j)][0] == 'IsingXX':
                qml.IsingXX(q_weights[layer][j][1], wires=current_design['enta' + str(j)][1])
            else:
                qml.IsingZZ(q_weights[layer][j][1], wires=current_design['enta' + str(j)][1])

    # return [qml.expval(qml.PauliZ(i)) for i in range(args.n_qubits)]
    return qml.expval(hamiltonian)

net = [0, 0, 2, 2, 1, 0, 2, 0, 0, 1, 1, 0, 1, 1, 0, 6, 0, 1, 4, 3, 2, 2]
design = translator(net)
input = nn.Parameter(torch.rand(args.n_qubits * 3))
weight = nn.Parameter(torch.rand(design['layer_repe'] * args.n_qubits * 2))
print(quantum_net(input,weight,design))
qml.draw_mpl(quantum_net, decimals=1, style="black_white", fontsize="x-small")(input, weight, design)
plt.show()