import matplotlib.pyplot as plt
from QFT_params import *
from QFT_utils import *

# importing Qiskit
from qiskit import Aer, execute

# import basic plot tools
from qiskit.visualization import plot_histogram


if __name__ == "__main__":

    qc = QuantumCircuit(NUMBER_OF_QUBITS)

    # initialize
    qc.append(QFT_initialize(NUMBER_OF_QUBITS, "initialize\nstates"),range(NUMBER_OF_QUBITS))


    # N qubits QFT
    for i in range(NUMBER_OF_QUBITS):
        for j in range(i, NUMBER_OF_QUBITS):
            if i == j:
                multiple_segment_operation(qc, i, Hadamard_params[coupler_type], dw=width_error)
            else:
                qc.cp(np.pi / (2 ** (j - i)), i, j)

        if i!=NUMBER_OF_QUBITS-1:
            qc.barrier()

    qc.measure_all()

    qasm_simulator = Aer.get_backend('qasm_simulator')
    shots = 100000
    results = execute(qc, backend=qasm_simulator, shots=shots).result()
    answer = results.get_counts()
    print("output results:",sorted(answer.items(), key=lambda x:x[1], reverse=True))
    plot_histogram(answer)
    qc.draw(output='mpl', filename='circuit_visualizaion\circuit_drawing.png', style=font_sizes, scale = 0.8)

    plt.show()
