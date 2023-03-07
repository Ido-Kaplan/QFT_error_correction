import matplotlib.pyplot as plt
from QFT_params import *
from QFT_utils import *

# importing Qiskit
from qiskit import Aer, QuantumCircuit, execute, assemble

# import basic plot tools
from qiskit.visualization import plot_histogram, plot_bloch_multivector


sim = Aer.get_backend('qasm_simulator')

def measure_only_qbits(dic,start1,end1,start2,end2):
    ret_dic = {}
    for key,value in dic.items():
        new_key = key[start1:end1]+key[start2:end2]
        ret_dic[new_key] = ret_dic.get(new_key,0) + value
    return ret_dic




if __name__ == "__main__":

    qc = QuantumCircuit(2)

    # initialize to state QFT_inverse|00>
    qc.h(1)
    qc.cp(-np.pi/2,0,1)
    qc.h(0)

    qc.barrier()

    # perform hadamard on first qubit
    perform_h_gate(qc, 0, width_error, use_uniform_coupler)

    # perform R2 on second qubit
    qc.cp(np.pi / 2, 0, 1)

    # perform hadamard on second qubit
    perform_h_gate(qc, 1, width_error, use_uniform_coupler)


    qc.measure_all()


    qasm_simulator = Aer.get_backend('qasm_simulator')
    shots = 10000
    results = execute(qc, backend=qasm_simulator, shots=shots).result()
    answer = results.get_counts()
    answer_only_qbits = measure_only_qbits(answer,start1=0,end1=2,start2=2,end2=3)
    print(answer_only_qbits)
    plot_histogram(answer)
    # qc.draw(output ='mpl', filename ='circuit_visualizaion\circuit_drawing.png')

    plt.show()
