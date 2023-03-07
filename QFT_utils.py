import numpy as np


eta_c = np.array(0.0528)
Uc_mat = np.sqrt(1-eta_c) * np.eye(2) - 1j * np.sqrt(eta_c) * np.array([[0,1],[1,0]])

# this function emulates the coupling affecting the photons before/after propagating through the directional coupler
def Uc(qc, qubit):
    qc.rx(0.4637, qubit)
    return


# polynomial function which derives detuning parameters from waveguide widths
def derive_detuning_from_width(parameters, index, error_w1=0.0, error_w2=0.0):
    return 3.94502808 * (parameters[index + 1] + error_w1) - 18.0203544 * (
                (parameters[index + 1] + error_w1) ** 2) + 27.94843595 * ((parameters[index + 1] + error_w1) ** 3) \
           - 15.42295066 * ((parameters[index + 1] + error_w1) ** 4) - 3.94502818 * (parameters[index] + error_w2) + \
           18.02035521 * ((parameters[index] + error_w2) ** 2) - 27.94843797 * (
                       (parameters[index] + error_w2) ** 3) + 15.42295233 * ((parameters[index] + error_w2) ** 4)


# polynomial function which derives coupling parameters from waveguide widths
def derive_coupling_from_width(parameters, index, error_w1=0.0, error_w2=0.0):
    return 0.38044405 - 1.48138422 * (parameters[index] + parameters[index + 1] + error_w1 + error_w2) + 2.51783632 * (
                (parameters[index] + parameters[index + 1] + error_w1 + error_w2) ** 2) \
           - 1.9993113 * ((parameters[index] + parameters[index + 1] + error_w1 + error_w2) ** 3) + 0.60771393 * (
                       (parameters[index] + parameters[index + 1] + error_w1 + error_w2) ** 4)


# this function emulates a single segment operation on input qubit 
def single_segment_operation(qc, qubit, w0, w1, t, cbit=0, dw=0.0, conditional_operation=False):
    delta = derive_detuning_from_width([w0, w1], 0, error_w1=dw, error_w2=dw)
    omega = derive_coupling_from_width([w0, w1], 0, error_w1=dw, error_w2=dw)

    vx = omega * 2*t
    vy = 0
    vz = delta * t *2

    if conditional_operation:
        qc.rv(vx, vy, vz, qubit).c_if(cbit, 1)
    else:
        qc.rv(vx, vy, vz, qubit)


# this function emulates multiple segments operation on input qubit
def multiple_segment_operation(qc, qubit, params, dw=0.0, cbit=0, conditional_operation=False):
    Uc(qc, qubit)
    single_segment_operation(qc, qubit, params[0], params[1], params[2], dw=dw, cbit=cbit,
                             conditional_operation=conditional_operation)
    for i in range(3, len(params), 3):
        single_segment_operation(qc, qubit, params[i], params[i + 1], params[i + 2], dw=dw, cbit=cbit,
                                 conditional_operation=conditional_operation)
    Uc(qc, qubit)

