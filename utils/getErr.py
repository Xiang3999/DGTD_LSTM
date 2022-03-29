import numpy as np
from utils.utils import compute_err
from scipy.io import savemat, loadmat


class Error():
    def __init__(self, _nt=263):
        self.mortime_error_Ez = np.zeros(_nt)
        self.mortime_error_Hy = np.zeros(_nt)
        self.portime_error_Ez = np.zeros(_nt)
        self.portime_error_Hy = np.zeros(_nt)


error = Error()


def get_err(nt, n_dof):
    """
    Input: the number of time instances nt,
           the degree of freedom ndof.
    Output: the relative error between the DGTD and POD-DL-ROM solution
    """

    time_parameter_pod = loadmat('../data/timeparameterPOD.mat')
    dgtd_time = loadmat('../data/DGTDtime')
    mor_time_error = np.zeros(1, nt)
    pro_time_error = np.zeros(1, nt)
    zero_dgtd_time = np.zeros(n_dof, 2)  # to compute the relative error
    for i in range(nt):
        mor_time = np.hstack((mor_time.Hye[:, i], mor_time.Eze[:, i]))
        dgtd_time = np.hstack((dgtd_time.Hye[:, i], dgtd_time.Eze[:, i]))  # snapshot
        pro_mor_time = np.hstack(
            (time_parameter_pod.Basis.Hy * (np.transpose(time_parameter_pod.Basis.Hy) * dgtd_time.Hye[:, i]),
             time_parameter_pod.Basis.Ez * (np.transpose(time_parameter_pod.Basis.Ez) * dgtd_time.Eze[:, i])))
        mor_err_hy, mor_err_ez = compute_err(mor_time, dgtd_time)
        pro_err_hy, pro_err_ez = compute_err(pro_mor_time, dgtd_time)
        repro_err_hy, repro_err_ez = compute_err(zero_dgtd_time, dgtd_time)
        error.mortime_errorEz[i] = mor_err_ez / repro_err_ez
        error.mortime_errorHy[i] = mor_err_hy / repro_err_hy
        error.protime_errorEz[i] = pro_err_ez / repro_err_ez
        error.protime_errorHy[i] = pro_err_hy / repro_err_hy

    return error
