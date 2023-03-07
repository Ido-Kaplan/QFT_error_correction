import numpy as np
import torch

from w1w2_error_optimization import *
from matplotlib import pyplot as plt

from scipy import linalg
import scipy.optimize as optimization


plt.rcParams.update({'font.size': 24})
global_counter = [0]

def naive_params_getter(rho=rho):
    n=1
    for i in range(len(X_matrices)):
        mat = X_matrices[i]
        if bool((mat==rho).all()):
            n=i


    if n==0:
        if g==1.0:
            naive_param=torch.tensor([0.45000726442145256, 0.3920128146837242, 20.460941933519397])
        if g==1.2:
            naive_param = torch.tensor([0.4599586076322415, 0.42574387215662707, 29.275918231440382])
            naive_param = torch.tensor([0.460, 0.426, 29.276])


    if full_etch:
        if n==0:
            naive_param = torch.tensor([0.5958758758758759, 0.5658458458458459, 15.55009028355428])
        if n==1:
            naive_param = torch.tensor([0.6, 0.6, 14.365])


    elif n==1:
        # if g==1.0:
        #     omega = photon_interp.get_coupling(0.15, 0.4, 0.4, g)
        #     naive_param = torch.tensor([0.4,0.4,0.5*pi/omega])

        if g==1.2:
            # naive_param = torch.tensor([0.45,0.45,0.5*pi/omega])
            naive_param = torch.tensor([0.45, 0.45, 39.72])#39.990322290988225])
            # naive_param = torch.tensor([0.495, 0.495, 43.5])#39.990322290988225])
            # naive_param = torch.tensor([0.4, 0.4, 34.7])#39.990322290988225])
    elif n==2:
        if g==1.0:
            naive_param = torch.tensor([0.4,0.4,34.35])


        if g==1.2:
            # naive_param = torch.tensor([0.40, 0.40, 59.486378490387715])
            naive_param = torch.tensor([0.40, 0.40, 10.0436])
    elif n==3:
        if g==1.2:
            naive_param = torch.tensor([0.4499933751340871, 0.44999294014311475, 17.639281114131705])#[0.43421720161407235, 0.43421720161407235, 5.21916957019675])
            naive_param = torch.tensor([0.450, 0.450, 17.639])#[0.43421720161407235, 0.43421720161407235, 5.21916957019675])
        elif g==1.0:
            naive_param = torch.tensor([0.38019802372081823, 0.38019868448198896, 8.65975922715397])
    return naive_param



def graph_sketch_2d_error(param,g=g,num_of_points_used=9,loss_func = lambda x,rho: loss_fn_Fidelity(x,rho),rho=rho,deterministic = False,width_error=width_error,etch_error=etch_error):
    print(param)
    param = torch.tensor(param)
    param_dict   = {}

    errors_w=np.linspace(-width_error*3,width_error*3,num_of_points_used)#*0.4
    errors_e=np.linspace(-etch_error*3,etch_error*3,num_of_points_used)+0.15



    for width_error in errors_w:
        for etch_error in errors_e:
            if width_error == 0 and etch_error == 0:
                print(return_mat_generic(param,error_w=0,error_etching=etch_error,deterministic=False))
            if len(param)==3:
                param_fidelity = (1-loss_func(return_mat_generic(param,error_w=width_error,error_etching=etch_error,deterministic=deterministic,g=g),rho=rho))
            else:
                param_fidelity = (1-loss_func(return_mat_generic(param,error_w=width_error,error_etching=etch_error,deterministic=deterministic,g=g),rho=Uc_dag@rho@Uc_dag))
            param_dict[(width_error,etch_error)] = param_fidelity


    Graph_disp_3d(errors_w,errors_e,param_dict,"param graph",xlabel="width error",ylabel="etch error")




def graph_sketch(param,rho=rho,num_of_points_used=64,to_multiplot=False,deterministic=True,loss_func = lambda x,rho: loss_fn_Fidelity(x,rho),y_label = "Fidelity", etch=False,uncorrelated_errors = False,w2_uncorrelated=False,analytic_vals = True,debug=False):
    # print("params checked:",param)
    param = torch.tensor(param)
    param_dict   = {}
    naive_dict   = {}
    analytic_dict = {}


    errors=np.linspace(-width_error*3,width_error*3,num_of_points_used)#*0.4
    errors_e=np.linspace(-etch_error*3,etch_error*3,num_of_points_used)+0.15

    if etch:
        errors=errors_e

    n = 1
    for i in range(len(X_matrices)):
        mat = X_matrices[i]
        if bool((mat == rho).all()):
            n = i

    if n==1:
        analytic_param = torch.tensor([0.4857,0.4345,47.1117/2,0.4057,0.4896,40.5109/2,0.4857,0.4345,47.1117/2])
    elif n==0:
        analytic_param = torch.tensor([0.486,0.379,29.6481/2,0.31,0.5,53.75/2,0.486,0.379,29.6481/2])
    else:
        analytic_param = param
    naive_param = naive_params_getter(rho)
    # naive_param[-1]*=2
    # print("naive params used:",list(naive_param))
    for curr_error in errors:
        w2_uncorrelated_factor = 1.0
        if w2_uncorrelated:
            w2_uncorrelated_factor = np.random.normal(1,0.2)
        # if curr_error == 0:
            # print(Uc@return_mat_generic(param,error_w1=curr_error,error_w2=curr_error*w2_uncorrelated_factor,error_etching=0.15,debug=debug,uncorrelated_errors=uncorrelated_errors)@Uc)
            # print(Uc@return_mat_generic(naive_param,error_w1=curr_error,error_w2=curr_error*w2_uncorrelated_factor,error_etching=0.15,deterministic=True,debug=True,uncorrelated_errors=uncorrelated_errors)@Uc)

        if etch:
            param_fidelity = (1-loss_func(return_mat_generic(param,error_w1=0,error_w2=0,error_etching=curr_error,uncorrelated_errors=uncorrelated_errors),rho=Uc_dag@rho@Uc_dag))#(1-loss_func(return_mat_generic(param,curr_error,curr_error,deterministic=deterministic,debug_opt = True),rho=rho))
            naive_fidelity = (1-loss_func(return_mat_generic(naive_param,error_w1=0,error_w2=0,error_etching=curr_error,uncorrelated_errors=uncorrelated_errors),rho=Uc_dag@rho@Uc_dag))#(1-loss_func(Uc_tensor@return_mat_generic(naive_param,curr_error,curr_error,deterministic=deterministic)@Uc_dag_tensor,rho=rho))
        else:
            analytic_fidelity = (1-loss_func(return_mat_generic(analytic_param,error_w1=curr_error,error_w2=curr_error*w2_uncorrelated_factor,error_etching=0.15,uncorrelated_errors=uncorrelated_errors),rho=Uc_dag@rho@Uc_dag))#(1-loss_func(return_mat_generic(param,curr_error,curr_error,deterministic=deterministic,debug_opt = True),rho=rho))
            param_fidelity = (1-loss_func(return_mat_generic(param,error_w1=curr_error,error_w2=curr_error*w2_uncorrelated_factor,error_etching=0.15,uncorrelated_errors=uncorrelated_errors),rho=Uc_dag@rho@Uc_dag))#(1-loss_func(return_mat_generic(param,curr_error,curr_error,deterministic=deterministic,debug_opt = True),rho=rho))
            naive_fidelity = (1-loss_func(return_mat_generic(naive_param,error_w1=curr_error,error_w2=curr_error*w2_uncorrelated_factor,error_etching=0.15,uncorrelated_errors=uncorrelated_errors),rho=Uc_dag@rho@Uc_dag))#(1-loss_func(Uc_tensor@return_mat_generic(naive_param,curr_error,curr_error,deterministic=deterministic)@Uc_dag_tensor,rho=rho))
        param_dict[curr_error] = param_fidelity
        naive_dict[curr_error] = naive_fidelity
        analytic_dict[curr_error] = analytic_fidelity

    param_values = np.array(list(param_dict.values()))
    analytic_values = np.array(list(analytic_dict.values()))
    naive_values = np.array(list(naive_dict.values()))
    norm_vec = np.array(get_norm_dist_vector(0, width_error, num_of_points_used))
    print("Average Param Fidelity - normal distribution:",np.sum(param_values*norm_vec))
    print("Average Uniform Fidelity - normal distribution:",np.sum(naive_values*norm_vec))
    print("Average Param Fidelity - uniform distribution:",np.mean(param_values))
    print("Average Uniform Fidelity - uniform distribution:",np.mean(naive_values))
    if not etch:
        x = np.array(list(param_dict.keys()))#*100/0.4
    else:
        x = np.array(list(param_dict.keys()))
    x = x * 1000


    if to_multiplot:
        return x,[naive_values,param_values,analytic_values],"δw[nm]","Fidelity",["uniform",
                                                                                       "perturbative ",
                                                                                       "non perturbative"],["r","royalblue","k--"],False, False



    if n==1:
        plt.title("Fidelity of uniform VS segmented coupler for iX gate")

    if n==0:
        plt.title("Fidelity of uniform VS segmented coupler for Hadamard gate")

    plt.plot(x,param_values, 'b'   ,label='segmented coupler - optimized parameters')#non-perturbative solution')
    if analytic_vals:
        plt.plot(x,analytic_values, 'g'   ,label='segmented coupler - perturbative solution')
    plt.plot(x,naive_values, 'k--' ,label='uniform coupler')
    if etch:
        plt.xlabel("error etch [um]")
    else:
        plt.xlabel("δw[nm]")#"error width [um]")#"% Error")

    plt.title("")
    plt.ylabel(y_label)
    plt.legend()
    plt.show()







def correlation_graph_sketch(param,num_of_points_used=21,loss_func = lambda x,rho: loss_fn_Fidelity(x,rho),rho=rho,y_label = "Fidelity",analytic_vals = False,average_width_error=width_error,print_stuff = True):
    if print_stuff:
        print("params checked:",param)
    param = torch.tensor(param)
    param_dict   = {}
    naive_dict   = {}
    analytic_dict = {}


    uncorrelation_range=np.linspace(0.0,uncorrelation_factor,num_of_points_used)

    analytic_param = torch.tensor([0.4857,0.4345,47.1117/2,0.4057,0.4896,40.5109/2,0.4857,0.4345,47.1117/2])
    # analytic_param = torch.tensor([0.371,0.420,25.17,0.427,0.361,26.725,0.391,0.450,23.755])
    naive_param = naive_params_getter()
    # naive_param[-1]*=2

    if print_stuff:
        print("naive params used:",list(naive_param))
    for uncorrelration_percentage in uncorrelation_range:
        error_w1 = average_width_error*(1-uncorrelration_percentage/2)
        error_w2 = average_width_error*(1+uncorrelration_percentage/2)
        if print_stuff:
            print(error_w2,error_w2)
        if uncorrelration_percentage == 1.0 and print_stuff:
            print(Uc@return_mat_generic(param,error_w1=error_w1,error_w2=error_w2,error_etching=0.15,deterministic=True)@Uc)
        analytic_fidelity = (1-loss_func(return_mat_generic(analytic_param,error_w1=error_w1,error_w2=error_w2,error_etching=0.15,deterministic=True),rho=Uc_dag@rho@Uc_dag))
        param_fidelity = (1-loss_func(return_mat_generic(param,error_w1=error_w1,error_w2=error_w2,error_etching=0.15,deterministic=True),rho=Uc_dag@rho@Uc_dag))
        naive_fidelity = (1-loss_func(return_mat_generic(naive_param,error_w1=error_w1,error_w2=error_w2,error_etching=0.15,deterministic=True),rho=Uc_dag@rho@Uc_dag))
        param_dict[uncorrelration_percentage] = param_fidelity
        naive_dict[uncorrelration_percentage] = naive_fidelity
        analytic_dict[uncorrelration_percentage] = analytic_fidelity

    param_values = np.array(list(param_dict.values()))
    analytic_values = np.array(list(analytic_dict.values()))
    naive_values = np.array(list(naive_dict.values()))
    if print_stuff:
        print("Average Param Fidelity - uniform distribution:",np.mean(param_values))
        print("Average Uniform Fidelity - uniform distribution:",np.mean(naive_values))
    x = np.array(list(param_dict.keys())) # when both waveguides are at 5% diff from center than the total diff is 10%
    plt.plot(x,param_values, 'b'   ,label='segmented coupler - optimized partly-correlated parameters')
    if analytic_vals:
        plt.plot(x,analytic_values, 'r'   ,label='segmented coupler - optimized fully-correlated parameters')
    plt.plot(x,naive_values, 'k--' ,label='uniform coupler')
    plt.xlabel("uncorrelation percentage %")
    plt.ylabel(y_label)
    plt.legend()
    plt.show()



# multiple pulses generation
def covariance_error_graph(param,num_of_points_used=31,num_of_sigmas=3,loss_func = lambda x,rho: loss_fn_Fidelity(x,rho),rho=rho,deterministic = True, partially_uncorrelated = True):
    print(param)
    param=torch.tensor(param)
    param_dict   = {}
    naive_dict    = {}
    param_list_5_precent_error = []
    naive_list_5_precent_error = []


    average_errors = np.linspace(-width_error*num_of_sigmas,width_error*num_of_sigmas,num_of_points_used)
    diff_errors = np.linspace(-uncorrelation_factor,uncorrelation_factor,num_of_points_used)
    # errors_w0=np.linspace(-width_error*num_of_sigmas,width_error*num_of_sigmas,num_of_points_used)
    # errors_w1=np.linspace(-width_error*num_of_sigmas,width_error*num_of_sigmas,num_of_points_used)
    # if partially_uncorrelated:
    #     errors_w1 = np.linspace(1-uncorrelation_factor, 1+uncorrelation_factor, num_of_points_used)

    naive_params = naive_params_getter()

    for average_error in average_errors:
        for diff_error in diff_errors:
            error_w0 = average_error*(1-diff_error/2)
            error_w1 = average_error * (1 + diff_error/2)

            if partially_uncorrelated:
                param_fidelity = (1-loss_func(return_mat_generic(param,error_w0,error_w1,deterministic = deterministic),rho=Uc_dag@rho@Uc_dag))
                naive_fidelity = (1-loss_func(return_mat_generic(naive_params,error_w0,error_w1,deterministic = deterministic),rho=Uc_dag@rho@Uc_dag))
            else:
                param_fidelity = (1-loss_func(return_mat_generic(param,error_w0,error_w1,deterministic = deterministic),rho=Uc_dag@rho@Uc_dag))
                naive_fidelity = (1-loss_func(return_mat_generic(naive_params,error_w0,error_w1,deterministic = deterministic),rho=Uc_dag@rho@Uc_dag))


            naive_dict[(average_error,diff_error)] = naive_fidelity
            param_dict[(average_error,diff_error)] = param_fidelity

            if diff_error< 10**-5 + uncorrelation_factor/3 and diff_error> -10**-5 + uncorrelation_factor/3:
                naive_list_5_precent_error.append(naive_fidelity)
                param_list_5_precent_error.append(param_fidelity)

    naive_mat = np.zeros((len(average_errors),len(diff_errors)))+0.0
    param_mat = np.zeros((len(average_errors),len(diff_errors)))+0.0
    for row in range(len(average_errors)):
        for col in range(len(diff_errors)):
            naive_mat[row][col]=naive_dict[(average_errors[row],diff_errors[col])]
            param_mat[row][col]=param_dict[(average_errors[row],diff_errors[col])]
    naive_mat=naive_mat.T
    param_mat=param_mat.T
    width_normal_vec = np.array(get_norm_dist_vector(0, width_error, len(average_errors)))
    uncorrelation_normal_vec = np.array(get_norm_dist_vector(0, uncorrelation_factor/3, len(diff_errors)))
    norm_dist_vec_2d = np.zeros((len(average_errors),len(diff_errors)))+0.0
    for i in range(len(width_normal_vec)):
        for j in range(len(uncorrelation_normal_vec)):
            norm_dist_vec_2d[i][j]=width_normal_vec[i]*uncorrelation_normal_vec[j]
    print("Average naive fidelity:",np.mean(naive_mat))
    print("Average param fidelity:",np.mean(param_mat))
    print("Average naive Fidelity - normal distribution:",np.sum(naive_mat*norm_dist_vec_2d))
    print("Average param Fidelity - normal distribution:",np.sum(param_mat*norm_dist_vec_2d))


    min_width = min(list(average_errors))#-width_error*3
    max_width = max(list(average_errors))#width_error*3

    min_axis_1 = -width_error*3
    max_axis_1 = width_error*3
    if partially_uncorrelated:
        min_axis_1 = -diff_errors[-1]*100
        max_axis_1 =  diff_errors[-1]*100

    fig, axs = plt.subplots(1, 3, figsize=(10, 5))

    naive_mat_img = axs[0].imshow(naive_mat,extent=(min_width, max_width,min_axis_1, max_axis_1), origin='lower',clim=(0.999, 1.0), aspect=abs(0.01*min_width/uncorrelation_factor))
    param_mat_img = axs[1].imshow(param_mat,extent=(min_width, max_width,min_axis_1, max_axis_1), origin='lower',clim=(0.999, 1.0), aspect=abs(0.01*min_width/uncorrelation_factor))
    diff_mat_img = axs[2].imshow(param_mat-naive_mat,extent=(min_width, max_width,min_axis_1, max_axis_1), origin='lower',  aspect=abs(0.01*min_width/uncorrelation_factor))

    # contour values
    X, Y = np.meshgrid(np.linspace(min_width, max_width, len(average_errors)),
                       np.linspace(min_axis_1, max_axis_1, len(diff_errors)))

    axs[2].contour(X, Y, param_mat-naive_mat, levels=[-1*(10**-5),0],colors = 'black')

    axs[0].title.set_text('Uniform coupler')
    axs[0].set_xlabel('dw [um]')
    axs[0].set_ylabel('uncorrelation percentage %')

    axs[1].title.set_text('Segmented coupler')
    axs[1].set_xlabel('dw [um]')
    axs[1].set_ylabel('uncorrelation percentage %')

    axs[2].title.set_text('Difference matrix')
    axs[2].set_xlabel('dw [um]')
    axs[2].set_ylabel('uncorrelation percentage %')

    plt.colorbar(naive_mat_img, ax=axs[0])
    plt.colorbar(param_mat_img, ax=axs[1])
    plt.colorbar(diff_mat_img, ax=axs[2])

    # axs[2].imshow(param_mat-naive_mat,extent=(min_width, max_width,min_axis_1, max_axis_1), origin='lower', clim=(0, np.max(np.abs(param_mat-naive_mat))), aspect=abs(min_width/uncorrelation_factor))
    print("max fidelity increase:",np.max(param_mat-naive_mat),"max fidelity decrease:",-np.min(param_mat-naive_mat))
    plt.show()

    # 5% error graphs:
    plt.plot(average_errors,np.array(naive_list_5_precent_error),"r",label="uniform fidelity")
    plt.plot(average_errors,np.array(param_list_5_precent_error),"b",label="segmented fidelity")
    plt.legend()
    plt.show()

    # Graph_disp_3d(errors_w0,errors_w1,param_dict,"param graph")


def Graph_disp_3d(x,y,dic,title,xlabel="delta_delta",ylabel="delta_omega"):
    Z = np.zeros((len(x),len(x)))+0.0
    cross_section = np.zeros(len(x))+0.0
    x_cross_section= np.zeros(len(x))+0.0
    for row in range(len(x)):
        cross_section[row]=dic[(x[row],y[row])]
        x_cross_section[row] = (x[row]+y[row])/2
        for col in range(len(x)):
            Z[row][col]=dic[(x[row],y[col])]
    Z=Z.T
    plt.imshow(Z,interpolation = 'none')
    plt.title(title, fontsize = 16)
    plt.clim(0.94, 1.0)
    plt.colorbar()
    plt.xlabel(xlabel, fontsize = 12)
    # plt.xticks(range(len(x))[::10],x.astype(np.float16)[::10], rotation=60)
    plt.xticks(range(len(x)),x.astype(np.float16), rotation=60)
    plt.ylabel(ylabel, fontsize = 12)
    # plt.yticks(range(len(y))[::10], y.astype(np.float16)[::10])
    plt.yticks(range(len(y)), y.astype(np.float16))
    plt.tick_params(labelsize=12)
    plt.tight_layout()
    print(np.sum(Z)/(len(x)**2))
    plt.show()
    # print(sum(dic.values()))

def Norm_Fidelity_Vector(sigma, rho=torch.tensor([[0+0j,1.0+0j],[1.0+0j,0+0j]])):
    return torch.abs((torch_trace(torch.conj(rho.T)@sigma)))*0.5


def return_mat_diff_robust(params,error_w1=((0.0,0.0,0.0)),error_w2=((0.0,0.0,0.0)),etch=0.15,deterministic = True,uncorrelated_errors = False):
    num_of_points_used = len(error_w1)
    mat = torch.eye(2).repeat(num_of_points_used,1).reshape(num_of_points_used,2,2)
    mat = mat.type(torch.complex128)
    for index in range(0,len(params),3):
        mat = mat@(get_R_robust(params,error_w1=error_w1[:,index//3],error_w2=error_w2[:,index//3],etch=etch,index=index))
    mat = mat.type(torch.complex64)
    return mat*global_factor


def return_mat_diff_errors(params,error_w1=(0.0,0.0,0.0),error_w2=(0.0,0.0,0.0),deterministic = False,error_etching = 0.15,g=g,debug=False,uncorrelated_errors=False):
    mat = torch.eye(2)+0*1j
    for index in range(0,len(params),3):
        mat = mat@(get_R_mat(params,error_w1=error_w1[index//3],error_w2=error_w2[index//3],index=index,error_etching = error_etching,debug=debug))

    return mat*global_factor



# def get_fidelity_vs_rho_between_segments(param,loss_func = lambda x,rho: loss_fn_Fidelity(x,rho),rho=rho,deterministic = True):
#     param=torch.tensor(param)
#     naive_params = naive_params_getter()
#
#
#     N = 10000
#     sigma = (width_error)**2
#     uniform_vals = []
#     segmented_vals = []
#     x=[]
#     for cross_cov in np.linspace(0,1,20):
#         cov = sigma*np.array([[1, cross_cov,cross_cov], [cross_cov, 1,cross_cov],[cross_cov,cross_cov,1]])
#         cov_eigenvalues = linalg.eigh(cov)[0]
#         if (np.array(cov_eigenvalues)<0.0).any():
#             print("THIS MAT ISN'T PSD:\n",cov)
#             print("eigenvalues:",cov_eigenvalues)
#             print("rho:",cross_cov)
#             continue
#         x.append(cross_cov)
#         uniform_fidelity = []
#         segmented_fidelity = []
#         for i in range(N):
#             cur_errors = torch.tensor(np.random.multivariate_normal([0,0,0], cov, 1)).squeeze()
#             param_fidelity = (1 - loss_func(return_mat_diff_errors(param, error_w1=cur_errors, error_w2=cur_errors, error_etching=0.15,deterministic=deterministic),rho=Uc_dag @ rho @ Uc_dag))
#             naive_fidelity = (1 - loss_func(return_mat_diff_errors(naive_params, error_w1=cur_errors, error_w2=cur_errors,error_etching=0.15, deterministic=deterministic),rho=Uc_dag @ rho @ Uc_dag))
#             segmented_fidelity.append(param_fidelity)
#             uniform_fidelity.append(naive_fidelity)
#
#         uniform_vals.insert(0,np.mean(np.array(uniform_fidelity)))
#         segmented_vals.insert(0, np.mean(np.array(segmented_fidelity)))
#
#     uniform_vals = np.array(uniform_vals)
#     segmented_vals = np.array(segmented_vals)
#     # x = np.linspace(0,1,len(uniform_vals))
#
#     plt.title("Average uniform VS segmented fidelity for different correlations between segments")
#     plt.xlabel("correlation between segments")
#     plt.ylabel("fidelity")
#     plt.plot(x,uniform_vals,'r',label="average uniform fidelity")
#     plt.plot(x,segmented_vals,'b',label="average segmented fidelity")
#     plt.legend()
#     plt.show()


def get_rho_crit_between_segments(param,loss_func = lambda x,rho: Norm_Fidelity_Vector(x,rho),rho=rho,deterministic = True, return_rho_crit=False,get_mean=True):
    param=torch.tensor(param)
    naive_params = naive_params_getter()

    N = 10000
    rho_crit = []
    sigmas = (np.linspace(0.5,4,10)**2)*width_error**2#np.array([width_error**2,(1.1*width_error)**2])#
    num_of_segments = len(param)//3

    for sigma in sigmas:
        uniform_vals = []
        segmented_vals = []
        x=[]

        rho_arr = np.linspace(0,1,51)
        for cross_cov in rho_arr:
            cov = sigma*(np.eye(num_of_segments) + (np.ones((num_of_segments,num_of_segments))-np.eye(num_of_segments))*cross_cov )
            cov_eigenvalues = linalg.eigh(cov)[0]
            if (np.array(cov_eigenvalues)<- 10**-10).any():
                print("THIS MAT ISN'T PSD:\n",cov)
                print("eigenvalues:",cov_eigenvalues)
                print("rho:",cross_cov)
                continue
            x.append(cross_cov)
            cur_errors = torch.tensor(np.random.multivariate_normal([0]*num_of_segments, cov, N)).squeeze()

            param_fidelity = (loss_func(return_mat_diff_robust(param, error_w1=cur_errors, error_w2=cur_errors, etch=0.15,deterministic=deterministic),rho=Uc_dag @ rho @ Uc_dag))
            naive_fidelity = (loss_func(return_mat_diff_robust(naive_params, error_w1=cur_errors, error_w2=cur_errors,etch=0.15, deterministic=deterministic),rho=Uc_dag @ rho @ Uc_dag))

            if get_mean:
                uniform_vals.append(torch.mean(naive_fidelity))
                segmented_vals.append(torch.mean(param_fidelity))
            else:
                uniform_vals.append(torch.std(naive_fidelity))
                segmented_vals.append(torch.std(param_fidelity))


        uniform_vals = np.array(uniform_vals)
        segmented_vals = np.array(segmented_vals)
        rho_crit.append(rho_arr[np.argmin(np.abs(uniform_vals-segmented_vals))])

    rho_crit = np.array(rho_crit)
    if return_rho_crit:
        return np.average(rho_crit)

    print("AVERAGE RHO CRIT:",np.average(rho_crit))
    plt.title("Rho crit VS sigma; num of segments:"+str(num_of_segments))
    plt.xlabel("sigma")
    plt.ylabel("rho crit")
    plt.plot((sigmas**0.5)/width_error,rho_crit,'r',label="rho crit")

    plt.legend()
    plt.show()

def get_rho_crit_between_waveguides(param,loss_func = lambda x,rho: Norm_Fidelity_Vector(x,rho),rho=rho,deterministic = True,return_rho_crit=False,get_mean=True):
    param=torch.tensor(param)
    naive_params = naive_params_getter()

    N = 10000
    rho_crit = []
    sigmas = (np.linspace(1,1.01,10)**2)*width_error**2#(np.linspace(0.5,4,10)**2)*width_error**2
    num_of_segments = len(param)//3

    for sigma in sigmas:
        uniform_vals = []
        segmented_vals = []
        x=[]

        rho_arr = np.linspace(0.9,1,101)
        for cross_cov in rho_arr:
            cov = sigma * np.array([[1, cross_cov], [cross_cov, 1]])
            all_errors = torch.tensor(np.random.multivariate_normal([0, 0], cov, N)).squeeze()
            error_w1 = all_errors[:, 0]
            error_w2 = all_errors[:, 1]
            error_w1 = torch.vstack([error_w1] * int(len(param) // 3)).T
            error_w2 = torch.vstack([error_w2] * int(len(param) // 3)).T
            param_fidelity = (loss_func(return_mat_diff_robust(param, error_w1=error_w1, error_w2=error_w2, etch=0.15,
                                                               deterministic=deterministic), rho=Uc_dag @ rho @ Uc_dag))
            naive_fidelity = (loss_func(
                return_mat_diff_robust(naive_params, error_w1=error_w1, error_w2=error_w2, etch=0.15,
                                       deterministic=deterministic), rho=Uc_dag @ rho @ Uc_dag))
            x.append(cross_cov)

            if get_mean:
                uniform_vals.append(torch.mean(naive_fidelity))
                segmented_vals.append(torch.mean(param_fidelity))
            else:
                uniform_vals.append(torch.std(naive_fidelity))
                segmented_vals.append(torch.std(param_fidelity))


        uniform_vals = np.array(uniform_vals)
        segmented_vals = np.array(segmented_vals)
        rho_crit.append(rho_arr[np.argmin(np.abs(uniform_vals-segmented_vals))])

        x0 = x[0]
        x1 = x[-1]
        a0_uniform = uniform_vals[0]
        a1_uniform = uniform_vals[-1]
        a0_segmented = segmented_vals[0]
        a1_segmented = segmented_vals[-1]

        alpha_uniform = a0_uniform - x0 * ((a1_uniform - a0_uniform) / (x1 - x0))
        beta_uniform = ((a1_uniform - a0_uniform) / (x1 - x0))
        alpha_segmented = a0_segmented - x0 * ((a1_segmented - a0_segmented) / (x1 - x0))
        beta_segmented = ((a1_segmented - a0_segmented) / (x1 - x0))
        rho_crit_analytic = (alpha_uniform - alpha_segmented) / (beta_segmented - beta_uniform)



        if return_rho_crit:
            return rho_crit[0]


    plt.title("Rho crit (between segments) VS sigma; num of segments:"+str(num_of_segments))
    plt.xlabel("sigma")
    plt.ylabel("rho crit")
    plt.plot((sigmas**0.5)/width_error,np.array(rho_crit),'r',label="rho crit")

    plt.legend()
    plt.show()


def get_fidelity_vs_rho_between_segments_and_inside_segments(param,loss_func = lambda x,rho: Norm_Fidelity_Vector(x,rho),rho=rho,deterministic = True):
    param=torch.tensor(param)
    naive_params = naive_params_getter()

    N = 50000#50000
    sigma = (10*width_error)**2

    rho_crit_between_waveguides_arr = []
    rho_crit_between_segments_arr = []

    # for sigma in [(width_error)**2]:
    # for cross_cov_inside_segment in np.linspace(0.996,1.0,21):
    for cross_cov_inside_segment in np.linspace(0.996,1.0,21):
    # for sigma in [(0.5*width_error)**2,(width_error)**2,(2*width_error)**2,(3*width_error)**2,(4*width_error)**2]:
        uniform_vals = []
        segmented_vals = []
        x=[]

        num_of_segments = len(param)//3
        calculate_mean = True
        cross_cov_arr = np.linspace(0.5,1.0,401)
        # cross_cov_inside_segment = 1.0
        for cross_cov in cross_cov_arr:
            old_cov_mat = False
            if old_cov_mat:
                inside_segment_rho_mat = []
                between_segments_rho_mat = []
                for i in range(num_of_segments*2):
                    inside_segment_rho_mat.append([])
                    between_segments_rho_mat.append([])
                    for j in range(num_of_segments*2):
                        if abs(i-j)%2==1 and ((i>j and i%2==1) or (j>i and j%2==1)):
                            between_segments_rho_mat[-1].append(0)
                            inside_segment_rho_mat[-1].append(cross_cov_inside_segment)
                            if abs(i-j)!=1:
                                inside_segment_rho_mat[-1][-1]*=cross_cov
                        elif i==j:
                            inside_segment_rho_mat[-1].append(0)
                            between_segments_rho_mat[-1].append(0)
                        else:
                            inside_segment_rho_mat[-1].append(0)
                            between_segments_rho_mat[-1].append(cross_cov)

                        # if (i-j == 1 and i%2==1) or (j-i==1 and j%2==1) :#(i-j==1 and i%2==1) or (i-j==-1 and i%2==0):
                        #     inside_segment_rho_mat[-1].append(cross_cov)
                        #     between_segments_rho_mat[-1].append(0)
                        # elif i==j:
                        #     inside_segment_rho_mat[-1].append(0)
                        #     between_segments_rho_mat[-1].append(0)
                        # else:
                        #     inside_segment_rho_mat[-1].append(0)
                        #     between_segments_rho_mat[-1].append(1.0)

                    inside_segment_rho_mat = np.array(inside_segment_rho_mat)
                    between_segments_rho_mat = np.array(between_segments_rho_mat)

                    cov = (width_error**2)*(np.eye(num_of_segments*2) + inside_segment_rho_mat + between_segments_rho_mat)
                    # print(cov/(width_error**2))
            else:
                cov = (width_error ** 2) * np.ones((6,6)) *\
          np.array([[1 if (i+j)%2==0 else cross_cov_inside_segment for i in range(6)] for j in range(6)]) * \
          np.array([[1 if i==j or (i-j==1 and j%2==0) or (j-i==1 and i%2==0) else cross_cov for i in range(6)] for j in range(6)])



            cov_eigenvalues = linalg.eigh(cov)[0]
            if (np.array(cov_eigenvalues)<- 10**-10).any():
                print("THIS MAT ISN'T PSD!")#\n",cov)
                # print("eigenvalues:",cov_eigenvalues)
                # print("rho:",cross_cov)
                continue
            x.append(cross_cov)
            # cur_errors = torch.tensor(np.random.multivariate_normal([0]*num_of_segments, cov, N)).squeeze()
            cur_errors = torch.tensor(np.random.multivariate_normal([0] * len(cov), cov, N)).squeeze()
            if len(cov)==num_of_segments:
                error_w1 = cur_errors
                error_w2 = cur_errors
            else:
                error_w1 = cur_errors[:, 1::2]
                error_w2 = cur_errors[:, ::2]

            param_fidelity = (loss_func(return_mat_diff_robust(param, error_w1=error_w1, error_w2=error_w2, etch=0.15,deterministic=deterministic),rho=Uc_dag @ rho @ Uc_dag))
            naive_fidelity = (loss_func(return_mat_diff_robust(naive_params, error_w1=error_w1, error_w2=error_w2,etch=0.15, deterministic=deterministic),rho=Uc_dag @ rho @ Uc_dag))

            if calculate_mean:
                uniform_vals.append(torch.mean(naive_fidelity))
                segmented_vals.append(torch.mean(param_fidelity))
            else:
                uniform_vals.append(torch.std(naive_fidelity))
                segmented_vals.append(torch.std(param_fidelity))


        uniform_vals = np.array(uniform_vals)
        segmented_vals = np.array(segmented_vals)

        x = np.linspace(0,1,len(uniform_vals))
        rho_as_displayed =np.linspace(1,0,len(uniform_vals))

        # plt.gca().invert_xaxis()

        if len(uniform_vals)==0:
            continue
        rho_crit = cross_cov_arr[np.argmin(np.abs(uniform_vals - segmented_vals))]
        # plt.axvline(x=rho_crit,color='black', linestyle="--")
        # print("for sigma:",sigma/(width_error**2),"the rho crit is:",rho_crit)

        x0 = x[0]
        x1 = x[-1]
        a0_uniform = uniform_vals[0]
        a1_uniform = uniform_vals[-1]
        a0_segmented = segmented_vals[0]
        a1_segmented = segmented_vals[-1]

        alpha_uniform = a0_uniform - x0 * ((a1_uniform - a0_uniform) / (x1 - x0))
        beta_uniform = ((a1_uniform - a0_uniform) / (x1 - x0))
        alpha_segmented = a0_segmented - x0 * ((a1_segmented - a0_segmented) / (x1 - x0))
        beta_segmented = ((a1_segmented - a0_segmented) / (x1 - x0))
        rho_crit_analytic = (alpha_uniform - alpha_segmented) / (beta_segmented - beta_uniform)

        print("for cross_cov_inside_segment:", cross_cov_inside_segment, "the rho between segments is:", rho_crit,
                "the rho between segments is ANALYTIC:", rho_crit_analytic)

        rho_crit_between_segments_arr.append(rho_crit)
        rho_crit_between_waveguides_arr.append(cross_cov_inside_segment)

        # print("for cross_cov_inside_segment:", cross_cov_inside_segment, "the rho between segments is:", rho_crit_analytic)

        # if calculate_mean:
        #
        #     plt.title("Average uniform VS segmented fidelity for different correlations between segments")
        #     plt.xlabel("rho")
        #     plt.ylabel("mean fidelity ")
        #     plt.plot(x,uniform_vals,label="mean uniform fidelity for sigma:"+str((sigma**0.5)/width_error))
        #     plt.plot(x,segmented_vals,label="mean segmented fidelity for sigma:"+str((sigma**0.5)/width_error))
        #     # approx_line = 0.9999873638155-(0.0004326105120200374)*rho_as_displayed
        #     # plt.plot(x,approx_line,'g',label="approximation graph")
        # else:
        #     plt.title("std uniform VS segmented fidelity for different correlations between segments")
        #     plt.xlabel("rho")
        #     plt.ylabel("std of fidelity")
        #     plt.plot(x,uniform_vals,'r',label="std uniform fidelity")
        #     plt.plot(x,segmented_vals,'b',label="std segmented fidelity")
        #     approx_line = 0.00046394*(1-x)
        #     plt.plot(x,approx_line,'g',label="approximation graph")

    # fit graph to crit values:

    rho_crit_between_waveguides_arr = np.array(rho_crit_between_waveguides_arr)
    rho_crit_between_segments_arr = np.array(rho_crit_between_segments_arr)

    estimate_func = lambda x, a0, a1, a2, a3: a0 + a1 * x +  a2 * (x ** 2) +  0 * a3 * (x ** 3)

    x0   = np.array([1.0]*4)

    xFit = optimization.curve_fit(estimate_func, rho_crit_between_waveguides_arr,rho_crit_between_segments_arr, x0, sigma=None)[0]

    # plt.title("critical correlation")
    plt.xlabel("ρ̄")
    plt.ylabel("ρ")
    plt.plot(rho_crit_between_waveguides_arr,rho_crit_between_segments_arr,'r',label="numeric approximation",linewidth=3)
    plt.plot(rho_crit_between_waveguides_arr,estimate_func(rho_crit_between_waveguides_arr,xFit[0],xFit[1],xFit[2],xFit[3]),'b--',label="polynomial fit",linewidth=3)

    print("Fit values:",xFit)

    plt.legend(loc = "lower left")
    plt.subplot_tool()
    plt.show()

def get_rho_crit_analytic_and_numeric(x,cross_cov_arr,uniform_vals,segmented_vals):
    rho_crit = cross_cov_arr[np.argmin(np.abs(uniform_vals - segmented_vals))]
    # print("for sigma:",sigma/(width_error**2),"the rho crit is:",rho_crit)

    x0 = x[0]
    x1 = x[-1]
    a0_uniform = uniform_vals[0]
    a1_uniform = uniform_vals[-1]
    a0_segmented = segmented_vals[0]
    a1_segmented = segmented_vals[-1]

    alpha_uniform = a0_uniform - x0 * ((a1_uniform - a0_uniform) / (x1 - x0))
    beta_uniform = ((a1_uniform - a0_uniform) / (x1 - x0))
    alpha_segmented = a0_segmented - x0 * ((a1_segmented - a0_segmented) / (x1 - x0))
    beta_segmented = ((a1_segmented - a0_segmented) / (x1 - x0))
    rho_crit_analytic = (alpha_uniform - alpha_segmented) / (beta_segmented - beta_uniform)
    return rho_crit,rho_crit_analytic


def get_fidelity_vs_rho_between_segments(param,loss_func = lambda x,rho: Norm_Fidelity_Vector(x,rho),rho=rho,deterministic = True,calculate_mean = True,calculate_both=False,which_rho_crit_is_bigger=False,save_fig=False):
    param=torch.tensor(param)
    naive_params = naive_params_getter()

    N = 100000
    sigmas =  [(width_error)**2]#[(0.5*width_error)**2,(width_error)**2,(2*width_error)**2,(3*width_error)**2,(4*width_error)**2]
    rho_crit_colors = [ 'aqua', 'aquamarine', 'bisque', 'blanchedalmond', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'forestgreen', 'fuchsia', 'gainsboro', 'gold', 'goldenrod', 'gray', 'green', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey', 'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navy', 'olive', 'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'rebeccapurple', 'red', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue', 'slategray', 'slategrey', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'yellowgreen']
#["gfdgfdgf","blue","red","green","orange","pink"]
    for i in range(len(sigmas)):#[(width_error)**2]:
        sigma = sigmas[i]
        mean_uniform_vals = []
        mean_segmented_vals = []
        std_uniform_vals = []
        std_segmented_vals = []

        x=[]

        num_of_segments = len(param)//3
        cross_cov_arr = np.linspace(0.0,1.0,25)
        for cross_cov in cross_cov_arr:
            # cov = sigma*(np.eye(num_of_segments) + (np.ones((num_of_segments,num_of_segments))-np.eye(num_of_segments))*cross_cov )
            # cov = sigma*torch.tensor([[1,cross_cov,cross_cov],[cross_cov,1,cross_cov],[cross_cov,cross_cov,1]])

            cov = []
            for i in range(num_of_segments):
                cur_row = []
                for j in range(num_of_segments):
                    if i==j:
                        cur_row.append(1)
                    else:
                        cur_row.append(cross_cov)
                cov.append(cur_row)

            cov = sigma*torch.tensor(cov)#[[1,cross_cov,cross_cov],[cross_cov,1,cross_cov],[cross_cov,cross_cov,1]])

            cov_eigenvalues = linalg.eigh(cov)[0]
            if (np.array(cov_eigenvalues)<- 10**-10).any():
                print("THIS MAT ISN'T PSD!")#\n",cov)
                continue
            x.append(cross_cov)
            cur_errors = torch.tensor(np.random.multivariate_normal([0] * len(cov), cov, N)).squeeze()
            if len(cov)==num_of_segments:
                error_w1 = cur_errors
                error_w2 = cur_errors
            else:
                error_w1 = cur_errors[:, 1::2]
                error_w2 = cur_errors[:, ::2]

            param_fidelity = (loss_func(return_mat_diff_robust(param, error_w1=error_w1, error_w2=error_w2, etch=0.15,deterministic=deterministic),rho=Uc_dag @ rho @ Uc_dag))
            naive_fidelity = (loss_func(return_mat_diff_robust(naive_params, error_w1=error_w1, error_w2=error_w2,etch=0.15, deterministic=deterministic),rho=Uc_dag @ rho @ Uc_dag))

            mean_uniform_vals.append(torch.mean(naive_fidelity))
            mean_segmented_vals.append(torch.mean(param_fidelity))
            std_uniform_vals.append(torch.std(naive_fidelity))
            std_segmented_vals.append(torch.std(param_fidelity))

        mean_uniform_vals=np.array(mean_uniform_vals)
        mean_segmented_vals=np.array(mean_segmented_vals)
        std_uniform_vals=np.array(std_uniform_vals)
        std_segmented_vals=np.array(std_segmented_vals)


        mean_rho_crit, mean_rho_crit_anayltic = get_rho_crit_analytic_and_numeric(x,cross_cov_arr,mean_uniform_vals,mean_segmented_vals)
        std_rho_crit, std_rho_crit_anayltic = get_rho_crit_analytic_and_numeric(x,cross_cov_arr,std_uniform_vals,std_segmented_vals)


        x = np.linspace(0,1,len(mean_uniform_vals))
        rho_as_displayed =np.linspace(1,0,len(mean_uniform_vals))

        # plt.gca().invert_xaxis()

        if len(mean_uniform_vals)==0:
            continue

        #
        # print("for sigma:", sigma, "the rho between segments is:", rho_crit_analytic)
        # print("NUMERIC RHO CRIT:",rho_crit)

        if calculate_both:
            if which_rho_crit_is_bigger:
                return mean_rho_crit>std_rho_crit
            plt.axvline(x=mean_rho_crit, linestyle="--",color="black", label="mean critical correlation value",linewidth=3)# - sigma:"+str((sigma**0.5)/width_error),linewidth=3)
            plt.axvline(x=std_rho_crit, linestyle="--",color="gray", label="std critical correlation value",linewidth=3)# - sigma:"+str((sigma**0.5)/width_error),linewidth=3)

            plt.xlabel("ρ")
            plt.ylabel("1-mean fidelity/std fidelity")
            plt.plot(x, 1-mean_uniform_vals,color="red", label="mean fidelity - uniform coupler",linewidth=3)#, sigma:" + str((sigma ** 0.5) / width_error),linewidth=3)
            plt.plot(x, 1-mean_segmented_vals,color="blue",label="mean fidelity - segmented coupler",linewidth=3)#, sigma:" + str((sigma ** 0.5) / width_error),linewidth=3)
            plt.plot(x, std_uniform_vals,color="green", label="std uniform fidelity",linewidth=3)# - sigma:" + str((sigma ** 0.5) / width_error),linewidth=3)
            plt.plot(x, std_segmented_vals,color="orange", label="std segmented fidelity",linewidth=3)# - sigma:" + str((sigma ** 0.5) / width_error),linewidth=3)
            # plt.get_current_fig_manager().full_screen_toggle()


            print("CORRELATION BETWEEN UNIFORM VALUES:",np.mean(std_uniform_vals)/np.mean(1-mean_uniform_vals))

        elif calculate_mean:
            if (sigma**0.5)/width_error==1.0:
                plt.axvline(x=mean_rho_crit, linestyle="--",color="black", label="critical correlation",linewidth=3)

            # plt.title("Average uniform VS segmented fidelity for different correlations between segments")
            plt.xlabel("ρ")
            plt.ylabel("mean fidelity")
            plt.plot(x,mean_uniform_vals,label="uniform - sigma:"+str((sigma**0.5)/width_error),linewidth=3)
            plt.plot(x,mean_segmented_vals,label="CP design - sigma:"+str((sigma**0.5)/width_error),linewidth=3)
            # approx_line = 0.9999873638155-(0.0004326105120200374)*rho_as_displayed
            # plt.plot(x,approx_line,'g',label="approximation graph")
        else:
            plt.axvline(x=std_rho_crit_anayltic, linestyle="--",color="black", label="critical correlation value - sigma:"+str((sigma**0.5)/width_error),linewidth=3)
            # plt.title("std uniform VS segmented fidelity for different correlations between segments")
            plt.xlabel("ρ")
            plt.ylabel("std of fidelity")
            plt.plot(x,std_uniform_vals,color="red",label="std uniform fidelity - sigma:"+str((sigma**0.5)/width_error),linewidth=3)
            plt.plot(x,std_segmented_vals,color="blue",label="std segmented fidelity - sigma:"+str((sigma**0.5)/width_error),linewidth=3)

    plt.legend(loc="lower left",ncol=1, fontsize=16, framealpha=1.0)
    # plt.legend(bbox_to_anchor =(0.48, 1.15), loc='upper center',ncol=4, fontsize=16)
    # plt.legend()

    if save_fig:
        fig = plt.gcf()
        fig.set_size_inches((15, 11), forward=False)
        try:
            plt.savefig("uniform_and_segmented_std_and_mean_figures\\"+str([int(val*1000) for val in param]).replace("[","").replace("]","").replace(" ","").replace(",","_")+".png")
        except:
            try:
                plt.savefig("uniform_and_segmented_std_and_mean_figures\\"+str(global_counter[0])+".png")
                global_counter[0]+=1
                plt.close()
            except:
                plt.close()
                return

        plt.close()
    else:
        plt.show()


def get_fidelity_vs_sigma_between_segments(param,loss_func = lambda x,rho: Norm_Fidelity_Vector(x,rho),rho=rho,deterministic = True,cross_cov = 0.9):
    param=torch.tensor(param)
    naive_params = naive_params_getter()

    N = 300000//len(param)
    sigmas = np.linspace(0.5,4,20)*(width_error)**2
    uniform_vals = []
    segmented_vals = []

    for sigma in sigmas:

        num_of_segments = len(param)//3
        calculate_mean = True
        cov = sigma*(np.eye(num_of_segments) + (np.ones((num_of_segments,num_of_segments))-np.eye(num_of_segments))* cross_cov)


        cov_eigenvalues = linalg.eigh(cov)[0]
        if (np.array(cov_eigenvalues)<- 10**-10).any():
            print("THIS MAT ISN'T PSD!")#\n",cov)
            continue
        cur_errors = torch.tensor(np.random.multivariate_normal([0] * len(cov), cov, N)).squeeze()
        error_w1 = cur_errors
        error_w2 = cur_errors

        param_fidelity = (loss_func(return_mat_diff_robust(param, error_w1=error_w1, error_w2=error_w2, etch=0.15,deterministic=deterministic),rho=Uc_dag @ rho @ Uc_dag))
        naive_fidelity = (loss_func(return_mat_diff_robust(naive_params, error_w1=error_w1, error_w2=error_w2,etch=0.15, deterministic=deterministic),rho=Uc_dag @ rho @ Uc_dag))

        if calculate_mean:
            uniform_vals.append(torch.mean(naive_fidelity))
            segmented_vals.append(torch.mean(param_fidelity))
        else:
            uniform_vals.append(torch.std(naive_fidelity))
            segmented_vals.append(torch.std(param_fidelity))


    uniform_vals = np.array(uniform_vals)
    segmented_vals = np.array(segmented_vals)

    x = sigmas/(width_error**2)

    plt.title("Average uniform VS segmented fidelity for different sigma values")
    plt.xlabel("sigma")
    plt.ylabel("mean fidelity ")
    plt.plot(x,uniform_vals,label="mean uniform fidelity for rho:"+str(cross_cov))
    plt.plot(x,segmented_vals,label="mean segmented fidelity for rho:"+str(cross_cov))

    plt.legend()
    plt.show()

def get_fidelity_vs_sigma_between_segments_and_inside_segment(param,loss_func = lambda x,rho: Norm_Fidelity_Vector(x,rho),rho=rho,deterministic = True,cross_cov=0.99,two_params = False):
    if two_params:
        analytic_params = torch.tensor(param[1])
        param = torch.tensor(param[0])
    else:
        param=torch.tensor(param)
    naive_params = naive_params_getter()

    N = 100000
    sigmas = np.linspace(0,(4*width_error)**2,40)
    uniform_vals = []
    segmented_vals = []
    analytic_vals = []
    x=[]

    num_of_segments = len(param)//3
    if two_params:
        num_of_segments = max(len(param) // 3, len(analytic_params) // 3)

    calculate_mean = True
    for sigma in sigmas:
        cov = sigma*(np.eye(num_of_segments*2) + (np.ones((num_of_segments*2,num_of_segments*2))-np.eye(num_of_segments*2))*cross_cov )
        x.append(cross_cov)
        cur_errors = torch.tensor(np.random.multivariate_normal([0]*num_of_segments*2, cov, N)).squeeze()
        error_w1 = cur_errors[:,:num_of_segments]
        error_w2 = cur_errors[:,num_of_segments:]
        param_fidelity = (loss_func(return_mat_diff_robust(param, error_w1=error_w1, error_w2=error_w2, etch=0.15,deterministic=deterministic),rho=Uc_dag @ rho @ Uc_dag))
        naive_fidelity = (loss_func(return_mat_diff_robust(naive_params, error_w1=error_w1, error_w2=error_w2,etch=0.15, deterministic=deterministic),rho=Uc_dag @ rho @ Uc_dag))
        if two_params:
            analytic_fidelity = (loss_func(
                return_mat_diff_robust(analytic_params, error_w1=error_w1, error_w2=error_w2, etch=0.15,
                                       deterministic=deterministic), rho=Uc_dag @ rho @ Uc_dag))
            analytic_vals.append(torch.mean(analytic_fidelity))


        if calculate_mean:
            uniform_vals.append(torch.mean(naive_fidelity))
            segmented_vals.append(torch.mean(param_fidelity))
        else:
            uniform_vals.append(torch.std(naive_fidelity))
            segmented_vals.append(torch.std(param_fidelity))


    uniform_vals = np.array(uniform_vals)
    segmented_vals = np.array(segmented_vals)

    x = sigmas

    plt.title("Average uniform VS segmented fidelity for different sigma values")
    plt.xlabel("sigma^2 [nm^2]")
    plt.ylabel("mean fidelity ")
    plt.plot(x,uniform_vals,'r',label="mean uniform fidelity for rho:"+str(cross_cov))
    plt.plot(x,segmented_vals,'b',label="mean segmented fidelity - non perturbative solution for rho:"+str(cross_cov))
    if two_params:
        plt.plot(x, analytic_vals, 'g',
                 label="mean segmented fidelity - perturbative solution for rho:"+str(cross_cov))

    plt.legend()
    plt.show()







def get_fidelity_vs_rho_inside_segment(param,loss_func = lambda x,rho: Norm_Fidelity_Vector(x,rho),rho=rho,deterministic = True, calculate_mean = True):
    param=torch.tensor(param)
    naive_params = naive_params_getter()

    N = 100000
    rho_crit_colors = [ 'aqua', 'aquamarine', 'bisque',  'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'forestgreen', 'fuchsia', 'gainsboro', 'gold', 'goldenrod', 'gray', 'green', 'greenyellow', 'grey', 'honeydew', 'indianred', 'indigo', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon','lime', 'limegreen', 'linen', 'magenta', 'maroon', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navy', 'olive', 'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'rebeccapurple', 'red', 'rosybrown', 'royalblue', 'saddlebrown', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'skyblue', 'slateblue', 'slategray', 'slategrey', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'yellowgreen']
    x = np.linspace(0.99,1,51)
    for sigma in [(0.5*width_error)**2,(width_error)**2,(2*width_error)**2,(3*width_error)**2,(4*width_error)**2]:#[(width_error)**2]:
        uniform_vals = []
        segmented_vals = []


        for cross_cov in x:
            cov = sigma * np.array([[1, cross_cov], [cross_cov, 1]])
            all_errors = torch.tensor(np.random.multivariate_normal([0, 0], cov, N)).squeeze()
            error_w1 = all_errors[:, 0]
            error_w2 = all_errors[:, 1]
            error_w1 = torch.vstack([error_w1] * int(len(param) // 3)).T
            error_w2 = torch.vstack([error_w2] * int(len(param) // 3)).T

            param_fidelity = (loss_func(return_mat_diff_robust(param, error_w1=error_w1, error_w2=error_w2, etch=0.15,
                                                               deterministic=deterministic), rho=Uc_dag @ rho @ Uc_dag))
            naive_fidelity = (loss_func(
                return_mat_diff_robust(naive_params, error_w1=error_w1, error_w2=error_w2, etch=0.15,
                                       deterministic=deterministic), rho=Uc_dag @ rho @ Uc_dag))

            if calculate_mean:
                uniform_vals.append(torch.mean(naive_fidelity))
                segmented_vals.append(torch.mean(param_fidelity))
            else:
                uniform_vals.append(torch.std(naive_fidelity))
                segmented_vals.append(torch.std(param_fidelity))




        uniform_vals = np.array(uniform_vals)
        segmented_vals = np.array(segmented_vals)

        rho_crit = x[np.argmin(np.abs(uniform_vals - segmented_vals))]
        print(np.abs(uniform_vals - segmented_vals))
        print(np.argmin(np.abs(uniform_vals - segmented_vals)))
        print(x[np.argmin(np.abs(uniform_vals - segmented_vals))])
        if sigma == (width_error)**2:
            plt.axvline(x=rho_crit,color="black", linestyle="--",linewidth=3,label="critical correlation")# for sigma:"+str((sigma**0.5)/width_error))
        # plt.axvline(x=rho_crit,color=rho_crit_colors[np.random.randint(0,len(rho_crit_colors))], linestyle="--",linewidth=3,label="critical correlation for sigma:"+str((sigma**0.5)/width_error))

        # calculate rho_crit analytically
        # x0 = x[0]
        # x1 = x[-1]
        # a0_uniform = uniform_vals[0]
        # a1_uniform = uniform_vals[-1]
        # a0_segmented = segmented_vals[0]
        # a1_segmented = segmented_vals[-1]

        # alpha_uniform = a0_uniform - x0*((a1_uniform - a0_uniform)/(x1-x0))
        # beta_uniform = ((a1_uniform - a0_uniform)/(x1-x0))
        # alpha_segmented = a0_segmented - x0*((a1_segmented - a0_segmented)/(x1-x0))
        # beta_segmented = ((a1_segmented - a0_segmented)/(x1-x0))
        # rho_crit_analytic = (alpha_uniform-alpha_segmented)/(beta_segmented-beta_uniform)
        # plt.axvline(x=rho_crit_analytic,color='black', linestyle="--")

        if calculate_mean:
            # if (sigma**0.5)/width_error==1.0:
            #     plt.axvline(x=mean_rho_crit, linestyle="--",color="black", label="critical correlation",linewidth=3)

            # plt.title("Average uniform VS segmented fidelity for different correlations between segments")
            plt.xlabel("ρ")
            plt.ylabel("mean fidelity")
            plt.plot(x,uniform_vals,linewidth=3,label="uniform - sigma:"+str((sigma**0.5)/width_error))
            plt.plot(x,segmented_vals,linewidth=3,label="CP design - sigma:"+str((sigma**0.5)/width_error))
            # approx_line = 0.9999873638155-(0.0004326105120200374)*rho_as_displayed
            # plt.plot(x,approx_line,'g',label="approximation graph")


            # plt.title("Average uniform VS segmented fidelity for different correlations between waveguides")
            # plt.xlabel("rho")
            # plt.ylabel("mean fidelity ")
            # plt.plot(x,uniform_vals,label="mean uniform fidelity for sigma:"+str((sigma**0.5)/width_error))
            # plt.plot(x,segmented_vals,label="mean segmented fidelity for sigma:"+str((sigma**0.5)/width_error))
        else:
            plt.title("uniform VS segmented coupler fidelity standard deviation for different correlations between waveguides")
            plt.xlabel("rho")
            plt.ylabel("fidelity standard deviation")
            plt.plot(x,uniform_vals,label="uniform fidelity standard deviation for sigma:"+str((sigma**0.5)/width_error),linewidth=3,color="red")
            plt.plot(x,segmented_vals,label="segmented fidelity standard deviation for sigma:"+str((sigma**0.5)/width_error),linewidth=3,color="blue")


    plt.subplot_tool()

    plt.legend(loc="lower right",ncol=1, fontsize=16, framealpha=1.0)
    plt.show()

def get_fidelity_vs_sigma_perfect_correlation(param,argv=None,loss_func = lambda x,rho: Norm_Fidelity_Vector(x,rho),deterministic = True,two_params = False, to_multiplot=False,rho=rho,get_std = False):
    if argv!=None:
        if len(argv)==1:
            argv.append(False)
        rho, get_std = argv


    two_params = two_params or len(param)==2
    if two_params:
        analytic_params = torch.tensor(param[1])
        param = torch.tensor(param[0])
    else:
        param=torch.tensor(param)
    naive_params = naive_params_getter(rho)

    N = 100000
    uniform_vals = []
    segmented_vals = []
    analytic_vals = []
    x=[]
    num_of_segments = len(param)//3

    for sigma in np.linspace(0,(width_error*3),51):
        cov = sigma*np.array([[1.0]*num_of_segments]*num_of_segments)
        x.append(sigma)
        cur_errors =  np.hstack([np.random.normal(0, sigma, (N,1))]*3) #torch.tensor(np.random.multivariate_normal([0.0]*num_of_segments, cov, N)).squeeze() #
        param_fidelity = (loss_func(return_mat_diff_robust(param, error_w1=cur_errors, error_w2=cur_errors, etch=0.15,deterministic=deterministic),rho=Uc_dag @ rho @ Uc_dag))
        naive_fidelity = (loss_func(return_mat_diff_robust(naive_params, error_w1=cur_errors, error_w2=cur_errors,etch=0.15, deterministic=deterministic),rho=Uc_dag @ rho @ Uc_dag))
        if two_params:
            analytic_fidelity = (loss_func(return_mat_diff_robust(analytic_params, error_w1=cur_errors, error_w2=cur_errors, etch=0.15,deterministic=deterministic), rho=Uc_dag @ rho @ Uc_dag))
            if get_std:
                analytic_vals.append(torch.var(analytic_fidelity))
            else:
                analytic_vals.append(torch.mean(analytic_fidelity))

        if get_std:
            uniform_vals.append(torch.std(naive_fidelity))
            segmented_vals.append(torch.std(param_fidelity))
        else:
            uniform_vals.append(torch.mean(naive_fidelity))
            segmented_vals.append(torch.mean(param_fidelity))


    uniform_vals = np.array(uniform_vals)
    segmented_vals = np.array(segmented_vals)
    analytic_vals = np.array(analytic_vals)

    x = np.linspace(0,(width_error*3),len(uniform_vals))*1000

    if to_multiplot:
        if get_std:
            return x,[uniform_vals,segmented_vals,analytic_vals],"σ [nm]","fidelity standard deviation ",["uniform",
                                                                                           "non perturbative",
                                                                                           "perturbative"],["r","royalblue","k--"],False, False
        else:
            return x,[uniform_vals,segmented_vals,analytic_vals],"σ [nm]","mean fidelity",["uniform",
                                                                                           "non perturbative",
                                                                                           "perturbative"],["r","royalblue","k--"],False, False



    plt.xlabel("σ [nm]")
    if get_std:
        plt.title("uniform VS segmented - standard deviation of fidelity")
        plt.ylabel("standard deviation of fidelity")
        plt.plot(x,uniform_vals,'r',label="uniform coupler - standard deviation of fidelity",linewidth=3)# for sigma:"+str(sigma**0.5))
        plt.plot(x,segmented_vals,'b',label="non perturbative solution - standard deviation of fidelity",linewidth=3)# for sigma:"+str(sigma**0.5))

        if two_params:
            plt.plot(x, analytic_vals, 'g', label="perturbative solution - standard deviation of fidelity",linewidth=3)  # for sigma:"+str(sigma**0.5))

    else:
        plt.title("Average uniform VS segmented fidelity ")
        plt.ylabel("mean fidelity ")
        plt.plot(x,uniform_vals,'r',label="mean uniform fidelity",linewidth=3)# for sigma:"+str(sigma**0.5))
        plt.plot(x,segmented_vals,'b',label="mean segmented fidelity - non perturbative solution",linewidth=3)# for sigma:"+str(sigma**0.5))
        if two_params:
            plt.plot(x, analytic_vals, 'g', label="mean segmented fidelity - perturbative solution",linewidth=3)  # for sigma:"+str(sigma**0.5))

    plt.legend()
    plt.show()



def get_fidelity_vs_sigma_with_rho_between_waveguide_inside_segment(param,loss_func = lambda x,rho: Norm_Fidelity_Vector(x,rho),rho=rho,deterministic = True,two_params = False,cross_cov = 0.9,rho_between_sigma = True):
    if two_params:
        analytic_params = torch.tensor(param[1])
        param = torch.tensor(param[0])
    else:
        param=torch.tensor(param)
    naive_params = naive_params_getter()

    N = 100000
    uniform_vals = []
    segmented_vals = []
    analytic_vals = []
    x=[]

    for sigma in np.linspace(0,(width_error*3)**2,20):
        cov = sigma*np.array([[1, cross_cov],[cross_cov, 1]])
        x.append(sigma)
        all_errors = torch.tensor(np.random.multivariate_normal([0,0], cov, N)).squeeze()
        error_w1 = all_errors[:,0]
        error_w2 = all_errors[:,1]
        error_w1 = torch.vstack([error_w1]*int(len(param)//3)).T
        error_w2 = torch.vstack([error_w2]*int(len(param)//3)).T
        if not rho_between_sigma:
            error_w2 = error_w1

        param_fidelity = (loss_func(return_mat_diff_robust(param, error_w1=error_w1, error_w2=error_w2, etch=0.15,deterministic=deterministic),rho=Uc_dag @ rho @ Uc_dag))
        naive_fidelity = (loss_func(return_mat_diff_robust(naive_params, error_w1=error_w1, error_w2=error_w2,etch=0.15, deterministic=deterministic),rho=Uc_dag @ rho @ Uc_dag))
        if two_params:
            analytic_fidelity = (loss_func(return_mat_diff_robust(analytic_params, error_w1=error_w1, error_w2=error_w2, etch=0.15,deterministic=deterministic), rho=Uc_dag @ rho @ Uc_dag))
            analytic_vals.append(torch.mean(analytic_fidelity))

        uniform_vals.append(torch.mean(naive_fidelity))
        segmented_vals.append(torch.mean(param_fidelity))


    uniform_vals = np.array(uniform_vals)
    segmented_vals = np.array(segmented_vals)
    analytic_vals = np.array(analytic_vals)

    x = np.linspace(0,(width_error*3),20)*1000

    plt.title("Average uniform VS segmented fidelity with rho:"+str(cross_cov))
    plt.xlabel("σ [nm]")
    plt.ylabel("mean fidelity ")
    plt.plot(x,uniform_vals,'r',label="mean uniform fidelity")# for sigma:"+str(sigma**0.5))
    plt.plot(x,segmented_vals,'b',label="mean segmented fidelity - non perturbative solution")# for sigma:"+str(sigma**0.5))
    if two_params:
        plt.plot(x, analytic_vals, 'g', label="mean segmented fidelity - perturbative solution")  # for sigma:"+str(sigma**0.5))

    plt.legend()
    plt.show()

# def get_average_error_partially_correlated(param,loss_func = lambda x,rho: loss_fn_Fidelity(x,rho),rho=rho,deterministic = True):
#     param=torch.tensor(param)
#     naive_params = torch.tensor(naive_params_getter())
#
#     N = 400
#     sigma = width_error**2
#     cross_cov = width_error**2
#     uniform_vals = []
#     segmented_vals = []
#
#     for cross_cov in np.linspace(0,sigma,10):
#         cov = np.array([[sigma, cross_cov], [cross_cov, sigma]])
#         uniform_fidelity = []
#         segmented_fidelity = []
#         for i in range(N):
#             cur_errors = torch.tensor(np.random.multivariate_normal([0,0], cov, 1))
#             error_w0 = cur_errors[0,0]
#             error_w1 = cur_errors[0,1]
#
#             param_fidelity = (1 - loss_func(return_mat_generic(param, error_w1=error_w0, error_w2=error_w1, error_etching=0.15,deterministic=deterministic),rho=Uc_dag @ rho @ Uc_dag))
#             naive_fidelity = (1 - loss_func(return_mat_generic(naive_params, error_w1=error_w0, error_w2=error_w1,error_etching=0.15, deterministic=deterministic),rho=Uc_dag @ rho @ Uc_dag))
#             segmented_fidelity.append(param_fidelity)
#             uniform_fidelity.append(naive_fidelity)
#
#         # print("for rho=",cross_cov/sigma,"the average uniform fidelity is:",np.mean(np.array(uniform_fidelity)))
#         # print("the average segmented fidelity is:",np.mean(np.array(segmented_fidelity)),"\n\n")
#         uniform_vals.insert(0,np.mean(np.array(uniform_fidelity)))
#         segmented_vals.insert(0, np.mean(np.array(segmented_fidelity)))
#
#     uniform_vals = np.array(uniform_vals)
#     segmented_vals = np.array(segmented_vals)
#     x = np.linspace(0,1,len(uniform_vals))
#
#
#     plt.plot(x,uniform_vals)
#     plt.plot(x,segmented_vals)
#     plt.show()


def min_psi_loss(sigma,rho):
    num_of_psis = 10000
    psi_vals = np.random.rand(num_of_psis, 2) + 1j*np.random.rand(num_of_psis, 2)
    normalization = np.sqrt(np.sum(np.abs(psi_vals)**2, axis=1))
    psi_vals = (psi_vals.T / normalization).T

    G_ideal = np.array(rho)
    G_estimated = np.array(sigma)
    G_estimated_psi = np.transpose(G_estimated@np.transpose(psi_vals))
    G_ideal_psi = np.transpose(G_ideal@np.transpose(psi_vals))
    F_psi = np.abs(np.sum(np.conj(G_ideal_psi)*G_estimated_psi,axis=1))
    return 1-min(F_psi)







def plot_logical_error_threshold(param,error=width_error,loss_func = lambda x,rho: min_psi_loss(x,rho),to_multiplot=False):
    param = torch.tensor(param)
    naive_param = naive_params_getter()
    print(error)
    optimized_threshold = loss_func(return_mat_generic(param, error_w1=error, error_w2=error, error_etching=0.15),rho=Uc_dag @ rho @ Uc_dag)
    uniform_threshold = loss_func(return_mat_generic(naive_param, error_w1=error, error_w2=error,error_etching=0.15),rho=Uc_dag @ rho @ Uc_dag)
    p = np.linspace(2*10**-5,1,1000)
    p_th = 0.57/100.0


    ys=[]
    graph_labels = []
    for d in [3,7,11,25,55]:
        de = (d+1)/2
        ys.append(0.03*((p/p_th)**de))
        graph_labels.append("d="+str(d))

    if to_multiplot:
        return p, ys, "Per-step error rate p", "Logical X error rate PL", graph_labels,False, True, [[optimized_threshold,"Error rate for segmented coupler","blue"],
                                                                                                   [uniform_threshold,"Error rate for uniform coupler","red"]]

    for i in range(len(ys)):
        plt.plot(p,ys[i],label=graph_labels[i])
    plt.axvline(x=optimized_threshold, color='blue', linestyle="--",label="Error rate for segmented coupler")
    plt.axvline(x=uniform_threshold, color='red', linestyle="--",label="Error rate for uniform coupler")
    plt.yscale('log')
    plt.xscale('log')
    plt.title("Logical error rate for width error "+str(int(100*1000*error)/100.0)+"[nm]")
    plt.xlabel("Per-step error rate p")
    plt.ylabel("Logical X error rate PL")
    plt.legend()
    plt.show()


def multiplot(funcs,params,*argv):

    font = {'family': 'serif',
            'weight': 'normal',
            'size': 20,
            }

    num_of_graphs = len(funcs)
    xs=[0]*num_of_graphs
    ys=[[0]]*num_of_graphs
    x_labels =[0]*num_of_graphs
    y_labels =[0]*num_of_graphs
    graph_labels=[[0]]*num_of_graphs
    graph_colors=[[0]]*num_of_graphs
    titles   =["("+chr(i+ord('a'))+")" for i in range(num_of_graphs)]
    log_axis =[False]*num_of_graphs
    linear_graphs = [False]*num_of_graphs
    for i in range(num_of_graphs):
        if len(argv)!=0:
            xs[i],ys[i],x_labels[i],y_labels[i],graph_labels[i],graph_colors[i],log_axis[i], linear_graphs[i] = funcs[i](params[i],argv[0][i],to_multiplot=True)
        else:
            xs[i],ys[i],x_labels[i],y_labels[i],graph_labels[i],graph_colors[i],log_axis[i], linear_graphs[i] = funcs[i](params[i],to_multiplot=True)


    graphs = []
    fig, axs = plt.subplots(max(2,num_of_graphs), 1, figsize=(10, 5))

    #
    # percentage_change = [(1-7*10**-4),(1-9*10**-4),(1-1.7*10**-4)]

    change_exp = True#False

    # params for std graph:
    percentage_change = [5,5,2]*10
    head_width = [350,350,700]*10
    head_length = [10,10,5]*10
    std_graph = True

    # params for mean fidelity graph
    # percentage_change = [(1 - 5 * 10 ** -4), (1 - 4 * 10 ** -4), (1 - 2 * 10 ** -4)]
    # head_width = [700,700,1400]
    # head_length = [5,5,5]
    mean_graph = False

    # params for deterministic plot
    # percentage_change = [(1 - 7 * 10 ** -4), (1 - 9 * 10 ** -4), (1 - 2 * 10 ** -4)]
    # head_length = [5,5,5]
    # head_width = [700,700,3500]

    arrow_direction = int(percentage_change[0]>1)*2-1


    if arrow_direction==1:
        added_dist = -10**-4
        extra_dist = 0.000035
    else:
        added_dist = 0.0001
        extra_dist = -0.00003




    for i in range(num_of_graphs):
        if std_graph:
            added_height = np.max(ys[i])/8
        else:
            added_height = 0
        for j in range(len(ys[i])):
            y=ys[i][j]

            graph_label = graph_labels[i][j]
            if change_exp:
                label_location = int(len(y)*(4**(j+1))/(4**(0.25+len(ys[i]))) + 0)
            else:
                # label_location = len(y)*(4+8*j)//30#(2+((j+1)%3)*2)//10
                # label_location = len(y)*(1+8*j)//30#(2+((j+1)%3)*2)//10
                # label_location = len(y)*(j)//100#(2+((j+1)%3)*2)//10
                if std_graph:
                    label_location = len(y)*(1+8*j)//30#(2+((j+1)%3)*2)//10
                elif mean_graph:
                    label_location = len(y)*(2+((j+1)%3)*2)//10
                else:
                    label_location = len(y) * (14 + 24 * j) // 90


            if graph_colors[i]:
                graphs.append(axs[i].plot(xs[i],y,graph_colors[i][j],label=graph_label,linewidth=3))
                axs[i].text(xs[i][label_location], percentage_change[j]*(added_height+y[label_location]), graph_label,color=graph_colors[i][j].replace("-",""), fontdict=font)
                arrow_length = y[label_location+1] - percentage_change[j]*(added_height+y[label_location])-added_dist +extra_dist
                axs[i].arrow(xs[i][label_location+1], percentage_change[j]*(added_height+y[label_location])+added_dist, 0.0, arrow_length, fc=graph_colors[i][j].replace("-",""), ec=graph_colors[i][j].replace("-",""), length_includes_head=True, head_width=arrow_length*head_width[j], head_length=-arrow_direction*arrow_length/head_length[j], width = arrow_length/50)



            else:
                print("@@",len(xs[i]),len(y),label_location, xs[i][label_location],graph_label,j)

                graphs.append(axs[i].plot(xs[i],y,label=graph_label,linewidth=3))
                # axs[i].text(xs[i][label_location], percentage_change[j]*(added_height+y[label_location]), graph_label, fontdict=font)
                # arrow_length = y[label_location+1] - percentage_change[j]*(added_height+y[label_location])-added_dist +extra_dist
                # axs[i].arrow(xs[i][label_location+1], percentage_change[j]*(added_height+y[label_location])+added_dist, 0.0, arrow_length, length_includes_head=True, head_width=arrow_length*head_width[j], head_length=-arrow_direction*arrow_length/head_length[j], width = arrow_length/50)


            axs[i].set_ylabel(y_labels[i])
            plt.xlabel(x_labels[i])
            # axs[i].set_xlabel(x_labels[i])
            axs[i].set_title(titles[i], loc='left')
            # axs[i].title.set_text(titles[i])
            if log_axis[i]:
                axs[i].set_xscale('log')
                axs[i].set_yscale('log')
        if linear_graphs[i]:
            for line, label,color in linear_graphs[i]:
                axs[i].axvline(x=line, color=color, linestyle="--", label=label,linewidth=3)

    ncol = len(ys[0])
    if linear_graphs[0]:
        ncol+=len(linear_graphs[0])
    plt.legend(bbox_to_anchor =(0.48, 2.54), loc='upper center',ncol=4, fontsize=17)
    # plt.plot()
    plt.show()


def show_graph_sketch_and_rho_crit(param):
    graph_sketch(param)
    get_rho_crit_between_segments(param)


# TEST VALUES HERE ! ##

# get_fidelity_vs_rho_between_segments([0.4123561603338839, 0.4684399760889638, 22.58339856324674, 0.44888189103572584, 0.38190578964245886, 24.790637953178546, 0.4121402359466627, 0.4684539351182967, 22.519408811200226],calculate_both=True,save_fig=False)
#
#
get_fidelity_vs_rho_between_segments_and_inside_segments([0.4123561603338839, 0.4684399760889638, 22.58339856324674, 0.44888189103572584, 0.38190578964245886, 24.790637953178546, 0.4121402359466627, 0.4684539351182967, 22.519408811200226])



# get_fidelity_vs_sigma_perfect_correlation([0.4123561603338839, 0.4684399760889638, 22.58339856324674, 0.44888189103572584, 0.38190578964245886, 24.790637953178546, 0.4121402359466627, 0.4684539351182967, 22.519408811200226],get_std=False)
#
# exit(0)
#
#
#
#
# # SMOL X^2/3
#
# graph_sketch([0.351, 0.46, 10.234, 0.459, 0.34, 17.039, 0.349, 0.46, 10.1305])

#


# get_fidelity_vs_rho_between_segments([0.4123561603338839, 0.4684399760889638, 22.58339856324674, 0.44888189103572584, 0.38190578964245886, 24.790637953178546, 0.4121402359466627, 0.4684539351182967, 22.519408811200226],calculate_mean=False)
#
#
#
# get_fidelity_vs_rho_inside_segment([0.4123561603338839, 0.4684399760889638, 22.58339856324674, 0.44888189103572584, 0.38190578964245886, 24.790637953178546, 0.4121402359466627, 0.4684539351182967, 22.519408811200226])
#
#
# get_fidelity_vs_rho_between_segments([0.4123561603338839, 0.4684399760889638, 22.58339856324674, 0.44888189103572584, 0.38190578964245886, 24.790637953178546, 0.4121402359466627, 0.4684539351182967, 22.519408811200226])
# get_fidelity_vs_rho_between_segments([0.4123561603338839, 0.4684399760889638, 22.58339856324674, 0.44888189103572584, 0.38190578964245886, 24.790637953178546, 0.4121402359466627, 0.4684539351182967, 22.519408811200226],calculate_mean=False)
#
#
# get_fidelity_vs_rho_between_segments([0.4123561603338839, 0.4684399760889638, 22.58339856324674, 0.44888189103572584, 0.38190578964245886, 24.790637953178546, 0.4121402359466627, 0.4684539351182967, 22.519408811200226],calculate_both=True,which_rho_crit_is_bigger=True)
# get_fidelity_vs_rho_between_segments([0.4123561603338839, 0.4684399760889638, 22.58339856324674, 0.44888189103572584, 0.38190578964245886, 24.790637953178546, 0.4121402359466627, 0.4684539351182967, 22.519408811200226],calculate_both=True,which_rho_crit_is_bigger=True)
#
#
# get_fidelity_vs_rho_between_segments([0.4123561603338839, 0.4684399760889638, 22.58339856324674, 0.44888189103572584, 0.38190578964245886, 24.790637953178546, 0.4121402359466627, 0.4684539351182967, 22.519408811200226],calculate_mean=False)
#
# graph_sketch([0.4123561603338839, 0.4684399760889638, 22.58339856324674, 0.44888189103572584, 0.38190578964245886, 24.790637953178546, 0.4121402359466627, 0.4684539351182967, 22.519408811200226])


# logical_error print
# params = [0.375, 0.425, 24.627, 0.429, 0.363, 26.304, 0.391, 0.45, 23.315]
# multiplot([plot_logical_error_threshold,plot_logical_error_threshold],[params,params],[width_error*3,width_error*9])
#
# # LOGICAL ERROR GRAPH!
# for num_of_error in range(1,9):
# #     # plot_logical_error_threshold([0.43, 0.452, 35.145, 0.422, 0.325, 16.761, 0.43, 0.452, 35.164],error=num_of_error*width_error)
#     plot_logical_error_threshold([0.375, 0.425, 24.627, 0.429, 0.363, 26.304, 0.391, 0.45, 23.315],error=num_of_error*width_error)

##



# correlation lines graphs

#
good_sols = [[0.4078786493638883, 0.46825936701386556, 15.945254615749228, 0.35077107480581726, 0.3310684612696264, 13.631718877358498, 0.48207103451636885, 0.43686620540451526, 25.09018774099587, 0.31122280355567394, 0.49000471060752315, 25.662718423819328], [0.4505351379994267, 0.38826609498756004, 20.456808103629374, 0.3849420585056215, 0.4776800613404739, 19.006635324787254, 0.49001695101388404, 0.4508545564072578, 29.958845706735534], [0.490000974565544, 0.416958552415462, 18.970442542944586, 0.3899026414858302, 0.48183859405086615, 17.59245241812798, 0.49001734165221905, 0.4618254362493522, 25.798576740566702, 0.42633385996692136, 0.3529567061274612, 3.4586161760376375], [0.48419692872998354, 0.49003856983729044, 31.007130139611093, 0.4899370181680883, 0.37479733481473276, 12.357913459275848, 0.35134232525604286, 0.4900167281291688, 17.677442433306904, 0.48990718540478706, 0.30999816733612084, 6.731884528051035], [0.44502072202393017, 0.4900172278030804, 24.185315968593898, 0.4899215277162655, 0.4080790495208661, 20.00226038111015, 0.42298433760226734, 0.4887881195518482, 14.007429407464803, 0.43898768709263825, 0.4898580686893114, 5.765149697934222], [0.44335111236666097, 0.4899936468732835, 15.985793474083613, 0.3802280496979356, 0.4229793106662404, 7.680617716768731, 0.4898809403963844, 0.39294393093967506, 17.177541082504973, 0.42070141242528314, 0.47774614829560974, 15.251675379270992, 0.44479695737009667, 0.4891772204236198, 7.510063597641028], [0.3557215276138723, 0.4899577979198726, 9.232087833282467, 0.49000878104298407, 0.3577984177192525, 16.821648168596198, 0.3440119436854569, 0.3966565239641836, 4.5050635382572795, 0.34976867294552366, 0.49000222718891895, 6.460410798531012, 0.4900186931345109, 0.486944114484119, 29.384425575494866], [0.4830028697174599, 0.4900217321059289, 28.31751141373733, 0.490006164356519, 0.3785727257301726, 12.564267531732812, 0.3671312732017394, 0.4899668628300347, 19.031935359672385, 0.4899958899622806, 0.3261574667558483, 3.8159248011832516, 0.4900068494021503, 0.3276903010753792, 3.249365781617045], [0.46962760643158313, 0.3923550229792377, 15.912387030422842, 0.36737765563552144, 0.4900162891767246, 14.117925464120665, 0.43378108291981954, 0.42681428866713317, 12.580100528735093, 0.4900109651640346, 0.32818071787592473, 4.393587098687932, 0.49000642870029215, 0.48999776401082334, 19.279545886820376], [0.39119535337562483, 0.3521178392335188, 9.208019507712297, 0.3347588728165619, 0.4900063507723213, 6.556188470520895, 0.3534137636550046, 0.3599002483702824, 4.667370327054041, 0.49001430756675574, 0.4875921434748693, 13.416124814866627, 0.4898713518340907, 0.39671325876058716, 16.329078846668356, 0.404685949274003, 0.48984528362040897, 14.21686386687132], [0.37532131209120495, 0.3161426480505013, 5.379485070953928, 0.3616039343345253, 0.49000615988228247, 11.394252090029093, 0.48994880302508936, 0.4471431863214312, 14.385074334685823, 0.49000649841374305, 0.38095337557331843, 6.507585960525352, 0.4899943791588328, 0.4899821150848651, 18.27808035275998, 0.3998911968012786, 0.49000396874888746, 7.025509318157876], [0.4899842384919998, 0.41353594810139693, 14.10524335559397, 0.3693541789365745, 0.35081378866467633, 5.67205007304949, 0.38645365064028625, 0.4870386526925473, 3.170834014760875, 0.39120031387460813, 0.49000153159207943, 14.17337226867034, 0.48001305679480116, 0.4289680118337063, 12.761419528978502, 0.40431317741272416, 0.3740581199785003, 14.933884435232343], [0.47138318624842696, 0.3482647157216656, 5.120144641970602, 0.36333716787210735, 0.3971423030966492, 11.642935998478178, 0.36093793674106256, 0.4720089385116773, 7.466290918649489, 0.33612672662033016, 0.33822304932363284, 10.99993459861888, 0.49000387724447786, 0.37569402336360197, 10.403414617636953, 0.39764751015045824, 0.386183116894503, 9.424708393484101, 0.37354476472685433, 0.4900026378881069, 5.5800940872382405], [0.4899805466051705, 0.42661634326195597, 11.256265296089827, 0.48991055935454475, 0.391959779313791, 7.114328874640341, 0.366497309117764, 0.46322840720850966, 6.817498878894238, 0.39674761950369064, 0.4900014333835372, 4.835321072785262, 0.3847898276207167, 0.48995989174354926, 2.268995002070706, 0.4900085707820353, 0.4899804102389883, 21.439867901053873, 0.48996830404276687, 0.4189168602634784, 10.119371932589631], [0.44230654144549336, 0.4899811221322436, 13.92913518263482, 0.387372898691488, 0.4059568831941857, 19.67951308769255, 0.4899973612538663, 0.3392339995389336, 7.457173364013847, 0.34316919037640525, 0.33169647848187234, 7.021706969518023, 0.342451700365025, 0.3100240304908447, 5.077600909531252, 0.3704533998871962, 0.4900009933778116, 6.39949543194415, 0.3925891480756989, 0.4899837984350204, 5.028518001436224], [0.4284070015163467, 0.37964035660035766, 11.922690569188072, 0.3464620661629162, 0.49001619258826357, 9.71824071929061, 0.354512581808995, 0.34214511497250816, 16.794544410873574, 0.4900053529980091, 0.35389919410868553, 7.029646861717133, 0.35082661533685017, 0.3339076156842542, 8.307550969158331, 0.37418972964595376, 0.42312665125215626, 3.100586082333871, 0.3722083823256544, 0.4899767208083484, 5.002818263234302, 0.3640782057945299, 0.4899924845058518, 0.8569761698728483], [0.49000128010538635, 0.3931246165938364, 2.701907967758845, 0.49000136872317784, 0.438260838892711, 7.4986406225053095, 0.4737954509442827, 0.49000697724473935, 15.58729631318117, 0.39316632771410975, 0.48997773349695545, 8.445485448812686, 0.3838260946930856, 0.485706119043538, 4.139986834168411, 0.46796578683119194, 0.3922892989859477, 13.753685600772021, 0.4899919632612284, 0.37387302666204586, 6.269920108665261, 0.3355822070252522, 0.46450122556581874, 4.591076326387242], [0.4899539022266312, 0.404452980279825, 11.195746180593275, 0.32611821334236185, 0.3366261196148439, 6.1490460144990395, 0.39181261440033155, 0.4816944772112681, 9.421816089881924, 0.42109674755482157, 0.4892930660409956, 6.384156560255005, 0.31000850109682954, 0.3262557888628904, 3.5583280599397638, 0.48994055157208516, 0.3954504167928782, 3.8246138818060187, 0.48996159023146885, 0.42986295534194746, 8.65342302462022, 0.48996614077601586, 0.4607113270076308, 4.653720715500921, 0.4899597792708386, 0.4900199147206831, 5.754967794784634], [0.4899451819211886, 0.3960731417578244, 10.929743096802024, 0.310004226458976, 0.310021818271634, 3.08735102452071, 0.36601608351613657, 0.4899842132612909, 9.725760170275983, 0.3746778112988212, 0.30999396774732824, 2.485312311698105, 0.3530749198081622, 0.48994530146481774, 5.1498746837007126, 0.46485583126437796, 0.3940918698832095, 1.2171206492051097, 0.47831412158575787, 0.3903019504697859, 4.822018917239791, 0.48997852519901935, 0.40254891587937597, 6.817808704825667, 0.4900098505487066, 0.49001429616623, 16.422654728766197], [0.35101987844141264, 0.3211771474862096, 8.648641996681201, 0.3438984889022184, 0.48999895590371745, 7.51427938177141, 0.34994117432849875, 0.31639711382436, 4.155331157166843, 0.3340442072927037, 0.3616206566925584, 8.089385203542578, 0.48998386939416805, 0.3655485635310682, 6.250447520362793, 0.48999006405048945, 0.38234054139577367, 2.8891756935322026, 0.4767672931387426, 0.38174158570910355, 2.5254863660451505, 0.31300151251583846, 0.31000801791287175, 8.200892542072031, 0.39165066386072966, 0.4899784127475899, 8.779511404072826], [0.4894531492131131, 0.3819190833110828, 9.227616015164491, 0.36500635196726333, 0.45640404181294175, 5.028908734162112, 0.37827862268573054, 0.4899228913599279, 1.9457870495152125, 0.4250216080947812, 0.48993671564590907, 10.120672881794706, 0.48989423055494963, 0.4899619222835277, 14.556520563685854, 0.3100675482143107, 0.3852861011996656, 0.7428292025568742, 0.48991467694192, 0.42867659812409636, 4.5732687473268765, 0.4898851561929981, 0.36757027630382383, 8.766886497692091, 0.3539306296905296, 0.4481390573566617, 3.7469835782322853, 0.37308559847802575, 0.48996269034514256, 3.455946787820463], [0.39509287001115706, 0.48997214551406826, 8.878822861131995, 0.4219297790756182, 0.3838615347658652, 9.51432243896958, 0.4900026149399389, 0.40004561625998036, 5.5822773932306315, 0.48990086910274694, 0.3970544804162401, 6.125532165663011, 0.3099955377850742, 0.36707548313834737, 2.33647785927038, 0.4899081572864703, 0.48442010833663895, 6.356498652410978, 0.42469632960298354, 0.482997181680074, 4.297900198412114, 0.3832967528621542, 0.49000498233311457, 8.989169771630385, 0.35793647486288094, 0.32260190156937346, 5.783009033613527, 0.4798253680902198, 0.3606747030058691, 1.945721648952983], [0.4899965333314182, 0.39588908948014195, 5.278478958900618, 0.4899904445310291, 0.3794310498461512, 4.402107467337745, 0.36020072639785194, 0.37112135545099095, 7.8824712281674145, 0.37517133819306986, 0.4142306719209146, 4.133334625072027, 0.38636069346256907, 0.49000662932945965, 8.946327423399106, 0.3703292528743356, 0.4243822741832125, 4.688326136724182, 0.4240605053055548, 0.3888980228142543, 4.206167930565121, 0.4262377959695018, 0.38579057741542483, 8.229241367558354, 0.4900057734601456, 0.3894813856637193, 5.762521764091104, 0.39079583937682694, 0.3966518373061701, 8.525191172615683], [0.3578061333609822, 0.48996154337556364, 5.41583488721898, 0.3340719980779991, 0.3280595115031476, 4.725794315186921, 0.3900391329884979, 0.3650874360463609, 11.019596278809376, 0.49001705593595285, 0.3731657262076613, 4.9084860653363105, 0.49000818407413677, 0.37200986782387074, 2.041899546769627, 0.37250477416705086, 0.35372997103154413, 7.824339461183544, 0.3755843710940572, 0.41258469558168664, 4.576236854884805, 0.37750685451866206, 0.4559275958124303, 2.110889680256257, 0.3705894076946486, 0.48998040441330803, 9.776222506959925, 0.3962773401900766, 0.3339109432071819, 8.178764146637773], [0.4900144519550591, 0.4538573833360233, 16.880853343479497, 0.41444824254433227, 0.4899356193522601, 20.572524122508025, 0.48992926951098825, 0.39899218769832057, 20.81415962968269, 0.34539547047110697, 0.48984673197592143, 5.803557262905264], [0.48990723865067753, 0.4024180453654614, 13.868402347551672, 0.386388382798052, 0.49000777874865087, 19.72073876797583, 0.4704736120180246, 0.3680911708850038, 9.771237387085863, 0.49001577904041893, 0.48267436609904646, 21.55767929404585], [0.48993535237861296, 0.42541399358864684, 16.061762038239653, 0.48080506791555305, 0.3920801717161837, 4.765471327671189, 0.37980323095534113, 0.48990071557299625, 13.357147743798356, 0.4900106366622313, 0.48189754646240635, 25.258451884455763, 0.4899218383435218, 0.3982811912463881, 6.642974361838336], [0.49001213171835467, 0.4870699269908171, 17.199112019138717, 0.40992670511795604, 0.4899244363176163, 12.01125425127586, 0.31000327285001095, 0.3526679070207612, 2.6626087456164087, 0.49001204549370503, 0.3831987014332968, 17.425133323296006, 0.4017766487904076, 0.4847835245538323, 13.581577447466435], [0.48673145888365305, 0.4188611324077908, 17.851778650347228, 0.37571772904447676, 0.45865411085612406, 18.13642748366388, 0.3531045064116263, 0.35523271240226556, 8.626714587530486, 0.4900025479460482, 0.3620231878605106, 6.816874945951037, 0.4898582108789675, 0.48801287328990123, 12.945452218781485], [0.4432763989856993, 0.35675843928383383, 14.479333753297789, 0.38163054787454453, 0.43924197670593257, 13.390082960635194, 0.34869202962261076, 0.43215951662516117, 12.135848461454136, 0.4393141467001658, 0.43655836225326067, 13.097585919015966, 0.41043454569166676, 0.3365459546885445, 15.283816632592151, 0.4009526454699384, 0.42498785177159065, 12.5639392694591], [0.44521953768040007, 0.4181647450389625, 8.106206393473437, 0.46541906103799235, 0.4117766033515148, 16.70889221785402, 0.3876389666803998, 0.44718264597208623, 7.807587141583117, 0.38516209941957374, 0.48993391886403703, 9.14574352838583, 0.3508924841246903, 0.3518480435147185, 8.634999145454698, 0.48991482888586024, 0.4133711060208541, 13.049433210783947], [0.48993524878142575, 0.40610425618288826, 16.361065417602777, 0.3514821309326474, 0.44735759621421894, 6.9909485848153015, 0.4322245703559765, 0.4899593907493206, 13.358584759922438, 0.4899217063600477, 0.4899193290412987, 10.54757951791599, 0.4899177101359812, 0.4363037382733211, 10.347369393169346, 0.4898148936503421, 0.44808327375856766, 7.306274298608665], [0.48986495601148855, 0.42971769702552165, 8.602156954251875, 0.4899563355879665, 0.390078444620468, 5.389734854469799, 0.3473345225225397, 0.35996818768545363, 7.974427552728339, 0.3896585684355042, 0.48993387965736945, 13.56673503575021, 0.42005487752476023, 0.3927219444189698, 12.035125510112934, 0.48993109575146826, 0.41077226504118336, 6.631850973903519, 0.44212441767450994, 0.4235360100339529, 8.327753525852456], [0.4899743178494117, 0.48994469331831, 10.845258455983455, 0.4899067705504947, 0.4069118156528337, 13.651646099919839, 0.39416262859189066, 0.46187527320614546, 12.987149617010122, 0.38933406135639864, 0.46710670687610456, 11.256713896449128, 0.48412562402835146, 0.39335100231326386, 4.106899413376413, 0.4898666073995908, 0.4107718064484896, 4.627577078738048, 0.4899063405280247, 0.42406042260117494, 6.620924607592127], [0.46919267376926477, 0.4341707806539149, 13.610671020367198, 0.4758949427310145, 0.41714270935946063, 10.946121262936689, 0.40421654016329167, 0.47747327688260877, 10.818800259776475, 0.3782823852362964, 0.49000040779355686, 5.836479936568663, 0.3300968089597959, 0.34026495289283226, 8.32000246303515, 0.4900042166235813, 0.3832125419978572, 9.072331557651816, 0.49000156729127287, 0.477487816083942, 4.632867928898483], [0.4898350101501829, 0.4410427136731721, 11.268689971745284, 0.4896473688690374, 0.426600167829901, 1.8329092085460266, 0.4711698073068761, 0.4243330728442367, 11.120440753100373, 0.3852735932455391, 0.43105706711672404, 6.614084483963657, 0.3649618439844372, 0.490005200637316, 8.847585965316862, 0.4047521748428902, 0.38844216825798916, 14.56394485413605, 0.4899584588691228, 0.4190377784700151, 11.003689843368605], [0.310703303678147, 0.48852771797531336, 3.638256057946547, 0.48996533109724305, 0.36589113838111514, 6.216038208181314, 0.48995882206227065, 0.3628370136341664, 6.233081492980115, 0.3510786336767456, 0.3693251173264606, 13.588573259737919, 0.3707925974013723, 0.4900034967592899, 9.11767949250194, 0.3135900377665224, 0.31000151672763016, 9.212640559208896, 0.48996453319395233, 0.3497874743475121, 6.295290159084871, 0.32619487215626436, 0.3492075470916044, 5.302679256312116], [0.49000183064927466, 0.48997347491237636, 8.495193754179077, 0.49000480862813284, 0.38077911220841754, 10.225431089874135, 0.3112797226564738, 0.32382046546014537, 4.158495371154267, 0.38190233040658594, 0.49000063390449533, 11.179232734307204, 0.4859239130771675, 0.4899878546421875, 5.547197033837867, 0.48993028218237733, 0.4899453229705984, 11.012332904295334, 0.4899885881577805, 0.423507062894297, 3.9695724470239253, 0.4899308408546565, 0.433271696126463, 7.6021437977625075], [0.38619471452365406, 0.48994903777470483, 4.429892711946836, 0.3828431165686722, 0.4899378900003819, 4.40590220073269, 0.3813432890079537, 0.3932414966289075, 9.532563107542636, 0.4899175810142076, 0.36245149275598837, 9.302109538137925, 0.4899214132388476, 0.4830493031906162, 13.01669092256932, 0.44616231137653367, 0.48994786724129225, 15.374246965084726, 0.4113939158522655, 0.4899747968239865, 2.6585357365561797, 0.4899714773713467, 0.48236807720386976, 5.838716140261131], [0.3891461287177071, 0.4652068987278226, 11.301677353941972, 0.48958411553516096, 0.3285245811917197, 0.32364203004258685, 0.48990716498750503, 0.3524541759448258, 10.429744576092746, 0.31017512478604564, 0.35262460085296277, 4.38808316732301, 0.4898647743259909, 0.4899410875416682, 15.845251215798255, 0.3802119270502494, 0.48990244778605274, 3.958890971083242, 0.377585028164169, 0.48981400422704824, 7.062328951134506, 0.43897199636151113, 0.4012979544420181, 5.304128401587483, 0.48991034768478237, 0.36659502672499306, 5.358644252798127], [0.4314867601709951, 0.4881618078979771, 6.448578531946578, 0.3816057143977657, 0.49001388286890557, 8.100442007412035, 0.3422656688037798, 0.3100038011005163, 7.306724668071675, 0.48989258472341557, 0.3691443864821728, 5.234724921705097, 0.3741379458411296, 0.3796448292616293, 6.768877855164719, 0.4820849055238324, 0.3995784619120344, 6.161386395470187, 0.3499185462915067, 0.3099984752766742, 4.929716853306935, 0.37148119001940066, 0.49000901954953974, 4.473320566896902, 0.43676773647596057, 0.4899181190766149, 12.886576016089082], [0.41246060332581275, 0.48087690824408136, 5.707538260408177, 0.3985375120220683, 0.4751976884000087, 3.464967697112496, 0.43229061175845795, 0.46326365201351477, 6.01876427247799, 0.4403617138967059, 0.3937917077908119, 7.461179452485023, 0.48997924386796193, 0.39516752153575285, 10.98718983602355, 0.3580955517544916, 0.36784211933404426, 8.042611715466851, 0.3915551003896971, 0.48992138563932613, 7.055272186223247, 0.4059211725070775, 0.4899000372129483, 6.886342934698332, 0.42446650422355836, 0.3530167509992757, 5.658541017588077], [0.3502686956206866, 0.41321210429780575, 8.046249110352607, 0.4899372962699584, 0.3984102961093206, 10.895134408659224, 0.4874624585927388, 0.40564617086451793, 6.316982306303645, 0.31011019576631854, 0.31009794401337226, 4.276265450331019, 0.397158335582673, 0.48990739040229814, 11.158386607758993, 0.3718644090135073, 0.48990328949097367, 2.642174763723453, 0.36268843510142545, 0.34581077139700267, 6.378561860261468, 0.3478337354315515, 0.36419901154073353, 4.45125017461364, 0.48994766689076175, 0.3682752919979976, 4.779877420750192], [0.3620202394040196, 0.30999001056964726, 6.515646479713635, 0.36743406626955344, 0.4900052647092252, 7.656445307504825, 0.358093999896412, 0.48999577361330643, 2.766133132534059, 0.3518533497701253, 0.48917358753833146, 0.15036908043975353, 0.351795702439222, 0.33770411711153775, 9.277158268072347, 0.4900059576865853, 0.48482603716799255, 4.832262330058134, 0.48995335921062644, 0.37109991550345345, 8.912939082771636, 0.31003561393315826, 0.3858598628796972, 1.496950260845474, 0.3227228338926469, 0.3100262408721275, 10.209314928474479, 0.36259414538990453, 0.4899908024913845, 6.31414593504553], [0.402168437090471, 0.3620309628562035, 9.458615658397353, 0.37618300848360064, 0.4898740554391704, 1.9873817498339852, 0.3725774649081281, 0.4898955679264603, 0.8584986950249204, 0.3770147350024901, 0.489888371069256, 7.696162244685234, 0.3402825097462799, 0.34422355308503105, 9.416145450071316, 0.4811097229537471, 0.3814966880003858, 6.402003966925297, 0.4899074691633041, 0.3913182094573099, 7.230171358634915, 0.3229558406625603, 0.35310734664043153, 4.206743482984614, 0.3783184325574192, 0.3100041739224191, 1.867111647635112, 0.40175069409807357, 0.488476401830128, 9.616480576178457], [0.4899031784110593, 0.42344058306414656, 9.7712733161996, 0.48993327493523725, 0.38695358512169176, 3.689981920104874, 0.3100158495120504, 0.3175093712536458, 5.692458765018262, 0.37813456936314754, 0.48987869699727604, 5.395286794460379, 0.38390466458169853, 0.46666221113278117, 4.709540905030441, 0.39052338429929445, 0.4898929550440216, 2.3259271717470407, 0.3542927256151439, 0.36344722764580434, 8.946821383722536, 0.489951228366107, 0.39090029970808965, 2.6369113453758564, 0.4805559538324487, 0.4050653052585458, 8.502627623389243, 0.4899543966150178, 0.49000317094378837, 8.53325093107713], [0.4710444153641434, 0.48857506128189315, 2.210158763080094, 0.38641226664656547, 0.4322206093346122, 5.370290731677069, 0.45040474328158525, 0.43878544047284596, 5.215953421436679, 0.4900017499454618, 0.3793111184408306, 4.672827014072089, 0.48987070856960707, 0.39704694341605873, 5.859512730840217, 0.3282239971838765, 0.3248881695607437, 8.972398867858317, 0.37397276282413805, 0.4900048327170915, 9.957193641305494, 0.3588216526228981, 0.3749651343556677, 10.623586700637304, 0.4760738318618033, 0.37316488319346747, 3.136377977891727, 0.48770964502553876, 0.3743340559344012, 4.398202333828513], [0.30999651455708344, 0.37752011635295607, 4.0850682201859065, 0.4899185245096112, 0.3926848181334291, 5.794807793615932, 0.48996249114477114, 0.38419235801544505, 7.062182616174948, 0.3100714595429921, 0.3153935927489086, 7.04033710403867, 0.3892211217136896, 0.48982801839509305, 7.954407203878833, 0.41244722554344926, 0.48988842802630633, 4.530721491730959, 0.3350750741688957, 0.33123027074297035, 7.944506970201824, 0.30999710669057245, 0.3611324975414733, 2.623023117678471, 0.4900013691933604, 0.38730949552247607, 4.183654714491074, 0.48985759577167115, 0.41566181675205743, 5.482964200789404],  [0.48990468136597476, 0.48989803168428364, 4.191519379859537, 0.3994472174245939, 0.48990577540073194,
              6.5700125311491435, 0.3742008321167432, 0.4899111707704912, 4.02409614728479, 0.33274044708620737,
              0.31539745029240557, 8.541264533827022, 0.4899324294143398, 0.3809384913264347, 4.065822701836131,
              0.48993798067557337, 0.38415127566643054, 2.4205322740061574, 0.4899357704616709, 0.4537286185469366,
              6.886543037530365, 0.4899071167871404, 0.48989139729098913, 5.425645958787709, 0.3556690297623494,
              0.3100608745197753, 5.389827361399279, 0.38069519960616127, 0.4899189341284121, 7.069496107849479,
              0.4443739763613496, 0.4898720603806001, 5.795467091690683],
             [0.48995295860171073, 0.3388561076340338, 3.6499271625110645, 0.35111800888820754, 0.3339653345513143,
              7.889711230596213, 0.34994861450926196, 0.4900041246265891, 1.5306021395896539, 0.3593748609946108,
              0.48997737717597034, 9.808230378428224, 0.3100562578521264, 0.3100190185565044, 6.519119051784218,
              0.4877389339602742, 0.36800291864721774, 3.0963429556269886, 0.4899677103756792, 0.370332899283357,
              5.464116962421383, 0.43627307664749926, 0.3687504664720643, 5.729207267595089, 0.3100337067897325,
              0.4451701025210608, 1.7833328076110744, 0.3100377838554978, 0.4683595517434905, 0.9413652840891314,
              0.480239453459539, 0.4899885906708005, 14.283816638306869],
             [0.4393801155005078, 0.35803512316378655, 7.518028436265498, 0.3811730873030597, 0.4900062354020321,
              8.486199039778597, 0.38211302450996576, 0.48431052416641196, 6.3009123132681, 0.3967190316893371,
              0.3774534065380249, 7.361473387135906, 0.489813475826907, 0.4028718212744727, 3.3272734437789713,
              0.48993108653590023, 0.40274454276151084, 1.997482185065881, 0.489976674797302, 0.4025778196816335,
              3.0240788524424707, 0.48994790471832833, 0.40485388881978895, 3.3804392348631507, 0.49000394380444273,
              0.4777168039419924, 9.688227933775421, 0.39720447488791494, 0.3766971118372878, 2.509140309905403,
              0.4114955717752003, 0.4900012726593989, 8.156228669776743],
             [0.36282280733778255, 0.41292980843805344, 9.96066782577591, 0.4897938236419987, 0.3738475702681613,
              4.979034299381809, 0.48992081616990096, 0.36676920864468315, 6.869463795377664, 0.32124278611821905,
              0.3393800568787, 8.096328985512475, 0.35509499352775087, 0.31000997024619004, 2.352130005394892,
              0.3637988889390188, 0.48994425658569857, 6.13102813906779, 0.3719329236657658, 0.4899058493472557,
              1.405044527875096, 0.36387710286942826, 0.4182562344630219, 5.876761644351992, 0.3100919275139154,
              0.31016949034298014, 2.743138795195032, 0.39835818036057663, 0.36598343053402227, 6.427719689506836,
              0.4899790698844862, 0.3831549660317065, 4.76670287096813],
             [0.489818820982682, 0.4033128212266339, 1.0832024977797252, 0.48997320527191784, 0.40043776381444113,
              1.0522846726690271, 0.4899528743206475, 0.41981617430528145, 7.176935067578075, 0.4898774572235727,
              0.4899782740883536, 4.411215023893114, 0.34502737180058535, 0.3297575465678402, 7.732581429629253,
              0.36787045667682544, 0.4899966790726211, 8.191888152179747, 0.36896108905232167, 0.4145837950694932,
              7.030092504255445, 0.47267366687612655, 0.4167437496806489, 5.609753194481441, 0.48997104366269073,
              0.48663883554633997, 5.813981486587041, 0.48999522166230314, 0.3622841955095757, 8.54820311711139,
              0.31637532346358477, 0.3732667115399899, 3.6147460662138027, 0.3370868218175341, 0.47855425631727827,
              1.2881072262052469],
             [0.4897038227720889, 0.42530713926840946, 9.421574190931969, 0.49000036510566336, 0.3843973077294998,
              3.163620918597408, 0.48996152579189794, 0.37328241421679276, 0.7471151041577254, 0.4900054504552517,
              0.37436555906286645, 2.951562860250104, 0.3318948794457905, 0.39886492292287423, 9.275213023313496,
              0.38263895663692676, 0.4286786386778972, 6.068390009630746, 0.4596682088192545, 0.49000886965604634,
              5.593895409001841, 0.44403701438902954, 0.478398201102938, 7.107376163836627, 0.3140489160065818,
              0.3420745531246892, 7.184108032588701, 0.49000688049726804, 0.35060114015686844, 2.644410503691117,
              0.4900124507110106, 0.3640785678698384, 6.771762268632184, 0.3100169534724131, 0.3624473796154316,
              2.2630348501774207],
             [0.35115544563894574, 0.3788064090531317, 11.74748745494703, 0.4899280181069946, 0.3515691281165707,
              5.578005533514658, 0.4899196194635862, 0.35543946447520974, 2.400418624672683, 0.36389326693757934,
              0.34895524137196865, 5.504298395768515, 0.34676700258064824, 0.39161053649695365, 3.382337984824315,
              0.35512990434566905, 0.3100779580569308, 3.21410383627609, 0.3624764484705675, 0.4899037503965818,
              4.403662566553203, 0.37272103705470677, 0.48990347704715886, 7.059788589935114, 0.3154600743751555,
              0.310064404974567, 2.6428657932225685, 0.3101758581294091, 0.3101219513786292, 0.5161667805145485,
              0.3929860186145768, 0.38894636578979025, 4.312534523562387, 0.4899201563465129, 0.40025480441078765,
              9.088543029689895],
             [0.4144604771168035, 0.49000047111827544, 2.1174756824541556, 0.36924760278477164, 0.4235108224221492,
              2.7615850819711594, 0.38118456853735394, 0.3183852705684873, 0.7182875296094094, 0.3778977656013114,
              0.4687577611657409, 6.632369464337516, 0.4095075982274415, 0.4220795931669447, 8.13762651616856,
              0.49000110322787294, 0.3667752725006263, 4.480912579648704, 0.4621797804199597, 0.37856105044234745,
              7.235696685537144, 0.48406365620287634, 0.3980423047070126, 4.8528667021914655, 0.38067184657208186,
              0.40480301194652174, 6.597804303548145, 0.403044507185543, 0.4882497403887868, 6.678750466515181,
              0.3887535332366637, 0.44121877603988835, 7.701511703435791, 0.48998193969838605, 0.49000287089180505,
              5.9273407170882555, 0.4654681548947141, 0.3176617313924301, 0.6950236303979027],
             [0.38562550517650934, 0.48993749338784365, 6.430848227809948, 0.3287497111327071, 0.3385818395114381,
              4.871379011785025, 0.4899346389724743, 0.3866203902711362, 5.035113200767746, 0.48993952403675634,
              0.3955719645687945, 4.355638948202186, 0.48999650124226624, 0.4899109217640844, 3.639185664062145,
              0.489918993075661, 0.3919141501995353, 4.816524482272421, 0.3453851989086181, 0.349283914739601,
              6.74639049827634, 0.3813827367532843, 0.4899153339224078, 3.3568873533706607, 0.3808518079514152,
              0.489976248364909, 2.7774044207877377, 0.3686483580670303, 0.49001067511704904, 4.983737785632193,
              0.3481526949923493, 0.3100255404458115, 3.2423699251756672, 0.3333429158617717, 0.37224356470122894,
              1.921349589290597, 0.4355912871675461, 0.3675537558296084, 6.547684320675409],
             [0.4899882962905715, 0.3535885512865311, 2.5176658841757775, 0.35416001734540165, 0.3339596428641694,
              8.27026820568468, 0.3695817460972922, 0.4838349088737901, 4.346521379259331, 0.3760453628965973,
              0.48998684881528, 2.823760266144447, 0.3730738522422067, 0.48991884362679167, 2.746297466751289,
              0.3660929578352296, 0.3916605393558343, 5.348727748518155, 0.3763512332261946, 0.3444333908404621,
              6.638317572613122, 0.3100005851999646, 0.35716604968768484, 2.2988817880718506, 0.48998282498413676,
              0.36602272898380067, 5.375292371733304, 0.4899120584905482, 0.37235337928144074, 4.237986243146673,
              0.3100023218545918, 0.3099942139458817, 7.572115141002483, 0.36780211862982726, 0.4898157770511252,
              1.8714194928062498, 0.37465658411543806, 0.48996689389473075, 3.704126097767125],
             [0.4899117740276517, 0.4173562786122085, 3.3885611490194134, 0.48434220694620855, 0.3938646185871623,
              7.267249497782205, 0.31006999199791907, 0.36705789107602843, 1.983018665598231, 0.3512339673275817,
              0.3397856496103971, 6.130621132343382, 0.3838309426120323, 0.48993217143166323, 5.431330726023829,
              0.3854293442735174, 0.48992066781750687, 3.7751152016977305, 0.386864157483712, 0.4899287510975005,
              1.2548700748969575, 0.37858816960474684, 0.4301610779649788, 4.105981480638493, 0.369065053947914,
              0.3578624879216868, 4.577300620428302, 0.4100525242593546, 0.3736562862541523, 6.789291682412596,
              0.4899225063957803, 0.3736899870571915, 7.615258204760439, 0.3100703734292419, 0.35545510048933493,
              3.911641716557099, 0.4104809963945954, 0.443610975426279, 2.980038725326272],
             [0.3969434339007151, 0.3838166283605892, 5.180608551419924, 0.4335617350292142, 0.4762321599876688,
              5.214746621933782, 0.39656883583176833, 0.47862141082902865, 5.742076652874075, 0.3767095418845349,
              0.4433610245849945, 5.06281404158168, 0.38892472174395387, 0.3954706402863074, 7.108303142803797,
              0.4887968933862228, 0.4059258793891961, 4.829942193969091, 0.48455229362936075, 0.3652949449354489,
              6.307149407216503, 0.4309313118230985, 0.4072938345231555, 0.4811608899246177, 0.47918133736125046,
              0.3619866878117183, 4.323476998227918, 0.3590905647672751, 0.41006860043185384, 6.81835328258069,
              0.37511757739538776, 0.4279928240580072, 7.504089169754409, 0.40257640084876073, 0.4198878565967388,
              3.179698223268595, 0.3900449605745338, 0.4789441134236008, 3.1842118060290066],
             [0.48993845780991296, 0.46247867159559225, 3.9156573238170385, 0.44884504682175297, 0.3498276036862569,
              8.388384341678822, 0.365966951276985, 0.48448800662898106, 7.958726523764687, 0.4035062386744267,
              0.4897704400477469, 6.119304227160873, 0.382116379015902, 0.44953244170969225, 7.20094222627414,
              0.481551885297713, 0.3711289624931322, 6.1520481777805465, 0.4750401132435771, 0.3560798602965143,
              0.6597419010257065, 0.3821921505252139, 0.3727044139316969, 5.5840121918150585, 0.31577241926312,
              0.36952581808689694, 1.035156686216862, 0.4439801814250082, 0.3549639734562646, 1.1442889841023802,
              0.48988057617671443, 0.35530591212384627, 4.821136388148577, 0.44916370220629814, 0.3543776859511991,
              0.35691140517280745, 0.3809937830232837, 0.37055518188986797, 4.849324445706145, 0.38811991534111984,
              0.4898063209260054, 6.661709567686004],
             [0.48993833389376973, 0.4718874804715205, 4.514716205809352, 0.39989831135486253, 0.48998854385167395,
              6.91877891103932, 0.3899951437850453, 0.31005529625117817, 1.034877799309911, 0.3848366186756516,
              0.48998806643940296, 2.8031798076885366, 0.4899756035366366, 0.48999828409253426, 6.658389770473382,
              0.3320944815981765, 0.35446916313344123, 6.881260291453245, 0.4899528456692979, 0.3585169487099415,
              2.9346892253637265, 0.4900022025150753, 0.3718955421583252, 6.101585757245995, 0.4001373997475372,
              0.35777838021800284, 6.7107993633492935, 0.4049524741059373, 0.4564510489599597, 4.022844493886169,
              0.4213025580502027, 0.4898671090532011, 6.067309438146517, 0.41733764848683363, 0.4899313510652149,
              2.374068421078144, 0.4166778407009242, 0.4899770421243824, 1.7133619535091187, 0.4269127764984008,
              0.4899910235130432, 3.527694598360418],
             [0.39678084232306055, 0.4900023100338675, 6.267154438731166, 0.37546875820305936, 0.48990193579570546,
              2.789871931591499, 0.34747582287305545, 0.30999692636166126, 4.020296391296503, 0.31000353185254653,
              0.37267003226771217, 1.7099302860890042, 0.45399773623071477, 0.3846066103090697, 6.209582734158297,
              0.48997558514967593, 0.40046434737365794, 5.896583994952589, 0.4698562421736805, 0.39998061598263207,
              7.43206357720448, 0.3722479396516952, 0.47093808934622944, 4.643690957927286, 0.37671287401628384,
              0.3100084887338415, 2.5719086127962525, 0.37877991280563567, 0.48995285842254804, 3.9202017545934056,
              0.38659817728108664, 0.48998377159844964, 1.5506337841455542, 0.3885540338469207, 0.4899400497153279,
              3.725583346509564, 0.48910324937812266, 0.4900054928428705, 6.251039959527307, 0.4072336436137037,
              0.3455693616996822, 2.9310941088629723],
             [0.4899668990416778, 0.49001034937783866, 7.323380933855662, 0.48994264734830406, 0.46518524841630127,
              1.4075659923659403, 0.4900044862068687, 0.41455115375709434, 1.7948121256655172, 0.49000561893175687,
              0.3909733381610307, 3.2917394445216797, 0.4900026993310349, 0.3740308646341102, 1.638695147456226,
              0.31000118169092034, 0.38700075402753115, 2.241173814920853, 0.48994831589090027, 0.41075622860568345,
              5.299323782174968, 0.33227542959142586, 0.31000266911326607, 5.862093079733014, 0.3592552030863834,
              0.49000217829886533, 3.858958125533565, 0.372601319145587, 0.48998635819121433, 5.971324744198975,
              0.36206227068362395, 0.391450629664837, 7.999877763351171, 0.47465325950657106, 0.3666513868993784,
              3.729103952027034, 0.3100275540511809, 0.36381252094664773, 3.1893765307540223, 0.4900034667844639,
              0.36627294085226775, 6.382074566983543],
             [0.3712467663926016, 0.48991625009133494, 5.302909116442502, 0.31002830669732023, 0.3100303321075046,
              3.319134079273292, 0.44465545068142304, 0.3706828683323477, 7.004349164606357, 0.30999005331004437,
              0.3787274888872612, 1.3993370766309554, 0.4899806640886712, 0.37514769768485773, 0.9986762658221685,
              0.48997103926039404, 0.3773831768341255, 4.673503311709043, 0.4899636803985626, 0.3691839470226588,
              2.6062472553012657, 0.31001503901888344, 0.311503310205099, 6.281483846407911, 0.37208011952693965,
              0.4707840825563536, 5.261718826062686, 0.3736744682715208, 0.49000818692270104, 4.734295967525612,
              0.37251737937418705, 0.3100176847688695, 1.100723317594778, 0.3590411406325625, 0.4899434027996144,
              2.406820599446795, 0.3462595309855887, 0.3194005756465646, 4.86533465901054, 0.48997977673202564,
              0.48610796014367175, 4.314458234767566, 0.49000122871549695, 0.37863577621625255, 3.450838284732012],
             [0.4897955401984006, 0.3957174148327672, 1.1436212975587454, 0.4899758041039294, 0.3843119720030596,
              6.635487888078458, 0.3100106748368232, 0.3603355879121082, 3.0250452540270634, 0.3507700250561794,
              0.33148083379503723, 5.00185297351096, 0.34874775898827165, 0.3100034910065267, 2.0250876550353056,
              0.37338442986543985, 0.4899414676095228, 4.072368537971539, 0.3959759202826885, 0.48993211138213755,
              4.598181063386538, 0.3896751477690483, 0.45249364000818093, 5.039011753635301, 0.4897128698723574,
              0.48984318181409714, 5.346755729390163, 0.310026731334023, 0.33884993229779276, 3.4473819628034255,
              0.4898953720475299, 0.37185242046513006, 6.645557601466017, 0.48999918472263443, 0.3712581477648796,
              3.149197853617785, 0.3100259539763263, 0.36806118327437537, 1.6133113448343237, 0.4900060608239401,
              0.42825367228335126, 3.3132585535732577, 0.3225753556734166, 0.39807976353474916, 3.203878032068917],
             [0.38017584854067243, 0.48141072614962216, 0.9736882696640031, 0.3099974932867359, 0.3587592013430203,
              5.886402247570345, 0.4900008521752486, 0.36941944128299753, 6.069016242668071, 0.4892070995075183,
              0.38779753460930433, 4.257197310824882, 0.4197901998808548, 0.36520484742733694, 4.3473565199945625,
              0.3588949627579664, 0.40433192238478605, 1.9138582657861167, 0.36908573464875, 0.35127471589526155,
              1.9085695376265175, 0.3945237706365506, 0.47283167680642696, 4.4921444474317145, 0.40390493996458837,
              0.48997594327460364, 4.157062252552505, 0.3923121267493129, 0.38068570179655287, 2.101524327201304,
              0.39323992451455664, 0.4900009668153906, 5.3021976720661845, 0.4206427542200789, 0.4299150273038307,
              8.04885698765283, 0.309997253429165, 0.36220661261025, 2.4330818796382534, 0.490003461268628,
              0.3761488521307538, 5.195873523642898, 0.48999914587442894, 0.397991354620862, 2.914695306277708],
             [0.489951657992601, 0.3930325931724361, 3.4475453025236193, 0.4899119443890703, 0.37942947723707715,
              5.126457852097229, 0.3100899856681632, 0.3385145031253572, 3.7828581214211483, 0.3879727552906764,
              0.4446987425024284, 4.699801896278443, 0.3724851125635353, 0.31003499821814945, 2.2214226627344558,
              0.37577437028289434, 0.4899211895958603, 1.4414991701979145, 0.38873358612956077, 0.4899152842777072,
              4.66619982265, 0.3844841911124241, 0.48990863928899675, 5.485773282494271, 0.3101823114619721,
              0.3100639201391435, 7.370277055194195, 0.48992990492883687, 0.3623920490530886, 3.6138396641288026,
              0.48994546201857525, 0.3650077649689547, 1.554714513445268, 0.3101821446528183, 0.36993471566201097,
              0.10897004977408405, 0.4899440989853703, 0.3558842890699913, 4.067174418084421, 0.3100901139299991,
              0.346173472830401, 2.2649438445118735, 0.3283128077110719, 0.3500957173005433, 7.346051381351361],
             [0.490000575788899, 0.4899921035960129, 3.5214560426775745, 0.3983106988433548, 0.4899508423413132,
              3.220083051256339, 0.3781863403341405, 0.30999677642849965, 3.174328497516371, 0.3646475878456975,
              0.4900031754868388, 4.994390354087703, 0.35785344816008385, 0.489976875445735, 2.6136943508325206,
              0.32825995234149763, 0.325845082807512, 4.207577348029709, 0.46158070538149504, 0.3745947603139421,
              2.7085726195819526, 0.4899755004474546, 0.37914638152086577, 2.393959502224474, 0.48999343687752356,
              0.4572579182069473, 7.807502022838592, 0.4899521238698112, 0.37784425291607837, 5.962803439842179,
              0.3183138041333211, 0.3099956752303989, 5.28869920216561, 0.3813226346074328, 0.4899313395710246,
              1.0889893511618363, 0.40102025876741404, 0.48997884892735066, 4.784504928898162, 0.41903530624334323,
              0.48999219766890434, 5.79395105973067, 0.4513047595980821, 0.48998429343194644, 1.9452313816355729],
             [0.40062014496948645, 0.31006204246649977, 2.507764734319437, 0.38591754339805795, 0.48996318932879446,
              4.599409724297714, 0.38532279583653717, 0.48994382308980083, 5.8961887473044285, 0.39148406485496645,
              0.3940136084086776, 5.753156173213197, 0.3100582319740087, 0.33325739367514395, 5.634766669617743,
              0.48994794524199814, 0.36395154929507334, 5.851835057851424, 0.48993542151775316, 0.37801509847949416,
              4.190290050605838, 0.3896394214927218, 0.370011780989303, 4.983797679192807, 0.47372511270043044,
              0.4868283126800584, 6.170958954900873, 0.4034799872666968, 0.4538885331888263, 1.237139528500086,
              0.38565809604059886, 0.4898375900908391, 1.731663588810792, 0.38786719357208405, 0.489912075172615,
              2.621418818991697, 0.3851115595417169, 0.3100515239261438, 1.109897702897866, 0.3731153054283025,
              0.48996382666932947, 0.7434959979073213, 0.38124193816481583, 0.31006329890173856, 2.0087808854761433,
              0.38506301057595504, 0.4899855567578022, 4.714436598085283],
             [0.41046928786778447, 0.489605104661124, 0.7806885706856386, 0.4134130318927906, 0.4899966826945753,
              2.3027259023391475, 0.41669371059811294, 0.4899999326135198, 1.1873359190653663, 0.42331934801436727,
              0.4899216102157327, 4.879353064205422, 0.40368302088884994, 0.45183385877193316, 2.0296688609149722,
              0.37806877263340843, 0.3948724328748575, 2.2421024355559216, 0.3340726300739092, 0.337768062541599,
              4.033853093944094, 0.4844987527582264, 0.39078055530972966, 2.4451626446664823, 0.4898193186069383,
              0.396957520725777, 1.9405966767325922, 0.48999686609098114, 0.3946115173930738, 4.7866261244652515,
              0.49000528279315897, 0.3863129293865172, 4.724370817929416, 0.3547545227504301, 0.3642475988872717,
              8.100941378516694, 0.37147481750706807, 0.4724790100269367, 1.1455936572328764, 0.40334183407264146,
              0.48999985129996076, 6.424229798278518, 0.3894290485711592, 0.4899629604998267, 5.297715045798579,
              0.40041515224628244, 0.35064121127850983, 6.9649556454399075],
             [0.4023675284606508, 0.49000265733554155, 5.084134787062392, 0.39065118906833085, 0.4321992097486418,
              5.065551490924108, 0.4892226399381262, 0.48025619610402687, 3.3087387656844265, 0.4264585308507218,
              0.4057478381750567, 3.274887721933195, 0.4899383245383475, 0.441024741782262, 3.3786316402554024,
              0.48999913171671666, 0.37353474556101673, 4.326602395345672, 0.49000167153625457, 0.41301177778011416,
              6.310267641559139, 0.3742945783671273, 0.3451774136781355, 5.398130379935082, 0.35550967871107514,
              0.4315171405506627, 3.8479486202544306, 0.39161678945984973, 0.44394461205073216, 3.0894948374333437,
              0.382523669774453, 0.46907939801415954, 5.079086229901389, 0.3718861338048957, 0.4688697114945306,
              6.40474294754189, 0.44043249773116067, 0.3919482882161755, 1.5000503407076653, 0.4530427644384866,
              0.40309892005214104, 1.2728510622315938, 0.4504123389656077, 0.38714888015658433, 0.6311495494936012,
              0.44240150885329854, 0.3539004520113282, 5.132053559507349],
             [0.4136896036227163, 0.4899586738188198, 4.455180825931625, 0.39437336875018997, 0.4899958543115649,
              1.3135610978652024, 0.3897455790404837, 0.4899928867831927, 1.1001639119177236, 0.3785927184914636,
              0.4899736537515818, 3.6201732139104577, 0.3776296161826448, 0.3100500956856053, 4.679545708927697,
              0.31415311090776105, 0.3415509828058447, 6.633712965923163, 0.48999821373408353, 0.349046316685553,
              2.0843332606492044, 0.4899864299773383, 0.3591038902135678, 5.7293447146226, 0.489948290683844,
              0.3525306764988064, 1.2962830698742798, 0.31001030318440215, 0.3329351407394597, 4.377456968352328,
              0.3258817649797357, 0.3099953671764103, 5.979559134096714, 0.35855877377146667, 0.48998165588920756,
              4.966848211755081, 0.35620709732011524, 0.4900059216799994, 2.4447961855580234, 0.3501150203065103,
              0.4899444792257663, 0.7500456985036567, 0.3465082517485396, 0.48983229831439135, 0.667876724014064,
              0.34493714863241826, 0.31001571437589864, 6.210310050348572],
             [0.4898523457797903, 0.43659225833046217, 7.073631730122408, 0.4900028046160417, 0.41565898546397606,
              1.360258180201324, 0.4899714525144321, 0.4129510510346522, 2.243788301058341, 0.48991539008675056,
              0.41015344586001673, 3.165070467262865, 0.4869162958584517, 0.39696547763224554, 3.9492197723744056,
              0.35331968959958265, 0.42610354602535533, 4.650646825899728, 0.3477789803724824, 0.48949160360072785,
              0.9001257535575183, 0.48995127361322927, 0.48999742501003246, 4.748360850174972, 0.376201401585658,
              0.48997600195941315, 3.44045614009145, 0.47877146985880953, 0.49000460835481663, 5.3101875301540815,
              0.36704120676063273, 0.4899959834485471, 2.411126132678338, 0.4817893963832175, 0.49000614475344667,
              5.201555929493592, 0.3433722787007948, 0.38173160650092264, 5.138934349031105, 0.4899966359205031,
              0.4139541661836064, 4.262166396960236, 0.4900010854315742, 0.38949318171703295, 5.276516077541331,
              0.4899116164644155, 0.4341652890126223, 4.221344840281823],
             [0.4703148466518698, 0.3538856743871915, 2.5516436881109166, 0.48945977796121265, 0.4887270442469598,
              4.004836977918679, 0.4005389974044279, 0.48880374119893555, 3.7807060792957436, 0.39842061556771585,
              0.4899808580573504, 0.6087497621179415, 0.3873159041335541, 0.489989272382653, 3.1389758457740493,
              0.3844294200140011, 0.49000739750474276, 4.534945312522921, 0.37357039377589585, 0.3317552826352288,
              3.035258926117322, 0.4899370619436404, 0.4900044125637976, 4.690256376119664, 0.3817421615322119,
              0.37737229562669405, 6.071350202812577, 0.4635747873366627, 0.36916602965716366, 1.662125597814595,
              0.4899975073989722, 0.37642407655020654, 2.6524135555110546, 0.4899386285398053, 0.3730261370128287,
              5.5991248196983525, 0.31014112486280954, 0.31000415122246666, 3.4294576133248937, 0.38493726555307234,
              0.40250817788376286, 5.227965106486146, 0.4041715074671722, 0.48990214392515125, 2.478888113202652,
              0.4123164732114947, 0.48993479102742105, 6.076829444194381, 0.4173407966591019, 0.48989210283167756,
              1.3419507535043214],
             [0.3907027739442337, 0.48990239336629293, 4.694950008804375, 0.37204639207498, 0.48994643884754474,
              4.47182168578299, 0.3541636924254441, 0.3187136623590233, 4.772621801171587, 0.4273689811437515,
              0.4897424956076338, 2.736954491710041, 0.39816369428408166, 0.38064192939118424, 2.936606402390337,
              0.4187888759216737, 0.3800239859945794, 4.5765958041668, 0.48984422424056107, 0.38460272721986605,
              2.639400881311549, 0.4898368483348456, 0.3866897483722995, 3.7782032283988576, 0.48982955228395364,
              0.38823647930595806, 0.8249418660755659, 0.4897787826850141, 0.404822625487304, 4.128269611923956,
              0.3256759580985302, 0.33357905023460527, 5.179966281112111, 0.363529065383404, 0.48977358891966727,
              3.439343052112016, 0.3692900391911298, 0.4898395915860924, 1.1459082036437729, 0.3699504109947999,
              0.31001155003944303, 3.7914148160046603, 0.3680260533431084, 0.49001495861841116, 2.115312564384521,
              0.37983506026678937, 0.48998200955101884, 3.790209546767126, 0.4140055471794777, 0.41831000777182503,
              5.498099060490211],
             [0.4452766159016302, 0.30999908173315593, 1.6235585948507665, 0.4884615761311136, 0.4899111849462316,
              3.0590686324850487, 0.3920258441218534, 0.48989685692324786, 3.8641930455111475, 0.3907136930784954,
              0.48980211720766265, 0.7882910709796422, 0.39434689571843284, 0.48989866583886843, 3.761641818911994,
              0.40088829158805384, 0.4898992135482752, 1.822900128104617, 0.4898263222208469, 0.4898928558215813,
              7.357810239810159, 0.48991454657703143, 0.48989274177887876, 8.463380557929012, 0.43551354809606396,
              0.378545146157807, 6.78187129456595, 0.48993051247258557, 0.37870809825855273, 1.7640756736884913,
              0.4899269896986662, 0.37862791710785515, 2.2686924082715296, 0.4899139699284989, 0.37413492735108866,
              3.3940588774141647, 0.32822117307751186, 0.35149264352092385, 3.884876583902032, 0.388671684431788,
              0.3129413110236938, 1.2303734322390603, 0.374904630730218, 0.48989206823195497, 6.522736615648768,
              0.38590791133505203, 0.3100394505948656, 1.5052290561383423, 0.3787538441762704, 0.48990777766431587,
              3.842028535817196],
             [0.32292192570475126, 0.41709736119703894, 4.39366524099234, 0.48342824418500613, 0.34016157151110366,
              1.182432263829919, 0.4899954472319592, 0.35808788598954205, 4.876224068879283, 0.48999368414078515,
              0.48986290976738556, 5.21933557302682, 0.47815571338764923, 0.3536556542360909, 3.9295578292132496,
              0.35558012812654594, 0.40297691014725256, 3.8659425388015176, 0.4898871167254052, 0.4900052562364777,
              6.79893057223388, 0.4844275540897663, 0.48999095404243515, 3.9410927564179468, 0.3699600675764938,
              0.48998133596085514, 1.3011857192412357, 0.37033852476009665, 0.4900012512427084, 2.0305508492147504,
              0.37269145053725344, 0.4899776546643363, 1.0184138232918805, 0.41693913230162183, 0.489909166265546,
              5.816184552311012, 0.39051544625343637, 0.445413998162367, 2.82897818573319, 0.3959016524297833,
              0.39030991973786355, 0.9732738690079409, 0.310014067404412, 0.33265900505959395, 3.86963173733186,
              0.48998975452223875, 0.37882590466810884, 4.685477005036166, 0.4899976753435406, 0.3872225866165408,
              4.702790142864391],
             [0.489980266893055, 0.3641530709354098, 2.1562190113371913, 0.4899765917253442, 0.3550023074878364,
              3.8945963474228975, 0.31004320371047306, 0.32527063010921287, 6.444916754321164, 0.36661650480750124,
              0.48994793294166705, 3.835992511800404, 0.4899162884780037, 0.489981012429214, 2.237305977781937,
              0.4004738388389593, 0.4899661182414973, 3.4242579676179776, 0.48995029572992477, 0.4899577565322769,
              7.408527977958831, 0.3881979374839574, 0.48996758422345765, 2.985733506217131, 0.317411686854281,
              0.3905538537932492, 4.107040673455837, 0.4074097669738432, 0.35115974359450225, 1.1470058349540029,
              0.3106657423285668, 0.35351302009755337, 0.14270283463412617, 0.4899496722644398, 0.3682135859943933,
              2.3044806838172525, 0.4899534815277446, 0.3792687458392563, 3.5218428143698732, 0.48995149174722347,
              0.3892848020072536, 3.1456128912677985, 0.489898427483155, 0.40325333859062806, 4.433895636110889,
              0.48717815854361807, 0.48992256017710195, 7.3933709423094776, 0.33831557286143693, 0.48993777694227947,
              2.3950238135295363],
             [0.38590472893372896, 0.48993603938107977, 0.8341948471135991, 0.4030522895638374, 0.4831842410746083,
              6.046811573539046, 0.4566016853182083, 0.3965011751601825, 3.951718381308669, 0.4899577966170705,
              0.3916585794038647, 4.355672311136602, 0.451835132348137, 0.38062588454109275, 4.493128064646023,
              0.48999524277627665, 0.38038190421103585, 1.6822119021524544, 0.4899972114514192, 0.37003339342526226,
              2.5740693757749247, 0.31001194376478425, 0.3525530405794049, 3.0291113163018215, 0.32195043511377963,
              0.31492254548801085, 5.319710939554817, 0.355802998046995, 0.48999679092315, 4.989068738635326,
              0.3612699530954457, 0.489983249518317, 0.4825174914901039, 0.35800575747027796, 0.4899771116791891,
              3.438346896270523, 0.3434399095017107, 0.334499621803856, 2.0306895051091316, 0.3981186720223307,
              0.3100101182177344, 1.912121374205936, 0.32639826356457335, 0.4539059780356436, 1.5515256725760347,
              0.39344479113467234, 0.4355042373861547, 4.468958119569352, 0.4779161684217916, 0.37981302289358626,
              4.727694666017201, 0.443729735903566, 0.37979529304179127, 5.2895034131106575],
             [0.35015440751492344, 0.451137561034985, 5.790678873804662, 0.4322909347785997, 0.3798400844730429,
              4.7609811474465875, 0.48985071430327, 0.4007121320715922, 4.321923360743322, 0.4898593900974593,
              0.40110815931219895, 3.7708079561711902, 0.4898839195811101, 0.40500009099123024, 1.2012379786554415,
              0.4883005264516337, 0.40526969125383355, 1.5114890533577268, 0.48991643062643636, 0.48984632018953833,
              8.359627803111014, 0.36716474924462417, 0.3351434259950028, 5.083667180461027, 0.37174962030876063,
              0.489865407654502, 1.3537341699062726, 0.37818202508731685, 0.4898814747362288, 1.5641658882851535,
              0.38137541681655, 0.48990996958946026, 1.5501167787160304, 0.38366444379202186, 0.4898838245891414,
              3.783024312848283, 0.3842472489029418, 0.48986150126329686, 1.8215114495296798, 0.36536321391177345,
              0.4258685412592016, 4.809722244589718, 0.3100294967525463, 0.31019176841977614, 1.8586644071395668,
              0.44764609909616093, 0.3746059493462686, 5.089017111690682, 0.48993506119578595, 0.39070156975341774,
              4.15835293104023, 0.4858939769637265, 0.38831766632663317, 0.0777273024364779],
             [0.3980805350557421, 0.4899787501062282, 4.422517161753464, 0.3870452126126357, 0.4899583182117732,
              4.296146489279812, 0.35902455046317205, 0.33998953516040037, 2.0571419809607967, 0.33524126517533637,
              0.31009797768473646, 2.351206456869212, 0.4090451078448584, 0.3275998627028469, 1.2238720996523589,
              0.44606220420201736, 0.3426582214618742, 0.5858540017974154, 0.4899784615571005, 0.48729358184236815,
              6.681014257834539, 0.4899702486132299, 0.437532552926448, 3.3912644742977975, 0.48996327511457916,
              0.38809853593880667, 4.125620494237734, 0.48995754009759834, 0.3785596092142466, 4.4615227377158,
              0.31000292852120254, 0.3379850822087992, 5.335498634086355, 0.35805442503385726, 0.3100130346461882,
              2.1547017175132024, 0.3680362051434657, 0.48995905694121134, 3.3335732642006284, 0.3705013380170254,
              0.48997988521614666, 1.7395043649071542, 0.36648159241109646, 0.48999015233999904, 1.8015233063894316,
              0.3581825849728614, 0.48999230644706004, 3.321143784752059, 0.355324330581549, 0.31003628255145604,
              4.671768205779813, 0.40640677602222985, 0.3542746578869731, 1.9819296902002526],
             [0.4064854628796029, 0.48994424162330225, 5.177563041901484, 0.4238614500212057, 0.48991832367662386,
              5.530881626013334, 0.43674375173308716, 0.44407649893289897, 5.113357018957098, 0.43338375214708075,
              0.39338530688372986, 2.4176745471823433, 0.48993715868527904, 0.38642330917029444, 3.9942273474353134,
              0.4899524663973754, 0.3699807281806338, 3.7134478817384804, 0.4899530973508331, 0.35815184939774714,
              3.1759381617512314, 0.31003406955968493, 0.38114719273783154, 3.2485970666383452, 0.4555976997980131,
              0.3450735649490541, 0.47766722273993517, 0.44796934170026415, 0.38510576251322537, 2.8727714780182647,
              0.4894364473597657, 0.48994762275707965, 2.384779522324738, 0.38798585325755974, 0.3106740207616243,
              0.05822912921940415, 0.36300545881027796, 0.3980118157052183, 4.681082583294394, 0.3652565397900613,
              0.48996880998974157, 1.229535023632604, 0.3642025092895489, 0.48996748170242965, 5.675676241377136,
              0.37254777149009277, 0.310022774770276, 2.6323932089121516, 0.35712076481777216, 0.43442891900380326,
              5.5073246003783645, 0.489939469598939, 0.3480965330653638, 3.766537573761024],
             [0.38125137005444737, 0.31012476801232597, 4.162769585203381, 0.36184508162830614, 0.4899666615222176,
              3.0432099603290346, 0.3700820102589543, 0.4899593385396435, 3.3518741939994947, 0.36812809621028936,
              0.31003618901802543, 3.4923564102585907, 0.3636448369020776, 0.48995260730756535, 4.5380475138743925,
              0.3695249252887852, 0.4898752912805532, 0.9098077195031423, 0.3338752401802274, 0.3581592009169235,
              5.803355268462237, 0.4835287055812533, 0.3653269613585473, 0.6551055384836004, 0.4899139553288808,
              0.38358967804838706, 1.449683291646256, 0.489931610577101, 0.38578843194175394, 1.449529400170093,
              0.4899723347821474, 0.39215272838550996, 2.720829390418149, 0.48996007952892473, 0.3908892184753352,
              5.346001791131132, 0.4899146067712135, 0.37791809534690174, 2.2430345338818305, 0.3100013674752796,
              0.31595577057767565, 6.749832882117467, 0.34784268231877136, 0.48998210769033956, 0.899111619377041,
              0.3563945997933206, 0.3100095844479476, 2.903007062745956, 0.3772243437961096, 0.4899911781741571,
              4.269398208740584, 0.3909957068525393, 0.48997080738232146, 3.0607565703551067],
             [0.4900021109495824, 0.39988888154716407, 4.349455945378614, 0.4899916983370542, 0.3888776545932396,
              2.2908953244448447, 0.41674254543824374, 0.3847455821467088, 4.477323775720505, 0.31000532622497057,
              0.40598068031599954, 1.3221926142410472, 0.4899436358902603, 0.4899875670114639, 5.257362453849228,
              0.4343826356736509, 0.438597407567172, 0.07172799546554173, 0.4890292120675557, 0.489998270250868,
              2.7479672618082414, 0.3497955130137437, 0.31167173039923485, 3.4099844887803585, 0.3503656422486602,
              0.4899999888576157, 3.23999345764167, 0.36031876891278614, 0.48999393249699563, 3.1517773725226435,
              0.36234036453688745, 0.48996280515806295, 2.1082932306824214, 0.3531311119729841, 0.48986878827782987,
              1.3000278317864518, 0.31001347680017927, 0.3100108555761185, 6.955499741847397, 0.48998955808922057,
              0.3595840731823087, 4.3822643100704655, 0.4899911331303373, 0.359906476289549, 2.885237074255183,
              0.4898460517784112, 0.3614985235604272, 0.10948554818346078, 0.4899994416151328, 0.35359343271173804,
              1.6297058586481914, 0.30999094424157436, 0.35764873270196335, 3.862256262502461, 0.4899973699145874,
              0.47898221330792273, 5.951016504156543],
             [0.39882082540581715, 0.4899527692268382, 3.585735734082971, 0.37931116461843506, 0.31002592913107474,
              3.4189307931934763, 0.3715678947243403, 0.4244285226999398, 3.9590227519974226, 0.37994022567007085,
              0.4889272486960492, 0.10641028556792166, 0.38391997827561647, 0.48997911572526587, 1.9406051775232858,
              0.38868362545507246, 0.48995107961350237, 1.1774187716233018, 0.3974524738996894, 0.48991202303654063,
              3.264878224082926, 0.3697263715753749, 0.40282590898405163, 5.533769412493987, 0.4471245004640601,
              0.3795745191801112, 4.612064901483425, 0.48995115421238933, 0.3937280806633127, 4.985890253546467,
              0.4900081096724629, 0.38038050835274223, 5.120554323690993, 0.4899863866537076, 0.36335199532828893,
              1.0518125409408272, 0.3206002748824308, 0.33411910201227313, 6.850263675735336, 0.37239436673990495,
              0.31109057562452946, 0.11585833511063055, 0.36525830807554316, 0.31004246866049284, 1.4838195708476678,
              0.3810018420672578, 0.48997591870586443, 1.4194419583995999, 0.3949974820752353, 0.48999011555655697,
              3.7381482361379574, 0.40685333963488485, 0.4899779643215948, 3.4149830586735166, 0.4499688283130323,
              0.48993886960581806, 4.444340587162342],
             [0.48999215172858607, 0.48998625336696494, 4.990590974482651, 0.31003961456972995, 0.3721536859998495,
              2.7123779704706235, 0.310085252411554, 0.360028998225517, 0.8722006741932705, 0.4899698991444982,
              0.3538322001977074, 1.5850484510329212, 0.4899677535000737, 0.36203020217137927, 3.493522523372624,
              0.4899105734042357, 0.36305141846582745, 4.809227691836651, 0.33803399624546154, 0.40721025788634746,
              4.585577323882547, 0.3538817770394144, 0.31003109047153693, 4.7245542731318695, 0.36667692200866425,
              0.4899823577446053, 4.490858040476675, 0.37251917177901495, 0.4899822585299075, 4.882456941562206,
              0.3590163620440691, 0.3847519935155783, 3.5378141675170833, 0.358601114711356, 0.37906335426511845,
              2.7090483830253245, 0.3168167346575665, 0.35987544337083055, 0.9864424544370466, 0.4141950382050397,
              0.40070102278223024, 2.9369965001884952, 0.48456964685427284, 0.4197774386807093, 3.0296514390222793,
              0.4898885086349028, 0.41673097272462545, 2.4419882897193146, 0.48999051452431913, 0.412989256007163,
              3.5155721873237313, 0.48997991646163574, 0.4158300623719224, 1.4462437333500693, 0.4899566300601561,
              0.42267991698323387, 1.7689632675015174],
             [0.4090979348064933, 0.48999286246808377, 1.7992481116265744, 0.38815529019658973, 0.4316427176121985,
              1.1618541121802797, 0.38483134782487477, 0.48997094006341907, 3.0101440213967203, 0.4274092342417681,
              0.4757373013218335, 3.9940837245911194, 0.35487868916256776, 0.4232815897880746, 3.9373638763342163,
              0.4899275219432857, 0.4058217169528648, 4.781989627342637, 0.4899951185802263, 0.38091433582087686,
              4.499614150111107, 0.48986389239297534, 0.39143111359196436, 5.050470936554559, 0.4745798996150346,
              0.4036138394441343, 0.29168900981945445, 0.46621069787577707, 0.3646280111483138, 4.90933173830387,
              0.3379118856934313, 0.4410006877123368, 4.836089300145317, 0.4209019633706476, 0.48975506666666635,
              2.683018357543464, 0.4490801513251742, 0.4899857240640428, 1.0163078684888511, 0.4141803059907916,
              0.49000078965843813, 5.068210800221204, 0.3618494766094911, 0.4758870921858588, 3.9539763900872344,
              0.4343352415661824, 0.30999203580707735, 2.3867326760754857, 0.48998410036589995, 0.48999790508246965,
              2.2317059394933874, 0.4898911324054768, 0.43471695379201725, 3.0065153694186915, 0.47120614091275564,
              0.48999933987270067, 4.866846161542365]
             ]
# get_fidelity_vs_rho_between_segments([0.4123561603338839, 0.4684399760889638, 22.58339856324674, 0.44888189103572584, 0.38190578964245886, 24.790637953178546, 0.4121402359466627, 0.4684539351182967, 22.519408811200226],calculate_both=True,save_fig=True)

for sol in good_sols:
    get_fidelity_vs_rho_between_segments(
        sol, calculate_both=True,
        save_fig=False)

exit(0)
# rho_mean_bigger_than_rho_std = 0
# for sol in good_sols:
#     cur_rho_mean_bigger_than_rho_std = get_fidelity_vs_rho_between_segments(sol, calculate_both=True,which_rho_crit_is_bigger=True)
#     print(cur_rho_mean_bigger_than_rho_std)
#     if not cur_rho_mean_bigger_than_rho_std:
#         get_fidelity_vs_rho_between_segments(sol, calculate_both=True)
#
#     rho_mean_bigger_than_rho_std+=int(cur_rho_mean_bigger_than_rho_std)
#
# print("Total rho mean bigger than rho std:",rho_mean_bigger_than_rho_std/len(good_sols))

rho_crit_values = []
for i in range(21):
    rho_crit_values.append([])


for i  in range(len(good_sols)):
    param = good_sols[i]
    cur_rho_crit = get_rho_crit_between_segments(param,return_rho_crit=True,get_mean=True)
    rho_crit_values[len(param)//3].append(cur_rho_crit)
    print(str(100*i/len(good_sols))+"% completed")


for n in range(len(rho_crit_values)):
    print("For n=",n,"the rho_crit values are:",rho_crit_values[n])

exit(0)


# get deterministic error graph
# multiplot([graph_sketch,graph_sketch],[[0.4123561603338839,0.4684399760889638,
#                                                                                                    22.58339856324674,
#                                                                                                    0.44888189103572584,
#                                                                                                    0.38190578964245886,
#                                                                                                    24.790637953178546,
#                                                                                                    0.4121402359466627,
#                                                                                                    0.4684539351182967,
#                                                                                                    22.519408811200226],
# [0.43, 0.452, 35.145, 0.422, 0.325, 16.761, 0.43, 0.452, 35.164]],[X_matrices[1],X_matrices[0]])
#
#

# get mean fidelity graph
multiplot([get_fidelity_vs_sigma_perfect_correlation,get_fidelity_vs_sigma_perfect_correlation],[[[0.4123561603338839,
                                                                                                   0.4684399760889638,
                                                                                                   22.58339856324674,
                                                                                                   0.44888189103572584,
                                                                                                   0.38190578964245886,
                                                                                                   24.790637953178546,
                                                                                                   0.4121402359466627,
                                                                                                   0.4684539351182967,
                                                                                                   22.519408811200226],
                                                                                                  [0.4857, 0.4345,
                                                                                                   47.1117 / 2, 0.4057,
                                                                                                   0.4896, 40.5109 / 2,
                                                                                                   0.4857, 0.4345,
                                                                                                   47.1117 / 2]],
[[0.43, 0.452, 35.145, 0.422, 0.325, 16.761, 0.43, 0.452, 35.164],
                         [0.486,0.379,29.6481/2,0.31,0.5,53.75/2,0.486,0.379,29.6481/2]]
                                                                                                 ],[[X_matrices[1]],[X_matrices[0]]])



# get fidelity std graph
multiplot([get_fidelity_vs_sigma_perfect_correlation,get_fidelity_vs_sigma_perfect_correlation],[[[0.4123561603338839,
                                                                                                   0.4684399760889638,
                                                                                                   22.58339856324674,
                                                                                                   0.44888189103572584,
                                                                                                   0.38190578964245886,
                                                                                                   24.790637953178546,
                                                                                                   0.4121402359466627,
                                                                                                   0.4684539351182967,
                                                                                                   22.519408811200226],
                                                                                                  [0.4857, 0.4345,
                                                                                                   47.1117 / 2, 0.4057,
                                                                                                   0.4896, 40.5109 / 2,
                                                                                                   0.4857, 0.4345,
                                                                                                   47.1117 / 2]],
[[0.43, 0.452, 35.145, 0.422, 0.325, 16.761, 0.43, 0.452, 35.164],
                         [0.486,0.379,29.6481/2,0.31,0.5,53.75/2,0.486,0.379,29.6481/2]]
                                                                                                 ],[[X_matrices[1],True],[X_matrices[0],True]])















# graph_sketch([0.3595872305286688, 0.423966038112, 19.814614591704306, 0.4082308339826729, 0.34096396102039617, 24.391501788111917, 0.3623212599042267, 0.41735389882165125, 21.797835705477677])

get_fidelity_vs_rho_between_segments([0.36, 0.424, 19.815, 0.408, 0.341, 24.392, 0.362, 0.417, 21.798])
get_fidelity_vs_rho_between_segments([0.3595872305286688, 0.423966038112, 19.814614591704306, 0.4082308339826729, 0.34096396102039617, 24.391501788111917, 0.3623212599042267, 0.41735389882165125, 21.797835705477677])


get_fidelity_vs_rho_between_segments([0.375, 0.425, 24.627, 0.429, 0.363, 26.304, 0.391, 0.45, 23.315])
get_fidelity_vs_rho_between_segments([0.3549059865097893, 0.4076683508526762, 24.33495062380004, 0.39994931209161194, 0.3330226836023824, 26.181028787028406, 0.3508193147974641, 0.407940953597921, 23.180983161325706])
get_fidelity_vs_rho_between_segments([0.3595872305286688, 0.423966038112, 19.814614591704306, 0.4082308339826729, 0.34096396102039617, 24.391501788111917, 0.3623212599042267, 0.41735389882165125, 21.797835705477677])




get_fidelity_vs_sigma_perfect_correlation([[0.43, 0.452, 35.145, 0.422, 0.325, 16.761, 0.43, 0.452, 35.164],
[0.4338653257709232, 0.45561932450540205, 35.76781638162929, 0.4254381702476171, 0.32774182892322284, 16.72245785219494, 0.43393005449682337, 0.4556423331477066, 35.787730365263286]],two_params=True,get_std=False)
                         # [0.486,0.379,29.6481/2,0.31,0.5,53.75/2,0.486,0.379,29.6481/2]],two_params=True,get_std=False)



get_fidelity_vs_sigma_perfect_correlation([[0.43, 0.452, 35.145, 0.422, 0.325, 16.761, 0.43, 0.452, 35.164],
                         [0.486,0.379,29.6481/2,0.31,0.5,53.75/2,0.486,0.379,29.6481/2]],two_params=True,get_std=False)


get_fidelity_vs_sigma_perfect_correlation([[0.4123561603338839, 0.4684399760889638, 22.58339856324674, 0.44888189103572584, 0.38190578964245886, 24.790637953178546, 0.4121402359466627, 0.4684539351182967, 22.519408811200226],
                                          [0.4857,0.4345,47.1117/2,0.4057,0.4896,40.5109/2,0.4857,0.4345,47.1117/2]],two_params=True,get_std=True)



graph_sketch([0.43, 0.452, 35.145, 0.422, 0.325, 16.761, 0.43, 0.452, 35.164],analytic_vals=True)

graph_sketch([0.375, 0.425, 24.627, 0.429, 0.363, 26.304, 0.391, 0.45, 23.315],analytic_vals=True)




# get_fidelity_vs_sigma_perfect_correlation()









# graph_sketch([0.48, 0.326, 7.64, 0.32, 0.478, 14.201, 0.48, 0.324, 7.59])


# graph_sketch([0.486,0.379,29.6481/2,0.31,0.5,53.75/2,0.486,0.379,29.6481/2])
#
# get_fidelity_vs_sigma_perfect_correlation([[0.43, 0.452, 35.145, 0.422, 0.325, 16.761, 0.43, 0.452, 35.164],
#                          [0.486,0.379,29.6481/2,0.31,0.5,53.75/2,0.486,0.379,29.6481/2]],two_params=True)



# X SOLUTIONS

graph_sketch([0.375, 0.425, 24.627, 0.429, 0.363, 26.304, 0.391, 0.45, 23.315],analytic_vals=False)


# good_sols = [[0.4078786493638883, 0.46825936701386556, 15.945254615749228, 0.35077107480581726, 0.3310684612696264, 13.631718877358498, 0.48207103451636885, 0.43686620540451526, 25.09018774099587, 0.31122280355567394, 0.49000471060752315, 25.662718423819328], [0.4505351379994267, 0.38826609498756004, 20.456808103629374, 0.3849420585056215, 0.4776800613404739, 19.006635324787254, 0.49001695101388404, 0.4508545564072578, 29.958845706735534], [0.490000974565544, 0.416958552415462, 18.970442542944586, 0.3899026414858302, 0.48183859405086615, 17.59245241812798, 0.49001734165221905, 0.4618254362493522, 25.798576740566702, 0.42633385996692136, 0.3529567061274612, 3.4586161760376375], [0.48419692872998354, 0.49003856983729044, 31.007130139611093, 0.4899370181680883, 0.37479733481473276, 12.357913459275848, 0.35134232525604286, 0.4900167281291688, 17.677442433306904, 0.48990718540478706, 0.30999816733612084, 6.731884528051035], [0.44502072202393017, 0.4900172278030804, 24.185315968593898, 0.4899215277162655, 0.4080790495208661, 20.00226038111015, 0.42298433760226734, 0.4887881195518482, 14.007429407464803, 0.43898768709263825, 0.4898580686893114, 5.765149697934222], [0.44335111236666097, 0.4899936468732835, 15.985793474083613, 0.3802280496979356, 0.4229793106662404, 7.680617716768731, 0.4898809403963844, 0.39294393093967506, 17.177541082504973, 0.42070141242528314, 0.47774614829560974, 15.251675379270992, 0.44479695737009667, 0.4891772204236198, 7.510063597641028], [0.3557215276138723, 0.4899577979198726, 9.232087833282467, 0.49000878104298407, 0.3577984177192525, 16.821648168596198, 0.3440119436854569, 0.3966565239641836, 4.5050635382572795, 0.34976867294552366, 0.49000222718891895, 6.460410798531012, 0.4900186931345109, 0.486944114484119, 29.384425575494866], [0.4830028697174599, 0.4900217321059289, 28.31751141373733, 0.490006164356519, 0.3785727257301726, 12.564267531732812, 0.3671312732017394, 0.4899668628300347, 19.031935359672385, 0.4899958899622806, 0.3261574667558483, 3.8159248011832516, 0.4900068494021503, 0.3276903010753792, 3.249365781617045], [0.46962760643158313, 0.3923550229792377, 15.912387030422842, 0.36737765563552144, 0.4900162891767246, 14.117925464120665, 0.43378108291981954, 0.42681428866713317, 12.580100528735093, 0.4900109651640346, 0.32818071787592473, 4.393587098687932, 0.49000642870029215, 0.48999776401082334, 19.279545886820376], [0.39119535337562483, 0.3521178392335188, 9.208019507712297, 0.3347588728165619, 0.4900063507723213, 6.556188470520895, 0.3534137636550046, 0.3599002483702824, 4.667370327054041, 0.49001430756675574, 0.4875921434748693, 13.416124814866627, 0.4898713518340907, 0.39671325876058716, 16.329078846668356, 0.404685949274003, 0.48984528362040897, 14.21686386687132], [0.37532131209120495, 0.3161426480505013, 5.379485070953928, 0.3616039343345253, 0.49000615988228247, 11.394252090029093, 0.48994880302508936, 0.4471431863214312, 14.385074334685823, 0.49000649841374305, 0.38095337557331843, 6.507585960525352, 0.4899943791588328, 0.4899821150848651, 18.27808035275998, 0.3998911968012786, 0.49000396874888746, 7.025509318157876], [0.4899842384919998, 0.41353594810139693, 14.10524335559397, 0.3693541789365745, 0.35081378866467633, 5.67205007304949, 0.38645365064028625, 0.4870386526925473, 3.170834014760875, 0.39120031387460813, 0.49000153159207943, 14.17337226867034, 0.48001305679480116, 0.4289680118337063, 12.761419528978502, 0.40431317741272416, 0.3740581199785003, 14.933884435232343], [0.47138318624842696, 0.3482647157216656, 5.120144641970602, 0.36333716787210735, 0.3971423030966492, 11.642935998478178, 0.36093793674106256, 0.4720089385116773, 7.466290918649489, 0.33612672662033016, 0.33822304932363284, 10.99993459861888, 0.49000387724447786, 0.37569402336360197, 10.403414617636953, 0.39764751015045824, 0.386183116894503, 9.424708393484101, 0.37354476472685433, 0.4900026378881069, 5.5800940872382405], [0.4899805466051705, 0.42661634326195597, 11.256265296089827, 0.48991055935454475, 0.391959779313791, 7.114328874640341, 0.366497309117764, 0.46322840720850966, 6.817498878894238, 0.39674761950369064, 0.4900014333835372, 4.835321072785262, 0.3847898276207167, 0.48995989174354926, 2.268995002070706, 0.4900085707820353, 0.4899804102389883, 21.439867901053873, 0.48996830404276687, 0.4189168602634784, 10.119371932589631], [0.44230654144549336, 0.4899811221322436, 13.92913518263482, 0.387372898691488, 0.4059568831941857, 19.67951308769255, 0.4899973612538663, 0.3392339995389336, 7.457173364013847, 0.34316919037640525, 0.33169647848187234, 7.021706969518023, 0.342451700365025, 0.3100240304908447, 5.077600909531252, 0.3704533998871962, 0.4900009933778116, 6.39949543194415, 0.3925891480756989, 0.4899837984350204, 5.028518001436224], [0.4284070015163467, 0.37964035660035766, 11.922690569188072, 0.3464620661629162, 0.49001619258826357, 9.71824071929061, 0.354512581808995, 0.34214511497250816, 16.794544410873574, 0.4900053529980091, 0.35389919410868553, 7.029646861717133, 0.35082661533685017, 0.3339076156842542, 8.307550969158331, 0.37418972964595376, 0.42312665125215626, 3.100586082333871, 0.3722083823256544, 0.4899767208083484, 5.002818263234302, 0.3640782057945299, 0.4899924845058518, 0.8569761698728483], [0.49000128010538635, 0.3931246165938364, 2.701907967758845, 0.49000136872317784, 0.438260838892711, 7.4986406225053095, 0.4737954509442827, 0.49000697724473935, 15.58729631318117, 0.39316632771410975, 0.48997773349695545, 8.445485448812686, 0.3838260946930856, 0.485706119043538, 4.139986834168411, 0.46796578683119194, 0.3922892989859477, 13.753685600772021, 0.4899919632612284, 0.37387302666204586, 6.269920108665261, 0.3355822070252522, 0.46450122556581874, 4.591076326387242], [0.4899539022266312, 0.404452980279825, 11.195746180593275, 0.32611821334236185, 0.3366261196148439, 6.1490460144990395, 0.39181261440033155, 0.4816944772112681, 9.421816089881924, 0.42109674755482157, 0.4892930660409956, 6.384156560255005, 0.31000850109682954, 0.3262557888628904, 3.5583280599397638, 0.48994055157208516, 0.3954504167928782, 3.8246138818060187, 0.48996159023146885, 0.42986295534194746, 8.65342302462022, 0.48996614077601586, 0.4607113270076308, 4.653720715500921, 0.4899597792708386, 0.4900199147206831, 5.754967794784634], [0.4899451819211886, 0.3960731417578244, 10.929743096802024, 0.310004226458976, 0.310021818271634, 3.08735102452071, 0.36601608351613657, 0.4899842132612909, 9.725760170275983, 0.3746778112988212, 0.30999396774732824, 2.485312311698105, 0.3530749198081622, 0.48994530146481774, 5.1498746837007126, 0.46485583126437796, 0.3940918698832095, 1.2171206492051097, 0.47831412158575787, 0.3903019504697859, 4.822018917239791, 0.48997852519901935, 0.40254891587937597, 6.817808704825667, 0.4900098505487066, 0.49001429616623, 16.422654728766197], [0.35101987844141264, 0.3211771474862096, 8.648641996681201, 0.3438984889022184, 0.48999895590371745, 7.51427938177141, 0.34994117432849875, 0.31639711382436, 4.155331157166843, 0.3340442072927037, 0.3616206566925584, 8.089385203542578, 0.48998386939416805, 0.3655485635310682, 6.250447520362793, 0.48999006405048945, 0.38234054139577367, 2.8891756935322026, 0.4767672931387426, 0.38174158570910355, 2.5254863660451505, 0.31300151251583846, 0.31000801791287175, 8.200892542072031, 0.39165066386072966, 0.4899784127475899, 8.779511404072826], [0.4894531492131131, 0.3819190833110828, 9.227616015164491, 0.36500635196726333, 0.45640404181294175, 5.028908734162112, 0.37827862268573054, 0.4899228913599279, 1.9457870495152125, 0.4250216080947812, 0.48993671564590907, 10.120672881794706, 0.48989423055494963, 0.4899619222835277, 14.556520563685854, 0.3100675482143107, 0.3852861011996656, 0.7428292025568742, 0.48991467694192, 0.42867659812409636, 4.5732687473268765, 0.4898851561929981, 0.36757027630382383, 8.766886497692091, 0.3539306296905296, 0.4481390573566617, 3.7469835782322853, 0.37308559847802575, 0.48996269034514256, 3.455946787820463], [0.39509287001115706, 0.48997214551406826, 8.878822861131995, 0.4219297790756182, 0.3838615347658652, 9.51432243896958, 0.4900026149399389, 0.40004561625998036, 5.5822773932306315, 0.48990086910274694, 0.3970544804162401, 6.125532165663011, 0.3099955377850742, 0.36707548313834737, 2.33647785927038, 0.4899081572864703, 0.48442010833663895, 6.356498652410978, 0.42469632960298354, 0.482997181680074, 4.297900198412114, 0.3832967528621542, 0.49000498233311457, 8.989169771630385, 0.35793647486288094, 0.32260190156937346, 5.783009033613527, 0.4798253680902198, 0.3606747030058691, 1.945721648952983], [0.4899965333314182, 0.39588908948014195, 5.278478958900618, 0.4899904445310291, 0.3794310498461512, 4.402107467337745, 0.36020072639785194, 0.37112135545099095, 7.8824712281674145, 0.37517133819306986, 0.4142306719209146, 4.133334625072027, 0.38636069346256907, 0.49000662932945965, 8.946327423399106, 0.3703292528743356, 0.4243822741832125, 4.688326136724182, 0.4240605053055548, 0.3888980228142543, 4.206167930565121, 0.4262377959695018, 0.38579057741542483, 8.229241367558354, 0.4900057734601456, 0.3894813856637193, 5.762521764091104, 0.39079583937682694, 0.3966518373061701, 8.525191172615683], [0.3578061333609822, 0.48996154337556364, 5.41583488721898, 0.3340719980779991, 0.3280595115031476, 4.725794315186921, 0.3900391329884979, 0.3650874360463609, 11.019596278809376, 0.49001705593595285, 0.3731657262076613, 4.9084860653363105, 0.49000818407413677, 0.37200986782387074, 2.041899546769627, 0.37250477416705086, 0.35372997103154413, 7.824339461183544, 0.3755843710940572, 0.41258469558168664, 4.576236854884805, 0.37750685451866206, 0.4559275958124303, 2.110889680256257, 0.3705894076946486, 0.48998040441330803, 9.776222506959925, 0.3962773401900766, 0.3339109432071819, 8.178764146637773], [0.4900144519550591, 0.4538573833360233, 16.880853343479497, 0.41444824254433227, 0.4899356193522601, 20.572524122508025, 0.48992926951098825, 0.39899218769832057, 20.81415962968269, 0.34539547047110697, 0.48984673197592143, 5.803557262905264], [0.48990723865067753, 0.4024180453654614, 13.868402347551672, 0.386388382798052, 0.49000777874865087, 19.72073876797583, 0.4704736120180246, 0.3680911708850038, 9.771237387085863, 0.49001577904041893, 0.48267436609904646, 21.55767929404585], [0.48993535237861296, 0.42541399358864684, 16.061762038239653, 0.48080506791555305, 0.3920801717161837, 4.765471327671189, 0.37980323095534113, 0.48990071557299625, 13.357147743798356, 0.4900106366622313, 0.48189754646240635, 25.258451884455763, 0.4899218383435218, 0.3982811912463881, 6.642974361838336], [0.49001213171835467, 0.4870699269908171, 17.199112019138717, 0.40992670511795604, 0.4899244363176163, 12.01125425127586, 0.31000327285001095, 0.3526679070207612, 2.6626087456164087, 0.49001204549370503, 0.3831987014332968, 17.425133323296006, 0.4017766487904076, 0.4847835245538323, 13.581577447466435], [0.48673145888365305, 0.4188611324077908, 17.851778650347228, 0.37571772904447676, 0.45865411085612406, 18.13642748366388, 0.3531045064116263, 0.35523271240226556, 8.626714587530486, 0.4900025479460482, 0.3620231878605106, 6.816874945951037, 0.4898582108789675, 0.48801287328990123, 12.945452218781485], [0.4432763989856993, 0.35675843928383383, 14.479333753297789, 0.38163054787454453, 0.43924197670593257, 13.390082960635194, 0.34869202962261076, 0.43215951662516117, 12.135848461454136, 0.4393141467001658, 0.43655836225326067, 13.097585919015966, 0.41043454569166676, 0.3365459546885445, 15.283816632592151, 0.4009526454699384, 0.42498785177159065, 12.5639392694591], [0.44521953768040007, 0.4181647450389625, 8.106206393473437, 0.46541906103799235, 0.4117766033515148, 16.70889221785402, 0.3876389666803998, 0.44718264597208623, 7.807587141583117, 0.38516209941957374, 0.48993391886403703, 9.14574352838583, 0.3508924841246903, 0.3518480435147185, 8.634999145454698, 0.48991482888586024, 0.4133711060208541, 13.049433210783947], [0.48993524878142575, 0.40610425618288826, 16.361065417602777, 0.3514821309326474, 0.44735759621421894, 6.9909485848153015, 0.4322245703559765, 0.4899593907493206, 13.358584759922438, 0.4899217063600477, 0.4899193290412987, 10.54757951791599, 0.4899177101359812, 0.4363037382733211, 10.347369393169346, 0.4898148936503421, 0.44808327375856766, 7.306274298608665], [0.48986495601148855, 0.42971769702552165, 8.602156954251875, 0.4899563355879665, 0.390078444620468, 5.389734854469799, 0.3473345225225397, 0.35996818768545363, 7.974427552728339, 0.3896585684355042, 0.48993387965736945, 13.56673503575021, 0.42005487752476023, 0.3927219444189698, 12.035125510112934, 0.48993109575146826, 0.41077226504118336, 6.631850973903519, 0.44212441767450994, 0.4235360100339529, 8.327753525852456], [0.4899743178494117, 0.48994469331831, 10.845258455983455, 0.4899067705504947, 0.4069118156528337, 13.651646099919839, 0.39416262859189066, 0.46187527320614546, 12.987149617010122, 0.38933406135639864, 0.46710670687610456, 11.256713896449128, 0.48412562402835146, 0.39335100231326386, 4.106899413376413, 0.4898666073995908, 0.4107718064484896, 4.627577078738048, 0.4899063405280247, 0.42406042260117494, 6.620924607592127], [0.46919267376926477, 0.4341707806539149, 13.610671020367198, 0.4758949427310145, 0.41714270935946063, 10.946121262936689, 0.40421654016329167, 0.47747327688260877, 10.818800259776475, 0.3782823852362964, 0.49000040779355686, 5.836479936568663, 0.3300968089597959, 0.34026495289283226, 8.32000246303515, 0.4900042166235813, 0.3832125419978572, 9.072331557651816, 0.49000156729127287, 0.477487816083942, 4.632867928898483], [0.4898350101501829, 0.4410427136731721, 11.268689971745284, 0.4896473688690374, 0.426600167829901, 1.8329092085460266, 0.4711698073068761, 0.4243330728442367, 11.120440753100373, 0.3852735932455391, 0.43105706711672404, 6.614084483963657, 0.3649618439844372, 0.490005200637316, 8.847585965316862, 0.4047521748428902, 0.38844216825798916, 14.56394485413605, 0.4899584588691228, 0.4190377784700151, 11.003689843368605], [0.310703303678147, 0.48852771797531336, 3.638256057946547, 0.48996533109724305, 0.36589113838111514, 6.216038208181314, 0.48995882206227065, 0.3628370136341664, 6.233081492980115, 0.3510786336767456, 0.3693251173264606, 13.588573259737919, 0.3707925974013723, 0.4900034967592899, 9.11767949250194, 0.3135900377665224, 0.31000151672763016, 9.212640559208896, 0.48996453319395233, 0.3497874743475121, 6.295290159084871, 0.32619487215626436, 0.3492075470916044, 5.302679256312116], [0.49000183064927466, 0.48997347491237636, 8.495193754179077, 0.49000480862813284, 0.38077911220841754, 10.225431089874135, 0.3112797226564738, 0.32382046546014537, 4.158495371154267, 0.38190233040658594, 0.49000063390449533, 11.179232734307204, 0.4859239130771675, 0.4899878546421875, 5.547197033837867, 0.48993028218237733, 0.4899453229705984, 11.012332904295334, 0.4899885881577805, 0.423507062894297, 3.9695724470239253, 0.4899308408546565, 0.433271696126463, 7.6021437977625075], [0.38619471452365406, 0.48994903777470483, 4.429892711946836, 0.3828431165686722, 0.4899378900003819, 4.40590220073269, 0.3813432890079537, 0.3932414966289075, 9.532563107542636, 0.4899175810142076, 0.36245149275598837, 9.302109538137925, 0.4899214132388476, 0.4830493031906162, 13.01669092256932, 0.44616231137653367, 0.48994786724129225, 15.374246965084726, 0.4113939158522655, 0.4899747968239865, 2.6585357365561797, 0.4899714773713467, 0.48236807720386976, 5.838716140261131], [0.3891461287177071, 0.4652068987278226, 11.301677353941972, 0.48958411553516096, 0.3285245811917197, 0.32364203004258685, 0.48990716498750503, 0.3524541759448258, 10.429744576092746, 0.31017512478604564, 0.35262460085296277, 4.38808316732301, 0.4898647743259909, 0.4899410875416682, 15.845251215798255, 0.3802119270502494, 0.48990244778605274, 3.958890971083242, 0.377585028164169, 0.48981400422704824, 7.062328951134506, 0.43897199636151113, 0.4012979544420181, 5.304128401587483, 0.48991034768478237, 0.36659502672499306, 5.358644252798127], [0.4314867601709951, 0.4881618078979771, 6.448578531946578, 0.3816057143977657, 0.49001388286890557, 8.100442007412035, 0.3422656688037798, 0.3100038011005163, 7.306724668071675, 0.48989258472341557, 0.3691443864821728, 5.234724921705097, 0.3741379458411296, 0.3796448292616293, 6.768877855164719, 0.4820849055238324, 0.3995784619120344, 6.161386395470187, 0.3499185462915067, 0.3099984752766742, 4.929716853306935, 0.37148119001940066, 0.49000901954953974, 4.473320566896902, 0.43676773647596057, 0.4899181190766149, 12.886576016089082], [0.41246060332581275, 0.48087690824408136, 5.707538260408177, 0.3985375120220683, 0.4751976884000087, 3.464967697112496, 0.43229061175845795, 0.46326365201351477, 6.01876427247799, 0.4403617138967059, 0.3937917077908119, 7.461179452485023, 0.48997924386796193, 0.39516752153575285, 10.98718983602355, 0.3580955517544916, 0.36784211933404426, 8.042611715466851, 0.3915551003896971, 0.48992138563932613, 7.055272186223247, 0.4059211725070775, 0.4899000372129483, 6.886342934698332, 0.42446650422355836, 0.3530167509992757, 5.658541017588077], [0.3502686956206866, 0.41321210429780575, 8.046249110352607, 0.4899372962699584, 0.3984102961093206, 10.895134408659224, 0.4874624585927388, 0.40564617086451793, 6.316982306303645, 0.31011019576631854, 0.31009794401337226, 4.276265450331019, 0.397158335582673, 0.48990739040229814, 11.158386607758993, 0.3718644090135073, 0.48990328949097367, 2.642174763723453, 0.36268843510142545, 0.34581077139700267, 6.378561860261468, 0.3478337354315515, 0.36419901154073353, 4.45125017461364, 0.48994766689076175, 0.3682752919979976, 4.779877420750192], [0.3620202394040196, 0.30999001056964726, 6.515646479713635, 0.36743406626955344, 0.4900052647092252, 7.656445307504825, 0.358093999896412, 0.48999577361330643, 2.766133132534059, 0.3518533497701253, 0.48917358753833146, 0.15036908043975353, 0.351795702439222, 0.33770411711153775, 9.277158268072347, 0.4900059576865853, 0.48482603716799255, 4.832262330058134, 0.48995335921062644, 0.37109991550345345, 8.912939082771636, 0.31003561393315826, 0.3858598628796972, 1.496950260845474, 0.3227228338926469, 0.3100262408721275, 10.209314928474479, 0.36259414538990453, 0.4899908024913845, 6.31414593504553], [0.402168437090471, 0.3620309628562035, 9.458615658397353, 0.37618300848360064, 0.4898740554391704, 1.9873817498339852, 0.3725774649081281, 0.4898955679264603, 0.8584986950249204, 0.3770147350024901, 0.489888371069256, 7.696162244685234, 0.3402825097462799, 0.34422355308503105, 9.416145450071316, 0.4811097229537471, 0.3814966880003858, 6.402003966925297, 0.4899074691633041, 0.3913182094573099, 7.230171358634915, 0.3229558406625603, 0.35310734664043153, 4.206743482984614, 0.3783184325574192, 0.3100041739224191, 1.867111647635112, 0.40175069409807357, 0.488476401830128, 9.616480576178457], [0.4899031784110593, 0.42344058306414656, 9.7712733161996, 0.48993327493523725, 0.38695358512169176, 3.689981920104874, 0.3100158495120504, 0.3175093712536458, 5.692458765018262, 0.37813456936314754, 0.48987869699727604, 5.395286794460379, 0.38390466458169853, 0.46666221113278117, 4.709540905030441, 0.39052338429929445, 0.4898929550440216, 2.3259271717470407, 0.3542927256151439, 0.36344722764580434, 8.946821383722536, 0.489951228366107, 0.39090029970808965, 2.6369113453758564, 0.4805559538324487, 0.4050653052585458, 8.502627623389243, 0.4899543966150178, 0.49000317094378837, 8.53325093107713], [0.4710444153641434, 0.48857506128189315, 2.210158763080094, 0.38641226664656547, 0.4322206093346122, 5.370290731677069, 0.45040474328158525, 0.43878544047284596, 5.215953421436679, 0.4900017499454618, 0.3793111184408306, 4.672827014072089, 0.48987070856960707, 0.39704694341605873, 5.859512730840217, 0.3282239971838765, 0.3248881695607437, 8.972398867858317, 0.37397276282413805, 0.4900048327170915, 9.957193641305494, 0.3588216526228981, 0.3749651343556677, 10.623586700637304, 0.4760738318618033, 0.37316488319346747, 3.136377977891727, 0.48770964502553876, 0.3743340559344012, 4.398202333828513], [0.30999651455708344, 0.37752011635295607, 4.0850682201859065, 0.4899185245096112, 0.3926848181334291, 5.794807793615932, 0.48996249114477114, 0.38419235801544505, 7.062182616174948, 0.3100714595429921, 0.3153935927489086, 7.04033710403867, 0.3892211217136896, 0.48982801839509305, 7.954407203878833, 0.41244722554344926, 0.48988842802630633, 4.530721491730959, 0.3350750741688957, 0.33123027074297035, 7.944506970201824, 0.30999710669057245, 0.3611324975414733, 2.623023117678471, 0.4900013691933604, 0.38730949552247607, 4.183654714491074, 0.48985759577167115, 0.41566181675205743, 5.482964200789404]]


# get_fidelity_vs_rho_between_segments([0.358,0.375,10.727/2,0.374,0.35,16.653/2,0.436,0.45,30.176/2,0.396,0.379,10.266/2])
#
# get_fidelity_vs_rho_between_segments([0.48999215172858607, 0.48998625336696494, 4.990590974482651, 0.31003961456972995, 0.3721536859998495, 2.7123779704706235, 0.310085252411554, 0.360028998225517, 0.8722006741932705, 0.4899698991444982, 0.3538322001977074, 1.5850484510329212, 0.4899677535000737, 0.36203020217137927, 3.493522523372624, 0.4899105734042357, 0.36305141846582745, 4.809227691836651, 0.33803399624546154, 0.40721025788634746, 4.585577323882547, 0.3538817770394144, 0.31003109047153693, 4.7245542731318695, 0.36667692200866425, 0.4899823577446053, 4.490858040476675, 0.37251917177901495, 0.4899822585299075, 4.882456941562206, 0.3590163620440691, 0.3847519935155783, 3.5378141675170833, 0.358601114711356, 0.37906335426511845, 2.7090483830253245, 0.3168167346575665, 0.35987544337083055, 0.9864424544370466, 0.4141950382050397, 0.40070102278223024, 2.9369965001884952, 0.48456964685427284, 0.4197774386807093, 3.0296514390222793, 0.4898885086349028, 0.41673097272462545, 2.4419882897193146, 0.48999051452431913, 0.412989256007163, 3.5155721873237313, 0.48997991646163574, 0.4158300623719224, 1.4462437333500693, 0.4899566300601561, 0.42267991698323387, 1.7689632675015174])
#
#
# get_fidelity_vs_rho_between_segments([0.48999215172858607, 0.48998625336696494, 4.990590974482651, 0.31003961456972995, 0.3721536859998495, 2.7123779704706235, 0.310085252411554, 0.360028998225517, 0.8722006741932705, 0.4899698991444982, 0.3538322001977074, 1.5850484510329212, 0.4899677535000737, 0.36203020217137927, 3.493522523372624, 0.4899105734042357, 0.36305141846582745, 4.809227691836651, 0.33803399624546154, 0.40721025788634746, 4.585577323882547, 0.3538817770394144, 0.31003109047153693, 4.7245542731318695, 0.36667692200866425, 0.4899823577446053, 4.490858040476675, 0.37251917177901495, 0.4899822585299075, 4.882456941562206, 0.3590163620440691, 0.3847519935155783, 3.5378141675170833, 0.358601114711356, 0.37906335426511845, 2.7090483830253245, 0.3168167346575665, 0.35987544337083055, 0.9864424544370466, 0.4141950382050397, 0.40070102278223024, 2.9369965001884952, 0.48456964685427284, 0.4197774386807093, 3.0296514390222793, 0.4898885086349028, 0.41673097272462545, 2.4419882897193146, 0.48999051452431913, 0.412989256007163, 3.5155721873237313, 0.48997991646163574, 0.4158300623719224, 1.4462437333500693, 0.4899566300601561, 0.42267991698323387, 1.7689632675015174]
# )
#
#
# get_rho_crit_between_segments([0.48999215172858607, 0.48998625336696494, 4.990590974482651, 0.31003961456972995, 0.3721536859998495, 2.7123779704706235, 0.310085252411554, 0.360028998225517, 0.8722006741932705, 0.4899698991444982, 0.3538322001977074, 1.5850484510329212, 0.4899677535000737, 0.36203020217137927, 3.493522523372624, 0.4899105734042357, 0.36305141846582745, 4.809227691836651, 0.33803399624546154, 0.40721025788634746, 4.585577323882547, 0.3538817770394144, 0.31003109047153693, 4.7245542731318695, 0.36667692200866425, 0.4899823577446053, 4.490858040476675, 0.37251917177901495, 0.4899822585299075, 4.882456941562206, 0.3590163620440691, 0.3847519935155783, 3.5378141675170833, 0.358601114711356, 0.37906335426511845, 2.7090483830253245, 0.3168167346575665, 0.35987544337083055, 0.9864424544370466, 0.4141950382050397, 0.40070102278223024, 2.9369965001884952, 0.48456964685427284, 0.4197774386807093, 3.0296514390222793, 0.4898885086349028, 0.41673097272462545, 2.4419882897193146, 0.48999051452431913, 0.412989256007163, 3.5155721873237313, 0.48997991646163574, 0.4158300623719224, 1.4462437333500693, 0.4899566300601561, 0.42267991698323387, 1.7689632675015174]
# )


# good_sols = [[0.48990468136597476, 0.48989803168428364, 4.191519379859537, 0.3994472174245939, 0.48990577540073194, 6.5700125311491435, 0.3742008321167432, 0.4899111707704912, 4.02409614728479, 0.33274044708620737, 0.31539745029240557, 8.541264533827022, 0.4899324294143398, 0.3809384913264347, 4.065822701836131, 0.48993798067557337, 0.38415127566643054, 2.4205322740061574, 0.4899357704616709, 0.4537286185469366, 6.886543037530365, 0.4899071167871404, 0.48989139729098913, 5.425645958787709, 0.3556690297623494, 0.3100608745197753, 5.389827361399279, 0.38069519960616127, 0.4899189341284121, 7.069496107849479, 0.4443739763613496, 0.4898720603806001, 5.795467091690683],
# [0.48995295860171073, 0.3388561076340338, 3.6499271625110645, 0.35111800888820754, 0.3339653345513143, 7.889711230596213, 0.34994861450926196, 0.4900041246265891, 1.5306021395896539, 0.3593748609946108, 0.48997737717597034, 9.808230378428224, 0.3100562578521264, 0.3100190185565044, 6.519119051784218, 0.4877389339602742, 0.36800291864721774, 3.0963429556269886, 0.4899677103756792, 0.370332899283357, 5.464116962421383, 0.43627307664749926, 0.3687504664720643, 5.729207267595089, 0.3100337067897325, 0.4451701025210608, 1.7833328076110744, 0.3100377838554978, 0.4683595517434905, 0.9413652840891314, 0.480239453459539, 0.4899885906708005, 14.283816638306869],
# [0.4393801155005078, 0.35803512316378655, 7.518028436265498, 0.3811730873030597, 0.4900062354020321, 8.486199039778597, 0.38211302450996576, 0.48431052416641196, 6.3009123132681, 0.3967190316893371, 0.3774534065380249, 7.361473387135906, 0.489813475826907, 0.4028718212744727, 3.3272734437789713, 0.48993108653590023, 0.40274454276151084, 1.997482185065881, 0.489976674797302, 0.4025778196816335, 3.0240788524424707, 0.48994790471832833, 0.40485388881978895, 3.3804392348631507, 0.49000394380444273, 0.4777168039419924, 9.688227933775421, 0.39720447488791494, 0.3766971118372878, 2.509140309905403, 0.4114955717752003, 0.4900012726593989, 8.156228669776743],
# [0.36282280733778255, 0.41292980843805344, 9.96066782577591, 0.4897938236419987, 0.3738475702681613, 4.979034299381809, 0.48992081616990096, 0.36676920864468315, 6.869463795377664, 0.32124278611821905, 0.3393800568787, 8.096328985512475, 0.35509499352775087, 0.31000997024619004, 2.352130005394892, 0.3637988889390188, 0.48994425658569857, 6.13102813906779, 0.3719329236657658, 0.4899058493472557, 1.405044527875096, 0.36387710286942826, 0.4182562344630219, 5.876761644351992, 0.3100919275139154, 0.31016949034298014, 2.743138795195032, 0.39835818036057663, 0.36598343053402227, 6.427719689506836, 0.4899790698844862, 0.3831549660317065, 4.76670287096813],
# [0.489818820982682, 0.4033128212266339, 1.0832024977797252, 0.48997320527191784, 0.40043776381444113, 1.0522846726690271, 0.4899528743206475, 0.41981617430528145, 7.176935067578075, 0.4898774572235727, 0.4899782740883536, 4.411215023893114, 0.34502737180058535, 0.3297575465678402, 7.732581429629253, 0.36787045667682544, 0.4899966790726211, 8.191888152179747, 0.36896108905232167, 0.4145837950694932, 7.030092504255445, 0.47267366687612655, 0.4167437496806489, 5.609753194481441, 0.48997104366269073, 0.48663883554633997, 5.813981486587041, 0.48999522166230314, 0.3622841955095757, 8.54820311711139, 0.31637532346358477, 0.3732667115399899, 3.6147460662138027, 0.3370868218175341, 0.47855425631727827, 1.2881072262052469],
# [0.4897038227720889, 0.42530713926840946, 9.421574190931969, 0.49000036510566336, 0.3843973077294998, 3.163620918597408, 0.48996152579189794, 0.37328241421679276, 0.7471151041577254, 0.4900054504552517, 0.37436555906286645, 2.951562860250104, 0.3318948794457905, 0.39886492292287423, 9.275213023313496, 0.38263895663692676, 0.4286786386778972, 6.068390009630746, 0.4596682088192545, 0.49000886965604634, 5.593895409001841, 0.44403701438902954, 0.478398201102938, 7.107376163836627, 0.3140489160065818, 0.3420745531246892, 7.184108032588701, 0.49000688049726804, 0.35060114015686844, 2.644410503691117, 0.4900124507110106, 0.3640785678698384, 6.771762268632184, 0.3100169534724131, 0.3624473796154316, 2.2630348501774207],
# [0.35115544563894574, 0.3788064090531317, 11.74748745494703, 0.4899280181069946, 0.3515691281165707, 5.578005533514658, 0.4899196194635862, 0.35543946447520974, 2.400418624672683, 0.36389326693757934, 0.34895524137196865, 5.504298395768515, 0.34676700258064824, 0.39161053649695365, 3.382337984824315, 0.35512990434566905, 0.3100779580569308, 3.21410383627609, 0.3624764484705675, 0.4899037503965818, 4.403662566553203, 0.37272103705470677, 0.48990347704715886, 7.059788589935114, 0.3154600743751555, 0.310064404974567, 2.6428657932225685, 0.3101758581294091, 0.3101219513786292, 0.5161667805145485, 0.3929860186145768, 0.38894636578979025, 4.312534523562387, 0.4899201563465129, 0.40025480441078765, 9.088543029689895],
# [0.4144604771168035, 0.49000047111827544, 2.1174756824541556, 0.36924760278477164, 0.4235108224221492, 2.7615850819711594, 0.38118456853735394, 0.3183852705684873, 0.7182875296094094, 0.3778977656013114, 0.4687577611657409, 6.632369464337516, 0.4095075982274415, 0.4220795931669447, 8.13762651616856, 0.49000110322787294, 0.3667752725006263, 4.480912579648704, 0.4621797804199597, 0.37856105044234745, 7.235696685537144, 0.48406365620287634, 0.3980423047070126, 4.8528667021914655, 0.38067184657208186, 0.40480301194652174, 6.597804303548145, 0.403044507185543, 0.4882497403887868, 6.678750466515181, 0.3887535332366637, 0.44121877603988835, 7.701511703435791, 0.48998193969838605, 0.49000287089180505, 5.9273407170882555, 0.4654681548947141, 0.3176617313924301, 0.6950236303979027],
# [0.38562550517650934, 0.48993749338784365, 6.430848227809948, 0.3287497111327071, 0.3385818395114381, 4.871379011785025, 0.4899346389724743, 0.3866203902711362, 5.035113200767746, 0.48993952403675634, 0.3955719645687945, 4.355638948202186, 0.48999650124226624, 0.4899109217640844, 3.639185664062145, 0.489918993075661, 0.3919141501995353, 4.816524482272421, 0.3453851989086181, 0.349283914739601, 6.74639049827634, 0.3813827367532843, 0.4899153339224078, 3.3568873533706607, 0.3808518079514152, 0.489976248364909, 2.7774044207877377, 0.3686483580670303, 0.49001067511704904, 4.983737785632193, 0.3481526949923493, 0.3100255404458115, 3.2423699251756672, 0.3333429158617717, 0.37224356470122894, 1.921349589290597, 0.4355912871675461, 0.3675537558296084, 6.547684320675409],
# [0.4899882962905715, 0.3535885512865311, 2.5176658841757775, 0.35416001734540165, 0.3339596428641694, 8.27026820568468, 0.3695817460972922, 0.4838349088737901, 4.346521379259331, 0.3760453628965973, 0.48998684881528, 2.823760266144447, 0.3730738522422067, 0.48991884362679167, 2.746297466751289, 0.3660929578352296, 0.3916605393558343, 5.348727748518155, 0.3763512332261946, 0.3444333908404621, 6.638317572613122, 0.3100005851999646, 0.35716604968768484, 2.2988817880718506, 0.48998282498413676, 0.36602272898380067, 5.375292371733304, 0.4899120584905482, 0.37235337928144074, 4.237986243146673, 0.3100023218545918, 0.3099942139458817, 7.572115141002483, 0.36780211862982726, 0.4898157770511252, 1.8714194928062498, 0.37465658411543806, 0.48996689389473075, 3.704126097767125],
# [0.4899117740276517, 0.4173562786122085, 3.3885611490194134, 0.48434220694620855, 0.3938646185871623, 7.267249497782205, 0.31006999199791907, 0.36705789107602843, 1.983018665598231, 0.3512339673275817, 0.3397856496103971, 6.130621132343382, 0.3838309426120323, 0.48993217143166323, 5.431330726023829, 0.3854293442735174, 0.48992066781750687, 3.7751152016977305, 0.386864157483712, 0.4899287510975005, 1.2548700748969575, 0.37858816960474684, 0.4301610779649788, 4.105981480638493, 0.369065053947914, 0.3578624879216868, 4.577300620428302, 0.4100525242593546, 0.3736562862541523, 6.789291682412596, 0.4899225063957803, 0.3736899870571915, 7.615258204760439, 0.3100703734292419, 0.35545510048933493, 3.911641716557099, 0.4104809963945954, 0.443610975426279, 2.980038725326272],
# [0.3969434339007151, 0.3838166283605892, 5.180608551419924, 0.4335617350292142, 0.4762321599876688, 5.214746621933782, 0.39656883583176833, 0.47862141082902865, 5.742076652874075, 0.3767095418845349, 0.4433610245849945, 5.06281404158168, 0.38892472174395387, 0.3954706402863074, 7.108303142803797, 0.4887968933862228, 0.4059258793891961, 4.829942193969091, 0.48455229362936075, 0.3652949449354489, 6.307149407216503, 0.4309313118230985, 0.4072938345231555, 0.4811608899246177, 0.47918133736125046, 0.3619866878117183, 4.323476998227918, 0.3590905647672751, 0.41006860043185384, 6.81835328258069, 0.37511757739538776, 0.4279928240580072, 7.504089169754409, 0.40257640084876073, 0.4198878565967388, 3.179698223268595, 0.3900449605745338, 0.4789441134236008, 3.1842118060290066],
# [0.48993845780991296, 0.46247867159559225, 3.9156573238170385, 0.44884504682175297, 0.3498276036862569, 8.388384341678822, 0.365966951276985, 0.48448800662898106, 7.958726523764687, 0.4035062386744267, 0.4897704400477469, 6.119304227160873, 0.382116379015902, 0.44953244170969225, 7.20094222627414, 0.481551885297713, 0.3711289624931322, 6.1520481777805465, 0.4750401132435771, 0.3560798602965143, 0.6597419010257065, 0.3821921505252139, 0.3727044139316969, 5.5840121918150585, 0.31577241926312, 0.36952581808689694, 1.035156686216862, 0.4439801814250082, 0.3549639734562646, 1.1442889841023802, 0.48988057617671443, 0.35530591212384627, 4.821136388148577, 0.44916370220629814, 0.3543776859511991, 0.35691140517280745, 0.3809937830232837, 0.37055518188986797, 4.849324445706145, 0.38811991534111984, 0.4898063209260054, 6.661709567686004],
# [0.48993833389376973, 0.4718874804715205, 4.514716205809352, 0.39989831135486253, 0.48998854385167395, 6.91877891103932, 0.3899951437850453, 0.31005529625117817, 1.034877799309911, 0.3848366186756516, 0.48998806643940296, 2.8031798076885366, 0.4899756035366366, 0.48999828409253426, 6.658389770473382, 0.3320944815981765, 0.35446916313344123, 6.881260291453245, 0.4899528456692979, 0.3585169487099415, 2.9346892253637265, 0.4900022025150753, 0.3718955421583252, 6.101585757245995, 0.4001373997475372, 0.35777838021800284, 6.7107993633492935, 0.4049524741059373, 0.4564510489599597, 4.022844493886169, 0.4213025580502027, 0.4898671090532011, 6.067309438146517, 0.41733764848683363, 0.4899313510652149, 2.374068421078144, 0.4166778407009242, 0.4899770421243824, 1.7133619535091187, 0.4269127764984008, 0.4899910235130432, 3.527694598360418],
# [0.39678084232306055, 0.4900023100338675, 6.267154438731166, 0.37546875820305936, 0.48990193579570546, 2.789871931591499, 0.34747582287305545, 0.30999692636166126, 4.020296391296503, 0.31000353185254653, 0.37267003226771217, 1.7099302860890042, 0.45399773623071477, 0.3846066103090697, 6.209582734158297, 0.48997558514967593, 0.40046434737365794, 5.896583994952589, 0.4698562421736805, 0.39998061598263207, 7.43206357720448, 0.3722479396516952, 0.47093808934622944, 4.643690957927286, 0.37671287401628384, 0.3100084887338415, 2.5719086127962525, 0.37877991280563567, 0.48995285842254804, 3.9202017545934056, 0.38659817728108664, 0.48998377159844964, 1.5506337841455542, 0.3885540338469207, 0.4899400497153279, 3.725583346509564, 0.48910324937812266, 0.4900054928428705, 6.251039959527307, 0.4072336436137037, 0.3455693616996822, 2.9310941088629723],
# [0.4899668990416778, 0.49001034937783866, 7.323380933855662, 0.48994264734830406, 0.46518524841630127, 1.4075659923659403, 0.4900044862068687, 0.41455115375709434, 1.7948121256655172, 0.49000561893175687, 0.3909733381610307, 3.2917394445216797, 0.4900026993310349, 0.3740308646341102, 1.638695147456226, 0.31000118169092034, 0.38700075402753115, 2.241173814920853, 0.48994831589090027, 0.41075622860568345, 5.299323782174968, 0.33227542959142586, 0.31000266911326607, 5.862093079733014, 0.3592552030863834, 0.49000217829886533, 3.858958125533565, 0.372601319145587, 0.48998635819121433, 5.971324744198975, 0.36206227068362395, 0.391450629664837, 7.999877763351171, 0.47465325950657106, 0.3666513868993784, 3.729103952027034, 0.3100275540511809, 0.36381252094664773, 3.1893765307540223, 0.4900034667844639, 0.36627294085226775, 6.382074566983543],
# [0.3712467663926016, 0.48991625009133494, 5.302909116442502, 0.31002830669732023, 0.3100303321075046, 3.319134079273292, 0.44465545068142304, 0.3706828683323477, 7.004349164606357, 0.30999005331004437, 0.3787274888872612, 1.3993370766309554, 0.4899806640886712, 0.37514769768485773, 0.9986762658221685, 0.48997103926039404, 0.3773831768341255, 4.673503311709043, 0.4899636803985626, 0.3691839470226588, 2.6062472553012657, 0.31001503901888344, 0.311503310205099, 6.281483846407911, 0.37208011952693965, 0.4707840825563536, 5.261718826062686, 0.3736744682715208, 0.49000818692270104, 4.734295967525612, 0.37251737937418705, 0.3100176847688695, 1.100723317594778, 0.3590411406325625, 0.4899434027996144, 2.406820599446795, 0.3462595309855887, 0.3194005756465646, 4.86533465901054, 0.48997977673202564, 0.48610796014367175, 4.314458234767566, 0.49000122871549695, 0.37863577621625255, 3.450838284732012],
# [0.4897955401984006, 0.3957174148327672, 1.1436212975587454, 0.4899758041039294, 0.3843119720030596, 6.635487888078458, 0.3100106748368232, 0.3603355879121082, 3.0250452540270634, 0.3507700250561794, 0.33148083379503723, 5.00185297351096, 0.34874775898827165, 0.3100034910065267, 2.0250876550353056, 0.37338442986543985, 0.4899414676095228, 4.072368537971539, 0.3959759202826885, 0.48993211138213755, 4.598181063386538, 0.3896751477690483, 0.45249364000818093, 5.039011753635301, 0.4897128698723574, 0.48984318181409714, 5.346755729390163, 0.310026731334023, 0.33884993229779276, 3.4473819628034255, 0.4898953720475299, 0.37185242046513006, 6.645557601466017, 0.48999918472263443, 0.3712581477648796, 3.149197853617785, 0.3100259539763263, 0.36806118327437537, 1.6133113448343237, 0.4900060608239401, 0.42825367228335126, 3.3132585535732577, 0.3225753556734166, 0.39807976353474916, 3.203878032068917],
# [0.38017584854067243, 0.48141072614962216, 0.9736882696640031, 0.3099974932867359, 0.3587592013430203, 5.886402247570345, 0.4900008521752486, 0.36941944128299753, 6.069016242668071, 0.4892070995075183, 0.38779753460930433, 4.257197310824882, 0.4197901998808548, 0.36520484742733694, 4.3473565199945625, 0.3588949627579664, 0.40433192238478605, 1.9138582657861167, 0.36908573464875, 0.35127471589526155, 1.9085695376265175, 0.3945237706365506, 0.47283167680642696, 4.4921444474317145, 0.40390493996458837, 0.48997594327460364, 4.157062252552505, 0.3923121267493129, 0.38068570179655287, 2.101524327201304, 0.39323992451455664, 0.4900009668153906, 5.3021976720661845, 0.4206427542200789, 0.4299150273038307, 8.04885698765283, 0.309997253429165, 0.36220661261025, 2.4330818796382534, 0.490003461268628, 0.3761488521307538, 5.195873523642898, 0.48999914587442894, 0.397991354620862, 2.914695306277708],
# [0.489951657992601, 0.3930325931724361, 3.4475453025236193, 0.4899119443890703, 0.37942947723707715, 5.126457852097229, 0.3100899856681632, 0.3385145031253572, 3.7828581214211483, 0.3879727552906764, 0.4446987425024284, 4.699801896278443, 0.3724851125635353, 0.31003499821814945, 2.2214226627344558, 0.37577437028289434, 0.4899211895958603, 1.4414991701979145, 0.38873358612956077, 0.4899152842777072, 4.66619982265, 0.3844841911124241, 0.48990863928899675, 5.485773282494271, 0.3101823114619721, 0.3100639201391435, 7.370277055194195, 0.48992990492883687, 0.3623920490530886, 3.6138396641288026, 0.48994546201857525, 0.3650077649689547, 1.554714513445268, 0.3101821446528183, 0.36993471566201097, 0.10897004977408405, 0.4899440989853703, 0.3558842890699913, 4.067174418084421, 0.3100901139299991, 0.346173472830401, 2.2649438445118735, 0.3283128077110719, 0.3500957173005433, 7.346051381351361],
# [0.490000575788899, 0.4899921035960129, 3.5214560426775745, 0.3983106988433548, 0.4899508423413132, 3.220083051256339, 0.3781863403341405, 0.30999677642849965, 3.174328497516371, 0.3646475878456975, 0.4900031754868388, 4.994390354087703, 0.35785344816008385, 0.489976875445735, 2.6136943508325206, 0.32825995234149763, 0.325845082807512, 4.207577348029709, 0.46158070538149504, 0.3745947603139421, 2.7085726195819526, 0.4899755004474546, 0.37914638152086577, 2.393959502224474, 0.48999343687752356, 0.4572579182069473, 7.807502022838592, 0.4899521238698112, 0.37784425291607837, 5.962803439842179, 0.3183138041333211, 0.3099956752303989, 5.28869920216561, 0.3813226346074328, 0.4899313395710246, 1.0889893511618363, 0.40102025876741404, 0.48997884892735066, 4.784504928898162, 0.41903530624334323, 0.48999219766890434, 5.79395105973067, 0.4513047595980821, 0.48998429343194644, 1.9452313816355729],
# [0.40062014496948645, 0.31006204246649977, 2.507764734319437, 0.38591754339805795, 0.48996318932879446, 4.599409724297714, 0.38532279583653717, 0.48994382308980083, 5.8961887473044285, 0.39148406485496645, 0.3940136084086776, 5.753156173213197, 0.3100582319740087, 0.33325739367514395, 5.634766669617743, 0.48994794524199814, 0.36395154929507334, 5.851835057851424, 0.48993542151775316, 0.37801509847949416, 4.190290050605838, 0.3896394214927218, 0.370011780989303, 4.983797679192807, 0.47372511270043044, 0.4868283126800584, 6.170958954900873, 0.4034799872666968, 0.4538885331888263, 1.237139528500086, 0.38565809604059886, 0.4898375900908391, 1.731663588810792, 0.38786719357208405, 0.489912075172615, 2.621418818991697, 0.3851115595417169, 0.3100515239261438, 1.109897702897866, 0.3731153054283025, 0.48996382666932947, 0.7434959979073213, 0.38124193816481583, 0.31006329890173856, 2.0087808854761433, 0.38506301057595504, 0.4899855567578022, 4.714436598085283],
# [0.41046928786778447, 0.489605104661124, 0.7806885706856386, 0.4134130318927906, 0.4899966826945753, 2.3027259023391475, 0.41669371059811294, 0.4899999326135198, 1.1873359190653663, 0.42331934801436727, 0.4899216102157327, 4.879353064205422, 0.40368302088884994, 0.45183385877193316, 2.0296688609149722, 0.37806877263340843, 0.3948724328748575, 2.2421024355559216, 0.3340726300739092, 0.337768062541599, 4.033853093944094, 0.4844987527582264, 0.39078055530972966, 2.4451626446664823, 0.4898193186069383, 0.396957520725777, 1.9405966767325922, 0.48999686609098114, 0.3946115173930738, 4.7866261244652515, 0.49000528279315897, 0.3863129293865172, 4.724370817929416, 0.3547545227504301, 0.3642475988872717, 8.100941378516694, 0.37147481750706807, 0.4724790100269367, 1.1455936572328764, 0.40334183407264146, 0.48999985129996076, 6.424229798278518, 0.3894290485711592, 0.4899629604998267, 5.297715045798579, 0.40041515224628244, 0.35064121127850983, 6.9649556454399075],
# [0.4023675284606508, 0.49000265733554155, 5.084134787062392, 0.39065118906833085, 0.4321992097486418, 5.065551490924108, 0.4892226399381262, 0.48025619610402687, 3.3087387656844265, 0.4264585308507218, 0.4057478381750567, 3.274887721933195, 0.4899383245383475, 0.441024741782262, 3.3786316402554024, 0.48999913171671666, 0.37353474556101673, 4.326602395345672, 0.49000167153625457, 0.41301177778011416, 6.310267641559139, 0.3742945783671273, 0.3451774136781355, 5.398130379935082, 0.35550967871107514, 0.4315171405506627, 3.8479486202544306, 0.39161678945984973, 0.44394461205073216, 3.0894948374333437, 0.382523669774453, 0.46907939801415954, 5.079086229901389, 0.3718861338048957, 0.4688697114945306, 6.40474294754189, 0.44043249773116067, 0.3919482882161755, 1.5000503407076653, 0.4530427644384866, 0.40309892005214104, 1.2728510622315938, 0.4504123389656077, 0.38714888015658433, 0.6311495494936012, 0.44240150885329854, 0.3539004520113282, 5.132053559507349],
# [0.4136896036227163, 0.4899586738188198, 4.455180825931625, 0.39437336875018997, 0.4899958543115649, 1.3135610978652024, 0.3897455790404837, 0.4899928867831927, 1.1001639119177236, 0.3785927184914636, 0.4899736537515818, 3.6201732139104577, 0.3776296161826448, 0.3100500956856053, 4.679545708927697, 0.31415311090776105, 0.3415509828058447, 6.633712965923163, 0.48999821373408353, 0.349046316685553, 2.0843332606492044, 0.4899864299773383, 0.3591038902135678, 5.7293447146226, 0.489948290683844, 0.3525306764988064, 1.2962830698742798, 0.31001030318440215, 0.3329351407394597, 4.377456968352328, 0.3258817649797357, 0.3099953671764103, 5.979559134096714, 0.35855877377146667, 0.48998165588920756, 4.966848211755081, 0.35620709732011524, 0.4900059216799994, 2.4447961855580234, 0.3501150203065103, 0.4899444792257663, 0.7500456985036567, 0.3465082517485396, 0.48983229831439135, 0.667876724014064, 0.34493714863241826, 0.31001571437589864, 6.210310050348572],
# [0.4898523457797903, 0.43659225833046217, 7.073631730122408, 0.4900028046160417, 0.41565898546397606, 1.360258180201324, 0.4899714525144321, 0.4129510510346522, 2.243788301058341, 0.48991539008675056, 0.41015344586001673, 3.165070467262865, 0.4869162958584517, 0.39696547763224554, 3.9492197723744056, 0.35331968959958265, 0.42610354602535533, 4.650646825899728, 0.3477789803724824, 0.48949160360072785, 0.9001257535575183, 0.48995127361322927, 0.48999742501003246, 4.748360850174972, 0.376201401585658, 0.48997600195941315, 3.44045614009145, 0.47877146985880953, 0.49000460835481663, 5.3101875301540815, 0.36704120676063273, 0.4899959834485471, 2.411126132678338, 0.4817893963832175, 0.49000614475344667, 5.201555929493592, 0.3433722787007948, 0.38173160650092264, 5.138934349031105, 0.4899966359205031, 0.4139541661836064, 4.262166396960236, 0.4900010854315742, 0.38949318171703295, 5.276516077541331, 0.4899116164644155, 0.4341652890126223, 4.221344840281823],
# [0.4703148466518698, 0.3538856743871915, 2.5516436881109166, 0.48945977796121265, 0.4887270442469598, 4.004836977918679, 0.4005389974044279, 0.48880374119893555, 3.7807060792957436, 0.39842061556771585, 0.4899808580573504, 0.6087497621179415, 0.3873159041335541, 0.489989272382653, 3.1389758457740493, 0.3844294200140011, 0.49000739750474276, 4.534945312522921, 0.37357039377589585, 0.3317552826352288, 3.035258926117322, 0.4899370619436404, 0.4900044125637976, 4.690256376119664, 0.3817421615322119, 0.37737229562669405, 6.071350202812577, 0.4635747873366627, 0.36916602965716366, 1.662125597814595, 0.4899975073989722, 0.37642407655020654, 2.6524135555110546, 0.4899386285398053, 0.3730261370128287, 5.5991248196983525, 0.31014112486280954, 0.31000415122246666, 3.4294576133248937, 0.38493726555307234, 0.40250817788376286, 5.227965106486146, 0.4041715074671722, 0.48990214392515125, 2.478888113202652, 0.4123164732114947, 0.48993479102742105, 6.076829444194381, 0.4173407966591019, 0.48989210283167756, 1.3419507535043214],
# [0.3907027739442337, 0.48990239336629293, 4.694950008804375, 0.37204639207498, 0.48994643884754474, 4.47182168578299, 0.3541636924254441, 0.3187136623590233, 4.772621801171587, 0.4273689811437515, 0.4897424956076338, 2.736954491710041, 0.39816369428408166, 0.38064192939118424, 2.936606402390337, 0.4187888759216737, 0.3800239859945794, 4.5765958041668, 0.48984422424056107, 0.38460272721986605, 2.639400881311549, 0.4898368483348456, 0.3866897483722995, 3.7782032283988576, 0.48982955228395364, 0.38823647930595806, 0.8249418660755659, 0.4897787826850141, 0.404822625487304, 4.128269611923956, 0.3256759580985302, 0.33357905023460527, 5.179966281112111, 0.363529065383404, 0.48977358891966727, 3.439343052112016, 0.3692900391911298, 0.4898395915860924, 1.1459082036437729, 0.3699504109947999, 0.31001155003944303, 3.7914148160046603, 0.3680260533431084, 0.49001495861841116, 2.115312564384521, 0.37983506026678937, 0.48998200955101884, 3.790209546767126, 0.4140055471794777, 0.41831000777182503, 5.498099060490211],
# [0.4452766159016302, 0.30999908173315593, 1.6235585948507665, 0.4884615761311136, 0.4899111849462316, 3.0590686324850487, 0.3920258441218534, 0.48989685692324786, 3.8641930455111475, 0.3907136930784954, 0.48980211720766265, 0.7882910709796422, 0.39434689571843284, 0.48989866583886843, 3.761641818911994, 0.40088829158805384, 0.4898992135482752, 1.822900128104617, 0.4898263222208469, 0.4898928558215813, 7.357810239810159, 0.48991454657703143, 0.48989274177887876, 8.463380557929012, 0.43551354809606396, 0.378545146157807, 6.78187129456595, 0.48993051247258557, 0.37870809825855273, 1.7640756736884913, 0.4899269896986662, 0.37862791710785515, 2.2686924082715296, 0.4899139699284989, 0.37413492735108866, 3.3940588774141647, 0.32822117307751186, 0.35149264352092385, 3.884876583902032, 0.388671684431788, 0.3129413110236938, 1.2303734322390603, 0.374904630730218, 0.48989206823195497, 6.522736615648768, 0.38590791133505203, 0.3100394505948656, 1.5052290561383423, 0.3787538441762704, 0.48990777766431587, 3.842028535817196],
# [0.32292192570475126, 0.41709736119703894, 4.39366524099234, 0.48342824418500613, 0.34016157151110366, 1.182432263829919, 0.4899954472319592, 0.35808788598954205, 4.876224068879283, 0.48999368414078515, 0.48986290976738556, 5.21933557302682, 0.47815571338764923, 0.3536556542360909, 3.9295578292132496, 0.35558012812654594, 0.40297691014725256, 3.8659425388015176, 0.4898871167254052, 0.4900052562364777, 6.79893057223388, 0.4844275540897663, 0.48999095404243515, 3.9410927564179468, 0.3699600675764938, 0.48998133596085514, 1.3011857192412357, 0.37033852476009665, 0.4900012512427084, 2.0305508492147504, 0.37269145053725344, 0.4899776546643363, 1.0184138232918805, 0.41693913230162183, 0.489909166265546, 5.816184552311012, 0.39051544625343637, 0.445413998162367, 2.82897818573319, 0.3959016524297833, 0.39030991973786355, 0.9732738690079409, 0.310014067404412, 0.33265900505959395, 3.86963173733186, 0.48998975452223875, 0.37882590466810884, 4.685477005036166, 0.4899976753435406, 0.3872225866165408, 4.702790142864391],
# [0.489980266893055, 0.3641530709354098, 2.1562190113371913, 0.4899765917253442, 0.3550023074878364, 3.8945963474228975, 0.31004320371047306, 0.32527063010921287, 6.444916754321164, 0.36661650480750124, 0.48994793294166705, 3.835992511800404, 0.4899162884780037, 0.489981012429214, 2.237305977781937, 0.4004738388389593, 0.4899661182414973, 3.4242579676179776, 0.48995029572992477, 0.4899577565322769, 7.408527977958831, 0.3881979374839574, 0.48996758422345765, 2.985733506217131, 0.317411686854281, 0.3905538537932492, 4.107040673455837, 0.4074097669738432, 0.35115974359450225, 1.1470058349540029, 0.3106657423285668, 0.35351302009755337, 0.14270283463412617, 0.4899496722644398, 0.3682135859943933, 2.3044806838172525, 0.4899534815277446, 0.3792687458392563, 3.5218428143698732, 0.48995149174722347, 0.3892848020072536, 3.1456128912677985, 0.489898427483155, 0.40325333859062806, 4.433895636110889, 0.48717815854361807, 0.48992256017710195, 7.3933709423094776, 0.33831557286143693, 0.48993777694227947, 2.3950238135295363],
# [0.38590472893372896, 0.48993603938107977, 0.8341948471135991, 0.4030522895638374, 0.4831842410746083, 6.046811573539046, 0.4566016853182083, 0.3965011751601825, 3.951718381308669, 0.4899577966170705, 0.3916585794038647, 4.355672311136602, 0.451835132348137, 0.38062588454109275, 4.493128064646023, 0.48999524277627665, 0.38038190421103585, 1.6822119021524544, 0.4899972114514192, 0.37003339342526226, 2.5740693757749247, 0.31001194376478425, 0.3525530405794049, 3.0291113163018215, 0.32195043511377963, 0.31492254548801085, 5.319710939554817, 0.355802998046995, 0.48999679092315, 4.989068738635326, 0.3612699530954457, 0.489983249518317, 0.4825174914901039, 0.35800575747027796, 0.4899771116791891, 3.438346896270523, 0.3434399095017107, 0.334499621803856, 2.0306895051091316, 0.3981186720223307, 0.3100101182177344, 1.912121374205936, 0.32639826356457335, 0.4539059780356436, 1.5515256725760347, 0.39344479113467234, 0.4355042373861547, 4.468958119569352, 0.4779161684217916, 0.37981302289358626, 4.727694666017201, 0.443729735903566, 0.37979529304179127, 5.2895034131106575],
# [0.35015440751492344, 0.451137561034985, 5.790678873804662, 0.4322909347785997, 0.3798400844730429, 4.7609811474465875, 0.48985071430327, 0.4007121320715922, 4.321923360743322, 0.4898593900974593, 0.40110815931219895, 3.7708079561711902, 0.4898839195811101, 0.40500009099123024, 1.2012379786554415, 0.4883005264516337, 0.40526969125383355, 1.5114890533577268, 0.48991643062643636, 0.48984632018953833, 8.359627803111014, 0.36716474924462417, 0.3351434259950028, 5.083667180461027, 0.37174962030876063, 0.489865407654502, 1.3537341699062726, 0.37818202508731685, 0.4898814747362288, 1.5641658882851535, 0.38137541681655, 0.48990996958946026, 1.5501167787160304, 0.38366444379202186, 0.4898838245891414, 3.783024312848283, 0.3842472489029418, 0.48986150126329686, 1.8215114495296798, 0.36536321391177345, 0.4258685412592016, 4.809722244589718, 0.3100294967525463, 0.31019176841977614, 1.8586644071395668, 0.44764609909616093, 0.3746059493462686, 5.089017111690682, 0.48993506119578595, 0.39070156975341774, 4.15835293104023, 0.4858939769637265, 0.38831766632663317, 0.0777273024364779],
# [0.3980805350557421, 0.4899787501062282, 4.422517161753464, 0.3870452126126357, 0.4899583182117732, 4.296146489279812, 0.35902455046317205, 0.33998953516040037, 2.0571419809607967, 0.33524126517533637, 0.31009797768473646, 2.351206456869212, 0.4090451078448584, 0.3275998627028469, 1.2238720996523589, 0.44606220420201736, 0.3426582214618742, 0.5858540017974154, 0.4899784615571005, 0.48729358184236815, 6.681014257834539, 0.4899702486132299, 0.437532552926448, 3.3912644742977975, 0.48996327511457916, 0.38809853593880667, 4.125620494237734, 0.48995754009759834, 0.3785596092142466, 4.4615227377158, 0.31000292852120254, 0.3379850822087992, 5.335498634086355, 0.35805442503385726, 0.3100130346461882, 2.1547017175132024, 0.3680362051434657, 0.48995905694121134, 3.3335732642006284, 0.3705013380170254, 0.48997988521614666, 1.7395043649071542, 0.36648159241109646, 0.48999015233999904, 1.8015233063894316, 0.3581825849728614, 0.48999230644706004, 3.321143784752059, 0.355324330581549, 0.31003628255145604, 4.671768205779813, 0.40640677602222985, 0.3542746578869731, 1.9819296902002526],
# [0.4064854628796029, 0.48994424162330225, 5.177563041901484, 0.4238614500212057, 0.48991832367662386, 5.530881626013334, 0.43674375173308716, 0.44407649893289897, 5.113357018957098, 0.43338375214708075, 0.39338530688372986, 2.4176745471823433, 0.48993715868527904, 0.38642330917029444, 3.9942273474353134, 0.4899524663973754, 0.3699807281806338, 3.7134478817384804, 0.4899530973508331, 0.35815184939774714, 3.1759381617512314, 0.31003406955968493, 0.38114719273783154, 3.2485970666383452, 0.4555976997980131, 0.3450735649490541, 0.47766722273993517, 0.44796934170026415, 0.38510576251322537, 2.8727714780182647, 0.4894364473597657, 0.48994762275707965, 2.384779522324738, 0.38798585325755974, 0.3106740207616243, 0.05822912921940415, 0.36300545881027796, 0.3980118157052183, 4.681082583294394, 0.3652565397900613, 0.48996880998974157, 1.229535023632604, 0.3642025092895489, 0.48996748170242965, 5.675676241377136, 0.37254777149009277, 0.310022774770276, 2.6323932089121516, 0.35712076481777216, 0.43442891900380326, 5.5073246003783645, 0.489939469598939, 0.3480965330653638, 3.766537573761024],
# [0.38125137005444737, 0.31012476801232597, 4.162769585203381, 0.36184508162830614, 0.4899666615222176, 3.0432099603290346, 0.3700820102589543, 0.4899593385396435, 3.3518741939994947, 0.36812809621028936, 0.31003618901802543, 3.4923564102585907, 0.3636448369020776, 0.48995260730756535, 4.5380475138743925, 0.3695249252887852, 0.4898752912805532, 0.9098077195031423, 0.3338752401802274, 0.3581592009169235, 5.803355268462237, 0.4835287055812533, 0.3653269613585473, 0.6551055384836004, 0.4899139553288808, 0.38358967804838706, 1.449683291646256, 0.489931610577101, 0.38578843194175394, 1.449529400170093, 0.4899723347821474, 0.39215272838550996, 2.720829390418149, 0.48996007952892473, 0.3908892184753352, 5.346001791131132, 0.4899146067712135, 0.37791809534690174, 2.2430345338818305, 0.3100013674752796, 0.31595577057767565, 6.749832882117467, 0.34784268231877136, 0.48998210769033956, 0.899111619377041, 0.3563945997933206, 0.3100095844479476, 2.903007062745956, 0.3772243437961096, 0.4899911781741571, 4.269398208740584, 0.3909957068525393, 0.48997080738232146, 3.0607565703551067],
# [0.4900021109495824, 0.39988888154716407, 4.349455945378614, 0.4899916983370542, 0.3888776545932396, 2.2908953244448447, 0.41674254543824374, 0.3847455821467088, 4.477323775720505, 0.31000532622497057, 0.40598068031599954, 1.3221926142410472, 0.4899436358902603, 0.4899875670114639, 5.257362453849228, 0.4343826356736509, 0.438597407567172, 0.07172799546554173, 0.4890292120675557, 0.489998270250868, 2.7479672618082414, 0.3497955130137437, 0.31167173039923485, 3.4099844887803585, 0.3503656422486602, 0.4899999888576157, 3.23999345764167, 0.36031876891278614, 0.48999393249699563, 3.1517773725226435, 0.36234036453688745, 0.48996280515806295, 2.1082932306824214, 0.3531311119729841, 0.48986878827782987, 1.3000278317864518, 0.31001347680017927, 0.3100108555761185, 6.955499741847397, 0.48998955808922057, 0.3595840731823087, 4.3822643100704655, 0.4899911331303373, 0.359906476289549, 2.885237074255183, 0.4898460517784112, 0.3614985235604272, 0.10948554818346078, 0.4899994416151328, 0.35359343271173804, 1.6297058586481914, 0.30999094424157436, 0.35764873270196335, 3.862256262502461, 0.4899973699145874, 0.47898221330792273, 5.951016504156543],
# [0.39882082540581715, 0.4899527692268382, 3.585735734082971, 0.37931116461843506, 0.31002592913107474, 3.4189307931934763, 0.3715678947243403, 0.4244285226999398, 3.9590227519974226, 0.37994022567007085, 0.4889272486960492, 0.10641028556792166, 0.38391997827561647, 0.48997911572526587, 1.9406051775232858, 0.38868362545507246, 0.48995107961350237, 1.1774187716233018, 0.3974524738996894, 0.48991202303654063, 3.264878224082926, 0.3697263715753749, 0.40282590898405163, 5.533769412493987, 0.4471245004640601, 0.3795745191801112, 4.612064901483425, 0.48995115421238933, 0.3937280806633127, 4.985890253546467, 0.4900081096724629, 0.38038050835274223, 5.120554323690993, 0.4899863866537076, 0.36335199532828893, 1.0518125409408272, 0.3206002748824308, 0.33411910201227313, 6.850263675735336, 0.37239436673990495, 0.31109057562452946, 0.11585833511063055, 0.36525830807554316, 0.31004246866049284, 1.4838195708476678, 0.3810018420672578, 0.48997591870586443, 1.4194419583995999, 0.3949974820752353, 0.48999011555655697, 3.7381482361379574, 0.40685333963488485, 0.4899779643215948, 3.4149830586735166, 0.4499688283130323, 0.48993886960581806, 4.444340587162342],
# [0.48999215172858607, 0.48998625336696494, 4.990590974482651, 0.31003961456972995, 0.3721536859998495, 2.7123779704706235, 0.310085252411554, 0.360028998225517, 0.8722006741932705, 0.4899698991444982, 0.3538322001977074, 1.5850484510329212, 0.4899677535000737, 0.36203020217137927, 3.493522523372624, 0.4899105734042357, 0.36305141846582745, 4.809227691836651, 0.33803399624546154, 0.40721025788634746, 4.585577323882547, 0.3538817770394144, 0.31003109047153693, 4.7245542731318695, 0.36667692200866425, 0.4899823577446053, 4.490858040476675, 0.37251917177901495, 0.4899822585299075, 4.882456941562206, 0.3590163620440691, 0.3847519935155783, 3.5378141675170833, 0.358601114711356, 0.37906335426511845, 2.7090483830253245, 0.3168167346575665, 0.35987544337083055, 0.9864424544370466, 0.4141950382050397, 0.40070102278223024, 2.9369965001884952, 0.48456964685427284, 0.4197774386807093, 3.0296514390222793, 0.4898885086349028, 0.41673097272462545, 2.4419882897193146, 0.48999051452431913, 0.412989256007163, 3.5155721873237313, 0.48997991646163574, 0.4158300623719224, 1.4462437333500693, 0.4899566300601561, 0.42267991698323387, 1.7689632675015174],
# [0.4090979348064933, 0.48999286246808377, 1.7992481116265744, 0.38815529019658973, 0.4316427176121985, 1.1618541121802797, 0.38483134782487477, 0.48997094006341907, 3.0101440213967203, 0.4274092342417681, 0.4757373013218335, 3.9940837245911194, 0.35487868916256776, 0.4232815897880746, 3.9373638763342163, 0.4899275219432857, 0.4058217169528648, 4.781989627342637, 0.4899951185802263, 0.38091433582087686, 4.499614150111107, 0.48986389239297534, 0.39143111359196436, 5.050470936554559, 0.4745798996150346, 0.4036138394441343, 0.29168900981945445, 0.46621069787577707, 0.3646280111483138, 4.90933173830387, 0.3379118856934313, 0.4410006877123368, 4.836089300145317, 0.4209019633706476, 0.48975506666666635, 2.683018357543464, 0.4490801513251742, 0.4899857240640428, 1.0163078684888511, 0.4141803059907916, 0.49000078965843813, 5.068210800221204, 0.3618494766094911, 0.4758870921858588, 3.9539763900872344, 0.4343352415661824, 0.30999203580707735, 2.3867326760754857, 0.48998410036589995, 0.48999790508246965, 2.2317059394933874, 0.4898911324054768, 0.43471695379201725, 3.0065153694186915, 0.47120614091275564, 0.48999933987270067, 4.866846161542365]
# ]


graph_sketch([ 0.4505,  0.3883, 20.4568,  0.3849,  0.4777, 19.0066,  0.4900,  0.4509,
        29.9588])




get_fidelity_vs_sigma_perfect_correlation([0.4123561603338839, 0.4684399760889638, 22.58339856324674, 0.44888189103572584, 0.38190578964245886, 24.790637953178546, 0.4121402359466627, 0.4684539351182967, 22.519408811200226])
get_fidelity_vs_rho_between_segments([0.4123561603338839, 0.4684399760889638, 22.58339856324674, 0.44888189103572584, 0.38190578964245886, 24.790637953178546, 0.4121402359466627, 0.4684539351182967, 22.519408811200226])



muhamad_solution = [0.37,0.4,9,0.38,0.355,15,0.37,0.4,9]
muhamad_solution = [0.358,0.375,10.727/2,0.374,0.35,16.653/2,0.436,0.45,30.176/2,0.396,0.379,10.266/2]

if len(muhamad_solution)//3 == 3:
    comp_sol = [0.31, 0.31, 8.666, 0.31, 0.31, 8.675, 0.31, 0.31, 8.265]
else:
    comp_sol = [0.31, 0.31, 6.405, 0.31, 0.31, 6.405, 0.31, 0.31, 6.404, 0.31, 0.31, 6.401]
    # comp_sol = [0.3102354139479926, 0.30998088427286624, 6.405512742258049, 0.31042356808373334, 0.3103938118570174, 6.405197365106319, 0.31031626054255107, 0.3100035671589348, 6.4038731523346, 0.31018263470001256, 0.3099819971257616, 6.401075837318065]



for cross_cov in np.linspace(0.96,0.94,3):
    get_fidelity_vs_sigma_between_segments_and_inside_segment([comp_sol,muhamad_solution],two_params=True,cross_cov=cross_cov)
    # get_fidelity_vs_sigma_between_segments_and_inside_segment([[0.31, 0.31, 8.666196051347727, 0.31, 0.31, 8.675204174783518, 0.31, 0.31, 8.265427128015922],[0.37,0.4,9,0.38,0.355,15,0.37,0.4,9]],two_params=True,cross_cov=cross_cov)
    # get_fidelity_vs_sigma_between_segments_and_inside_segment([[0.3099830796118722, 0.30994131432965116, 12.792205585464533, 0.30998818589277705, 0.3099363109038271, 12.792044643348913],[0.37,0.4,9,0.38,0.355,15,0.37,0.4,9]],two_params=True,cross_cov=cross_cov)
    # get_fidelity_vs_sigma_between_segments_and_inside_segment([[0.31005725465815, 0.31017728334854533, 8.666196051347727, 0.3100220366269435, 0.3100632157238241, 8.675204174783518, 0.3100185918140481, 0.3100947252654, 8.265427128015922],[0.37,0.4,9,0.38,0.355,15,0.37,0.4,9]],two_params=True,cross_cov=cross_cov)


get_fidelity_vs_sigma_with_rho_between_waveguide_inside_segment([0.3501394019625475, 0.34986515265991724, 8.646361647421411, 0.3500947276876372, 0.34989598157731877, 12.689042065449092, 0.3501150398869906, 0.3499022050802125, 8.258874693412901],cross_cov=0.95)


get_fidelity_vs_sigma_between_segments_and_inside_segment([0.3048558079950391, 0.3166344270506287, 12.964625742481324, 0.3043013024636271, 0.49523275937621253, 15.359169786949062, 0.3048494750348469, 0.31520360666966113, 13.855199239686687]
,cross_cov=0.95)


get_fidelity_vs_sigma_between_segments_and_inside_segment([0.30502071355184496, 0.30497438125933357, 7.860626359072013, 0.3049778972267658, 0.3049889773727485, 9.118872546502821, 0.3050166092594665, 0.3050115701891069, 8.117213796197799],cross_cov=0.95)
get_fidelity_vs_sigma_between_segments_and_inside_segment([0.30492634334519597, 0.3165681874787205, 12.64100637011489, 0.30432897524788194, 0.49528993572381785, 15.418579266329385, 0.3048944682695208, 0.31435348271404123, 14.103428569433072],
                                                     cross_cov=0.95)

for cross_cov in np.linspace(1.0,0.8,10):
    get_fidelity_vs_sigma_between_segments_and_inside_segment([0.37,0.4,9,0.38,0.355,15,0.37,0.4,9],cross_cov=cross_cov)
    # get_fidelity_vs_sigma_between_segments_and_inside_segment([0.4123561603338839, 0.4684399760889638, 22.58339856324674, 0.44888189103572584, 0.38190578964245886, 24.790637953178546, 0.4121402359466627, 0.4684539351182967, 22.519408811200226],cross_cov=cross_cov)

get_fidelity_vs_sigma_between_segments_and_inside_segment([0.4767934029012644, 0.4950729266522823, 19.42604971340568, 0.49502579482852316, 0.3045503649554255, 19.16315331941751, 0.4800858634998331, 0.49509311896084435, 21.9199859854395])

get_fidelity_vs_rho_inside_segment([0.4123561603338839, 0.4684399760889638, 22.58339856324674, 0.44888189103572584, 0.38190578964245886, 24.790637953178546, 0.4121402359466627, 0.4684539351182967, 22.519408811200226])


show_graph_sketch_and_rho_crit([0.4949870371467506, 0.44922310612028304, 25.14024989681448, 0.39143578856550904, 0.4651243413824442, 21.578678000756323, 0.4908441716715791, 0.4243014422455636, 19.084489767792483])

show_graph_sketch_and_rho_crit([0.495003426207621, 0.4492161700623223, 24.915237510504916, 0.3915655877946515, 0.4648913782184959, 21.504379392259704, 0.49280884428507804, 0.42575254075677954, 18.8699081555895])
show_graph_sketch_and_rho_crit([0.4950052072786813, 0.4490723340367375, 24.466471940254557, 0.39204000700842306, 0.4647548371999481, 21.35909784253759, 0.49487439529983873, 0.4271722344381356, 18.48128296292306])

show_graph_sketch_and_rho_crit([0.4949364961184291, 0.4271775889409281, 20.42386791135983, 0.35110679623189806, 0.40298015681200805, 11.650180634107032, 0.35304994295516895, 0.3936454248637407, 19.899342627161705, 0.49495852346180724, 0.40744395622765917, 15.446519373871642])

show_graph_sketch_and_rho_crit([0.494962764428793, 0.4616889610034848, 15.227621327495559, 0.43020689843793003, 0.3968295074564149, 17.65872304840612, 0.3605998340408903, 0.4949812509390264, 9.612041250106312, 0.33087156116229355, 0.3443615808447357, 11.268689797861304, 0.4950018540080521, 0.38245304370415, 9.658150515164033])

show_graph_sketch_and_rho_crit([0.4041983285531807, 0.3865397402516428, 13.12743996707401, 0.448675488311745, 0.3939311318917477, 15.729895457038843, 0.3900983378542343, 0.47224990012058615, 8.874141885401107, 0.37287496365932, 0.49499871415156277, 6.769402181484334, 0.3364637416463744, 0.33722672856768127, 9.85915278073113, 0.49500217689573717, 0.377558738525673, 8.511800330269885])


show_graph_sketch_and_rho_crit([0.31996599634671086, 0.3394643977545254, 10.329944097678684, 0.4949967218757157, 0.3373957618692811, 6.280073403181137, 0.3293779670075588, 0.3185751651316372, 13.862317629233129, 0.35525657404239935, 0.4950113375781988, 8.022483467035563, 0.352355222106872, 0.3717799589458264, 6.1021014041400665, 0.32952547468054527, 0.3460123298590381, 8.249439026129485, 0.4949743281699892, 0.3676759716768851, 7.507593912919326])

show_graph_sketch_and_rho_crit([0.32505894149867254, 0.36454525357207873, 9.549019830501441, 0.49500393693955014, 0.35234100248224326, 4.7479846403075605, 0.4947793078629472, 0.3700793427233448, 5.844306414232177, 0.3430318305114106, 0.35366628144207124, 10.806985172810137, 0.35582976386962667, 0.4950002478738905, 7.738724002661468, 0.34363083121999666, 0.30501246754128586, 5.774135229651312, 0.3383618884876705, 0.4070545495024479, 7.929726756527136, 0.49489932373528467, 0.3586342531927985, 7.124094388615766])

show_graph_sketch_and_rho_crit([0.3157957024372814, 0.3784578180938119, 7.859260562579883, 0.4950029565180474, 0.3605603201649062, 4.464586478959822, 0.49500438172941946, 0.3638953779669308, 5.531047093342255, 0.4151259185380105, 0.38587510954577803, 8.844681146088869, 0.33653800617404334, 0.4950071242751304, 7.404209948379284, 0.3843031193993374, 0.31294789578594745, 5.006100132673831, 0.42230459893631633, 0.4914496731296881, 4.213204750843547, 0.34618048062758716, 0.41461872135919986, 7.637624250320745, 0.49500650165909865, 0.361294289727803, 6.71098868679737]
)

show_graph_sketch_and_rho_crit([0.4948612772869107, 0.37186129740782004, 2.440562845661222, 0.49487093603711324, 0.38082287715273944, 6.339301518047025, 0.30497769883190884, 0.4948838314807059, 10.100306562961558, 0.30497482037309454, 0.4949241658677333, 9.946056615134832, 0.38339164892115574, 0.4949156687537984, 4.909449496386695, 0.3050534951703881, 0.49486752977112713, 2.256294618661801, 0.3050253589227202, 0.3592601964572544, 10.983729456010508, 0.4949538319276856, 0.3266418066826843, 9.521515924551577, 0.3476381083431094, 0.36622313355269576, 23.85416948275621]
)

show_graph_sketch_and_rho_crit([0.32655789822190723, 0.3654030453519463, 7.849944366264439, 0.49422675098412316, 0.3615539285423627, 4.458939376741187, 0.4943303132949491, 0.38137545026882214, 5.525144207853575, 0.40939455483485016, 0.3885629671122711, 8.853903826325466, 0.34849121514151904, 0.4949406674515929, 7.393341932831438, 0.3836944739851838, 0.31231741507368993, 5.018477252650058, 0.4160838950556577, 0.49473429253582524, 4.224181255951505, 0.3343773425237746, 0.42709276345066727, 7.64900649510996, 0.4947782903420237, 0.3777217421235069, 6.69300992337281, 0.4940240297108126, 0.38658981142542626, 3.798491390007658])
show_graph_sketch_and_rho_crit([0.3240626111315076, 0.3678016513982793, 7.853672334102588, 0.4949754394070764, 0.36512247680310767, 0.4981048997490922, 0.4949993034798956, 0.35816533452570304, 4.455436743935009, 0.49488286420972544, 0.3849858768735388, 5.521666599663448, 0.40754791536872736, 0.3905649964079289, 8.850694092270487, 0.3487687422383009, 0.4948947001176886, 7.39315872573684, 0.38439918706117204, 0.3111806298213546, 5.022358667778265, 0.4172549816937548, 0.49500273903718334, 4.22466782463799, 0.3345471100609641, 0.4267137792070889, 7.6495994274703065, 0.4949477827534899, 0.3794891446969312, 6.691787119936487, 0.4949609413706297, 0.3888577178844865, 3.797388688991232])



get_fidelity_vs_rho_between_segments([0.4123561603338839, 0.4684399760889638, 22.58339856324674, 0.44888189103572584, 0.38190578964245886, 24.790637953178546, 0.4121402359466627, 0.4684539351182967, 22.519408811200226])
get_rho_crit_between_segments([0.4123561603338839, 0.4684399760889638, 22.58339856324674, 0.44888189103572584, 0.38190578964245886, 24.790637953178546, 0.4121402359466627, 0.4684539351182967, 22.519408811200226])


get_fidelity_vs_sigma_with_rho_between_waveguide_inside_segment([0.375, 0.425, 24.627, 0.429, 0.363, 26.304, 0.391, 0.45, 23.315],cross_cov=0.999)



get_fidelity_vs_rho_between_segments([0.4123561603338839, 0.4684399760889638, 22.58339856324674, 0.44888189103572584, 0.38190578964245886, 24.790637953178546, 0.4121402359466627, 0.4684539351182967, 22.519408811200226])
show_graph_sketch_and_rho_crit([0.4123561603338839, 0.4684399760889638, 22.58339856324674, 0.44888189103572584, 0.38190578964245886, 24.790637953178546, 0.4121402359466627, 0.4684539351182967, 22.519408811200226])




# 0.2 Rho crit???
show_graph_sketch_and_rho_crit([0.3288047423885331, 0.37404004082652587, 6.969347591160029, 0.4949888878093993, 0.37812467479028433, 0.4597962258427138, 0.4949859974810792, 0.35848335581860485, 4.420476652237097, 0.4949551941266219, 0.3911507105514769, 5.477149208933636, 0.41304015957986395, 0.394053161071097, 8.465827012402269, 0.34613920790093444, 0.49491921797335536, 7.36691922815691, 0.3987919061858099, 0.3161825822662406, 5.01234677806876, 0.4301882400797335, 0.49496665010321356, 4.1804324003039275, 0.3338208157489045, 0.43512563469096965, 7.625081914612503, 0.49492168430956, 0.3772853022809718, 6.647579813349331, 0.4949693099458117, 0.3864037081313708, 3.748302531222952])


show_graph_sketch_and_rho_crit([0.36750885698557473, 0.3769333221567761, 5.916044894971842, 0.48820236730749283, 0.4420540939644419, 0.2978358686209202, 0.4881796643399609, 0.3504252072487141, 4.312250047200767, 0.4914222380020581, 0.4317984079813605, 5.30409884793528, 0.4164403342539, 0.410817263111361, 7.097380653899816, 0.33983770732156465, 0.4876712870687122, 7.303334874275747, 0.4439431731287936, 0.3170321897199937, 5.015186415270039, 0.44472116873557893, 0.4584343055957875, 4.056373892944522, 0.32456795568500707, 0.46991250398164464, 7.587240933669699, 0.49340317445506515, 0.372701665741392, 6.518892716548492, 0.4949951181618923, 0.34864599193857426, 3.623364805250513]
)

show_graph_sketch_and_rho_crit([0.3279212840923437, 0.4237249910277322, 23.871791237125382, 0.4848440064708275, 0.3068640876534375, 26.14984903826656, 0.3137777428088369, 0.38158112821226475, 22.493367195812002, 0.4685200494774908, 0.4539415093209, 23.16647511989675, 0.3616305398806132, 0.47340095542706984, 9.854052303588173, 0.4269073326981193, 0.4052374680084468, 11.856475515491619, 0.44263524012107414, 0.36608109237632036, 7.907220352387125, 0.4355304949583953, 0.37140766435712314, 7.903771704871876, 0.3914293299209549, 0.4171347616875201, 9.941081446280101, 0.3987398755375453, 0.41234164636692483, 11.96001916431495])


# 3
show_graph_sketch_and_rho_crit([0.375, 0.425, 24.627, 0.429, 0.363, 26.304, 0.391, 0.45, 23.315])

# 4
show_graph_sketch_and_rho_crit([0.3049914394588974, 0.35843152956670943, 23.884724369165987, 0.4346093235377034, 0.34068165011585755, 26.116330335687366, 0.335745981284574, 0.38646276122095896, 22.618136340397783, 0.4555226602620099, 0.46085531925520984, 23.410986260956506])

# 5
show_graph_sketch_and_rho_crit([0.3050725315505377, 0.36552754669295195, 23.841536673744205, 0.4539583205218973, 0.34019699690310484, 26.116939998563353, 0.3237983598484129, 0.3834558636008993, 22.57115015023786, 0.4657726611510986, 0.45240842924867986, 23.25819773951557, 0.3962920511055436, 0.4191514726255507, 9.913470363072094])

# 6
show_graph_sketch_and_rho_crit([0.3050579064725897, 0.3661619913758559, 23.83469429776636, 0.4664554868536189, 0.3326003688929066, 26.127419823493277, 0.31619421425458233, 0.38872093749338765, 22.5652539229944, 0.47154247182089065, 0.4499784863804988, 23.24646664291057, 0.39539676085766184, 0.4294229774157086, 9.899135760001998, 0.40391468561238114, 0.4026704312130444, 11.982464817562747])

# 7
show_graph_sketch_and_rho_crit([0.30534694831937703, 0.3638734621079549, 23.828086461698526, 0.4754902857320343, 0.3269590139766906, 26.133856279105053, 0.3108602966807631, 0.39405476666819045, 22.56478481782413, 0.47207180167060603, 0.4500687257784958, 23.234045137963626, 0.39309278871875686, 0.43538995579122947, 9.886280774375933, 0.40536215277698573, 0.4097632973838792, 11.96888342679344, 0.4028240013316684, 0.40084682696285806, 7.98770558484778])

# 8
show_graph_sketch_and_rho_crit([0.3050365998990641, 0.36277842569062846, 23.809117183951418, 0.49013375992880837, 0.3199563625695105, 26.14622716359668, 0.3086267110789621, 0.40337441867411056, 22.57134911633366, 0.4713398460079945, 0.4541500370349367, 23.20050545081343, 0.38569969083640365, 0.44150842429524506, 9.851826385632025, 0.4133713343081713, 0.42060545179956343, 11.928089881033953, 0.40952779215946433, 0.40374918636353213, 7.947314583618346, 0.3995265802280429, 0.4069135255602215, 7.959933340110966])

# 9
show_graph_sketch_and_rho_crit([0.32503677898794947, 0.40653147295891834, 23.85138193992788, 0.4912868402357231, 0.3180811471287855, 26.154974714023464, 0.30514026677858164, 0.3841098297531463, 22.496584718355376, 0.47211717717952606, 0.453607238498194, 23.186636835488073, 0.3727083600368826, 0.4623450414759941, 9.850454335464237, 0.42931451658334757, 0.40546766182178784, 11.882703138885992, 0.4339816937865994, 0.37756317010661256, 7.915362863823527, 0.42211194178824857, 0.3840874932080708, 7.918619894231507, 0.374921572078812, 0.4292563869539658, 9.975953008676127])

# 10
show_graph_sketch_and_rho_crit([0.32873019482482907, 0.42312752187261493, 23.871622315380794, 0.4840942210695591, 0.30691732642123093, 26.1497898045117, 0.3128294648702022, 0.3823304232681186, 22.494238918548373, 0.46804589117545164, 0.4545239169274493, 23.168929668858564, 0.36295717218931267, 0.47346857721365737, 9.854357926657507, 0.42695156696826103, 0.4049284007853587, 11.859396631664048, 0.442696747568052, 0.36554076665130536, 7.908401042653072, 0.4346804041138496, 0.37174921260967886, 7.905197169182533, 0.38957870039541387, 0.41876295339792363, 9.945773750104198, 0.3975194836640191, 0.4133999105115631, 11.96596615240738])

# 11
show_graph_sketch_and_rho_crit([0.3388555877269953, 0.42224901524803665, 23.869631303887136, 0.4772110695040392, 0.30499122894856023, 26.146283854190692, 0.31464864026995715, 0.39069155796649585, 22.50164844798598, 0.45748710620592753, 0.46118696890140076, 23.162869415119417, 0.35170804364272473, 0.48543083262590914, 9.86329898535269, 0.4156428554426872, 0.41865204038851916, 11.843535979215899, 0.4434380483438862, 0.35406576208708773, 7.90239823448728, 0.44017499910612784, 0.3542655536999793, 7.900282742310436, 0.39328119622505747, 0.41753578444882533, 9.929350069345876, 0.40399625294397207, 0.40755546748749655, 11.944793358163452, 0.3304730677112376, 0.36417920415332555, 14.988951724345085])


get_rho_crit_between_segments([0.375, 0.425, 24.627, 0.429, 0.363, 26.304, 0.391, 0.45, 23.315])
get_fidelity_vs_rho_between_segments([0.375, 0.425, 24.627, 0.429, 0.363, 26.304, 0.391, 0.45, 23.315])


# show_graph_sketch_and_rho_crit([0.4857,0.4345,47.1117/2,0.4057,0.4896,40.5109/2,0.4857,0.4345,47.1117/2])
#
# show_graph_sketch_and_rho_crit([0.49499741598414754, 0.3912490108365983, 13.465959975797968, 0.3703782734611008, 0.42954561975465294, 25.72668312904947, 0.4950127148146422, 0.4535166565850161, 27.331798533619843])
# show_graph_sketch_and_rho_crit([0.49499741598414754, 0.3912490108365983, 13.465959975797968, 0.3703782734611008, 0.42954561975465294, 25.72668312904947, 0.4950127148146422, 0.4535166565850161, 27.331798533619843])

get_fidelity_vs_rho_between_segments([0.4123561603338839, 0.4684399760889638, 22.58339856324674, 0.44888189103572584, 0.38190578964245886, 24.790637953178546, 0.4121402359466627, 0.4684539351182967, 22.519408811200226])


show_graph_sketch_and_rho_crit([0.49498251719027164, 0.39276906625889396, 13.719570501645642, 0.37061410718300086, 0.4299131900178186, 25.778377540067282, 0.4950119451368309, 0.45340390167504013, 27.40282680998478])


# 3
show_graph_sketch_and_rho_crit([0.4123561603338839, 0.4684399760889638, 22.58339856324674, 0.44888189103572584, 0.38190578964245886, 24.790637953178546, 0.4121402359466627, 0.4684539351182967, 22.519408811200226])
# 4
show_graph_sketch_and_rho_crit([0.3983475776462084, 0.47957826389865243, 21.308876383137637, 0.4091330214989601, 0.34745649879885004, 23.346689569261187, 0.46047734703648935, 0.45852085508171286, 18.11086046394327, 0.3535729830333715, 0.40493940370086656, 17.51056847223798])
# 5
show_graph_sketch_and_rho_crit([0.38864158536467, 0.48095511907649574, 21.15476637192908, 0.37232383690860354, 0.31328646359605555, 23.22207005950408, 0.46878299648334126, 0.4597762066941142, 17.781606175063224, 0.3510712999743093, 0.3839274693294444, 17.100879481804785, 0.42896665183057275, 0.489958594591821, 7.277495318818816])
# 6
show_graph_sketch_and_rho_crit([0.4221250710391486, 0.4825243183243031, 21.31621820772236, 0.4765129631768031, 0.4036964490856236, 11.986935409197127, 0.4949393480848293, 0.3983443996829274, 8.080708893980049, 0.44127766509282457, 0.46209725459036594, 10.969169450665023, 0.42736072520022017, 0.48753998215177863, 8.627041010183557, 0.38335448211660966, 0.42304881305173725, 7.929118503172335])
# 7
show_graph_sketch_and_rho_crit([0.41519328751480356, 0.4840717565896009, 21.233002444073218, 0.4804928046215981, 0.3978387192043475, 11.92766611673881, 0.47736419204222147, 0.38498416053504275, 8.033133955392003, 0.4566128889816549, 0.45802916429994556, 10.725169377515533, 0.44810306447701354, 0.47252819993509515, 8.456727911167212, 0.32001124597477204, 0.40411544373696856, 7.839565607513642, 0.41454564537737093, 0.40842827696490197, 7.853917770713693])
# 8
show_graph_sketch_and_rho_crit([0.40873020669545107, 0.4877399022170756, 21.032928615752784, 0.494900297602489, 0.37393325156479185, 11.878442055891375, 0.43845258292656936, 0.39688150189478666, 7.9945157031457965, 0.4949174778978498, 0.41125361708768315, 10.445637182068896, 0.4949224219544562, 0.493514208669622, 6.941945631122341, 0.3049910493619962, 0.49255444677287474, 7.958058447968866, 0.4541508463847319, 0.40385304426832785, 7.625263458776286, 0.424752492027294, 0.4334552895506495, 20.746367571584805])
# 9
show_graph_sketch_and_rho_crit([0.40505816354772195, 0.4869567116763057, 21.016101034193134, 0.49429572802124117, 0.37817678834729007, 11.88531901412757, 0.4575476205935828, 0.38648937185277443, 7.998634311036691, 0.4948913519265683, 0.39525793945901305, 10.460655991061612, 0.4873332372720022, 0.49486226044028536, 6.957788711930871, 0.3050813371629978, 0.49488863164186503, 8.00424037890391, 0.4434012109772869, 0.42423934912158673, 7.589869536414149, 0.4357458779475116, 0.43019295321438533, 20.695037497845174, 0.3946403287759486, 0.4239031901999343, 7.949000889539569])
# 10


graph_sketch([0.3967458972088381, 0.4948842862118363, 20.98573031303606, 0.36410696972407336, 0.3051205688160912, 23.0997551636834, 0.47334247514177674, 0.4454552710099809, 17.38287034168034, 0.3082330676036072, 0.3536989450703947, 16.361073510012513, 0.4435983038379559, 0.4327620562442545, 6.221265071442101, 0.4189758336981387, 0.4651554971582529, 7.107259727200773])
get_rho_crit_between_segments([0.3967458972088381, 0.4948842862118363, 20.98573031303606, 0.36410696972407336, 0.3051205688160912, 23.0997551636834, 0.47334247514177674, 0.4454552710099809, 17.38287034168034, 0.3082330676036072, 0.3536989450703947, 16.361073510012513, 0.4435983038379559, 0.4327620562442545, 6.221265071442101, 0.4189758336981387, 0.4651554971582529, 7.107259727200773])

graph_sketch([0.4221250710391486, 0.4825243183243031, 21.31621820772236, 0.4765129631768031, 0.4036964490856236, 11.986935409197127, 0.4949393480848293, 0.3983443996829274, 8.080708893980049, 0.44127766509282457, 0.46209725459036594, 10.969169450665023, 0.42736072520022017, 0.48753998215177863, 8.627041010183557, 0.38335448211660966, 0.42304881305173725, 7.929118503172335])
get_rho_crit_between_segments([0.4221250710391486, 0.4825243183243031, 21.31621820772236, 0.4765129631768031, 0.4036964490856236, 11.986935409197127, 0.4949393480848293, 0.3983443996829274, 8.080708893980049, 0.44127766509282457, 0.46209725459036594, 10.969169450665023, 0.42736072520022017, 0.48753998215177863, 8.627041010183557, 0.38335448211660966, 0.42304881305173725, 7.929118503172335])





# cross correlation between waveguides

# graph_sketch([0.3171985557839368, 0.4002042074232191, 11.060922925048065, 0.35736818864217956, 0.49492218042001485, 16.3508566093385, 0.40539712950217305, 0.3051417576802826, 9.245860517879567, 0.3531403451123125, 0.33629637169703874, 32.26381688374536, 0.4185115913935556, 0.4540208898327493, 33.167014393276354, 0.48648246479458307, 0.41818805367282114, 24.820762489453582, 0.30509233547242665, 0.45806799513505564, 16.15169611924246])
#
# graph_sketch([0.3050752868341029, 0.3809214498146033, 11.097863537111204, 0.36323356862195544, 0.4949376712786362, 16.323231617005177, 0.4218347880317291, 0.3178084056082972, 9.21674740182078, 0.37502217877764954, 0.36136653867429536, 32.27510304800072, 0.44679676523848344, 0.4811795270315393, 33.0551670116274, 0.48986762372041487, 0.4188383580571737, 24.80219205297953, 0.30510107151037125, 0.4662733190011161, 16.165267377489307]
# )
graph_sketch([0.38864158536467, 0.48095511907649574, 21.15476637192908, 0.37232383690860354, 0.31328646359605555, 23.22207005950408, 0.46878299648334126, 0.4597762066941142, 17.781606175063224, 0.3510712999743093, 0.3839274693294444, 17.100879481804785, 0.42896665183057275, 0.489958594591821, 7.277495318818816])
get_rho_crit_between_segments([0.38864158536467, 0.48095511907649574, 21.15476637192908, 0.37232383690860354, 0.31328646359605555, 23.22207005950408, 0.46878299648334126, 0.4597762066941142, 17.781606175063224, 0.3510712999743093, 0.3839274693294444, 17.100879481804785, 0.42896665183057275, 0.489958594591821, 7.277495318818816])


graph_sketch([0.38984471733929293, 0.4813418339756648, 21.159786456133904, 0.373833565673715, 0.3150870046905925, 23.226542299426708, 0.46974572416668586, 0.4595410220924057, 17.797124948292726, 0.3519353278522169, 0.3866588994163364, 17.12663067646259, 0.43006872468515805, 0.4870565735677387, 7.299687745827188]
)

get_rho_crit_between_segments([0.38984471733929293, 0.4813418339756648, 21.159786456133904, 0.373833565673715, 0.3150870046905925, 23.226542299426708, 0.46974572416668586, 0.4595410220924057, 17.797124948292726, 0.3519353278522169, 0.3866588994163364, 17.12663067646259, 0.43006872468515805, 0.4870565735677387, 7.299687745827188]
)


# 3
graph_sketch([0.4123561603338839, 0.4684399760889638, 22.58339856324674, 0.44888189103572584, 0.38190578964245886, 24.790637953178546, 0.4121402359466627, 0.4684539351182967, 22.519408811200226])
# 4
graph_sketch([0.3983475776462084, 0.47957826389865243, 21.308876383137637, 0.4091330214989601, 0.34745649879885004, 23.346689569261187, 0.46047734703648935, 0.45852085508171286, 18.11086046394327, 0.3535729830333715, 0.40493940370086656, 17.51056847223798])
# 5
graph_sketch([0.42912628043335616, 0.4803972437883214, 21.344899058391913, 0.4803427524812507, 0.39985261260188754, 11.990344177552883, 0.48002393878003996, 0.4060178760546134, 8.084573663139848, 0.41982831199117354, 0.480203892548481, 11.070003741523246, 0.42781420299327483, 0.48016157361073974, 8.695381049501721])
# 6
graph_sketch([0.4221250710391486, 0.4825243183243031, 21.31621820772236, 0.4765129631768031, 0.4036964490856236, 11.986935409197127, 0.4949393480848293, 0.3983443996829274, 8.080708893980049, 0.44127766509282457, 0.46209725459036594, 10.969169450665023, 0.42736072520022017, 0.48753998215177863, 8.627041010183557, 0.38335448211660966, 0.42304881305173725, 7.929118503172335])
# 7
graph_sketch([0.41519328751480356, 0.4840717565896009, 21.233002444073218, 0.4804928046215981, 0.3978387192043475, 11.92766611673881, 0.47736419204222147, 0.38498416053504275, 8.033133955392003, 0.4566128889816549, 0.45802916429994556, 10.725169377515533, 0.44810306447701354, 0.47252819993509515, 8.456727911167212, 0.32001124597477204, 0.40411544373696856, 7.839565607513642, 0.41454564537737093, 0.40842827696490197, 7.853917770713693])
# 8
graph_sketch([0.40873020669545107, 0.4877399022170756, 21.032928615752784, 0.494900297602489, 0.37393325156479185, 11.878442055891375, 0.43845258292656936, 0.39688150189478666, 7.9945157031457965, 0.4949174778978498, 0.41125361708768315, 10.445637182068896, 0.4949224219544562, 0.493514208669622, 6.941945631122341, 0.3049910493619962, 0.49255444677287474, 7.958058447968866, 0.4541508463847319, 0.40385304426832785, 7.625263458776286, 0.424752492027294, 0.4334552895506495, 20.746367571584805])
# 9
graph_sketch([0.40505816354772195, 0.4869567116763057, 21.016101034193134, 0.49429572802124117, 0.37817678834729007, 11.88531901412757, 0.4575476205935828, 0.38648937185277443, 7.998634311036691, 0.4948913519265683, 0.39525793945901305, 10.460655991061612, 0.4873332372720022, 0.49486226044028536, 6.957788711930871, 0.3050813371629978, 0.49488863164186503, 8.00424037890391, 0.4434012109772869, 0.42423934912158673, 7.589869536414149, 0.4357458779475116, 0.43019295321438533, 20.695037497845174, 0.3946403287759486, 0.4239031901999343, 7.949000889539569])


exit(0)


graph_sketch([0.4621507810315183, 0.4950410372267429, 20.128292699850256, 0.4947616888820305, 0.4938693034064936, 11.077034443725866, 0.4908751886210259, 0.36802307519775423, 7.655645524648641, 0.4949859520010946, 0.49455717171061486, 9.028101008696185, 0.3847839808715234, 0.4928131769528457, 7.919878471289421])
get_rho_crit_between_segments([0.4621507810315183, 0.4950410372267429, 20.128292699850256, 0.4947616888820305, 0.4938693034064936, 11.077034443725866, 0.4908751886210259, 0.36802307519775423, 7.655645524648641, 0.4949859520010946, 0.49455717171061486, 9.028101008696185, 0.3847839808715234, 0.4928131769528457, 7.919878471289421])

for i in np.linspace(1,0,10):
    get_fidelity_vs_sigma_with_rho_between_waveguide_inside_segment(
        [0.4621507810315183, 0.4950410372267429, 20.128292699850256, 0.4947616888820305, 0.4938693034064936, 11.077034443725866, 0.4908751886210259, 0.36802307519775423, 7.655645524648641, 0.4949859520010946, 0.49455717171061486, 9.028101008696185, 0.3847839808715234, 0.4928131769528457, 7.919878471289421]
        , cross_cov=i,rho_between_sigma=False)

get_fidelity_vs_sigma_with_rho_between_waveguide_inside_segment([0.4621507810315183, 0.4950410372267429, 20.128292699850256, 0.4947616888820305, 0.4938693034064936, 11.077034443725866, 0.4908751886210259, 0.36802307519775423, 7.655645524648641, 0.4949859520010946, 0.49455717171061486, 9.028101008696185, 0.3847839808715234, 0.4928131769528457, 7.919878471289421])

get_fidelity_vs_rho_between_segments_and_inside_segment([0.4621507810315183, 0.4950410372267429, 20.128292699850256, 0.4947616888820305, 0.4938693034064936, 11.077034443725866, 0.4908751886210259, 0.36802307519775423, 7.655645524648641, 0.4949859520010946, 0.49455717171061486, 9.028101008696185, 0.3847839808715234, 0.4928131769528457, 7.919878471289421])

get_fidelity_vs_rho_between_segments([0.4621507810315183, 0.4950410372267429, 20.128292699850256, 0.4947616888820305, 0.4938693034064936, 11.077034443725866, 0.4908751886210259, 0.36802307519775423, 7.655645524648641, 0.4949859520010946, 0.49455717171061486, 9.028101008696185, 0.3847839808715234, 0.4928131769528457, 7.919878471289421])
get_rho_crit_between_segments([0.4621507810315183, 0.4950410372267429, 20.128292699850256, 0.4947616888820305, 0.4938693034064936, 11.077034443725866, 0.4908751886210259, 0.36802307519775423, 7.655645524648641, 0.4949859520010946, 0.49455717171061486, 9.028101008696185, 0.3847839808715234, 0.4928131769528457, 7.919878471289421])


graph_sketch([0.4576308149529887, 0.4950355377954988, 20.536067828127695, 0.49501418029259975, 0.47890121383942025, 11.500670217961924, 0.49285427936791215, 0.3816325056376912, 7.7685465230818345, 0.4948749950186994, 0.49369705985682, 9.562067895223432, 0.384172163733987, 0.494001348446914, 8.134200497594964])

get_rho_crit_between_segments([0.4397448170967127, 0.49126453704348294, 21.236498441691214, 0.4785438544509673, 0.40818488427401417, 11.923309913194293, 0.48500819524314875, 0.40941499343393944, 8.02139971113244, 0.4360960523254088, 0.47795001878161536, 10.968609172628558, 0.4240516941828197, 0.4949706423516087, 8.597936386959853]
)

graph_sketch([0.4397448170967127, 0.49126453704348294, 21.236498441691214, 0.4785438544509673, 0.40818488427401417, 11.923309913194293, 0.48500819524314875, 0.40941499343393944, 8.02139971113244, 0.4360960523254088, 0.47795001878161536, 10.968609172628558, 0.4240516941828197, 0.4949706423516087, 8.597936386959853]
)

get_rho_crit_between_segments([0.4397448170967127, 0.49126453704348294, 21.236498441691214, 0.4785438544509673, 0.40818488427401417, 11.923309913194293, 0.48500819524314875, 0.40941499343393944, 8.02139971113244, 0.4360960523254088, 0.47795001878161536, 10.968609172628558, 0.4240516941828197, 0.4949706423516087, 8.597936386959853]
)


get_rho_crit_between_segments([0.43771322534790896, 0.4890681663756395, 21.458378535783126, 0.4831908353564383, 0.4030008175988442, 12.027344276414247, 0.4783512860245597, 0.40557379430601503, 8.130875620378315, 0.4239274237827817, 0.48613869515565217, 11.161074670427878, 0.4425879513779154, 0.49224786869250403, 8.818089394636766]
)

graph_sketch([0.43771322534790896, 0.4890681663756395, 21.458378535783126, 0.4831908353564383, 0.4030008175988442, 12.027344276414247, 0.4783512860245597, 0.40557379430601503, 8.130875620378315, 0.4239274237827817, 0.48613869515565217, 11.161074670427878, 0.4425879513779154, 0.49224786869250403, 8.818089394636766]
)


graph_sketch([0.4312520582160209, 0.3704288529706803, 11.480083683378327, 0.35736622881472996, 0.46563768958734675, 2.6248324700592747, 0.41186146856721223, 0.3701862176867522, 7.552024943590138, 0.42370786554226314, 0.4950068625850025, 5.003372286966519, 0.3721804039533026, 0.47045129546080594, 8.476481431100607, 0.4113149586831622, 0.3724647754190614, 7.627290578526644, 0.4950148505342179, 0.3992896693040892, 7.551573274789143])
get_fidelity_vs_sigma_with_rho_between_waveguide_inside_segment([0.4312520582160209, 0.3704288529706803, 11.480083683378327, 0.35736622881472996, 0.46563768958734675, 2.6248324700592747, 0.41186146856721223, 0.3701862176867522, 7.552024943590138, 0.42370786554226314, 0.4950068625850025, 5.003372286966519, 0.3721804039533026, 0.47045129546080594, 8.476481431100607, 0.4113149586831622, 0.3724647754190614, 7.627290578526644, 0.4950148505342179, 0.3992896693040892, 7.551573274789143])
get_fidelity_vs_sigma_perfect_correlation([0.4312520582160209, 0.3704288529706803, 11.480083683378327, 0.35736622881472996, 0.46563768958734675, 2.6248324700592747, 0.41186146856721223, 0.3701862176867522, 7.552024943590138, 0.42370786554226314, 0.4950068625850025, 5.003372286966519, 0.3721804039533026, 0.47045129546080594, 8.476481431100607, 0.4113149586831622, 0.3724647754190614, 7.627290578526644, 0.4950148505342179, 0.3992896693040892, 7.551573274789143])
get_fidelity_vs_rho_between_segments_and_inside_segment([0.4312520582160209, 0.3704288529706803, 11.480083683378327, 0.35736622881472996, 0.46563768958734675, 2.6248324700592747, 0.41186146856721223, 0.3701862176867522, 7.552024943590138, 0.42370786554226314, 0.4950068625850025, 5.003372286966519, 0.3721804039533026, 0.47045129546080594, 8.476481431100607, 0.4113149586831622, 0.3724647754190614, 7.627290578526644, 0.4950148505342179, 0.3992896693040892, 7.551573274789143])

get_fidelity_vs_rho_between_segments_and_inside_segment([0.4312520582160209, 0.3704288529706803, 11.480083683378327, 0.35736622881472996, 0.46563768958734675, 2.6248324700592747, 0.41186146856721223, 0.3701862176867522, 7.552024943590138, 0.42370786554226314, 0.4950068625850025, 5.003372286966519, 0.3721804039533026, 0.47045129546080594, 8.476481431100607, 0.4113149586831622, 0.3724647754190614, 7.627290578526644, 0.4950148505342179, 0.3992896693040892, 7.551573274789143]
)

get_fidelity_vs_rho_between_segments_and_inside_segment([0.495001742333748, 0.49504974491892934, 5.708667357147962, 0.3050316011280475, 0.3051015090616144, 4.921392110292514, 0.4949726951042391, 0.49500704644231347, 5.729941965563583, 0.494976817530088, 0.4950080416108346, 5.833709290563668, 0.4949797443552196, 0.49500364573187466, 5.832832260759274, 0.4949811835625705, 0.49500231715570453, 6.021974965587263, 0.4949822692316577, 0.49499820435820674, 5.776402435363985])
get_fidelity_vs_rho_between_segments_and_inside_segment([0.495001742333748, 0.49504974491892934, 5.708667357147962, 0.3050316011280475, 0.3051015090616144, 4.921392110292514, 0.4949726951042391, 0.49500704644231347, 5.729941965563583, 0.494976817530088, 0.4950080416108346, 5.833709290563668, 0.4949797443552196, 0.49500364573187466, 5.832832260759274, 0.4949811835625705, 0.49500231715570453, 6.021974965587263, 0.4949822692316577, 0.49499820435820674, 5.776402435363985])


graph_sketch([0.3050752868341029, 0.3809214498146033, 11.097863537111204, 0.36323356862195544, 0.4949376712786362, 16.323231617005177, 0.4218347880317291, 0.3178084056082972, 9.21674740182078, 0.37502217877764954, 0.36136653867429536, 32.27510304800072, 0.44679676523848344, 0.4811795270315393, 33.0551670116274, 0.48986762372041487, 0.4188383580571737, 24.80219205297953, 0.30510107151037125, 0.4662733190011161, 16.165267377489307]
)


get_fidelity_vs_rho_between_segments_and_inside_segment([0.3050752868341029, 0.3809214498146033, 11.097863537111204, 0.36323356862195544, 0.4949376712786362, 16.323231617005177, 0.4218347880317291, 0.3178084056082972, 9.21674740182078, 0.37502217877764954, 0.36136653867429536, 32.27510304800072, 0.44679676523848344, 0.4811795270315393, 33.0551670116274, 0.48986762372041487, 0.4188383580571737, 24.80219205297953, 0.30510107151037125, 0.4662733190011161, 16.165267377489307]
)

get_fidelity_vs_sigma_with_rho_between_waveguide_inside_segment([0.3050752868341029, 0.3809214498146033, 11.097863537111204, 0.36323356862195544, 0.4949376712786362, 16.323231617005177, 0.4218347880317291, 0.3178084056082972, 9.21674740182078, 0.37502217877764954, 0.36136653867429536, 32.27510304800072, 0.44679676523848344, 0.4811795270315393, 33.0551670116274, 0.48986762372041487, 0.4188383580571737, 24.80219205297953, 0.30510107151037125, 0.4662733190011161, 16.165267377489307]
,cross_cov=0.88)

get_fidelity_vs_sigma_with_rho_between_waveguide_inside_segment([0.3050752868341029, 0.3809214498146033, 11.097863537111204, 0.36323356862195544, 0.4949376712786362, 16.323231617005177, 0.4218347880317291, 0.3178084056082972, 9.21674740182078, 0.37502217877764954, 0.36136653867429536, 32.27510304800072, 0.44679676523848344, 0.4811795270315393, 33.0551670116274, 0.48986762372041487, 0.4188383580571737, 24.80219205297953, 0.30510107151037125, 0.4662733190011161, 16.165267377489307]
,cross_cov=0.86)

get_fidelity_vs_sigma_with_rho_between_waveguide_inside_segment([0.3050752868341029, 0.3809214498146033, 11.097863537111204, 0.36323356862195544, 0.4949376712786362, 16.323231617005177, 0.4218347880317291, 0.3178084056082972, 9.21674740182078, 0.37502217877764954, 0.36136653867429536, 32.27510304800072, 0.44679676523848344, 0.4811795270315393, 33.0551670116274, 0.48986762372041487, 0.4188383580571737, 24.80219205297953, 0.30510107151037125, 0.4662733190011161, 16.165267377489307]
,cross_cov=0.85)

get_fidelity_vs_sigma_perfect_correlation([0.3050752868341029, 0.3809214498146033, 11.097863537111204, 0.36323356862195544, 0.4949376712786362, 16.323231617005177, 0.4218347880317291, 0.3178084056082972, 9.21674740182078, 0.37502217877764954, 0.36136653867429536, 32.27510304800072, 0.44679676523848344, 0.4811795270315393, 33.0551670116274, 0.48986762372041487, 0.4188383580571737, 24.80219205297953, 0.30510107151037125, 0.4662733190011161, 16.165267377489307]
)


get_fidelity_vs_sigma_perfect_correlation([0.30499609425356683, 0.37904223371129814, 10.923101074977955, 0.36662285894132585, 0.4950055286096269, 16.103474220970526, 0.4522692484662292, 0.34217825318712564, 9.043718668708932, 0.3845441083517853, 0.37280877817772, 31.799798222221977, 0.4615666948647048, 0.49493444841148454, 32.280310593521115, 0.4906987716030837, 0.4181809737792144, 24.574420050068987, 0.30566870018223113, 0.4726119337152859, 16.1518212491611])
get_fidelity_vs_sigma_with_rho_between_waveguide_inside_segment([0.30499609425356683, 0.37904223371129814, 10.923101074977955, 0.36662285894132585, 0.4950055286096269, 16.103474220970526, 0.4522692484662292, 0.34217825318712564, 9.043718668708932, 0.3845441083517853, 0.37280877817772, 31.799798222221977, 0.4615666948647048, 0.49493444841148454, 32.280310593521115, 0.4906987716030837, 0.4181809737792144, 24.574420050068987, 0.30566870018223113, 0.4726119337152859, 16.1518212491611])
get_rho_crit_between_segments([0.30499425995188384, 0.3853941111092362, 10.666022233558081, 0.3684995492390418, 0.49500907858150134, 15.738810655979782, 0.47672013769847554, 0.3428694054936823, 8.781079685161078, 0.3847519249205483, 0.37769084823598487, 30.93360052957545, 0.4658356474677453, 0.49501412113672044, 30.78491153781439, 0.48987608960494644, 0.41586446558261136, 24.171144969333113, 0.31833673152958936, 0.49485913909582574, 16.168238087160546])
get_fidelity_vs_sigma_perfect_correlation([0.30499425995188384, 0.3853941111092362, 10.666022233558081, 0.3684995492390418, 0.49500907858150134, 15.738810655979782, 0.47672013769847554, 0.3428694054936823, 8.781079685161078, 0.3847519249205483, 0.37769084823598487, 30.93360052957545, 0.4658356474677453, 0.49501412113672044, 30.78491153781439, 0.48987608960494644, 0.41586446558261136, 24.171144969333113, 0.31833673152958936, 0.49485913909582574, 16.168238087160546])

get_fidelity_vs_sigma_with_rho_between_waveguide_inside_segment([0.3049592078252063, 0.39864340288046085, 9.590584378067303, 0.3741504908462965, 0.4950150364413998, 14.230850975710753, 0.4950058578913172, 0.3356675469530955, 7.440005407071657, 0.3412756309108918, 0.3382406725100196, 24.50010709020479, 0.4634225589495381, 0.49503232849898965, 22.23718090706331, 0.4303783523759058, 0.36642784685543106, 22.19908858896488, 0.32364117324952374, 0.49503812925449286, 15.616087005765761])
get_fidelity_vs_sigma_perfect_correlation([0.3049592078252063, 0.39864340288046085, 9.590584378067303, 0.3741504908462965, 0.4950150364413998, 14.230850975710753, 0.4950058578913172, 0.3356675469530955, 7.440005407071657, 0.3412756309108918, 0.3382406725100196, 24.50010709020479, 0.4634225589495381, 0.49503232849898965, 22.23718090706331, 0.4303783523759058, 0.36642784685543106, 22.19908858896488, 0.32364117324952374, 0.49503812925449286, 15.616087005765761])

get_fidelity_vs_sigma_with_rho_between_waveguide_inside_segment([0.3703948095581563, 0.49202426190933923, 7.446050461432128, 0.3920685989417935, 0.4950284493391749, 11.97259268786633, 0.39851487566315486, 0.3049791289780496, 4.892605346939941, 0.3440261463655985, 0.3049693816589955, 16.568787458420587, 0.43917244967109154, 0.49502725176520807, 14.078786799730263, 0.3629313152622814, 0.30496458713044483, 18.278363115509467, 0.3520430400603849, 0.4950391212020905, 14.272589786445486])

get_fidelity_vs_sigma_with_rho_between_waveguide_inside_segment([0.35978367408813955, 0.49500653357325014, 7.0744244567390595, 0.44222923920800383, 0.4950177187258164, 11.64825134498668, 0.3727142929004049, 0.304977533743382, 2.9950463291404166, 0.33484463546043075, 0.3049394155743317, 11.940355859504525, 0.49186726134753184, 0.49503013188825024, 8.674602954403777, 0.3431249591677441, 0.3049255618807241, 16.53640841288321, 0.3553717787005296, 0.4950459676733395, 11.90696761520731])

graph_sketch([0.35564234771621517, 0.4950152860036643, 7.06117706808245, 0.45774528143561005, 0.49500919027625984, 11.67577212595414, 0.36974787866275827, 0.3049697269181035, 2.7206120340169386, 0.3333475557173307, 0.304925286144566, 11.265680117401244, 0.4950116737215546, 0.49502476591960864, 6.100062142426473, 0.3458974452361681, 0.30494604322955066, 16.219900868890186, 0.35962728896673557, 0.4950565535220146, 11.82230533866897]
             )
get_fidelity_vs_sigma_with_rho_between_waveguide_inside_segment([0.35564234771621517, 0.4950152860036643, 7.06117706808245, 0.45774528143561005, 0.49500919027625984, 11.67577212595414, 0.36974787866275827, 0.3049697269181035, 2.7206120340169386, 0.3333475557173307, 0.304925286144566, 11.265680117401244, 0.4950116737215546, 0.49502476591960864, 6.100062142426473, 0.3458974452361681, 0.30494604322955066, 16.219900868890186, 0.35962728896673557, 0.4950565535220146, 11.82230533866897]
)



# many pulses solutions:


get_rho_crit_between_segments([0.39918509942323377, 0.4925656411890484, 21.025428357855183, 0.47995814686828026, 0.3881928753775338, 11.860108099819962, 0.45846688970287275, 0.3448974138756441, 7.977038165018285, 0.489193035034335, 0.38139712237753265, 10.436890495973115, 0.47143307632334985, 0.4948798022671166, 6.959839798063948, 0.31987207082397157, 0.4944579172538013, 8.010555832767633, 0.42243382751847575, 0.44364195162553555, 7.576918808625301, 0.4443739533721619, 0.42108986116335767, 20.667421074323038, 0.411588479932799, 0.4076875460005157, 7.924728433330188, 0.3670504719750586, 0.4301383966529856, 7.979324873824293])

graph_sketch([0.39894191359768494, 0.4925756406608287, 21.028059843388725, 0.4796610455090371, 0.3878609652071817, 11.862659951095425, 0.4588419746796103, 0.3450698017343846, 7.979207484405615, 0.4882750995061775, 0.38144273386450445, 10.439220651460058, 0.47063406316680256, 0.49488789884322526, 6.968270195044814, 0.32011373161933954, 0.4942058689858696, 8.01155091083647, 0.42244417957686126, 0.4437765370223318, 7.58299296465996, 0.4448177128303326, 0.42066853669277865, 20.675878988469616, 0.41103550335162303, 0.40782910753980967, 7.92986828477945, 0.3668147619657272, 0.4302461111481352, 7.9820666213106675])
get_rho_crit_between_segments([0.39894191359768494, 0.4925756406608287, 21.028059843388725, 0.4796610455090371, 0.3878609652071817, 11.862659951095425, 0.4588419746796103, 0.3450698017343846, 7.979207484405615, 0.4882750995061775, 0.38144273386450445, 10.439220651460058, 0.47063406316680256, 0.49488789884322526, 6.968270195044814, 0.32011373161933954, 0.4942058689858696, 8.01155091083647, 0.42244417957686126, 0.4437765370223318, 7.58299296465996, 0.4448177128303326, 0.42066853669277865, 20.675878988469616, 0.41103550335162303, 0.40782910753980967, 7.92986828477945, 0.3668147619657272, 0.4302461111481352, 7.9820666213106675])


get_rho_crit_between_segments([0.4047773387162763, 0.4843272224043092, 21.012854751884518, 0.49270357948000915, 0.37747308677222924, 11.896144680107868, 0.4765838108385891, 0.37603978531187326, 8.008651099150734, 0.4949011312905886, 0.3875358976804048, 10.473183113663724, 0.4808725755510949, 0.4942190053159332, 6.982232621692047, 0.3050129179676043, 0.49486898238541266, 8.030580614695367, 0.4291118206863942, 0.4415672587604942, 7.5673366923162355, 0.44305834364817137, 0.4295173865166342, 20.671010456531818, 0.3953851132350567, 0.4406595630911131, 7.92395745968175, 0.4069816998004283, 0.40674507449107056, 7.969420896184747])


graph_sketch([0.40505816354772195, 0.4869567116763057, 21.016101034193134, 0.49429572802124117, 0.37817678834729007, 11.88531901412757, 0.4575476205935828, 0.38648937185277443, 7.998634311036691, 0.4948913519265683, 0.39525793945901305, 10.460655991061612, 0.4873332372720022, 0.49486226044028536, 6.957788711930871, 0.3050813371629978, 0.49488863164186503, 8.00424037890391, 0.4434012109772869, 0.42423934912158673, 7.589869536414149, 0.4357458779475116, 0.43019295321438533, 20.695037497845174, 0.3946403287759486, 0.4239031901999343, 7.949000889539569])



get_rho_crit_between_segments([0.4950144396824467, 0.3050925429786619, 12.468243067288178, 0.49501279716610785, 0.3064953575990344, 13.358014952319502, 0.3755286145227356, 0.49490346117258976, 12.497847190972228, 0.35561488755176035, 0.320966684120064, 3.2213455594017533, 0.49486695755021326, 0.49502573938787237, 16.846600884381985, 0.40152078397595614, 0.420389768024985, 9.030379328381404, 0.3050228173480108, 0.3706415814359488, 2.356743362539853, 0.4949951503216461, 0.3690367849632777, 6.834976069631723])

graph_sketch([0.49501322983062435, 0.3050041297259664, 12.367363228320468, 0.4950119558613447, 0.30567637890218524, 13.257250836314025, 0.3733289814131301, 0.49491120128498167, 12.399184864040947, 0.3542528673307998, 0.3138289181743819, 3.3199528092637713, 0.4949275830782928, 0.49502834707260474, 16.22086150080458, 0.39906370817320413, 0.417214985626097, 8.955048192293155, 0.30500530455549324, 0.37186482738807536, 2.235599836882767, 0.49499344568653103, 0.3693150495209246, 6.608506748349908])

graph_sketch([0.49501753075723753, 0.30501280766908945, 12.244769465367327, 0.4950155930688981, 0.3051037282593012, 13.134714106353755, 0.37216035210556125, 0.49493959683806965, 12.278104036927225, 0.35212420460361465, 0.307391988818653, 3.424255183315075, 0.4936566315774386, 0.4950161518550521, 15.431145070472972, 0.3968478949758145, 0.41085201276771893, 8.856472559668264, 0.30502803110502347, 0.3739864182607201, 2.1012877754750745, 0.49502363142260064, 0.37143400041459784, 6.330278966221217])

graph_sketch([0.49503349397756, 0.30496374650130686, 11.934883339775272, 0.49502962903442516, 0.30496862502103694, 12.824855597698237, 0.37312738367722953, 0.4950041413579115, 12.002191745768132, 0.3511444577237263, 0.3049994905492732, 3.5947305055533123, 0.4945284374737623, 0.4950255318342596, 13.923350677607576, 0.38608773908083843, 0.3986273155833891, 8.602915501945624, 0.305052929786243, 0.3728403112048195, 1.7629767509761405, 0.4949998327619576, 0.37285528983432853, 5.730936682898375])

get_rho_crit_between_segments([0.4950411054780292, 0.3049476896499808, 11.840727931683977, 0.49504140782347683, 0.3049508238259842, 12.730703346377302, 0.373742783362558, 0.4949991319165337, 11.931229050356764, 0.3513494115817135, 0.30499702812961327, 3.6238763961059157, 0.4948529793641143, 0.49502653187398876, 13.527510985260733, 0.3848910700285866, 0.39664734756500203, 8.555643763437153, 0.3049825970733714, 0.37320508624337356, 1.6871127695311792, 0.4949759660230464, 0.3735616830543836, 5.5691170364476]
)

graph_sketch([0.49504449364674924, 0.30493748356244954, 11.730162984193798, 0.49504673764480056, 0.3049395260386945, 12.620141880682016, 0.3746643938155143, 0.4950081536545587, 11.852231045906775, 0.3519010263783789, 0.30500099931261093, 3.649526332773714, 0.49493846429019167, 0.4950275443305515, 13.08482505582598, 0.3834284077440875, 0.39390454936710856, 8.503224498869528, 0.3050007803819703, 0.3738396844295817, 1.6020477822628598, 0.4949658464987217, 0.3746475106517567, 5.38558131367577])

graph_sketch([0.49506276575444813, 0.30490855722422666, 11.535453697224327, 0.4950660127586513, 0.3049087605296442, 12.425438239615993, 0.37671409894458946, 0.49499510117328777, 11.72441501403554, 0.35374919784187725, 0.3049976093375933, 3.6797485844780007, 0.4949842187488931, 0.49502738654566003, 12.34725724286437, 0.3792888902010415, 0.38831902827158005, 8.418200547464728, 0.3050237778523175, 0.37437299357003534, 1.4611926813459517, 0.49490896278688135, 0.37603666398929686, 5.076650600264489]

)

get_rho_crit_between_segments([0.40929620549673534, 0.48726423680570635, 21.0549366117227, 0.4950038867488152, 0.37701930319653937, 11.884569496292167, 0.44366306231709696, 0.39498676595540927, 7.999485908587565, 0.4950055602410389, 0.41703328267252665, 10.449282633385131, 0.4950055624351917, 0.4897730547658753, 7.353678241863933, 0.30498390929194946, 0.4850965155531635, 7.948398445921493, 0.4458384154165538, 0.4054371599269293, 7.636290847129617, 0.4258505047239532, 0.43242863460283865, 20.790109848782258]
)

graph_sketch([0.4111054952267171, 0.48624188458221707, 21.098439998822755, 0.49487433502248585, 0.3852057679914591, 11.898453557139872, 0.45831960458134974, 0.3902976650188396, 8.010363813255102, 0.4948744546975858, 0.4280054848845795, 10.462007065280554, 0.4837408176650608, 0.4809584765701447, 8.026458858264066, 0.30500395923862467, 0.46669497716447605, 7.9222475693240115, 0.4310298974438204, 0.4081888222335771, 7.70818403092096, 0.4275116470594414, 0.4303311334196095, 20.87353794734128]
)
graph_sketch([0.4145356748897133, 0.48387851328577397, 21.17486047599251, 0.4925003557152381, 0.3918173535700069, 11.924726963619655, 0.4813792921694617, 0.3810352121430406, 8.031303195200739, 0.4812446491223916, 0.4395547981785629, 10.533990062777042, 0.4671199002004712, 0.4699019291499268, 8.330279399033145, 0.3049186359011222, 0.4376110351335616, 7.87543900317689, 0.41234442945313765, 0.416789074674999, 7.8014021201436945, 0.427817789480215, 0.42317269426324283, 20.957564898625478]
)

# 7 solutions
get_rho_crit_between_segments([0.41519328751480356, 0.4840717565896009, 21.233002444073218, 0.4804928046215981, 0.3978387192043475, 11.92766611673881, 0.47736419204222147, 0.38498416053504275, 8.033133955392003, 0.4566128889816549, 0.45802916429994556, 10.725169377515533, 0.44810306447701354, 0.47252819993509515, 8.456727911167212, 0.32001124597477204, 0.40411544373696856, 7.839565607513642, 0.41454564537737093, 0.40842827696490197, 7.853917770713693])

get_rho_crit_between_segments([0.4170966304784449, 0.4828331031779127, 21.276985884870385, 0.47456483589407006, 0.40503429182552325, 11.964394375784236, 0.4888745484412859, 0.39292338221123313, 8.062562216937492, 0.4623627150677457, 0.44943121255368734, 10.851538630957526, 0.44202102379105934, 0.4789289200476846, 8.536354340620692, 0.34370589651844996, 0.42205738706993795, 7.863350517741973, 0.40804984994940907, 0.41161114620388567, 7.921452268567251]
)


# 6 values
# get_rho_crit_between_segments([0.4221250710391486, 0.4825243183243031, 21.31621820772236, 0.4765129631768031, 0.4036964490856236, 11.986935409197127, 0.4949393480848293, 0.3983443996829274, 8.080708893980049, 0.44127766509282457, 0.46209725459036594, 10.969169450665023, 0.42736072520022017, 0.48753998215177863, 8.627041010183557, 0.38335448211660966, 0.42304881305173725, 7.929118503172335]
# )
#
#
# get_rho_crit_between_segments([0.421936993166263, 0.48207158669365724, 21.303339360551046, 0.47498170052082445, 0.4037068864565806, 11.978297156026139, 0.49482700241459254, 0.3967031214240495, 8.073629424524238, 0.4422176343563642, 0.46334746751036265, 10.897989411117527, 0.428021232432596, 0.4869043820603708, 8.611096554986423, 0.38115688079739296, 0.4227917580551108, 7.911117991999711])



graph_sketch([0.35975402380961286, 0.49504093498391905, 6.974056558835942, 0.4755333769476858, 0.49502690923669973, 11.792717574378532, 0.37475862218987904, 0.30499059910875076, 1.6879083265536163, 0.32216437779937623, 0.30491997563215123, 9.321438505671383, 0.49505054538166354, 0.44839293577837147, 4.807581631858494, 0.3473554985279565, 0.3049193735274608, 15.190145897058391, 0.3726713254429879, 0.49500453246045045, 12.043379023948939])
get_fidelity_vs_sigma_with_rho_between_waveguide_inside_segment([0.35975402380961286, 0.49504093498391905, 6.974056558835942, 0.4755333769476858, 0.49502690923669973, 11.792717574378532, 0.37475862218987904, 0.30499059910875076, 1.6879083265536163, 0.32216437779937623, 0.30491997563215123, 9.321438505671383, 0.49505054538166354, 0.44839293577837147, 4.807581631858494, 0.3473554985279565, 0.3049193735274608, 15.190145897058391, 0.3726713254429879, 0.49500453246045045, 12.043379023948939])

get_fidelity_vs_sigma_with_rho_between_waveguide_inside_segment([0.4784128962011589, 0.42128872277987023, 8.214246587421215, 0.39867281130273274, 0.4709247837964105, 13.215212677928685, 0.4454767994587347, 0.4869766886476547, 2.2203966894348075, 0.3596660899647406, 0.3565458623613169, 8.96101327487578, 0.4795228198745466, 0.49209843374971807, 5.488303927163904, 0.4628965049466531, 0.35567378169463826, 15.31925565728688, 0.38756811103077704, 0.49501113444719674, 12.767766516236442])

get_fidelity_vs_sigma_with_rho_between_waveguide_inside_segment([0.4687139371822258, 0.4126319544086875, 8.308566424912254, 0.4005270136505223, 0.4662496362377522, 13.281958054657016, 0.4344838036929285, 0.4938952373210389, 2.2813342189133268, 0.35954168997995, 0.3607539867703518, 8.979258550780518, 0.4743153475159846, 0.49296945963996214, 5.524463408787376, 0.46340393890776566, 0.35095110518473055, 15.328906431744533, 0.38739786311949675, 0.4950092113864273, 12.790430743210177])

get_fidelity_vs_sigma_with_rho_between_waveguide_inside_segment([0.4332031170636182, 0.4949680664301993, 21.46500393965912, 0.3981004935802919, 0.3398057568789833, 25.60949016604101, 0.4291653886805982, 0.4949644843900969, 20.450528495277474])

get_fidelity_vs_sigma_with_rho_between_waveguide_inside_segment([0.4098415034331622, 0.4647367066752521, 23.362303468913197, 0.40462470160249847, 0.34356382499659643, 25.86832955347541, 0.43047498917732546, 0.49504982350057464, 22.20524039806517])

get_fidelity_vs_sigma_with_rho_between_waveguide_inside_segment([0.421062682757598, 0.4040439174250062, 17.132513705934624, 0.4382739098017427, 0.4950262152859176, 9.604764338798365, 0.44271969061797084, 0.41365452586755264, 12.356210482812637])

get_fidelity_vs_sigma_with_rho_between_waveguide_inside_segment([0.48910129624672655, 0.4944721405450573, 0.03171906808571361, 0.4950415990484049, 0.49504641963989754, 19.136868306905296, 0.49504465691039456, 0.495062566826915, 24.26510943818125])
graph_sketch([0.48910129624672655, 0.4944721405450573, 0.03171906808571361, 0.4950415990484049, 0.49504641963989754, 19.136868306905296, 0.49504465691039456, 0.495062566826915, 24.26510943818125])

get_fidelity_vs_sigma_with_rho_between_waveguide_inside_segment([0.4950344730131765, 0.37001955245024765, 7.4491855113388645, 0.4306192490969078, 0.49503550023003645, 18.650491979802784, 0.4951098066518431, 0.4678869779678108, 25.160385927017657])
get_fidelity_vs_sigma_with_rho_between_waveguide_inside_segment([0.4950346506945494, 0.3597140655404756, 8.14793217016581, 0.42337613741815633, 0.4950411574548384, 19.51439856704482, 0.4950991696817273, 0.4671032504627519, 27.800160341467812])
graph_sketch([0.4950432338104017, 0.3529962050808162, 8.73290502427561, 0.41895168382877707, 0.495034060684127, 20.32625433873871, 0.4951040917787109, 0.4676687522357898, 30.393160662860517])
get_fidelity_vs_sigma_with_rho_between_waveguide_inside_segment([0.4950432338104017, 0.3529962050808162, 8.73290502427561, 0.41895168382877707, 0.495034060684127, 20.32625433873871, 0.4951040917787109, 0.4676687522357898, 30.393160662860517])

get_fidelity_vs_sigma_with_rho_between_waveguide_inside_segment([0.4950714820344568, 0.4950495219934343, 19.256253328770462, 0.3649787523426319, 0.3430774400742722, 3.576149758902229, 0.4034466281998886, 0.4145033126676537, 14.4650277718425, 0.4949486140522633, 0.41217165487894825, 1.2797505745529096])

get_fidelity_vs_sigma_with_rho_between_waveguide_inside_segment([0.49739772705203517, 0.49747974971921427, 19.068513623274026, 0.39090843731414643, 0.3387835186453003, 3.69955249022523, 0.3978272727094832, 0.4237503530154707, 14.805992590217063, 0.49508390562970217, 0.32564294461951204, 1.3897790523838396])
get_fidelity_vs_sigma_with_rho_between_waveguide_inside_segment([0.4950758832829501, 0.49504784641752786, 19.072417619956727, 0.38741910204330093, 0.34201900170525906, 3.6860414352284208, 0.39803211380561143, 0.42082034981654237, 14.769682284803174, 0.4949639083983064, 0.3424673375412999, 1.3747197086562366])
# get_fidelity_vs_sigma_with_rho_between_waveguide_inside_segment([0.4857,0.4345,47.1117/2,0.4057,0.4896,40.5109/2,0.4857,0.4345,47.1117/2])


# get_fidelity_vs_sigma_with_rho_between_waveguide_inside_segment([0.30490271123933715, 0.49503788371841884, 2.4846488202094044, 0.3040971254606763, 0.4100416395654931, 23.595765755748957, 0.49541855913654953, 0.49299328498498846, 24.65440887507644, 0.4952850837358106, 0.4950450962792285, 20.628306213722656])
# get_fidelity_vs_sigma_with_rho_between_waveguide_inside_segment([0.4857,0.4345,47.1117/2,0.4057,0.4896,40.5109/2,0.4857,0.4345,47.1117/2])
# get_fidelity_vs_sigma_with_rho_between_waveguide_inside_segment([0.4857,0.4345,47.1117/2,0.4057,0.4896,40.5109/2,0.4857,0.4345,47.1117/2])


# get_fidelity_vs_sigma_with_rho_between_waveguide_inside_segment([0.30490271123933715, 0.49503788371841884, 2.4846488202094044, 0.3040971254606763, 0.4100416395654931, 23.595765755748957, 0.49541855913654953, 0.49299328498498846, 24.65440887507644, 0.4952850837358106, 0.4950450962792285, 20.628306213722656])

# covariance_error_graph([0.375, 0.425, 24.627, 0.429, 0.363, 26.304, 0.391, 0.45, 23.315])
# get_fidelity_vs_sigma_with_rho_between_waveguide_inside_segment([0.39294820654288315, 0.4699797453536133, 17.82940942529142, 0.4695686340842414, 0.36263664136922696, 11.711143721434485, 0.4699910166024295, 0.4690093495282821, 24.25014581343141, 0.38506912987129105, 0.46992968034498406, 8.09438856604849])
# covariance_error_graph([0.38722030509297006, 0.4600062353046215, 17.510215229413344, 0.46000907081206066, 0.35801065973166524, 11.849761146535071, 0.45995126126692115, 0.4599579134795146, 21.620594744512832, 0.37379951200115347, 0.4598307209261217, 7.812769873811055])
get_fidelity_vs_sigma_with_rho_between_waveguide_inside_segment([0.3843398460250679, 0.4599973338635527, 17.624279542128168, 0.45999001605482726, 0.35463419001981866, 11.770466094301545, 0.4600080342485956, 0.45999543381746455, 23.230028374449343, 0.37177413220579614, 0.4599991992499739, 7.631346886677996])
get_fidelity_vs_sigma_with_rho_between_waveguide_inside_segment([0.3891805211693974, 0.45994919997355477, 17.57643656609791, 0.4599204560414965, 0.3595500042593564, 11.884797426681, 0.459886579248139, 0.4597980546157198, 20.749212171804704, 0.375325493972266, 0.4595543215321187, 7.9968590485999815])
get_fidelity_vs_sigma_with_rho_between_waveguide_inside_segment([0.38800689992046605, 0.4599901553629824, 17.623530450939306, 0.45951325315539143, 0.35722659638722987, 11.775933495797476, 0.45988352567898566, 0.45974893456818594, 21.447765853756234, 0.37500121002445047, 0.4599643351832685, 7.933676429648191])
get_fidelity_vs_sigma_with_rho_between_waveguide_inside_segment([0.39247424196608827, 0.4699386313598639, 17.850445624488543, 0.4691665141289171, 0.3606803273554273, 11.71875209628184, 0.46995921105273664, 0.4699443296099578, 24.382015185438156, 0.38672296870425377, 0.46964825079636185, 8.127456307296514])
get_fidelity_vs_sigma_with_rho_between_waveguide_inside_segment([0.3867982092960626, 0.4598736957469783, 17.665538953315902, 0.4594839688589636, 0.35623209332428873, 11.782902989730658, 0.45995757214802885, 0.4599720177574798, 22.10401071883383, 0.3735535030312501, 0.4600347797664802, 7.7950880988931495])
get_fidelity_vs_sigma_with_rho_between_waveguide_inside_segment([0.38581115913929054, 0.4599668949796587, 17.713895433079998, 0.45945323137910893, 0.35482666922563294, 11.728670360403973, 0.4599879014100011, 0.4596676646738616, 22.553711898218367, 0.3744048723649314, 0.45999080132823766, 7.932645897149784])
get_fidelity_vs_sigma_with_rho_between_waveguide_inside_segment([0.4857,0.4345,47.1117/2,0.4057,0.4896,40.5109/2,0.4857,0.4345,47.1117/2])
get_fidelity_vs_sigma_with_rho_between_waveguide_inside_segment([0.39707806426274206, 0.44999513579936934, 22.670671276651458, 0.44999333019456694, 0.37552196849943714, 22.50360067420975, 0.39383299185172577, 0.4499940540026893, 21.777894184923642],deterministic=True)
get_fidelity_vs_sigma_with_rho_between_waveguide_inside_segment([0.4553776748841885, 0.32522012269169365, 34.17483048362124, 0.3200370561790618, 0.35019175826906773, 38.70063808546897, 0.43449495992175324, 0.3299682353750312, 41.02314404464175])
get_fidelity_vs_sigma_with_rho_between_waveguide_inside_segment([0.4489331028434909, 0.35259358206666097, 3.0038377782049897, 0.37948876382560676, 0.36775949380946765, 0.887092273326516, 0.369304079650319, 0.4499850407977652, 13.433919674702144, 0.35016961483217734, 0.3501011724853802, 6.653191309038562, 0.44968285537358654, 0.3703048988618179, 5.200358237951969, 0.45001799003386567, 0.37367032417967144, 12.572892578625563, 0.37698361335665215, 0.449918469893154, 13.769192379321881])
get_fidelity_vs_sigma_with_rho_between_waveguide_inside_segment([0.3901854281480041, 0.44998080203042157, 12.233034353582427, 0.40891118366615875, 0.449951501063593, 8.482549368804055, 0.44998255994657405, 0.37525599214022387, 16.339551991256112, 0.3646452293984369, 0.35003520929155324, 6.704243944525012, 0.38138387264083123, 0.4499897633809829, 14.190727766881384])
get_fidelity_vs_sigma_with_rho_between_waveguide_inside_segment([0.3751472121258264, 0.45000530610407463, 17.055916050354035, 0.39670152714476814, 0.3499916748445693, 30.97729471496034, 0.3751425296303577, 0.45000579285495707, 17.055921588925163])
get_fidelity_vs_sigma_with_rho_between_waveguide_inside_segment([0.45001426828089686, 0.38781051681299894, 17.322120126877174, 0.3500658723738427, 0.3757482968341877, 8.701030912211532, 0.3560549266179467, 0.4497918902789044, 9.105750521357404, 0.35002071609721724, 0.3652066592277311, 10.546809105053798, 0.45002830188206305, 0.37612533225716716, 13.74531265911632])
get_fidelity_vs_sigma_with_rho_between_waveguide_inside_segment([0.36464183239553916, 0.449957597855721, 13.701845097603659, 0.41152433992938675, 0.35864009732673596, 26.395700501831804, 0.3879381787091325, 0.4499561766794087, 17.714055885816293])
get_fidelity_vs_sigma_with_rho_between_waveguide_inside_segment([0.39093675674915224, 0.44995623463407664, 21.386550460330955, 0.43004626352017317, 0.3679236326131606, 26.229087798116268, 0.3906326018233662, 0.4500200925268849, 21.29765125327227],deterministic=True)
get_fidelity_vs_sigma_with_rho_between_waveguide_inside_segment([0.371, 0.42, 25.17, 0.427, 0.361, 26.725, 0.391, 0.450, 23.755],deterministic=True)

get_fidelity_vs_sigma_with_rho_between_waveguide_inside_segment([0.4699568426672263, 0.3967909732189501, 16.279093184974965, 0.36959862521599446, 0.43475286953807984, 19.603792836168115, 0.469904677967996, 0.4502358522349852, 14.529743264249118, 0.46989921570122434, 0.3914776169323468, 8.218124850887664],deterministic=True)
get_fidelity_vs_sigma_with_rho_between_waveguide_inside_segment([0.39919327043938274, 0.45000070636933787, 14.174345296198025, 0.4499812944874085, 0.38749184746490345, 14.058221557091574, 0.3585948694436403, 0.3500093808546068, 4.191259982536617, 0.3943303495970257, 0.45000013540180467, 11.070572914949045],deterministic=True)



# [0.49, 0.437, 20.795785714285714, 0.412, 0.49, 19.665285714285716, 0.49, 0.437, 20.795785714285714]
# MORE X SOLUTIONS
# graph_sketch([0.4857,0.4345,47.1117/2,0.4057,0.4896,40.5109/2,0.4857,0.4345,47.1117/2])
# covariance_error_graph([0.4857,0.4345,47.1117/2,0.4057,0.4896,40.5109/2,0.4857,0.4345,47.1117/2])

get_fidelity_vs_sigma_with_rho_between_waveguide_inside_segment([0.38606823118764283, 0.41523567485920215, 27.67227982840459, 0.45333630427546706, 0.30648083776012547, 31.701048279284585, 0.4209872687217838, 0.49500083206139983, 18.810134253609228])
get_fidelity_vs_sigma_with_rho_between_waveguide_inside_segment([0.38606823118764283, 0.41523567485920215, 27.67227982840459, 0.45333630427546706, 0.30648083776012547, 31.701048279284585, 0.4209872687217838, 0.49500083206139983, 18.810134253609228])


graph_sketch([0.495, 0.443, 20.1035, 0.417, 0.495, 18.827, 0.495, 0.443, 20.1035])
graph_sketch([0.49, 0.437, 21.867, 0.412, 0.49, 20.737, 0.49, 0.437, 21.867])


covariance_error_graph([0.495, 0.443, 20.1035, 0.417, 0.495, 18.827, 0.495, 0.443, 20.1035])
covariance_error_graph([0.49, 0.437, 21.867, 0.412, 0.49, 20.737, 0.49, 0.437, 21.867])


covariance_error_graph([0.495, 0.443, 20.1035, 0.417, 0.495, 18.827, 0.495, 0.443, 20.1035])

graph_sketch([0.49511459724168566, 0.44287535985799253, 20.103488817677963, 0.4172102163742812, 0.4950810160388714, 18.937779857849428, 0.4951144868692877, 0.4428755033213033, 20.103499095975874])
covariance_error_graph([0.49511459724168566, 0.44287535985799253, 20.103488817677963, 0.4172102163742812, 0.4950810160388714, 18.937779857849428, 0.4951144868692877, 0.4428755033213033, 20.103499095975874])
covariance_error_graph([0.49, 0.437, 21.867, 0.412, 0.49, 20.737, 0.49, 0.437, 21.867])


covariance_error_graph([0.48979993346880457, 0.43730476976679844, 20.08157829170177, 0.41194563752639557, 0.49004265324274576, 18.95103789577923, 0.48979992646040815, 0.4373047884803068, 20.08157823447326])


graph_sketch([0.48979993346880457, 0.43730476976679844, 20.08157829170177, 0.41194563752639557, 0.49004265324274576, 18.95103789577923, 0.48979992646040815, 0.4373047884803068, 20.08157823447326])
covariance_error_graph([0.48979993346880457, 0.43730476976679844, 20.08157829170177, 0.41194563752639557, 0.49004265324274576, 18.95103789577923, 0.48979992646040815, 0.4373047884803068, 20.08157823447326])

graph_sketch([0.490,0.437,20.082,0.412,0.49,18.951,0.49,0.437,20.082])
covariance_error_graph([0.490,0.437,20.082,0.412,0.49,18.951,0.49,0.437,20.082])



for avg_error in np.linspace(0.1*width_error,3*width_error,10):
    print("curr error:",avg_error/width_error)
    correlation_graph_sketch([0.48998382230610527, 0.4374840043019467, 20.08182154630714, 0.4119062160245618, 0.48998240825149614, 18.950814306362524, 0.4899563136301337, 0.4374740373103104, 20.081510857308434],average_width_error=avg_error)




correlation_graph_sketch([0.48998382230610527, 0.4374840043019467, 20.08182154630714, 0.4119062160245618, 0.48998240825149614, 18.950814306362524, 0.4899563136301337, 0.4374740373103104, 20.081510857308434])




graph_sketch([0.48998382230610527, 0.4374840043019467, 20.08182154630714, 0.4119062160245618, 0.48998240825149614, 18.950814306362524, 0.4899563136301337, 0.4374740373103104, 20.081510857308434])

graph_sketch([0.375, 0.425, 24.627, 0.429, 0.363, 26.304, 0.391, 0.45, 23.315])
correlation_graph_sketch([0.375, 0.425, 24.627, 0.429, 0.363, 26.304, 0.391, 0.45, 23.315])


# covariance_error_graph([0.48998382230610527, 0.4374840043019467, 20.08182154630714, 0.4119062160245618, 0.48998240825149614, 18.950814306362524, 0.4899563136301337, 0.4374740373103104, 20.081510857308434])
# graph_sketch([0.48998382230610527, 0.4374840043019467, 20.08182154630714, 0.4119062160245618, 0.48998240825149614, 18.950814306362524, 0.4899563136301337, 0.4374740373103104, 20.081510857308434])
#
#
# graph_sketch([0.42259808476248933, 0.46999422143741215, 22.363398571472683, 0.46968008495261077, 0.3975015547802589, 16.099947752498196, 0.3542754932166928, 0.3300397739287245, 7.216540983619076, 0.38234122709661483, 0.46998097129619576, 11.293661085662324])
# graph_sketch([0.490001350189227, 0.43843898184833024, 18.371665326340167, 0.4162896739941937, 0.48998938568997635, 18.12334637457229, 0.49000449338423313, 0.43844597492396664, 18.37162011709061])
# graph_sketch([0.42515885825350663, 0.4700023218926401, 23.875964657472682, 0.46988556410172916, 0.38948170426843115, 14.490056141313511, 0.3492593950424428, 0.33004730669860083, 8.173755726133573, 0.37729032728967815, 0.46990333177503907, 10.432968577931522])
# graph_sketch([0.42639059303031074, 0.46993295587970607, 24.717901233184143, 0.46962704774543257, 0.37267115026362496, 11.49147283004339, 0.3465357551889045, 0.3304798770234779, 9.89626968045448, 0.3708939658454141, 0.4700037988665317, 9.11686463568933])
# graph_sketch([0.39791447605350266, 0.4599439949881493, 17.99232021590007, 0.45981698901157697, 0.37764389147395316, 14.763277793655485, 0.4600114328585026, 0.45988138975326603, 12.68426360993559, 0.3940589314047884, 0.45995826995805533, 11.615485142877835])
# graph_sketch([0.4009798285028621, 0.46996385538529245, 16.80965995431849, 0.46966831242398605, 0.3772679682957493, 12.544115343307293, 0.4699393023400225, 0.46994916218851984, 17.86320363022342, 0.3987428861887348, 0.4699495999536006, 9.374966879405248])


graph_sketch([0.375, 0.425, 24.627, 0.429, 0.363, 26.304, 0.391, 0.45, 23.315])
covariance_error_graph([0.375, 0.425, 24.627, 0.429, 0.363, 26.304, 0.391, 0.45, 23.315])

# for average_width_error in np.linspace(0.5*width_error,1.5*width_error,5):
#     print(average_width_error/width_error)
#     print(width_error)
#     correlation_graph_sketch([0.39366618154091204, 0.46997689577585905, 15.454192556465719, 0.3709754710789766, 0.33003567892508157, 7.771646734262678, 0.469973447870517, 0.46994870048099224, 7.475821947308521, 0.4684421536693974, 0.35939029662117816, 3.0349379562263876, 0.4691274087144081, 0.37991513286750617, 4.036540646160831, 0.35518947990571437, 0.33009838111368583, 10.94855731731937, 0.3897327235349098, 0.4700002883680059, 14.732204568744248],average_width_error=average_width_error,print_stuff=False)
#     # correlation_graph_sketch(
#     #     [0.48998382230610527, 0.4374840043019467, 20.08182154630714, 0.4119062160245618, 0.48998240825149614,
#     #      18.950814306362524, 0.4899563136301337, 0.4374740373103104, 20.081510857308434],average_width_error=average_width_error,print_stuff=False)

# correlation_graph_sketch([0.4699428474067878, 0.41567299075272157, 11.051893201166926, 0.4691318425095868, 0.4698782185560465, 9.642086390285877, 0.39452590898925954, 0.46928411635108075, 6.242170598010814, 0.38459657284867615, 0.4697892818255729, 6.782813262118064, 0.4699899235918775, 0.40455745735179843, 15.07668812707037, 0.33010263128482986, 0.42390436083684657, 0.8516816619610434])













graph_sketch([0.375, 0.425, 24.627, 0.4285, 0.3627, 26.415, 0.391, 0.45, 23.315])
graph_sketch([0.37495663810865404, 0.425102665411733, 24.627081071680657, 0.4284894225671782, 0.36267359008973465, 26.41523448373749, 0.3907773876902017, 0.4498837798597435, 23.31467475430432])

graph_sketch([0.3714368162863226, 0.42086093419967474, 24.95652161344538, 0.42664236708072134, 0.3608344210814691, 26.59624477599109, 0.38888068116357005, 0.44791465600873126, 23.560804843012345])

graph_sketch([0.3714368162863226, 0.42, 25.17, 0.427, 0.361, 26.725, 0.391, 0.450, 23.755],deterministic=True)

graph_sketch([0.37495663810865404, 0.425102665411733, 24.627081071680657, 0.4284894225671782, 0.36267359008973465, 26.41523448373749, 0.3907773876902017, 0.4498837798597435, 23.31467475430432])
graph_sketch([0.37495663810865404, 0.425102665411733, 24.627081071680657, 0.4284894225671782, 0.36267359008973465, 26.41523448373749, 0.3907773876902017, 0.4498837798597435, 23.31467475430432])


graph_sketch([0.3737693890279279, 0.4236714198883069, 24.738085173223507, 0.42785768259833484, 0.3620434413901478, 26.475317747474424, 0.39011254748936164, 0.4492016360894642, 23.39469554546146])

graph_sketch([0.37259583822085696, 0.422256760653877, 24.848467511214803, 0.42723898371215546, 0.36142737759012394, 26.53589858997494, 0.38947186061481454, 0.4485355071879864, 23.477058115201494])


graph_sketch([0.36145600617283496, 0.4092048439412886, 25.618507076566214, 0.4219698001265174, 0.3561080678374644, 26.943381664327276, 0.3853443290583383, 0.444227358472214, 24.162144167947236])

graph_sketch([0.36183002545489446, 0.40955677439522625, 25.61859193075411, 0.4223830375880404, 0.3565358792722044, 26.94655701789994, 0.38615087553625327, 0.4450705681993653, 24.170683591412907])

graph_sketch([0.362531792997336, 0.4102466866958547, 25.62202910025767, 0.4229706875726792, 0.3571454435043865, 26.950518759646403, 0.38713207184547294, 0.4460770136318619, 24.180086650109676])

graph_sketch([0.3636920983916463, 0.41140075259614967, 25.630020311021383, 0.4238440478665643, 0.35804469916669596, 26.95596651271265, 0.3884144692046665, 0.4473673214562013, 24.191643227812936])

graph_sketch([0.3648254242330844, 0.41256622316447344, 25.649821343571965, 0.4256371387040366, 0.35971244553596227, 26.975627642673565, 0.3893581274424301, 0.4482950697518193, 24.211568469166597],deterministic=True)



graph_sketch([0.44894413395120275, 0.38180775800702815, 18.566427867930688, 0.3711092959894884, 0.4481423214660076, 18.869637117367574, 0.41661857272360203, 0.4125516268545695, 7.524196016945221, 0.44932572550280964, 0.3983123140587879, 17.7738378920845],deterministic=True)


graph_sketch([0.3729766958348179, 0.44996228772488556, 16.678130077851723, 0.39496084620969796, 0.3500023981921909, 16.735058303945898, 0.4441436907878615, 0.3678161986158613, 11.369555961432345, 0.3990727760634982, 0.44997364975974247, 26.026292527476386],deterministic=True)
graph_sketch([0.371, 0.42, 25.17, 0.427, 0.361, 26.725, 0.391, 0.450, 23.755],deterministic=True)
graph_sketch([0.3707029299133514, 0.4196277831112225, 25.169312978388106, 0.42732370629232946, 0.36151809616250163, 26.725383387311485, 0.39053965663992773, 0.44962671926576286, 23.755183559163335],deterministic=True)
graph_sketch([0.3675460257226313, 0.41585942046737595, 25.394400927780282, 0.4259790118230765, 0.36021436347022273, 26.863529009571614, 0.39039337732722434, 0.44957736955432587, 23.95741820169217],deterministic=True)
graph_sketch([0.3648254242330844, 0.41256622316447344, 25.649821343571965, 0.4256371387040366, 0.35971244553596227, 26.975627642673565, 0.3893581274424301, 0.4482950697518193, 24.211568469166597],deterministic=True)
graph_sketch([0.3646971583366394, 0.412126362323761, 25.658046078016966, 0.4261331260204315, 0.3602679669857025, 26.980631436518383, 0.39101049304008484, 0.4499884843826294, 24.21756081418449],deterministic=True)
graph_sketch([0.3646971583366394, 0.412126362323761, 25.658046078016966, 0.4261331260204315, 0.3602679669857025, 26.980631436518383, 0.39101049304008484, 0.4499884843826294, 24.21756081418449])


covariance_error_graph([0.39366618154091204, 0.46997689577585905, 15.454192556465719, 0.3709754710789766, 0.33003567892508157, 7.771646734262678, 0.469973447870517, 0.46994870048099224, 7.475821947308521, 0.4684421536693974, 0.35939029662117816, 3.0349379562263876, 0.4691274087144081, 0.37991513286750617, 4.036540646160831, 0.35518947990571437, 0.33009838111368583, 10.94855731731937, 0.3897327235349098, 0.4700002883680059, 14.732204568744248])
covariance_error_graph([0.46992475191420463, 0.4122557799313141, 9.379380176785318,
                              0.4697554907666833, 0.46989602617018367, 13.346579454657903,
                              0.3912160443444747, 0.4698240628413758, 5.6864299993144565,
                              0.374143303931053, 0.46750260458333603, 6.026781089131028,
                              0.47000025492324043, 0.3988159815634961, 14.142048585746513,
                              0.3330855893912344, 0.4168933879596165, 0.8126533894520162])

#









# graph_sketch([0.371,0.35,2.49,0.35,0.353,24.87,0.371,0.35,2.49])

# graph_sketch([0.483,0.497,18.9057*0.5,0.311,0.300,28.6237*0.5,0.484,0.497,18.9057*0.5])


covariance_error_graph([0.39247424196608827, 0.4699386313598639, 17.850445624488543, 0.4691665141289171, 0.3606803273554273, 11.71875209628184, 0.46995921105273664, 0.4699443296099578, 24.382015185438156, 0.38672296870425377, 0.46964825079636185, 8.127456307296514])
# covariance_error_graph([0.3742327753484413, 0.4699010369108006, 14.69918218155685, 0.46908143416024095, 0.34395644086088134, 10.001856256214438, 0.4699463470038769, 0.4697239203396556, 27.726342559480734, 0.38168241552250276, 0.4692454570665041, 6.685535551001749])
covariance_error_graph([0.4799948461402813, 0.4154440105217039, 20.943386851910052, 0.34619748366677566, 0.45440535795712156, 7.2311029041272725, 0.43686802893587107, 0.4797862483430116, 19.422900394062143, 0.47992498256114535, 0.42085047085448063, 18.59506263522333])
covariance_error_graph([0.4457679739539582, 0.3865644189812155, 21.03985170034828, 0.3308363122895021, 0.43171672642505754, 7.22104285902275, 0.43301984800928595, 0.4798204513134276, 19.345742987305158, 0.4799749603247678, 0.41588527314849805, 18.493255126231954])
covariance_error_graph([0.3729766958348179, 0.44996228772488556, 16.678130077851723, 0.39496084620969796, 0.3500023981921909, 16.735058303945898, 0.4441436907878615, 0.3678161986158613, 11.369555961432345, 0.3990727760634982, 0.44997364975974247, 26.026292527476386],deterministic=True)




# get_fidelity_vs_rho_between_segments([0.375, 0.425, 24.627, 0.429, 0.363, 26.304, 0.391, 0.45, 23.315])

# graph_sketch([0.41930242121656386, 0.4800167385142259, 21.33276068949854, 0.4788646461673328, 0.3898075340435803, 12.002166684293243, 0.48004664602994884, 0.39155648561228923, 8.095779442407059, 0.43143912192582456, 0.47024047763370536, 11.034676773806412, 0.41949040106563296, 0.47985456326851283, 8.663257389910475, 0.3963098994435085, 0.4173829658781016, 7.952152730216549])
# get_rho_crit_between_segments([0.41930242121656386, 0.4800167385142259, 21.33276068949854, 0.4788646461673328, 0.3898075340435803, 12.002166684293243, 0.48004664602994884, 0.39155648561228923, 8.095779442407059, 0.43143912192582456, 0.47024047763370536, 11.034676773806412, 0.41949040106563296, 0.47985456326851283, 8.663257389910475, 0.3963098994435085, 0.4173829658781016, 7.952152730216549])
# graph_sketch([0.4123561603338839, 0.4684399760889638, 22.58339856324674, 0.44888189103572584, 0.38190578964245886, 24.790637953178546, 0.4121402359466627, 0.4684539351182967, 22.519408811200226])
# get_rho_crit_between_segments([0.4123561603338839, 0.4684399760889638, 22.58339856324674, 0.44888189103572584, 0.38190578964245886, 24.790637953178546, 0.4121402359466627, 0.4684539351182967, 22.519408811200226])
#
#
# graph_sketch([0.418163597367826, 0.4799436362589238, 21.31894310146513, 0.47413842502704434, 0.3953194635833085, 11.998269590940849, 0.480026220203823, 0.39185020699660056, 8.091968307805478, 0.44040700323160487, 0.4633868568853502, 10.978800803457915, 0.42172483161866714, 0.4799512781467989, 8.635790919669017, 0.38964045955414683, 0.42758273039852046, 7.906625641217369])
# get_rho_crit_between_segments([0.418163597367826, 0.4799436362589238, 21.31894310146513, 0.47413842502704434, 0.3953194635833085, 11.998269590940849, 0.480026220203823, 0.39185020699660056, 8.091968307805478, 0.44040700323160487, 0.4633868568853502, 10.978800803457915, 0.42172483161866714, 0.4799512781467989, 8.635790919669017, 0.38964045955414683, 0.42758273039852046, 7.906625641217369])



# 3 segment solutions:
graph_sketch([0.4857, 0.4345, 47.1117/2, 0.4057, 0.4896, 40.5109/2,0.4857,0.4345,47.1117/2])
get_fidelity_vs_sigma_perfect_correlation([[0.40141329364175965, 0.4558732386936501, 23.363544297718526, 0.4430577734799344, 0.37753391350660725, 25.871801240291756, 0.40264159713292996, 0.45985381795706487, 22.786005552777223],
                         [0.4857, 0.4345, 47.1117/2, 0.4057, 0.4896, 40.5109/2,0.4857,0.4345,47.1117/2]],two_params=True)

get_fidelity_vs_sigma_perfect_correlation([[0.375, 0.425, 24.627, 0.429, 0.363, 26.304, 0.391, 0.45, 23.315],
                         [0.4857, 0.4345, 47.1117/2, 0.4057, 0.4896, 40.5109/2,0.4857,0.4345,47.1117/2]],two_params=True)


graph_sketch([0.40141329364175965, 0.4558732386936501, 23.363544297718526, 0.4430577734799344, 0.37753391350660725, 25.871801240291756, 0.40264159713292996, 0.45985381795706487, 22.786005552777223])
get_fidelity_vs_sigma_perfect_correlation([0.40141329364175965, 0.4558732386936501, 23.363544297718526, 0.4430577734799344, 0.37753391350660725, 25.871801240291756, 0.40264159713292996, 0.45985381795706487, 22.786005552777223])
get_fidelity_vs_sigma_perfect_correlation([0.3721693222379981, 0.42021715362327955, 25.587050462625967, 0.4308363378516594, 0.3650240135034013, 26.93651239258484, 0.39457323199235184, 0.453313800508819, 24.165189838998252])

graph_sketch([0.3700497129890663, 0.4179109835465324, 25.633159651453667, 0.42972684640507075, 0.3638826825827091, 26.961140508355058, 0.39328949017251164, 0.4520768656257691, 24.19673661795223])

graph_sketch([0.3647678639497576, 0.4124472755341923, 25.648739478623273, 0.42545994527288106, 0.3596359820361771, 26.974803606398844, 0.3893056953934985, 0.4481890698380222, 24.210455508421806])
graph_sketch([0.3648254242330844, 0.41256622316447344, 25.649821343571965, 0.4256371387040366, 0.35971244553596227, 26.975627642673565, 0.3893581274424301, 0.4482950697518193, 24.211568469166597],deterministic=True)

get_fidelity_vs_sigma_perfect_correlation([0.3648254242330844, 0.41256622316447344, 25.649821343571965, 0.4256371387040366, 0.35971244553596227, 26.975627642673565, 0.3893581274424301, 0.4482950697518193, 24.211568469166597],deterministic=True)

get_fidelity_vs_sigma_perfect_correlation([0.4857, 0.4345, 47.1117/2, 0.4057, 0.4896, 40.5109/2,0.4857,0.4345,47.1117/2])
get_fidelity_vs_sigma_perfect_correlation([0.4123561603338839, 0.4684399760889638, 22.58339856324674, 0.44888189103572584, 0.38190578964245886, 24.790637953178546, 0.4121402359466627, 0.4684539351182967, 22.519408811200226])


graph_sketch([0.4123561603338839, 0.4684399760889638, 22.58339856324674, 0.44888189103572584, 0.38190578964245886, 24.790637953178546, 0.4121402359466627, 0.4684539351182967, 22.519408811200226])


get_rho_crit_between_segments([0.4857, 0.4345, 47.1117/2, 0.4057, 0.4896, 40.5109/2,0.4857,0.4345,47.1117/2])
get_rho_crit_between_segments([0.4123561603338839, 0.4684399760889638, 22.58339856324674, 0.44888189103572584, 0.38190578964245886, 24.790637953178546, 0.4121402359466627, 0.4684539351182967, 22.519408811200226])


get_rho_crit_between_segments([0.4123561603338839, 0.4684399760889638, 22.58339856324674, 0.44888189103572584, 0.38190578964245886, 24.790637953178546, 0.4121402359466627, 0.4684539351182967, 22.519408811200226])
graph_sketch([0.4057144198726955, 0.46123567272252014, 22.72144203365038, 0.4500484503354326, 0.38223117599095074, 24.833888895710313, 0.40549200256807183, 0.4612437708709565, 22.656887116827306])
get_rho_crit_between_segments([0.4057144198726955, 0.46123567272252014, 22.72144203365038, 0.4500484503354326, 0.38223117599095074, 24.833888895710313, 0.40549200256807183, 0.4612437708709565, 22.656887116827306])
get_rho_crit_between_segments([0.4044665315137576, 0.45998008221218173, 22.74614128917278, 0.44926724575315063, 0.3813770853356301, 24.86264877821265, 0.4042406655776807, 0.45998520623655714, 22.681519485122543])
get_rho_crit_between_segments([0.4880854229756083, 0.4368354436989885, 23.507550784267195, 0.4066188931959855, 0.4902948283176243, 20.241655303664444, 0.48808541293487523, 0.436835450879151, 23.507550937077013])



get_rho_crit_between_segments([0.4045, 0.46, 22.746, 0.4495, 0.3815, 24.8625, 0.404, 0.46, 22.6815])
graph_sketch([0.4044665315137576, 0.45998008221218173, 22.74614128917278, 0.44926724575315063, 0.3813770853356301, 24.86264877821265, 0.4042406655776807, 0.45998520623655714, 22.681519485122543])
get_rho_crit_between_segments([0.4044665315137576, 0.45998008221218173, 22.74614128917278, 0.44926724575315063, 0.3813770853356301, 24.86264877821265, 0.4042406655776807, 0.45998520623655714, 22.681519485122543])

get_rho_crit_between_segments([0.40439070992953535, 0.4599739463514366, 22.740211474366507, 0.44863170176777367, 0.3810459907795578, 24.95326639815811, 0.4040891053113204, 0.4599801942204744, 22.65420905077311])
get_rho_crit_between_segments([0.404380817996913, 0.45999410159953485, 22.7441770176085, 0.4481978305745901, 0.3808138324161273, 25.017136057440357, 0.4040159098103229, 0.4600037696474881, 22.639670692428705])
get_rho_crit_between_segments([0.4041895520184405, 0.4598491164765502, 22.828870946199903, 0.4456046320171482, 0.37900949520049393, 25.29511320886589, 0.4033842021945972, 0.4598779759315268, 22.59763924681398])
get_rho_crit_between_segments([0.40438139595170686, 0.45978611026325256, 22.996158255765433, 0.44475477090920196, 0.3784783920656712, 25.456289216656767, 0.40312109648005134, 0.459824984275742, 22.634389675878136])
get_rho_crit_between_segments([0.39392883382764315, 0.4469529228263021, 23.780867182410034, 0.4397906639246326, 0.3736078077703229, 25.893378691186946, 0.3997009765754595, 0.4573968669990418, 22.97743514721077])
graph_sketch([0.39392883382764315, 0.4469529228263021, 23.780867182410034, 0.4397906639246326, 0.3736078077703229, 25.893378691186946, 0.3997009765754595, 0.4573968669990418, 22.97743514721077])

get_rho_crit_between_segments([0.38604080806892493, 0.4372093428587025, 24.36462534346618, 0.43643759140298294, 0.37025835248439964, 26.177874653323933, 0.3979031075534459, 0.45648131274027925, 23.21486540410241])

get_rho_crit_between_segments([0.375, 0.425, 24.627, 0.429, 0.363, 26.304, 0.391, 0.45, 23.315])


get_rho_crit_between_segments([0.4880854229756083, 0.4368354436989885, 23.507550784267195, 0.4066188931959855, 0.4902948283176243, 20.241655303664444, 0.48808541293487523, 0.436835450879151, 23.507550937077013])


get_rho_crit_between_segments([0.48998382230610527, 0.4374840043019467, 20.08182154630714, 0.4119062160245618, 0.48998240825149614, 18.950814306362524, 0.4899563136301337, 0.4374740373103104, 20.081510857308434])


get_fidelity_vs_rho_between_segments([0.48998382230610527, 0.4374840043019467, 20.08182154630714, 0.4119062160245618, 0.48998240825149614, 18.950814306362524, 0.4899563136301337, 0.4374740373103104, 20.081510857308434])


get_fidelity_vs_rho_between_segments([0.4857, 0.4345, 47.1117/2, 0.4057, 0.4896, 40.5109/2,0.4857,0.4345,47.1117/2])
get_fidelity_vs_rho_between_segments([0.48998382230610527, 0.4374840043019467, 20.08182154630714, 0.4119062160245618, 0.48998240825149614, 18.950814306362524, 0.4899563136301337, 0.4374740373103104, 20.081510857308434])

get_fidelity_vs_sigma_perfect_correlation([0.4857, 0.4345, 47.1117/2, 0.4057, 0.4896, 40.5109/2,0.4857,0.4345,47.1117/2])
get_fidelity_vs_sigma_perfect_correlation([0.375, 0.425, 24.627, 0.429, 0.363, 26.304, 0.391, 0.45, 23.315])

graph_sketch([0.3648254242330844, 0.41256622316447344, 25.649821343571965, 0.4256371387040366, 0.35971244553596227, 26.975627642673565, 0.3893581274424301, 0.4482950697518193, 24.211568469166597],deterministic=True)

graph_sketch([0.375, 0.425, 24.627, 0.4285, 0.3627, 26.415, 0.391, 0.45, 23.315])
graph_sketch([0.37495663810865404, 0.425102665411733, 24.627081071680657, 0.4284894225671782, 0.36267359008973465, 26.41523448373749, 0.3907773876902017, 0.4498837798597435, 23.31467475430432])

graph_sketch([0.3714368162863226, 0.42086093419967474, 24.95652161344538, 0.42664236708072134, 0.3608344210814691, 26.59624477599109, 0.38888068116357005, 0.44791465600873126, 23.560804843012345])

graph_sketch([0.3714368162863226, 0.42, 25.17, 0.427, 0.361, 26.725, 0.391, 0.450, 23.755],deterministic=True)



get_fidelity_vs_rho_between_segments([0.4857, 0.4345, 47.1117/2, 0.4057, 0.4896, 40.5109/2,0.4857,0.4345,47.1117/2])

graph_sketch([0.375, 0.425, 24.627, 0.429, 0.363, 26.304, 0.391, 0.45, 23.315])

graph_sketch([0.4857, 0.4345, 47.1117/2, 0.4057, 0.4896, 40.5109/2,0.4857,0.4345,47.1117/2])




# get_average_error_partially_correlated([0.48998382230610527, 0.4374840043019467, 20.08182154630714, 0.4119062160245618, 0.48998240825149614, 18.950814306362524, 0.4899563136301337, 0.4374740373103104, 20.081510857308434])
# get_average_error_partially_correlated([0.375, 0.425, 24.627, 0.429, 0.363, 26.304, 0.391, 0.45, 23.315])



covariance_error_graph([0.48998382230610527, 0.4374840043019467, 20.08182154630714, 0.4119062160245618, 0.48998240825149614, 18.950814306362524, 0.4899563136301337, 0.4374740373103104, 20.081510857308434])
correlation_graph_sketch([0.48998382230610527, 0.4374840043019467, 20.08182154630714, 0.4119062160245618, 0.48998240825149614, 18.950814306362524, 0.4899563136301337, 0.4374740373103104, 20.081510857308434])
graph_sketch([0.48998382230610527, 0.4374840043019467, 20.08182154630714, 0.4119062160245618, 0.48998240825149614, 18.950814306362524, 0.4899563136301337, 0.4374740373103104, 20.081510857308434])


# 4 segment solutions:
get_rho_crit_between_segments([0.3983475776462084, 0.47957826389865243, 21.308876383137637, 0.4091330214989601, 0.34745649879885004, 23.346689569261187, 0.46047734703648935, 0.45852085508171286, 18.11086046394327, 0.3535729830333715, 0.40493940370086656, 17.51056847223798])
graph_sketch([0.3983475776462084, 0.47957826389865243, 21.308876383137637, 0.4091330214989601, 0.34745649879885004, 23.346689569261187, 0.46047734703648935, 0.45852085508171286, 18.11086046394327, 0.3535729830333715, 0.40493940370086656, 17.51056847223798])
get_rho_crit_between_segments([0.39338179126998707, 0.4800017617812801, 22.6589849647307, 0.38756955159345896, 0.3307199239320138, 24.681796291964414, 0.4555944703962892, 0.4484678413858951, 21.962864135172875, 0.33159060598240103, 0.37766173654950846, 19.88571362468307])
graph_sketch([0.39882011700995573, 0.4711939495438471, 22.716609223692206, 0.4438947801321474, 0.3754160814812457, 24.785233367435115, 0.4473643760800202, 0.4677067539582988, 22.208034259540707, 0.3200003240598114, 0.3801990691045598, 9.809633887639357])
graph_sketch([0.3975182400759075, 0.4701789080861246, 22.724832035123686, 0.44352692746947214, 0.37513861186422714, 24.801516579515596, 0.44688040014531544, 0.46658846888093025, 22.419175128900097, 0.31999772933424486, 0.3802948557032175, 9.859542301888395])
get_rho_crit_between_segments([0.3975182400759075, 0.4701789080861246, 22.724832035123686, 0.44352692746947214, 0.37513861186422714, 24.801516579515596, 0.44688040014531544, 0.46658846888093025, 22.419175128900097, 0.31999772933424486, 0.3802948557032175, 9.859542301888395])
graph_sketch([0.40557799337760175, 0.46131184208957665, 22.72000385340732, 0.4508301057684583, 0.38150678337264793, 24.8341523579116, 0.4054642682976072, 0.46122279370790087, 22.65530561627743, 0.39891862378684156, 0.4016883248923803, 0.9978414836984407])
get_rho_crit_between_segments([0.40557799337760175, 0.46131184208957665, 22.72000385340732, 0.4508301057684583, 0.38150678337264793, 24.8341523579116, 0.4054642682976072, 0.46122279370790087, 22.65530561627743, 0.39891862378684156, 0.4016883248923803, 0.9978414836984407])
graph_sketch([0.4057144198726955, 0.46123567272252014, 22.72144203365038, 0.4500484503354326, 0.38223117599095074, 24.833888895710313, 0.40549200256807183, 0.4612437708709565, 22.656887116827306, 0.4, 0.4, 1])
graph_sketch([0.405686498673865, 0.46136003248106183, 22.699638007552714, 0.44707214082522134, 0.37950801720227967, 24.81033221814294, 0.40553936350306086, 0.4614564151824904, 22.6400494189322, 0.4268164240364718, 0.3602661860836655, 0.0031752304290619776])
graph_sketch([0.4057773278996134, 0.4613204316301766, 22.720727508216605, 0.4496717850459148, 0.3818904314174803, 24.83244776213679, 0.40555796527790566, 0.4613326202565626, 22.65634008464508, 0.40211691711833164, 0.39678032627148274, 0.00234237911027316])
get_rho_crit_between_segments([0.4057773278996134, 0.4613204316301766, 22.720727508216605, 0.4496717850459148, 0.3818904314174803, 24.83244776213679, 0.40555796527790566, 0.4613326202565626, 22.65634008464508, 0.40211691711833164, 0.39678032627148274, 0.00234237911027316])
graph_sketch([0.4057144198726955, 0.46123567272252014, 22.72144203365038, 0.4500484503354326, 0.38223117599095074, 24.833888895710313, 0.40549200256807183, 0.4612437708709565, 22.656887116827306, 0.4, 0.4, 0.0001])
graph_sketch([0.3744674489906036, 0.34202825283054417, 27.49558354856328, 0.3847079753387891, 0.45497865461044945, 27.48039032241816, 0.4051489384776804, 0.3204091333730398, 27.84325568158009, 0.4590983369991817, 0.32951855879883635, 16.71574628754999])
graph_sketch([0.37123604519353554, 0.3389858674497928, 27.451880150482992, 0.3865124698174775, 0.45667371581930505, 27.48294061093258, 0.41016544552769457, 0.32319800822586736, 27.84294584987144, 0.45438425370853053, 0.3285282173064244, 16.719229059885915])
get_rho_crit_between_segments([0.37123604519353554, 0.3389858674497928, 27.451880150482992, 0.3865124698174775, 0.45667371581930505, 27.48294061093258, 0.41016544552769457, 0.32319800822586736, 27.84294584987144, 0.45438425370853053, 0.3285282173064244, 16.719229059885915])
get_fidelity_vs_rho_between_segments([0.37123604519353554, 0.3389858674497928, 27.451880150482992, 0.3865124698174775, 0.45667371581930505, 27.48294061093258, 0.41016544552769457, 0.32319800822586736, 27.84294584987144, 0.45438425370853053, 0.3285282173064244, 16.719229059885915])
graph_sketch([0.3618708277928045, 0.32999809534711405, 27.366149926688923, 0.3983620923485239, 0.4696935700786533, 27.499924718287232, 0.4243057173579634, 0.3300104246744588, 27.844585265604447, 0.44456744546469684, 0.33000161422809077, 16.72888029282068])
get_rho_crit_between_segments([0.3618708277928045, 0.32999809534711405, 27.366149926688923, 0.3983620923485239, 0.4696935700786533, 27.499924718287232, 0.4243057173579634, 0.3300104246744588, 27.844585265604447, 0.44456744546469684, 0.33000161422809077, 16.72888029282068])
graph_sketch([0.361860102694521, 0.32999587063880637, 27.365202952443077, 0.3983606356930735, 0.4697127336751565, 27.499942662359672, 0.42443947496440276, 0.3299880266099241, 27.84470820764701, 0.44466632609416606, 0.3299911936110271, 16.729006820103198])
get_fidelity_vs_rho_between_segments([0.361860102694521, 0.32999587063880637, 27.365202952443077, 0.3983606356930735, 0.4697127336751565, 27.499942662359672, 0.42443947496440276, 0.3299880266099241, 27.84470820764701, 0.44466632609416606, 0.3299911936110271, 16.729006820103198])
graph_sketch([0.36174743589047026, 0.33010475576680776, 27.025596212967617, 0.4022719234150374, 0.4699223580361367, 28.549242856198568, 0.4385564398974553, 0.3300506208966493, 24.686089773045932, 0.4591114347110751, 0.33007698058153784, 15.14379224064301])


# 5 segment solutoins:
graph_sketch([0.4147002601250747, 0.47832240870436715, 15.078054140257445, 0.3323811103689302, 0.31992814879577236, 18.326941054929133, 0.4800580011664064, 0.33612957212082417, 8.150247219483143, 0.32003593236898253, 0.37684584906075225, 14.21815710267353, 0.3199238608987444, 0.47998314261304176, 23.545810882689203])
get_fidelity_vs_rho_between_segments([0.4147002601250747, 0.47832240870436715, 15.078054140257445, 0.3323811103689302, 0.31992814879577236, 18.326941054929133, 0.4800580011664064, 0.33612957212082417, 8.150247219483143, 0.32003593236898253, 0.37684584906075225, 14.21815710267353, 0.3199238608987444, 0.47998314261304176, 23.545810882689203])
get_rho_crit_between_segments([0.4147002601250747, 0.47832240870436715, 15.078054140257445, 0.3323811103689302, 0.31992814879577236, 18.326941054929133, 0.4800580011664064, 0.33612957212082417, 8.150247219483143, 0.32003593236898253, 0.37684584906075225, 14.21815710267353, 0.3199238608987444, 0.47998314261304176, 23.545810882689203])
graph_sketch([0.4147002601250747, 0.47832240870436715, 15.078054140257445, 0.3323811103689302, 0.31992814879577236, 18.326941054929133, 0.4800580011664064, 0.33612957212082417, 8.150247219483143, 0.32003593236898253, 0.37684584906075225, 14.21815710267353, 0.3199238608987444, 0.47998314261304176, 23.545810882689203])
graph_sketch([0.42760360430790373, 0.4800058781613129, 22.819411134587313, 0.4800071422594536, 0.3970859035936021, 12.479991393688753, 0.4799925357378971, 0.405037323119654, 8.615038173277648, 0.42360362231373305, 0.4800007569595576, 12.388982003598633, 0.4309044265905175, 0.4800046164227577, 10.08616227889962])
get_rho_crit_between_segments([0.42742513301508256, 0.48000615621277853, 22.70820178384983, 0.4800017229604855, 0.3975450416609335, 12.429866202754514, 0.4799402392182052, 0.40600214254988365, 8.560227629824032, 0.42607197434124616, 0.48001041173569575, 12.281791400781461, 0.42820468778498516, 0.4799583819990902, 9.967699938926001])
graph_sketch([0.42742513301508256, 0.48000615621277853, 22.70820178384983, 0.4800017229604855, 0.3975450416609335, 12.429866202754514, 0.4799402392182052, 0.40600214254988365, 8.560227629824032, 0.42607197434124616, 0.48001041173569575, 12.281791400781461, 0.42820468778498516, 0.4799583819990902, 9.967699938926001])
get_fidelity_vs_rho_between_segments([0.42912628043335616, 0.4803972437883214, 21.344899058391913, 0.4803427524812507, 0.39985261260188754, 11.990344177552883, 0.48002393878003996, 0.4060178760546134, 8.084573663139848, 0.41982831199117354, 0.480203892548481, 11.070003741523246, 0.42781420299327483, 0.48016157361073974, 8.695381049501721])
get_rho_crit_between_segments([0.42912628043335616, 0.4803972437883214, 21.344899058391913, 0.4803427524812507, 0.39985261260188754, 11.990344177552883, 0.48002393878003996, 0.4060178760546134, 8.084573663139848, 0.41982831199117354, 0.480203892548481, 11.070003741523246, 0.42781420299327483, 0.48016157361073974, 8.695381049501721])






# H GATE, GAP = 1.2
# graph_sketch([0.379,0.486,29.6481,0.5,0.31,53.75,0.379,0.486,29.6481])
graph_sketch([0.43, 0.452, 35.145, 0.422, 0.325, 16.761, 0.43, 0.452, 35.164],analytic_vals=True)
get_fidelity_vs_sigma_perfect_correlation([[0.43, 0.452, 35.145, 0.422, 0.325, 16.761, 0.43, 0.452, 35.164],
                         [0.486,0.379,29.6481/2,0.31,0.5,53.75/2,0.486,0.379,29.6481/2]],two_params=True)
get_fidelity_vs_sigma_perfect_correlation([0.43, 0.452, 35.145, 0.422, 0.325, 16.761, 0.43, 0.452, 35.164])

graph_sketch([0.4302188750640788, 0.45199420366164966, 35.644830483507356, 0.4225437724163206, 0.32545578039755446, 16.76105109367905, 0.4302743188822319, 0.4520082748856469, 35.663868120526494])
get_fidelity_vs_sigma_perfect_correlation([0.4302188750640788, 0.45199420366164966, 35.644830483507356, 0.4225437724163206, 0.32545578039755446, 16.76105109367905, 0.4302743188822319, 0.4520082748856469, 35.663868120526494])
graph_sketch([0.42916614185706115, 0.45088935688589177, 35.675831863605026, 0.42374030058569795, 0.326477186855047, 16.761273544928866, 0.42921357399523075, 0.4508924729066878, 35.695131690905015])
get_fidelity_vs_sigma_perfect_correlation([0.42916614185706115, 0.45088935688589177, 35.675831863605026, 0.42374030058569795, 0.326477186855047, 16.761273544928866, 0.42921357399523075, 0.4508924729066878, 35.695131690905015])
graph_sketch([0.42877357231048646, 0.4503147792305109, 35.69072659536671, 0.43225255244304994, 0.3343088821112234, 16.767624473660927, 0.42881626999889116, 0.45031123457174616, 35.710173172500774])
get_fidelity_vs_sigma_perfect_correlation([0.42877357231048646, 0.4503147792305109, 35.69072659536671, 0.43225255244304994, 0.3343088821112234, 16.767624473660927, 0.42881626999889116, 0.45031123457174616, 35.710173172500774])
graph_sketch([0.4287101171302525, 0.4500933820176707, 35.695867925452, 0.4406586715136021, 0.3419247679256168, 16.774442964104402, 0.42875060181506686, 0.4500865675202648, 35.71539423893255])
get_fidelity_vs_sigma_perfect_correlation([0.4287101171302525, 0.4500933820176707, 35.695867925452, 0.4406586715136021, 0.3419247679256168, 16.774442964104402, 0.42875060181506686, 0.4500865675202648, 35.71539423893255])

graph_sketch([0.42870310919507043, 0.44991227882900986, 35.699363924356895, 0.44988688961434875, 0.3500097383819321, 16.782472713661058, 0.42874167400354446, 0.4499021993299708, 35.718967085855375],deterministic=True)
get_fidelity_vs_sigma_perfect_correlation([0.42870310919507043, 0.44991227882900986, 35.699363924356895, 0.44988688961434875, 0.3500097383819321, 16.782472713661058, 0.42874167400354446, 0.4499021993299708, 35.718967085855375],deterministic=True)

graph_sketch([0.428, 0.449, 35.838, 0.45, 0.35, 16.787, 0.428, 0.449, 35.767],deterministic=True)

graph_sketch([0.433, 0.454, 36.233, 0.4395, 0.34, 16.6525, 0.4325, 0.454, 36.105])
graph_sketch([0.486,0.379,29.6481/2,0.31,0.5,53.75/2,0.486,0.379,29.6481/2])
get_fidelity_vs_sigma_perfect_correlation([0.433, 0.454, 36.233, 0.4395, 0.34, 16.6525, 0.4325, 0.454, 36.105])
get_fidelity_vs_sigma_perfect_correlation([0.486,0.379,29.6481/2,0.31,0.5,53.75/2,0.486,0.379,29.6481/2])

get_fidelity_vs_sigma_perfect_correlation([0.433, 0.454, 36.233, 0.4395, 0.34, 16.6525, 0.4325, 0.454, 36.105])
graph_sketch([0.433, 0.454, 36.233, 0.4395, 0.34, 16.6525, 0.4325, 0.454, 36.105])
graph_sketch([0.433, 0.454, 36.233333333333334, 0.4395, 0.34, 16.6525, 0.4325, 0.454, 36.105333333333334])
# graph_sketch([0.433, 0.454, 36.28888888888889, 0.4395, 0.34, 16.6525, 0.4325, 0.454, 36.16088888888889] )
for i in np.linspace(-0.1,0.4,10):
    print("@@@\n\nPARAMS:",[0.433, 0.454, 36.0+i, 0.4395, 0.34, 16.6525, 0.4325, 0.454, 35.872+i],"\n\n")
    graph_sketch([0.433, 0.454, 36.0+i, 0.4395, 0.34, 16.6525, 0.4325, 0.454, 35.872+i])


graph_sketch([0.432648771166698, 0.45393558619610963, 35.93379585113319, 0.43930646548455, 0.3400142973696891, 16.65253975410869, 0.4324018750457524, 0.4538060703072634, 35.871956141077526])
graph_sketch([0.433, 0.454, 35.934, 0.439, 0.34, 16.653, 0.432, 0.454, 35.872])

graph_sketch([0.42870310919507043, 0.44991227882900986, 35.699363924356895, 0.44988688961434875, 0.3500097383819321, 16.782472713661058, 0.42874167400354446, 0.4499021993299708, 35.718967085855375],deterministic=True)
graph_sketch([0.428906030567466, 0.44999278272532545, 35.83951750371481, 0.4500011225709372, 0.34998935726148317, 16.776975596207524, 0.4286925215414807, 0.4499280593304558, 35.770745137698455],deterministic=True)
graph_sketch([0.4288865733841144, 0.44997245748482817, 35.83766048019133, 0.45000181881881646, 0.34998575815872524, 16.787378994157972, 0.42870864970853567, 0.4499585908126799, 35.767112180219414],deterministic=True)
graph_sketch([0.4288865733841144, 0.44997245748482817, 35.83766048019133, 0.45000181881881646, 0.34998575815872524, 16.787378994157972, 0.42870864970853567, 0.4499585908126799, 35.767112180219414])
graph_sketch([0.4288865733841144, 0.44997245748482817, 35.83766048019133, 0.45000181881881646, 0.34998575815872524, 16.787378994157972, 0.42870864970853567, 0.4499585908126799, 35.767112180219414],deterministic=True)
graph_sketch([0.4308243591869351, 0.44997451552993095, 36.085207263689895, 0.45001138034467275, 0.34999411348093856, 16.70660582125026, 0.42629953481012534, 0.44997440732743493, 34.19663952738418],deterministic=True)
graph_sketch([0.43343015067298074, 0.44999973685142963, 36.26863031406008, 0.44999359090212226, 0.3499797829822308, 16.58628049664455, 0.422641756854722, 0.4499893088027105, 31.88072137862819],deterministic=True)
graph_sketch([0.43633888225545736, 0.44999111440467765, 36.5109924189534, 0.44952017153569834, 0.349977006942013, 16.53903978365884, 0.4177145023456773, 0.44998731011331594, 29.188882100976535],deterministic=True)
graph_sketch([0.4382823919849238, 0.4500143200698091, 35.64645764197679, 0.4456753681178709, 0.3499932763733956, 16.904005666953857, 0.41208445800132176, 0.45001288305179044, 25.90911334624649],deterministic=True)
graph_sketch([0.4380053994436562, 0.45000503356010846, 34.731747872037715, 0.4445125131625745, 0.34998717960916387, 16.963537143171173, 0.4117043459097096, 0.45000527925333994, 25.145253370867856],deterministic=True)
graph_sketch([0.4372599275256075, 0.45000434915144827, 32.84038039377518, 0.4424399365034183, 0.34998059583261903, 17.03729575969475, 0.41143640466079745, 0.45000035267452165, 23.795772755827965],deterministic=True)
graph_sketch([0.43672083561942204, 0.45002105179483365, 31.512725543451158, 0.4410732107486153, 0.35000152256231065, 17.06397153542702, 0.4112607519765821, 0.44999482517403366, 22.861361288789443],deterministic=True)
graph_sketch([0.4360501582896607, 0.4500021868533522, 30.50405258689383, 0.4402929515434649, 0.3499848866110632, 17.067341119289875, 0.4119140574480584, 0.45000374654837716, 22.442705493367065],deterministic=True)
graph_sketch([0.4360501582896607, 0.4500021868533522, 30.50405258689383, 0.4402929515434649, 0.3499848866110632, 17.067341119289875, 0.4119140574480584, 0.45000374654837716, 22.442705493367065])
graph_sketch([0.4348397227323077, 0.4487055405802767, 30.360776289321464, 0.4401513471218532, 0.3500335422783487, 17.07442283314764, 0.410847272139333, 0.44937332189806634, 22.247213347157246])
graph_sketch([0.42722856486497696, 0.43829861045463425, 30.67683310053911, 0.4392046649707892, 0.3499848250074142, 17.296570444575313, 0.40072388087306654, 0.44578363108298547, 20.4364971075721])
graph_sketch([0.45000726442145256, 0.3920128146837242, 16.460941933519397],deterministic=True)









# X^(2/3), g=1.2 SOLUTIONS
# graph_sketch([0.34997820850601263, 0.45017245556584834, 12.049153244245835, 0.45021765954576803, 0.3492847973140296, 21.41421909582758, 0.3499782134787607, 0.45017245077941453, 12.049153283752622])
graph_sketch([0.351, 0.46, 10.234, 0.459, 0.34, 17.039, 0.349, 0.46, 10.1305])
graph_sketch([0.358,0.457,21.89/2,0.485,0.34,28.0211/2,0.358,0.457,21.89/2 ])
get_fidelity_vs_sigma_perfect_correlation([0.351, 0.46, 10.234, 0.459, 0.34, 17.039, 0.349, 0.46, 10.1305])
get_fidelity_vs_sigma_perfect_correlation([0.358,0.457,21.89/2,0.485,0.34,28.0211/2,0.358,0.457,21.89/2 ])

for i in np.linspace(-0.03,0.02,10):
    params = [0.351, 0.46, 10.253 + i, 0.459, 0.34, 17.039, 0.349, 0.46, 10.1495 + i]
    print("\n\nCURRENT PARAMS:",params)
    graph_sketch(params)

graph_sketch([0.351, 0.46, 10.253, 0.459, 0.34, 17.039, 0.349, 0.46, 10.15])
graph_sketch([0.351, 0.46, 10.253, 0.459, 0.34, 17.039, 0.349, 0.46, 10.15])
graph_sketch([0.351, 0.46, 10.253, 0.459, 0.34, 17.039, 0.349, 0.46, 10.15])
# graph_sketch([0.358,0.457,21.890/2,0.485,0.340,28.021/2,0.358,0.457,21.890/2])
# graph_sketch([0.35, 0.46, 10.227, 0.46, 0.34, 10.99+6.001,  0.349, 0.46, 10.091])
graph_sketch([0.35, 0.46, 10.227, 0.46, 0.34, 10.99, 0.458, 0.34, 6.001, 0.349, 0.46, 10.091])

graph_sketch([0.35106045460176055, 0.4601873505290526, 10.252949390800644, 0.45894586919168784, 0.3398854418265453, 17.038819067519047, 0.3492924540588603, 0.4600311605005704, 10.14968885824658])
graph_sketch([0.35085605315029, 0.4599677755108272, 10.253122440468244, 0.45910601444307764, 0.3400067796429244, 17.03939540581759, 0.3490395777440054, 0.45973067966596814, 10.150433789991629])

# graph_sketch([0.367,0.331,78.805/2,0.307,0.488,58.698/2,0.367,0.331,78.805/2])



graph_sketch([0.3504298747713646, 0.4600315826941112, 10.227424146583608, 0.45998818434911565, 0.340000361358356, 10.990275948805674, 0.457824378813563, 0.3400074154071737, 6.001373866810587, 0.34924986783645356, 0.46002859115589945, 10.09143189048948])
graph_sketch([0.3499980793728727, 0.46002905511207925, 10.171119602566291, 0.4599894581758452, 0.3399892113591185, 10.980439760914086, 0.4577084070693712, 0.3399956768103847, 5.992340824798818, 0.34885143733228463, 0.46002575829955006, 10.034501131896194])
graph_sketch([0.34926233243093735, 0.4600325132854062, 10.077242941329072, 0.46000942207630213, 0.3399714225700489, 10.959541728048348, 0.4575543700372911, 0.3399852504586509, 5.972718304684016, 0.3481867720405856, 0.4600313948318206, 9.941247648846867])
graph_sketch([0.3478601020555804, 0.460051666441122, 9.882585994582485, 0.46001767719736997, 0.3399496825081428, 10.904909277628622, 0.4577350055616089, 0.339978222227183, 5.920509171217218, 0.34662926124826604, 0.460052443676171, 9.744967461811347])
graph_sketch([0.35665493883924254, 0.470007759941924, 9.395524652456078, 0.4700041700179813, 0.34083220527510766, 10.216080311336986, 0.4616113388678011, 0.3468560956832125, 5.214427249554481, 0.3496244330754652, 0.4700105951308132, 8.70637899642459])
graph_sketch([0.36587258029214864, 0.4791919693173671, 9.330995190725284, 0.4799499788167555, 0.35068477845123625, 10.15962300837, 0.46841115010892276, 0.35360983165288956, 5.155227335124862, 0.35882376448409864, 0.479974072770937, 8.603558433240154])
graph_sketch([0.36614431428518757, 0.4795869287482039, 9.235838147946055, 0.4799709635357454, 0.34992086169997433, 10.051769006364413, 0.4686879859027928, 0.35293718027084653, 5.041716590650333, 0.35755407589415683, 0.47988396052073024, 8.461207772744766])
graph_sketch([0.3660782890035546, 0.4792784810920708, 9.156994372678746, 0.4799477530755549, 0.34925129149737977, 9.94946175996128, 0.4690493165582397, 0.35195379709963304, 4.93396218032762, 0.3562986089922904, 0.4799978809275408, 8.331052354755105])
graph_sketch([0.366709541776869, 0.47995872795099676, 9.101378076844322, 0.4799948473641447, 0.3488087413323974, 9.877458704956213, 0.4692899751349317, 0.3514787317486557, 4.858265501903875, 0.3554485769054573, 0.4798897800441224, 8.239408400793524])
graph_sketch([0.36768319799322297, 0.47941257913749774, 8.850420441758269, 0.4799847453531652, 0.346496002083806, 9.49511144033851, 0.4715576209550271, 0.3478417972032738, 4.459590139461305, 0.350126180958849, 0.4798009615187282, 7.765954703966594])
graph_sketch([0.36768319799322297, 0.47941257913749774, 8.850420441758269, 0.4799847453531652, 0.346496002083806, 9.49511144033851, 0.4715576209550271, 0.3478417972032738, 4.459590139461305, 0.350126180958849, 0.4798009615187282, 7.765954703966594],deterministic=False)
graph_sketch([0.36828699900856504, 0.4799697483746264, 8.83288246935401, 0.47993432402422875, 0.34630905391894495, 9.466132784614114, 0.4716170209618115, 0.34753967085939136, 4.429719716578902, 0.3499222376071614, 0.4799372824035813, 7.730456570095782])
graph_sketch([0.36970395589463434, 0.47928962798444413, 8.636206149203105, 0.47998260148128374, 0.34419249912305894, 9.114069883436, 0.47346626201143527, 0.3436939151932062, 4.070390499247527, 0.3447924848247887, 0.4798517902687964, 7.299296650244686])
graph_sketch([0.37714945043129877, 0.47991746837113014, 8.231541801044989, 0.47998117605066304, 0.3395765283401168, 8.26355552974452, 0.4771763785793367, 0.33444410038146444, 3.233656375121847, 0.3317976626736413, 0.4792946107148129, 6.245738063258318])
graph_sketch([0.38323331259500665, 0.47960768197014364, 8.03282821316078, 0.47992432863384044, 0.33752711910984223, 7.7360008032189915, 0.47929402564261214, 0.3294207984451023, 2.732766389125064, 0.3252585772736272, 0.4799751031137313, 5.61619077275757])
graph_sketch([0.38323331259500665, 0.47960768197014364, 8.03282821316078, 0.47992432863384044, 0.33752711910984223, 7.7360008032189915, 0.47929402564261214, 0.3294207984451023, 2.732766389125064, 0.3252585772736272, 0.4799751031137313, 5.61619077275757],deterministic=False)
graph_sketch([0.38941285247310325, 0.47965661433092815, 7.915739947896402, 0.4799978120276851, 0.3343129373720894, 7.284941697861633, 0.4741334370472006, 0.329303377661356, 2.30373709505688, 0.3229212676044159, 0.47945686206772553, 5.135900318781529])

graph_sketch([0.40485707113925634, 0.47925819134683445, 7.948702133799234, 0.47991959441014326, 0.33585332997793343, 6.592070959702064, 0.47699443978621, 0.3324527740524432, 1.6187533228105029, 0.32227905555891073, 0.47920324875565606, 4.433143700500642])
graph_sketch([0.4013552245293883, 0.4800022336516592, 8.280346543156071, 0.48000120961722853, 0.3571187572644042, 2.598972844394803, 0.48000211205951276, 0.3199991426698452, 6.303904778319018, 0.3199963587654268, 0.48000380490202416, 4.966843546240101])

graph_sketch([0.4435111972843085, 0.4480016978717862, 7.133178857011359, 0.4508419638138734, 0.42645714360352605, 1.490420136561047, 0.4443586966062847, 0.4380067052180818, 3.1630451280136187, 0.44553031353132605, 0.4505308375254055, 5.6233354899076184])
graph_sketch([0.44487655438637114, 0.3772637974073753, 2.732245210031812, 0.38302781053253204, 0.4495955180591617, 5.499803581833624, 0.4492089800115598, 0.4074322041170841, 6.951727608658986, 0.3583416724541845, 0.4344535253379768, 1.248604830152664],deterministic=True)
graph_sketch([0.371, 0.42, 25.17, 0.427, 0.361, 26.725, 0.391, 0.450, 23.755],deterministic=True)
graph_sketch([0.35637829175095753, 0.4495030638459303, 4.148772163247833, 0.4495967711663962, 0.3625335220334779, 8.199102062555868, 0.3688392586009835, 0.4493743564555026, 4.660584509780035],deterministic=True)















# # X^(2/3), g=1.0 SOLUTIONS
# graph_sketch([0.301, 0.496, 6.503, 0.484, 0.3, 12.191, 0.301, 0.496, 6.503])
graph_sketch([0.301, 0.495, 6.503, 0.484, 0.3, 12.191, 0.301, 0.495, 6.503])
graph_sketch([0.301, 0.4955, 6.503, 0.484, 0.3, 12.191, 0.301, 0.4955, 6.503])
graph_sketch([0.30098837416387425, 0.49556900204206894, 6.502633895499643, 0.4842802188697784, 0.3000821169237364, 12.191135458408832, 0.30098848187642396, 0.4955687409093018, 6.5026353938523105])
graph_sketch([0.3022371239092377, 0.4918280795588683, 6.686133432886128, 0.4857915222722036, 0.3000165802554212, 12.171188226027043, 0.30223703742962404, 0.49182816889177694, 6.686133024391732])
graph_sketch([0.3030109577970338, 0.4901582951681372, 6.7777854705665685, 0.48693046455025285, 0.3002194747911449, 12.1520141542733, 0.3030111225274684, 0.4901582951566965, 6.777785402069592])
graph_sketch([0.3105705705705706, 0.49885885885885883, 6.83247239187124, 0.5, 0.3103803803803804, 12.155159810294684, 0.3105705705705706, 0.49885885885885883, 6.832472305622959])
# graph_sketch([0.46817079488371777, 0.4643663700547029, 4.250532702511367, 0.43219127246610456, 0.4484920782035164, 1.8697276795048081, 0.46955317560013865, 0.4644611565022436, 3.4784464451963273])
# graph_sketch([0.46017913609565686, 0.4614846251770739, 4.6960152414186265, 0.4680946105611265, 0.462819868363503, 2.7758420451009465, 0.4585938362273937, 0.46284883113776726, 2.1386602346354757])
















# H GATE, GAP = 1.0

# MUHAMMAD PARAMETERS
# w1 = 0.425, w2 = 0.389 , t = 33.1167714613773`
# w1 = 0.359, w2 = 0.359 , t = 43.31043872285401`
# w1 = 0.389, w2 = 0.425 , t = 33.1167714613773`
graph_sketch([0.425,0.389,33.1167714613773,0.359,0.359,43.31043872285401,0.389,0.425,33.1167714613773])
graph_sketch([0.425,0.389,33.1167714613773,0.359,0.359,43.31043872285401,0.389,0.425,33.1167714613773],deterministic=True)


graph_sketch([0.45,0.39,16.06])


graph_sketch([0.49, 0.31, 3.05, 0.49, 0.49, 8.16, 0.49, 0.31, 3.05],deterministic=True)
graph_sketch([0.49, 0.31, 3.05, 0.49, 0.49, 8.16, 0.49, 0.31, 3.05],deterministic=True)
graph_sketch([0.49, 0.31, 3.05, 0.49, 0.49, 8.16, 0.49, 0.31, 3.05],deterministic=True)
graph_sketch([0.4899944880618881, 0.31005446806987735, 3.0492391966379944, 0.4900077035708945, 0.49001265984075715, 8.156021371638165, 0.4899250670485525, 0.31001836467477617, 3.0490917238921313],deterministic=True)
graph_sketch([0.4900279218371208, 0.30997616910353737, 3.050258997486797, 0.4899918441776742, 0.49000434315040076, 8.156211800486362, 0.48999105924935943, 0.31001475136019035, 3.0502829284080795],deterministic=True)
graph_sketch([0.48542836260938627, 0.3121221232592896, 3.664110910331813, 0.4615778496604576, 0.4883652513804258, 6.844797328627163, 0.4852553068006711, 0.3101921779749949, 3.585694185749986],deterministic=True)
graph_sketch([0.4502010050251256, 0.49, 20.986533977200043, 0.49, 0.31, 9.830689138978908, 0.4502010050251256, 0.49, 20.94522089273649],deterministic=True)









# X^0.5 GATE, GAP = 1.0
graph_sketch([0.49, 0.324, 12.345, 0.36, 0.438, 18.275, 0.438, 0.336, 18.1],deterministic=True)
# graph_sketch([0.48852481740637527, 0.3429122912659924, 10.548843488060502, 0.4162453993357681, 0.4893444452902599, 15.090071377131911, 0.4573912807540846, 0.3571574490025023, 14.513764817905031],deterministic=True,w2_uncorrelated=False)
graph_sketch([0.48852481740637527, 0.3429122912659924, 10.548843488060502, 0.4162453993357681, 0.4893444452902599, 15.090071377131911, 0.4573912807540846, 0.3571574490025023, 14.513764817905031],deterministic=True,w2_uncorrelated=False)
graph_sketch([0.48999292262772093, 0.32426984729249636, 12.345637935371109, 0.36036086685458574, 0.4384764952054197, 18.275395321515454, 0.43792351576206123, 0.3364609801548085, 18.098406356039874],deterministic=True,w2_uncorrelated=True)
graph_sketch([0.4564995407038932, 0.37112611568952497, 10.105235838691387, 0.42055148167320694, 0.46917093720993225, 12.16134464323643, 0.4247599994376589, 0.36458459514429736, 13.656704181161002])
graph_sketch([0.45818597104169256, 0.40902990581635873, 10.413473211069814, 0.4404059695935637, 0.4693983084401961, 11.693295129730163, 0.46694552743154455, 0.4312790556177872, 14.457630322270065],deterministic=True)
graph_sketch([0.47000723748297735, 0.3324223762201813, 13.028542455866027, 0.38272598584133044, 0.47000519886878395, 15.753977836685257, 0.4595909928329637, 0.3531463009228444, 16.229144075805895],deterministic=True)
covariance_error_graph([0.47002300171502914, 0.3299795200608889, 13.546535979267746, 0.3841050777256587, 0.47000995777088156, 17.027826929536058, 0.4700089894715167, 0.36396218930703467, 17.385461263438028],deterministic=True)
covariance_error_graph([0.49, 0.324, 12.345, 0.36, 0.438, 18.275, 0.438, 0.336, 18.098],deterministic=True)

graph_sketch([0.49, 0.324, 12.345, 0.36, 0.438, 18.275, 0.438, 0.336, 18.098],deterministic=True)
graph_sketch([0.48999292262772093, 0.32426984729249636, 12.345637935371109, 0.36036086685458574, 0.4384764952054197, 18.275395321515454, 0.43792351576206123, 0.3364609801548085, 18.098406356039874],deterministic=True)
graph_sketch([0.48999292262772093, 0.32426984729249636, 12.345637935371109, 0.36036086685458574, 0.4384764952054197, 18.275395321515454, 0.43792351576206123, 0.3364609801548085, 18.098406356039874])
graph_sketch([0.48999802206969806, 0.311693137079819, 12.113382468481731, 0.3503640936274459, 0.4222866767632392, 19.946494031651376, 0.4298142581887522, 0.32949098677530936, 18.965019097058256],deterministic=True)
graph_sketch([0.490, 0.312, 12.113, 0.350, 0.422, 19.946, 0.430, 0.329, 18.965],deterministic=True)
graph_sketch([0.4900017470056515, 0.31214027981179865, 12.0685001692943, 0.3635518846380174, 0.4364031390956112, 20.002494369482577, 0.427644061829707, 0.32831839593379364, 18.976310371963013],deterministic=True)
graph_sketch([0.4900017470056515, 0.31214027981179865, 12.0685001692943, 0.3635518846380174, 0.4364031390956112, 20.002494369482577, 0.427644061829707, 0.32831839593379364, 18.976310371963013])
graph_sketch([0.4889285700171445, 0.3115403503650666, 12.063214423002432, 0.3728434257647486, 0.44716197873654695, 20.026389357404934, 0.41677848101061304, 0.3185160141868275, 18.969204575706765],deterministic=True)
graph_sketch([0.49, 0.3101801801801802, 12.064987392425536, 0.37504504504504504, 0.45036036036036037, 20.03065837097168, 0.41000000000000003, 0.3114414414414414, 18.960899459838867],deterministic=True)








# X^(1/3) GATES
graph_sketch([0.47943779259441544, 0.47943528029278837, 5.808743419875872, 0.47848397780565605, 0.4777091337927788, 0.0031484310843805778, 0.4775674460894554, 0.479749596187687, 0.003155181801055345, 0.47777037145524937, 0.4783382926437178, 0.003153913073348592])


# EXCELLENT SOLUTION FOR X GATE, g=1.2
# graph_sketch([0.365, 0.412, 25.649821343571965, 0.426, 0.36, 26.975627642673565, 0.39, 0.450, 24.211568469166597],deterministic=True)

# w1=485.7/405.7/485.7      w2=434.5/489.6/434.5       z=47.1117/40.5109/47.1117

# graph_sketch([0.4857,0.4345,47.1117/2,0.4057,0.4896,40.5109/2,0.4857,0.4345,47.1117/2])



# X GATE GAP = 1.0

graph_sketch([0.460,0.350,10.6,0.340,0.460,13.36,0.46,0.35,10.6])
graph_sketch([0.4599896844378258, 0.3491056466374065, 10.600905685119361, 0.34000691436820757, 0.4596665117074058, 13.361161385100171, 0.4599889641512716, 0.34909676711185095, 10.600912460307528],uncorrelated_errors=False,deterministic=True)
graph_sketch([0.45993100176746987, 0.34939314827578577, 10.613006790355119, 0.3400550704898116, 0.4597175978974609, 13.34107405136965, 0.45993183967682255, 0.3493934508820339, 10.61298359412492],uncorrelated_errors=False,deterministic=True)

graph_sketch([0.47, 0.355, 9.91, 0.345, 0.462, 13.05, 0.47, 0.355, 9.905],deterministic=True)
graph_sketch([0.469, 0.355, 9.911, 0.345, 0.4615, 13.053, 0.469, 0.355, 9.905],deterministic=True)
graph_sketch([0.468754015475249, 0.35507846046553715, 9.911967531308253, 0.34489969804262366, 0.46150622724342105, 13.052733333243715, 0.4691042062008894, 0.3552559234311942, 9.905190319557043],deterministic=True)



# H GATE, GAP = 1.0
graph_sketch([0.45000726442145256, 0.3920128146837242, 16.460941933519397],deterministic=True)
graph_sketch([0.49, 0.31, 3.05, 0.49, 0.49, 8.16, 0.49, 0.31, 3.05],deterministic=True)
graph_sketch([0.49, 0.31, 3.05, 0.49, 0.49, 8.16, 0.49, 0.31, 3.05],deterministic=True)
graph_sketch([0.49, 0.31, 3.05, 0.49, 0.49, 8.16, 0.49, 0.31, 3.05],deterministic=True)
graph_sketch([0.4899944880618881, 0.31005446806987735, 3.0492391966379944, 0.4900077035708945, 0.49001265984075715, 8.156021371638165, 0.4899250670485525, 0.31001836467477617, 3.0490917238921313],deterministic=True)
graph_sketch([0.4900279218371208, 0.30997616910353737, 3.050258997486797, 0.4899918441776742, 0.49000434315040076, 8.156211800486362, 0.48999105924935943, 0.31001475136019035, 3.0502829284080795],deterministic=True)
graph_sketch([0.48542836260938627, 0.3121221232592896, 3.664110910331813, 0.4615778496604576, 0.4883652513804258, 6.844797328627163, 0.4852553068006711, 0.3101921779749949, 3.585694185749986],deterministic=True)
graph_sketch([0.4502010050251256, 0.49, 20.986533977200043, 0.49, 0.31, 9.830689138978908, 0.4502010050251256, 0.49, 20.94522089273649],deterministic=True)
graph_sketch([0.4288865733841144, 0.44997245748482817, 35.83766048019133, 0.45000181881881646, 0.34998575815872524, 16.787378994157972, 0.42870864970853567, 0.4499585908126799, 35.767112180219414],deterministic=True)
graph_sketch([0.4288865733841144, 0.44997245748482817, 35.83766048019133, 0.45000181881881646, 0.34998575815872524, 16.787378994157972, 0.42870864970853567, 0.4499585908126799, 35.767112180219414])
graph_sketch([0.4288865733841144, 0.44997245748482817, 35.83766048019133, 0.45000181881881646, 0.34998575815872524, 16.787378994157972, 0.42870864970853567, 0.4499585908126799, 35.767112180219414],deterministic=True)
graph_sketch([0.4308243591869351, 0.44997451552993095, 36.085207263689895, 0.45001138034467275, 0.34999411348093856, 16.70660582125026, 0.42629953481012534, 0.44997440732743493, 34.19663952738418],deterministic=True)
graph_sketch([0.43343015067298074, 0.44999973685142963, 36.26863031406008, 0.44999359090212226, 0.3499797829822308, 16.58628049664455, 0.422641756854722, 0.4499893088027105, 31.88072137862819],deterministic=True)
graph_sketch([0.43633888225545736, 0.44999111440467765, 36.5109924189534, 0.44952017153569834, 0.349977006942013, 16.53903978365884, 0.4177145023456773, 0.44998731011331594, 29.188882100976535],deterministic=True)
graph_sketch([0.4382823919849238, 0.4500143200698091, 35.64645764197679, 0.4456753681178709, 0.3499932763733956, 16.904005666953857, 0.41208445800132176, 0.45001288305179044, 25.90911334624649],deterministic=True)
graph_sketch([0.4380053994436562, 0.45000503356010846, 34.731747872037715, 0.4445125131625745, 0.34998717960916387, 16.963537143171173, 0.4117043459097096, 0.45000527925333994, 25.145253370867856],deterministic=True)
graph_sketch([0.4372599275256075, 0.45000434915144827, 32.84038039377518, 0.4424399365034183, 0.34998059583261903, 17.03729575969475, 0.41143640466079745, 0.45000035267452165, 23.795772755827965],deterministic=True)
graph_sketch([0.43672083561942204, 0.45002105179483365, 31.512725543451158, 0.4410732107486153, 0.35000152256231065, 17.06397153542702, 0.4112607519765821, 0.44999482517403366, 22.861361288789443],deterministic=True)
graph_sketch([0.4360501582896607, 0.4500021868533522, 30.50405258689383, 0.4402929515434649, 0.3499848866110632, 17.067341119289875, 0.4119140574480584, 0.45000374654837716, 22.442705493367065],deterministic=True)
graph_sketch([0.4360501582896607, 0.4500021868533522, 30.50405258689383, 0.4402929515434649, 0.3499848866110632, 17.067341119289875, 0.4119140574480584, 0.45000374654837716, 22.442705493367065])
graph_sketch([0.4348397227323077, 0.4487055405802767, 30.360776289321464, 0.4401513471218532, 0.3500335422783487, 17.07442283314764, 0.410847272139333, 0.44937332189806634, 22.247213347157246])
graph_sketch([0.42722856486497696, 0.43829861045463425, 30.67683310053911, 0.4392046649707892, 0.3499848250074142, 17.296570444575313, 0.40072388087306654, 0.44578363108298547, 20.4364971075721])







graph_sketch([0.46000544505470153, 0.3491895576484114, 10.607128229007563, 0.34007198165500846, 0.4598582721844998, 13.353364115568166, 0.45993649063384073, 0.34913215781544693, 10.607093187181153],uncorrelated_errors=True,deterministic=True)
graph_sketch([0.45999938256168654, 0.3491092128298756, 10.599152089922798, 0.33999870078115496, 0.4596351048039797, 13.363178621215933, 0.45999980322537104, 0.3491032966795937, 10.599149298467578],uncorrelated_errors=True,deterministic=True)
graph_sketch([0.45993100176746987, 0.34939314827578577, 10.613006790355119, 0.3400550704898116, 0.4597175978974609, 13.34107405136965, 0.45993183967682255, 0.3493934508820339, 10.61298359412492],deterministic=True)
graph_sketch([0.4600005025181419, 0.3492445724596729, 10.614312408297655, 0.34000366210771393, 0.4599009567114287, 13.344291906437022, 0.45998917180593507, 0.349259993324966, 10.614145902256865])
graph_sketch([0.4600005025181419, 0.3492445724596729, 10.614312408297655, 0.34000366210771393, 0.4599009567114287, 13.344291906437022, 0.45998917180593507, 0.349259993324966, 10.614145902256865],deterministic=True)
graph_sketch([0.4600005025181419, 0.3492445724596729, 10.614312408297655, 0.34000366210771393, 0.4599009567114287, 13.344291906437022, 0.45998917180593507, 0.349259993324966, 10.614145902256865])
graph_sketch([0.46000373360929975, 0.3491626981603497, 10.606599080732478, 0.3400061409908715, 0.459721983240385, 13.359285682896878, 0.45999585982890284, 0.3491672155516624, 10.60669387944497],deterministic=True)
graph_sketch([0.4599759801702898, 0.34736377967394777, 10.97180796110759, 0.340042422355771, 0.4496600901519544, 14.886031834676823, 0.45987327864866273, 0.3449917674791983, 10.814192613217074],deterministic=True)
graph_sketch([0.4599759801702898, 0.34736377967394777, 10.97180796110759, 0.340042422355771, 0.4496600901519544, 14.886031834676823, 0.45987327864866273, 0.3449917674791983, 10.814192613217074],deterministic=True)
graph_sketch([0.44034267713403785, 0.34400255211962477, 13.283426381404716, 0.34000942329879386, 0.45392679635179906, 15.62146752860127, 0.45464843501830743, 0.3442919255988222, 12.343105185804804])
graph_sketch([0.4500098173261105, 0.3499844272895847, 14.114123249101302, 0.3499744712371705, 0.45001132358846735, 18.29408927970707, 0.4500098121372316, 0.34998443248970024, 14.114122955815875],deterministic=True)
graph_sketch([0.43680152203410394, 0.34309604198759136, 13.777387346242058, 0.3399923232333641, 0.4553856616092375, 15.722135138448161, 0.45153912024709536, 0.3441858056160284, 12.834214987478507])
graph_sketch([0.43294816655935764, 0.3420368223380047, 14.32048840361325, 0.3399861399455109, 0.45709721740282155, 15.807870695951136, 0.44817448781546565, 0.3441382400538265, 13.388743523071847])
graph_sketch([0.43294816655935764, 0.3420368223380047, 14.32048840361325, 0.3399861399455109, 0.45709721740282155, 15.807870695951136, 0.44817448781546565, 0.3441382400538265, 13.388743523071847],deterministic=True)
graph_sketch([0.4280480480480481, 0.343963963963964, 15.048416999999999, 0.34012012012012016, 0.4583183183183184, 15.824121899999996, 0.46, 0.35717717717717723, 14.203622399999999])
graph_sketch([0.358978978978979, 0.44222222222222224, 15.3948, 0.46, 0.3452852852852853, 16.18836, 0.36090090090090093, 0.46, 14.53056])
graph_sketch([0.349980144208542, 0.4500140364385995, 14.114054415245713, 0.4500066922360174, 0.34997942767461143, 18.29453803842362, 0.34998019196980845, 0.4500140674781063, 14.114053918538215],deterministic=True)
graph_sketch([0.3499859021967333, 0.4500263882442742, 14.081988617697581, 0.45006775017894374, 0.34990994329216957, 18.242324170491585, 0.3499793833940727, 0.4500388062217378, 14.082183153463793],deterministic=True)
graph_sketch([0.3498639018152045, 0.45012828811013256, 13.910659468964084, 0.45026355971218457, 0.3497002655899217, 17.98501580703767, 0.34987417233758206, 0.45012168159874116, 13.914252268738679],deterministic=True)
graph_sketch([0.3613243025029149, 0.4490630855391503, 15.484053997436222, 0.4599525354656549, 0.34421939617225034, 16.34275892555141, 0.3490549114567034, 0.44435380255415735, 14.71633867289572])
graph_sketch([0.3613243025029149, 0.4490630855391503, 15.484053997436222, 0.4599525354656549, 0.34421939617225034, 16.34275892555141, 0.3490549114567034, 0.44435380255415735, 14.71633867289572],deterministic=True)

# X GATE FULL ETCH

graph_sketch([0.6262947618552157, 0.5600599823003919, 7.226245157112129, 0.5599275918129709, 0.6306971610229538, 9.389181875093783, 0.6264155135053913, 0.5600216561518722, 7.226490570965122])
graph_sketch([0.6294949494949496, 0.5624242424242425, 7.608276974819154, 0.5608080808080809, 0.6343434343434343, 9.563980173520331, 0.6294949494949496, 0.5624242424242425, 7.608260345851736],deterministic=True)
graph_sketch([0.6294949494949496, 0.5624242424242425, 7.608276974819154, 0.5608080808080809, 0.6343434343434343, 9.563980173520331, 0.6294949494949496, 0.5624242424242425, 7.608260345851736])
graph_sketch([0.5897303244002989, 0.5894779670380875, 9.33013671530001, 0.5650400935238006, 0.5668945571518392, 2.42507573302199, 0.5765361673605217, 0.5754777887801334, 3.590253742515112])
graph_sketch([0.6056456456456457, 0.6052452452452453, 10.306177028622658, 0.6083683683683684, 0.6142142142142142, 2.332808232139264, 0.5783383383383384, 0.5710510510510511, 1.3515567806814648])
graph_sketch([0.6277477477477478, 0.6322322322322322, 4.736569339181831, 0.573933933933934, 0.5681681681681682, 4.1744995200569015, 0.6094094094094095, 0.6135735735735736, 4.830115632764866])
graph_sketch([0.6162162162162163, 0.6333533533533534, 4.837425833437663, 0.5821821821821822, 0.56, 4.303481555618459, 0.5990790790790791, 0.6146146146146146, 4.989098251168205])
graph_sketch([0.6162162162162163, 0.6333533533533534, 4.837425833437663, 0.5821821821821822, 0.56, 4.303481555618459, 0.5990790790790791, 0.6146146146146146, 4.989098251168205],deterministic=True)
graph_sketch([0.5600867167524191, 0.560091175460495, 5.0892625692773334, 0.5599887874026306, 0.5600079127487745, 7.06327962154235, 0.5600456922531991, 0.5600819214253961, 4.671965953458962],deterministic=True)
graph_sketch([0.5600867167524191, 0.560091175460495, 5.0892625692773334, 0.5599887874026306, 0.5600079127487745, 7.06327962154235, 0.5600456922531991, 0.5600819214253961, 4.671965953458962])
graph_sketch([0.5780980980980981, 0.6340740740740741, 10.5967732963562, 0.6395195195195196, 0.5643243243243243, 11.143000551223754, 0.56, 0.6213413413413413, 10.001852991104126],deterministic=True)
graph_sketch([0.5780180180180181, 0.6339489489489489, 10.5967732963562, 0.64, 0.565, 11.143000551223754, 0.56, 0.6213363363363363, 10.001852991104126])

graph_sketch([0.5780180180180181, 0.6339489489489489, 10.5967732963562, 0.64, 0.565, 11.143000551223754, 0.56, 0.6213363363363363, 10.001852991104126])


# FULL ETCH H GATE
graph_sketch([0.5600385963133359, 0.5638326934452398, 1.235316270134832, 0.6350307044208114, 0.5852559984598615, 8.920122948950295, 0.5600163770646198, 0.5639813859228747, 1.229921070518535])
graph_sketch([0.5600385963133359, 0.5638326934452398, 1.235316270134832, 0.6350307044208114, 0.5852559984598615, 8.920122948950295, 0.5600163770646198, 0.5639813859228747, 1.229921070518535],deterministic=True)
graph_sketch([0.5600799650436739, 0.5908371069844924, 0.5391562426272287, 0.6323582352187354, 0.5850333475668991, 9.963425107926803, 0.5601012375329502, 0.5909983572505612, 0.5374546098053773])
graph_sketch([0.626290338028494, 0.5902574749521733, 4.158673693082275, 0.635779698077665, 0.5839604037131503, 5.9835440537549545, 0.5898291262274857, 0.6271492877850281, 0.5152766903230391])
graph_sketch([0.5605317552197443, 0.575805700727665, 1.024962924369133, 0.5982112111331118, 0.5610795457729382, 10.633388921649619, 0.5611369354619969, 0.5727660996897777, 1.114262601836484])
graph_sketch([0.5696197084042186, 0.5610880894161138, 5.3043747200103555, 0.6381063592965233, 0.5717323820699016, 6.294264516811712, 0.5652244507089831, 0.60917621647086, 1.8098960252162803])
graph_sketch([0.5672856879169672, 0.5638327883708587, 6.143324138638837, 0.6363934365542075, 0.5644345415888592, 6.4358086858851475, 0.562469793664378, 0.6218122521206019, 2.2287293182729475])
graph_sketch([0.5604555619610095, 0.6279669894767541, 2.632097392897859, 0.6083964021545324, 0.5604302994850615, 11.438684241849366, 0.5602111606111998, 0.5912782818642637, 3.8958737789661413],deterministic=True)
graph_sketch([0.5599986165173348, 0.6397459440411442, 3.3795415393507002, 0.6094384773076004, 0.5599948644636333, 11.934740005988694, 0.5599967004976981, 0.5830883212158574, 6.142471416762587])



# MUHAMAD VALUES
graph_sketch([0.45076557132722733, 0.3488935095056416, 33.091645834858504, 0.44079336290224347, 0.44983675316431637, 44.85991599214905, 0.3497849596265617, 0.42348064544220465, 32.6784756826935])
graph_sketch([ 0.3950,  0.3620, 32.62657122541451,  0.3785,  0.3785, 46.0015636690357,  0.3620,  0.3950,
               32.62657122541451])

graph_sketch([ 0.3950,  0.3620, 35.9697,  0.3785,  0.3785, 35.9697,  0.3620,  0.3950,
               35.9697])










# covariance_error_graph([0.3507129044215354, 0.4487859451590115, 0.25251681245119373, 0.34995546167364444, 0.3499045109464061, 28.7026871269869, 0.35023959178512526, 0.44869868471785607, 0.24999998640511673])
# # covariance_error_graph([0.38569987909894343, 0.4372875988875816, 0.34280706282345336, 0.34986563563832584, 0.34999059926401815, 28.638207788315274, 0.3499531397006422, 0.4221536377940909, 0.24851076284251794])
# # graph_sketch([0.3531780540943146, 0.4004172682762146, 28.765007840494018, 0.4110850989818573, 0.3500567408186383, 30.655880594564312, 0.35437071323394775, 0.40183594822883606, 28.77210441667025])
# graph_sketch([0.36469716309473044, 0.41212637429292914, 25.799906851067497, 0.426133127631129, 0.36026795864288946, 27.08565800482148, 0.39101049395155485, 0.44998847693717364, 24.364006804346815])
# # covariance_error_graph([0.35022503703845703, 0.3508457491642306, 5.990111755321123, 0.35132447346261325, 0.36072038815723245, 0.028039234478710668, 0.35002534775605504, 0.3499203447360068, 16.325861578904018, 0.3500787923997763, 0.3506547667045729, 7.128033548241433])
# covariance_error_graph([0.3836800390322265, 0.41763947039396715, 0.23073587225657025, 0.3500034123216104, 0.3499523309258727, 28.987544885585645, 0.35645072992832655, 0.3838206950401308, 0.28544990049819496])
# covariance_error_graph([0.3507912860184953, 0.39083305086759124, 8.201589391000496, 0.37767280072317794, 0.35000674595622794, 16.402935729228666, 0.35057037538216373, 0.3906276660852304, 8.20192111263818])
covariance_error_graph([0.3901185173375584, 0.349981570618563, 5.214992520399418, 0.35004485325096807, 0.3630012584572002, 10.425649730279673, 0.35013750650067765, 0.36307157363166664, 10.425469726831162, 0.39074397050676335, 0.35068692286184144, 5.213265104653216])
# covariance_error_graph([0.44998073922691056, 0.3502011855982789, 9.757899969765605, 0.39934676036975036, 0.43647211537610775, 17.770726503664626, 0.36322399567551356, 0.40055805160858987, 13.321511635356439, 0.44999514069268504, 0.35001021920920944, 9.61107147287333])
covariance_error_graph([0.37079082388463946, 0.35002684591062805, 2.487503375227385, 0.3499741795451948, 0.3524740420769906, 24.866615581587276, 0.3709545288860054, 0.35031186580629015, 2.4868966113086706])
covariance_error_graph([0.3507863324792457, 0.3735878483813331, 0.0031756128832262378, 0.34998889114566606, 0.3499887495770282, 29.536638930964113, 0.3513348874127779, 0.3747148592981032, 0.0031759146339758494])
covariance_error_graph([0.3502672899191126, 0.3707167820209726, 5.085303126610136, 0.3567861211335445, 0.35002541173299473, 20.07583204648323, 0.35030192508359337, 0.3707552551655727, 5.088480265631608])
covariance_error_graph([0.3502324750725061, 0.38111197051900997, 9.220138602138741, 0.38618610748807625, 0.35019432234228076, 11.185334019158926, 0.429687112154854, 0.44969000472742643, 14.63289491620164])
covariance_error_graph([0.4239185135685009, 0.4499711994720953, 18.703024688845453, 0.4499651050160983, 0.38622301741503623, 12.378605110400336, 0.39405470319180336, 0.4499952731129559, 10.576867666366137])
covariance_error_graph([0.45001045883226215, 0.38244261452127054, 15.412039065963356, 0.34999672741758336, 0.39188143941829473, 21.093693332406502, 0.35005353913833037, 0.4499766357225239, 5.730819401775913, 0.4500095696123918, 0.3586765285886688, 14.43178859868613])
covariance_error_graph([0.4499253327102188, 0.379929167376368, 13.71493119612403, 0.35001743787980655, 0.3925572021324958, 19.993176033815136, 0.35145984437300276, 0.4498652071373341, 5.025348533838426, 0.44980535096462676, 0.35797888347069656, 12.880680446374871])
covariance_error_graph([0.4499851260068771, 0.39596960069451476, 9.858948498379974, 0.3499637996771631, 0.3816164550813695, 16.410432800926696, 0.3507136119985974, 0.4494923109630479, 2.6963753971149806, 0.44975942102226074, 0.3785884676498902, 9.30326113231542])
covariance_error_graph([0.44945212174871174, 0.4494268035678733, 7.673891462580896, 0.3500502147697099, 0.34995655331906056, 15.000554067390784, 0.35022907926427466, 0.3521388246801894, 1.6118407229369163, 0.35608363525346104, 0.35567588212068485, 7.363749529629336])
covariance_error_graph([0.3584840693627785, 0.44259265286918026, 0.3166285472438265, 0.3500176048042603, 0.3499731144946457, 28.274213946939074, 0.35284024846321804, 0.4367518517373484, 0.3106524329204967])
# graph_sketch([0.3646971583366394, 0.412126362323761, 25.658046078016966, 0.4261331260204315, 0.3602679669857025, 26.980631436518383, 0.39101049304008484, 0.4499884843826294, 24.21756081418449])
# graph_sketch([0.3584840693627785, 0.44259265286918026, 0.3166285472438265, 0.3500176048042603, 0.3499731144946457, 28.274213946939074, 0.35284024846321804, 0.4367518517373484, 0.3106524329204967])


# graph_sketch_2d_error([0.3936175388299793, 0.34548336619490716, 69.45205303173597, 0.45102514561992996, 0.4360070566512364, 65.5248279488548, 0.4370459179248492, 0.45135842119774117, 39.66442789341747])
# graph_sketch([0.3500977884415713, 0.39682522416114807, 29.469168799256817, 0.4085565209388733, 0.3500956427857118, 32.14216150579926, 0.3500980566312617, 0.3966447114944458, 29.51642145426282])
# graph_sketch([0.34999779018906724, 0.3968252276977474, 29.469268936506776, 0.40855652737648074, 0.34999564895872004, 32.142261787398645, 0.3499980459632427, 0.39664471264909534, 29.51652147911157])
# graph_sketch([0.35047374539509646, 0.39734968331208864, 29.431995571177364, 0.409136178521305, 0.3499964061507012, 31.90468405640843, 0.3504956524909027, 0.39722224290949854, 29.472338892403755])
# graph_sketch([0.3531780540943146, 0.4004172682762146, 28.902319468891744, 0.4110850989818573, 0.3500567408186383, 30.780486906788386, 0.35437071323394775, 0.40183594822883606, 28.912821945522566])
# graph_sketch([0.4500029273568712, 0.3931783768675857, 16.971919974045406, 0.350613514707631, 0.45000287226458036, 13.334681128250828, 0.4500034111220633, 0.4081911993876085, 20.952925971374587],etch=True)
# graph_sketch([0.3531780540943146, 0.4004172682762146, 28.765007840494018, 0.4110850989818573, 0.3500567408186383, 30.655880594564312, 0.35437071323394775, 0.40183594822883606, 28.77210441667025])
graph_sketch([0.3531780540943146, 0.4004172682762146, 28.832127226015608, 0.4110850989818573, 0.3500567408186383, 30.714259469052156, 0.35437071323394775, 0.40183594822883606, 28.84224133162614])
graph_sketch([0.3531780540943146, 0.4004172682762146, 28.902319468891744, 0.4110850989818573, 0.3500567408186383, 30.780486906788386, 0.35437071323394775, 0.40183594822883606, 28.912821945522566])
graph_sketch([0.3510742921418899, 0.3980071352087352, 29.34090649286859, 0.4096155744105247, 0.3499957472063357, 31.654557664036485, 0.3512007461234598, 0.39805480709304814, 29.36875463102632])
graph_sketch([0.3524536324289393, 0.39957691999139006, 29.110682763208136, 0.4105456498177996, 0.34999768528215747, 31.137003883834083, 0.3530126843516391, 0.40020801137707734, 29.124687553402815])
graph_sketch([0.3531780540943146, 0.4004172682762146, 28.948954143251168, 0.4110850989818573, 0.35005673987413527, 30.82589366488561, 0.35437071323394775, 0.40183594822883606, 28.9595442252823])
graph_sketch([0.3531780540943146, 0.4004172682762146, 28.94665348531012, 0.4110850989818573, 0.3500567408186383, 30.823701596042753, 0.35437071323394775, 0.40183594822883606, 28.957243390527136])
graph_sketch([0.3531780540943146, 0.4004172682762146, 28.93955621368869, 0.4110850989818573, 0.3500567408186383, 30.816743587366883, 0.35437071323394775, 0.40183594822883606, 28.95014132653433])
graph_sketch([0.3510742921418899, 0.3980071352087352, 29.34090649286859, 0.4096155744105247, 0.3499957472063357, 31.654557664036485, 0.3512007461234598, 0.39805480709304814, 29.36875463102632],deterministic=True)
graph_sketch([0.34999779018906724, 0.3968252276977474, 29.469268936506776, 0.40855652737648074, 0.34999564895872004, 32.142261787398645, 0.3499980459632427, 0.39664471264909534, 29.51652147911157],deterministic=True)
graph_sketch([0.34999779018906724, 0.3968252276977474, 29.469268936506776, 0.40855652737648074, 0.34999564895872004, 32.142261787398645, 0.3499980459632427, 0.39664471264909534, 29.51652147911157])
graph_sketch([0.34999779018906724, 0.3968252276977474, 29.469268936506776, 0.40855652737648074, 0.34999564895872004, 32.142261787398645, 0.3499980459632427, 0.39664471264909534, 29.51652147911157])
graph_sketch_2d_error([0.3976977303353444, 0.34995456059677826, 70.31591350290749, 0.4500086355788198, 0.4348102467616029, 65.52153894986067, 0.4356250352167248, 0.45005729413781814, 39.671797931454684])
graph_sketch_2d_error( [0.3983494399074778, 0.3499859905597135, 69.35864537515874, 0.44993010866818534, 0.43483489569050954, 65.9465848517826, 0.43576412749682564, 0.4499268340582372, 40.14855712002849])
graph_sketch_2d_error([0.3944908480989966, 0.3454580891995282, 67.93367530608396, 0.4514442939556191, 0.43650437868347436, 66.40537950490608, 0.43775756090105816, 0.4516588189218884, 40.648276951330104])
# graph_sketch([0.40178951774014904, 0.3499587918517068, 64.81164594590791, 0.4499703040604929, 0.43476106380308294, 67.46499088857195, 0.4365925608651903, 0.45005273444008337, 42.248036024803376],etch=True)
# graph_sketch_2d_error([0.4142743385204239, 0.349933735006119, 52.943240539530464, 0.3656316559669494, 0.35002682253112744, 56.3411713165563, 0.3743334274784483, 0.3879609960672322, 39.43873223796911])
# graph_sketch_2d_error([0.35147706656024713, 0.3522118507639588, 2.3739811485521654, 0.3500527928825889, 0.34995867853108037, 24.69588570874256, 0.3507963258645947, 0.35167072816229694, 2.5017026190415548])
# graph_sketch_2d_error([0.35, 0.35, 0.5*float(pi/photon_interp.get_params(0.35, 0.35, g, error_w=0,error_etching=0.15,num_of_points=1)[1][0])])
# graph_sketch([0.35147706656024713, 0.3522118507639588, 2.3739811485521654, 0.3500527928825889, 0.34995867853108037, 24.69588570874256, 0.3507963258645947, 0.35167072816229694, 2.5017026190415548],etch=True)
# graph_sketch([0.35147706656024713, 0.3522118507639588, 2.3739811485521654, 0.3500527928825889, 0.34995867853108037, 24.69588570874256, 0.3507963258645947, 0.35167072816229694, 2.5017026190415548])
# graph_sketch_2d_error([0.35147706656024713, 0.3522118507639588, 2.3739811485521654, 0.3500527928825889, 0.34995867853108037, 24.69588570874256, 0.3507963258645947, 0.35167072816229694, 2.5017026190415548])
# graph_sketch_2d_error([0.45, 0.45, 0.5*float(pi/photon_interp.get_params(0.45, 0.45, g, error_w=0,error_etching=0.15,num_of_points=1)[1][0])])


# graph_sketch_2d_error([0.4500102872349207, 0.35637496136903507, 8.932033557471081, 0.41515614592062877, 0.4500254678101442, 30.97928199633609, 0.4500084940488647, 0.3563763172858014, 8.932033288646101])
# graph_sketch_2d_error([0.4500102872349207, 0.35637496136903507, 8.932033557471081, 0.41515614592062877, 0.4500254678101442, 30.97928199633609, 0.4500084940488647, 0.3563763172858014, 8.932033288646101],deterministic=True)
# graph_sketch([0.4500055006537908, 0.3499938087393055, 8.938570709153476, 0.3525964929035553, 0.3499922188681294, 31.148287805911828, 0.4500051136103323, 0.34999418152535494, 8.938570269177784],etch=True)
graph_sketch([0.35003113746643066, 0.41717100143432617, 29.67204171786717, 0.44994115829467773, 0.39957985281944275, 32.877690765760676, 0.3500586748123169, 0.41682231426239014, 29.79541678765386])

graph_sketch([0.4500055006537908, 0.3499938087393055, 8.938570709153476, 0.3525964929035553, 0.3499922188681294, 31.148287805911828, 0.4500051136103323, 0.34999418152535494, 8.938570269177784])
graph_sketch([0.4500055006537908, 0.3499938087393055, 8.938570709153476, 0.3525964929035553, 0.3499922188681294, 31.148287805911828, 0.4500051136103323, 0.34999418152535494, 8.938570269177784],deterministic=True)
# graph_sketch_2d_error([0.45, 0.45, 0.5*float(pi/photon_interp.get_params(0.45, 0.45, g, error_w=0,error_etching=0.15,num_of_points=1)[1][0])])
graph_sketch_2d_error([0.4500055006537908, 0.3499938087393055, 8.938570709153476, 0.3525964929035553, 0.3499922188681294, 31.148287805911828, 0.4500051136103323, 0.34999418152535494, 8.938570269177784])
graph_sketch([0.35003113746643066, 0.41717100143432617, 29.70302537340302, 0.44994115829467773, 0.39957985281944275, 32.930233374336126, 0.3500586748123169, 0.41682231426239014, 29.82736474968564])
graph_sketch_2d_error( [0.4498629407409305, 0.3500509053246032, 9.602998624086098, 0.3504625465379038, 0.3500028193165682, 29.865336197808467, 0.4499703678005375, 0.3500386265802902, 9.594438104760858])
graph_sketch([0.35003113746643066, 0.41717100143432617, 29.748328140499645, 0.44994115555845254, 0.39957985281944275, 32.99499442604281, 0.35005867762892756, 0.41682231426239014, 29.85626958985409])
graph_sketch([0.3500311502909338,0.41717101277895774, 29.755714381992135, 0.45000122062542247, 0.3995798619698717, 32.987696100225406,  0.34999862927994013, 0.41682231083400745, 29.86334657474153])

# graph_sketch_2d_error([0.45, 0.45, 0.5*float(pi/photon_interp.get_params(0.45, 0.45, g, error_w=0,error_etching=0.15,num_of_points=1)[1][0])])
# graph_sketch_2d_error([0.35, 0.35, 0.5*float(pi/photon_interp.get_params(0.35, 0.35, g, error_w=0,error_etching=0.15,num_of_points=1)[1][0])])
graph_sketch_2d_error([0.4158909927818151, 0.3501093742939977, 17.0560262004011, 0.3500350738306452, 0.36540657562505235, 24.137716044491555, 0.44074401680665115, 0.3504115883925847, 12.496509916843008])

graph_sketch_2d_error([0.3500311502909338,0.41717101277895774, 29.755714381992135, 0.45000122062542247, 0.3995798619698717, 32.987696100225406,  0.34999862927994013, 0.41682231083400745, 29.86334657474153])
# graph_sketch([0.3500311502909338,0.41717101277895774, 29.755714381992135, 0.45000122062542247, 0.3995798619698717, 32.987696100225406,  0.34999862927994013, 0.41682231083400745, 29.86334657474153],deterministic=True)
graph_sketch([0.3500311502909338,0.41717101277895774, 29.755714381992135, 0.45000122062542247, 0.3995798619698717, 32.987696100225406,  0.34999862927994013, 0.41682231083400745, 29.86334657474153])

# graph_sketch_2d_error([0.35011934705941833, 0.4500561272446929, 8.404025303704156, 0.3500212006740984, 0.35411884971459767, 32.32239469044294, 0.35004962292446506, 0.45007063852356005, 8.394605944006182])
# graph_sketch_2d_error([0.4500001908384628, 0.3501692850788468, 8.392166822042153, 0.35405236564325643, 0.3500206827396754, 32.3232579313339, 0.44967404374806774, 0.3499447194107645, 8.408059704274827])
#
# graph_sketch_2d_error([0.4499838864335426, 0.3500008429867195, 17.035526076358916, 0.41942290061399656, 0.4500085973276482, 30.90935684961619, 0.4499961098548342, 0.3500070790881249, 17.034902157236605])
# graph_sketch_2d_error([0.45000038402625314, 0.36107886534132827, 22.136939470200634, 0.41031811880178115, 0.45000526871371005, 32.95015511352784, 0.44999844374741116, 0.36108688412265033, 22.13697760121415])
# graph_sketch_2d_error([0.4499364149910008, 0.35006893483492074, 11.82294080505085, 0.4422638548893596, 0.449862947392169, 34.89566582064606, 0.44993655469053045, 0.35000226569738563, 11.82140772871478])
# BEST 1D GRAPH SKETCH

graph_sketch([0.41717101277895774, 0.3500311502909338, 29.755714381992135, 0.3995798619698717, 0.45000122062542247, 32.987696100225406, 0.41682231083400745, 0.34999862927994013, 29.86334657474153])

#

graph_sketch_2d_error([0.41717101277895774, 0.3500311502909338, 29.755714381992135, 0.3995798619698717, 0.45000122062542247, 32.987696100225406, 0.41682231083400745, 0.34999862927994013, 29.86334657474153])

graph_sketch_2d_error([0.4470577405144064, 0.3671558252304029, 24.525083279389566, 0.405522533900364, 0.45000234930757776, 31.247863852590033, 0.44643525478820234, 0.36723269121264634, 24.70065880937876])

graph_sketch([0.41596601201613237, 0.35004373072826017, 30.196647530460893, 0.3981905487481447, 0.449988471430241, 32.49546881708402, 0.41500022822336646, 0.34999572862351036, 30.530105700369617])
graph_sketch([0.4156442880630493, 0.3505974428042207, 30.4067670696162, 0.3978445827960968, 0.4499931335449219, 32.223334007840435, 0.41451820731163025, 0.350004106760025, 30.52010100133761])
graph_sketch([0.4152592486357615, 0.3500078389369911, 30.463985980227893, 0.39735836113917966, 0.4499824755087875, 32.202771908297066, 0.41387668501861075, 0.34999869608031275, 30.957794567366932])
graph_sketch([0.4130739062734044, 0.35000528867577385, 31.499312232405806, 0.3953964772364811, 0.45000410168700494, 31.905918285581357, 0.410080802927, 0.35000373681887614, 32.652776476404526])


graph_sketch([0.41292980474626884, 0.35000304085024964, 31.5821215838041, 0.3953375823406088, 0.4499832689197875, 31.94444075308051, 0.40982875224229304, 0.34999671367043383, 32.78499361923115])

graph_sketch([0.4216003453397909, 0.37343822495605256, 65.03540813701501/2, 0.3708526440963105, 0.4231670646237347, 65.04106148524343/2, 0.4314805652585642, 0.38567677022805735, 68.46935793558563/2])

graph_sketch_2d_error([0.45002032479074994, 0.39956193040582927, 21.928668884844704, 0.39250554908307606, 0.45000934723111385, 21.326277948341964, 0.4500197409766094, 0.40003149042226965, 22.091552038740698])
graph_sketch_2d_error([0.4500126673717791, 0.3981102610147584, 24.10955798695796, 0.39089832378643136, 0.4500169016815939, 23.383610870119956, 0.45001220650263807, 0.3985640863305713, 24.27355010769823])
graph_sketch_2d_error([0.44999858742545523, 0.39794796848826963, 25.45069273759925, 0.3909415021516029, 0.45001602870283364, 24.71361846481102, 0.4500234758979894, 0.39846497634411, 25.64616145356418])
# graph_sketch_2d_error([0.45, 0.45, 0.5*float(pi/photon_interp.get_params(0.45, 0.45, g, error_w=0,error_etching=0.15,num_of_points=1)[1][0])])




graph_sketch_2d_error([0.450018080929969, 0.39750152626358215, 25.53248005379224, 0.3917258187654568, 0.4499813474091576, 25.21963057419744, 0.4500150653074343, 0.398395727088094, 25.868498627983453])
graph_sketch_2d_error([0.4216003453397909, 0.37343822495605256, 65.03540813701501/2, 0.3708526440963105, 0.4231670646237347, 65.04106148524343/2, 0.4314805652585642, 0.38567677022805735, 68.46935793558563/2])
graph_sketch_2d_error([0.45001593407185914, 0.3971045360078082, 26.072147962209947, 0.3926305662306793, 0.44999251141699315, 26.219139088466278, 0.45000591224893005, 0.3986773932448644, 26.67285589211825])
graph_sketch_2d_error([0.44999843072119733, 0.34995770352109307, 37.535497362824536/2, 0.41711440508304587, 0.45004311547407294, 69.15790185084285/2, 0.44999311230859, 0.3499976984817224, 37.53340006496894/2, 0.4500645847345013, 0.450025567498015, 193.64210778387348/2])

# graph_sketch_2d_error([0.3993770480155945, 0.36786097288131714, 78.10065137042305, 0.367825984954834, 0.39935120940208435, 78.09120312775603, 0.4113065004348755, 0.380872905254364, 82.27090242920845])
# graph_sketch_2d_error([0.45001677426575293, 0.34997731322866693, 60.24456977631744, 0.4494215385095488, 0.45002548257914976, 112.02681486737862, 0.45001588582687285, 0.34997819434165983, 60.24456903504294])
# graph_sketch_2d_error([0.45001626740330947, 0.3499779256333395, 60.2493040062099, 0.44942968042698855, 0.4500170554002813, 112.01706777157744, 0.45001619692780104, 0.34997799609264824, 60.24930398107334])


graph_sketch_2d_error([0.4216003453397909, 0.37343822495605256, 65.03540813701501, 0.3708526440963105, 0.4231670646237347, 65.04106148524343, 0.4314805652585642, 0.38567677022805735, 68.46935793558563])
graph_sketch_2d_error([0.40678519887774134, 0.35810033197302243, 64.95886254366583, 0.3542172490651339, 0.40678387431505075, 64.93758349186096, 0.4054208773193648, 0.36040951276123845, 68.02644293319953])



graph_sketch_2d_error([0.3500716894141628, 0.3500716894141628, 27.686217943683538, 0.350072395461024, 0.350072395461024, 23.30936366837971, 0.35007303584681304, 0.35007303584681304, 18.993636447313715])
# graph_sketch_2d_error([0.35, 0.35, float(pi/photon_interp.get_params(0.35, 0.35, g, error_w=0,error_etching=0.15,num_of_points=1)[1][0])])





graph_sketch_2d_error([0.4267676767676768, 0.37626262626262624, 65.08387614201921, 0.37626262626262624, 0.4267676767676768, 65.07600260646336, 0.4409090909090909, 0.3924242424242424, 68.55908535767371])
graph_sketch_2d_error([0.4175218316657116, 0.34992612642624216, 64.82451656352723, 0.38366680611500514, 0.436465321094812, 64.98851545681286, 0.42689088325086316, 0.370233310823998, 68.31191733764453, 0.4377651852564504, 0.4500399870456333, 168.90131609435565])

graph_sketch_2d_error([0.4500277543436809, 0.39516628177540347, 50.31303427018406, 0.39517289549586715, 0.45002857950829, 50.31855416609019, 0.45002808163385133, 0.39918639192020106, 53.464261665570476])
graph_sketch_2d_error([0.4499532594094118, 0.39737857240711216, 45.63368419227566, 0.3975497050783538, 0.4500343031212521, 45.752814271063585, 0.4499843172283964, 0.39930520472245395, 47.09923011868644])
# graph_sketch_2d_error([0.43055795600634583, 0.3798656306486057, 64.88150752125696, 0.3798362456482817, 0.4305305495971275, 64.8761878673947, 0.43741706656013796, 0.38951408255348496, 68.40427690380373])
graph_sketch_2d_error([0.44696239079054434, 0.3948308708049211, 63.509916461535916, 0.39485884108727143, 0.4470073628583089, 63.50294703681719, 0.4386882607348899, 0.3914306538569219, 67.05821639546072])
graph_sketch_2d_error([0.4500220738215648, 0.39655455628712916, 58.60221347610167, 0.39662648858861527, 0.44997696089595607, 58.61192810784966, 0.45000180330478984, 0.3998385532659777, 61.44912523582034])
graph_sketch_2d_error([0.4500169846014156, 0.3971777259284086, 60.49538183392085, 0.3971841142022797, 0.45001683162836303, 60.500836183349584, 0.45001509765534803, 0.4006523683119173, 63.653745194201115])



# graph_sketch_2d_error([0.4257125761139675, 0.3750608551459574, 64.99410527148902, 0.3750748565522086, 0.42573743109867557, 64.98745115596365, 0.4312894881047037, 0.3835804094182106, 68.46175112196248])


# BEST X GATE
graph_sketch_2d_error([0.44995454137832963, 0.349951395785185, 60.08670040335893, 0.44926754338470287, 0.45016598854926143, 112.03698742807721, 0.45011208663537783, 0.34980739858671805, 60.08682492905684])
# graph_sketch_2d_error([0.4500624911536551, 0.3499218016551561, 60.08686208658856, 0.4494259675023352, 0.44999351961546735, 112.0374113636867, 0.4500616654836339, 0.34992261581752926, 60.08684420868123])
graph_sketch_2d_error([0.4500615708720576, 0.34992500235626406, 60.08827486977377, 0.4493752145833509, 0.4499623721226064, 112.03992181252575, 0.45006150379380555, 0.3499250741712414, 60.08836314140912])
graph_sketch_2d_error([0.4500227339814079, 0.3499703502872494, 60.17907254270819, 0.44943101305477484, 0.4500152091105594, 112.13349804210102, 0.45002269070994927, 0.34997039973243466, 60.179122988932704])
# graph_sketch_2d_error([0.44995454137832963, 0.349951395785185, 60.08670040335893, 0.44926754338470287, 0.45016598854926143, 112.03698742807721, 0.45011208663537783, 0.34980739858671805, 60.08682492905684],deterministic=True)
# graph_sketch_2d_error([0.4500496892122089, 0.3499388763535425, 60.11910596209083, 0.4494225214740278, 0.450007718124667, 112.06966256164945, 0.45004963884567833, 0.349938938487145, 60.119189165297314])
graph_sketch_2d_error([0.45001903597904636, 0.3499747126199972, 60.18093614492385, 0.44943418828495013, 0.45001709666822615, 112.15674028932261, 0.4500190949253575, 0.34997467182241127, 60.180937806681634])
# graph_sketch_2d_error([0.35, 0.35, float(pi/photon_interp.get_params(0.35, 0.35, g, error_w=0,error_etching=0.15,num_of_points=1)[1][0])])


graph_sketch_2d_error([0.39010334619185477, 0.3548057013999685, 90.35322534475743, 0.36537283565391604, 0.4066437070903748, 86.08434202112612, 0.39011224575124576, 0.3548446606698387, 90.39020747074869])
graph_sketch_2d_error([0.3984894510833731, 0.34997934716485213, 136.69415107422108, 0.4112563132452142, 0.4500562726274323, 44.45479572313751, 0.39850557520151547, 0.34997089330588876, 136.71300855716996])
graph_sketch_2d_error([0.3984894510833731, 0.34997934716485213, 136.69415107422108, 0.4112563132452142, 0.4500562726274323, 44.45479572313751, 0.39850557520151547, 0.34997089330588876, 136.71300855716996],deterministic=True)
graph_sketch_2d_error([0.45, 0.40897435897435896, 90.03344827802597, 0.3576923076923077, 0.3935897435897436, 84.85673370629938, 0.45, 0.40897435897435896, 90.07219803950386])
graph_sketch_2d_error([0.45, 0.40897435897435896, 90.03344827802597, 0.3576923076923077, 0.3935897435897436, 84.85673370629938, 0.45, 0.40897435897435896, 90.07219803950386],deterministic=True)

graph_sketch_2d_error([0.44978675675238416, 0.41045074887184774, 155.8259624201426, 0.4374103659628581, 0.450121350373347, 64.29822976780818, 0.4498906923275551, 0.410463300049408, 155.89775708579708],deterministic=True)
graph_sketch_2d_error([0.446225385017405, 0.40514389974633924, 153.20453680635288, 0.43440691876342546, 0.4501077282738329, 60.92527417642518, 0.446253250083401, 0.40525823226264535, 153.27357372842644],deterministic=True)



graph_sketch_2d_error([0.417633489479364, 0.365772795972942, 152.96304142124725, 0.3922171476262683, 0.45030520780000527, 50.70997142546867, 0.4179213451855304, 0.36545741673737353, 152.92768700557804])
graph_sketch_2d_error([0.417633489479364, 0.365772795972942, 152.96304142124725, 0.3922171476262683, 0.45030520780000527, 50.70997142546867, 0.4179213451855304, 0.36545741673737353, 152.92768700557804],deterministic=True)

# graph_sketch_2d_error([0.35, 0.35, pi/photon_interp.get_params(0.35, 0.35, g, error_w=0,error_etching=0.15,num_of_points=1)[1][0]],deterministic=True)



# graph_sketch_2d_error([0.4413413778013027, 0.4005137370173786, 155.75478568624234, 0.4286654723270866, 0.4496859091751056, 53.80351089580342, 0.44113230879253207, 0.40067175116802756, 155.75226878606722])
graph_sketch_2d_error([0.4413413778013027, 0.4005137370173786, 155.75478568624234, 0.4286654723270866, 0.4496859091751056, 53.80351089580342, 0.44113230879253207, 0.40067175116802756, 155.75226878606722])
graph_sketch_2d_error([0.4413413778013027, 0.4005137370173786, 155.75478568624234, 0.4286654723270866, 0.4496859091751056, 53.80351089580342, 0.44113230879253207, 0.40067175116802756, 155.75226878606722],deterministic=True)
graph_sketch_2d_error([0.4413413778013027, 0.4005137370173786, 155.75478568624234, 0.4286654723270866, 0.4496859091751056, 53.80351089580342, 0.44113230879253207, 0.40067175116802756, 155.75226878606722])

graph_sketch_2d_error([0.3820344032472623, 0.44078709459901116, 154.6715333122832, 0.38652392717883716, 0.3500041372679295, 72.88481617673861, 0.3819889266004091, 0.4408141766601171, 154.71825779821265])
graph_sketch_2d_error([0.3828515769319923, 0.4385296366491667, 155.7446720120503, 0.3982177121650603, 0.3499783374351309, 58.31008983308478, 0.3827849807636001, 0.43856789964914983, 155.79013455381084])
graph_sketch_2d_error([0.45033599367141136, 0.41389633946193066, 162.0999007993152, 0.4411801065988053, 0.44994926578118855, 68.05162728590192, 0.449972814095793, 0.41422632559851214, 162.07381824799427])
graph_sketch_2d_error([0.41348491347025895, 0.36042415402946043, 158.20016710286666, 0.35104706780593975, 0.40239935526834186, 59.055234412467925, 0.41335841710507426, 0.3600130336243682, 158.174793023064])
# graph_sketch_2d_error([0.4155317035380708, 0.3623430823298543, 158.27253597817077, 0.3504858628711794, 0.40367798218792816, 57.98372722855503, 0.415157894560961, 0.36232760845879786, 158.24701574198014])
# graph_sketch_2d_error([0.42443745625172313, 0.3713454697818225, 158.4572788254472, 0.35004652732306, 0.4048826095555877, 54.65368215376557, 0.4241759729429835, 0.37103843942159964, 158.43206068783965])
# graph_sketch_2d_error([0.35, 0.35, pi/photon_interp.get_params(0.35, 0.35, g, error_w=0,error_etching=0,num_of_points=1)[1][0]])
# graph_sketch_2d_error([0.3836683417085427, 0.35050251256281406, 195.80018869, 0.3515075376884422, 0.3836683417085427, 184.54213173, 0.38567839195979897, 0.35201005025125626, 195.88445972])
# graph_sketch([0.4053303181069862, 0.37425834742897873, 81.77996450874556, 0.37423724001555625, 0.4053013377385069, 81.77096498596387, 0.41904479842202846, 0.3905018037620236, 89.41754321970427])
graph_sketch_2d_error([0.3685121691246256, 0.36083914075562057, 13.859980046205061, 0.35006752590428386, 0.35324688037182134, 38.93384288641116, 0.35454241818044846, 0.35080083785473143, 29.33986944944547])


# graph_sketch_2d_error([0.35, 0.35, pi/photon_interp.get_params(0.35, 0.35, 0.8, error_w=0,error_etching=0,num_of_points=1)[1][0]],g=0.8)


graph_sketch_2d_error([0.3993770509332494, 0.36786096170456156, 78.90481790867072, 0.36782598456348914, 0.3993511997906787, 78.8984228737435, 0.411306508714983, 0.3808729009384435, 83.04975817409228])
graph_sketch_2d_error([0.3993770480155945, 0.36786097288131714, 78.10065137042305, 0.367825984954834, 0.39935120940208435, 78.09120312775603, 0.4113065004348755, 0.380872905254364, 82.27090242920845])
graph_sketch_2d_error([0.3980541592434589, 0.36652699724150267, 78.60490542504672, 0.36648976873405825, 0.39801426593204137, 78.60008777603184, 0.4094964417731159, 0.37864736334935456, 81.88328089362845])


graph_sketch_2d_error([0.39971509756383106, 0.3681986943848472, 78.99696256148955, 0.3681674643703996, 0.3996781304723429, 78.99110213809814, 0.4117263998263515, 0.3813926707542387, 83.34377375529733])
graph_sketch_2d_error([0.4024503529071808, 0.37108010053634644, 78.04720671982258, 0.37103480100631714, 0.40240782499313354, 77.98816701880722, 0.41519197821617126, 0.3857189416885376, 84.6294384315927])
graph_sketch_2d_error([0.4024503529071808, 0.37108010053634644, 79.42714157687804, 0.37103480100631714, 0.40240782499313354, 79.40807528393604, 0.41519197821617126, 0.3857189416885376, 85.94088996209024])
graph_sketch_2d_error([0.4024503442500182, 0.37108009673791603, 80.08158580300469, 0.3710348139483581, 0.4024078165867859, 80.07507880643332, 0.4151919813538934, 0.38571893608148233, 86.05229956941321])

graph_sketch_2d_error([0.4053303181069862, 0.37425834742897873, 81.77996450874556, 0.37423724001555625, 0.4053013377385069, 81.77096498596387, 0.41904479842202846, 0.3905018037620236, 89.41754321970427])
graph_sketch_2d_error([0.41096252214695506, 0.38075878611231767, 86.11582420200867, 0.3807575164583866, 0.41094460811913225, 86.11092755259328, 0.42640904912640354, 0.399555289643107, 96.52207960192091])

graph_sketch_2d_error([0.4134668558804942, 0.3839625145407938, 89.25818876659127, 0.38395354113354146, 0.41346087513488927, 89.25604667310014, 0.4303114918087049, 0.404375324087282, 101.10547924124775])
graph_sketch([0.4134668558804942, 0.3839625145407938, 89.25818876659127, 0.38395354113354146, 0.41346087513488927, 89.25604667310014, 0.4303114918087049, 0.404375324087282, 101.10547924124775])
graph_sketch([0.4134668558804942, 0.3839625145407938, 89.25818876659127, 0.38395354113354146, 0.41346087513488927, 89.25604667310014, 0.4303114918087049, 0.404375324087282, 101.10547924124775],deterministic=True)
graph_sketch([0.41136440455019685, 0.3824438073538559, 90.86492270359537, 0.38245958523775736, 0.4113677670177508, 90.86941093961029, 0.4295480379109651, 0.40402452296716285, 102.97391115486234],deterministic=True)
graph_sketch([0.41136440455019685, 0.3824438073538559, 90.86492270359537, 0.38245958523775736, 0.4113677670177508, 90.86941093961029, 0.4295480379109651, 0.40402452296716285, 102.97391115486234])
graph_sketch([0.41333120233721277, 0.38483754404573495, 92.75280919170497, 0.384806136496342, 0.4133902678054643, 92.76091166363759, 0.4314462752138237, 0.4064146220555132, 105.45888483636678],deterministic=True)
graph_sketch([0.41333120233721277, 0.38483754404573495, 92.75280919170497, 0.384806136496342, 0.4133902678054643, 92.76091166363759, 0.4314462752138237, 0.4064146220555132, 105.45888483636678])
graph_sketch([0.4236864489757444, 0.3972399306196176, 103.456433629278, 0.3972400717323842, 0.42368661365106236, 103.456514738865, 0.4393914917160473, 0.4163333675496143, 116.90727063226583])
graph_sketch([0.4236864489757444, 0.3972399306196176, 103.456433629278, 0.3972400717323842, 0.42368661365106236, 103.456514738865, 0.4393914917160473, 0.4163333675496143, 116.90727063226583],deterministic=True)
graph_sketch([0.42782790694677425, 0.4018711115750307, 106.49994287231534, 0.4018712457374776, 0.42782802914853235, 106.50003749818036, 0.44271506570069413, 0.4200880062670904, 120.00050708942084])
graph_sketch([0.42782790694677425, 0.4018711115750307, 106.49994287231534, 0.4018712457374776, 0.42782802914853235, 106.50003749818036, 0.44271506570069413, 0.4200880062670904, 120.00050708942084],deterministic=True)
graph_sketch([0.45, 0.4257575757575758, 123.30707788, 0.3883838383838384, 0.4095959595959596, 116.21720674, 0.45, 0.4257575757575758, 123.36014838])
graph_sketch([0.4281327724456787, 0.4020807445049286, 108.86783418202313, 0.4020809531211853, 0.4281327426433563, 108.8682536196051, 0.44298478960990906, 0.4202568829059601, 122.64283729015794])
# graph_sketch([0.44999998807907104, 0.42575758695602417, 121.95343860640861, 0.3883838355541229, 0.40959596633911133, 116.97613709278241, 0.44999998807907104, 0.42575758695602417, 121.95344108116379])
# graph_sketch([0.4425706569231666, 0.4197180611091713, 122.69112626976445, 0.4101584302991019, 0.4347461292046248, 115.23200652611426, 0.4426092392686082, 0.4197729333362817, 122.751553525953])
graph_sketch([0.42813276502955083, 0.40208073484191104, 106.66450825461582, 0.40208095691696016, 0.42813275290184705, 106.66460043282278, 0.442984801946244, 0.420256889034912, 120.17354154630745])
# graph_sketch([0.44999998807907104, 0.42575758695602417, 122.52487238239486, 0.3883838355541229, 0.40959596633911133, 116.91763445566227, 0.44999998807907104, 0.42575758695602417, 122.56685145560756])
# graph_sketch([0.4490650822372321, 0.4253295278668795, 123.2870052552814, 0.3906601364622087, 0.41274159461195137, 116.18728827858023, 0.4490593028682784, 0.42534283435641973, 123.34046623051356])
# graph_sketch([0.4499250993829017, 0.4500965872702585, 38.35119740484911, 0.3502843185339633, 0.3502803642335563, 57.127278325620075, 0.4499922217818146, 0.44992891891395703, 24.66960229377162])
# graph_sketch_2d_error([0.44964241452447995, 0.41253919447618304, 198.2427773611962, 0.3876379339372012, 0.4085219066391943, 96.7299061170322, 0.4496489609810086, 0.41254357686355236, 198.23651502579077])
# graph_sketch_2d_error([0.4413004738519009, 0.40481618991172513, 198.20341483294413, 0.38505274942366785, 0.4063057453399464, 96.70043687716093, 0.4413432688173254, 0.40485291714393395, 198.19711108530618])
# graph_sketch_2d_error([0.42970383167266846, 0.3938986659049988, 198.08834891506783, 0.3765721619129181, 0.3980318605899811, 96.75642044416962, 0.42968231439590454, 0.3938794434070587, 198.085799928548])
# graph_sketch_2d_error([0.42970383167266846, 0.3938986659049988, 198.6860262206259, 0.3765721619129181, 0.3980318605899811, 97.79005598910045, 0.42968231439590454, 0.3938794434070587, 198.6893224002543])
graph_sketch_2d_error([0.42665944597857336, 0.39096951956892745, 197.98231146050378, 0.3729850575763655, 0.3944416557296163, 96.47262997778576, 0.4266589262108397, 0.39096758470831383, 197.97602050065007])


graph_sketch_2d_error([0.45120654651250997, 0.4271931322006547, 204.77547648694699, 0.44675087425510585, 0.4514638420890208, 101.8611082290524, 0.45120917985215736, 0.42719055239406656, 204.77547661175797])
graph_sketch_2d_error([0.45068971127186785, 0.4265263834918984, 203.3473892411405, 0.44659433499316187, 0.4508237627655712, 99.99181326246035, 0.45068954246167453, 0.42652616946121047, 203.34593915593643])
graph_sketch_2d_error([0.45035534600716265, 0.4121397716989281, 198.87409291811124, 0.3830073775590843, 0.4033371850946712, 96.11079888713243, 0.45037820922329, 0.41211540708664984, 198.86891806625405])
graph_sketch_2d_error([0.4504914515724887, 0.4137686288203108, 198.10714878825596, 0.38549027881061554, 0.40475016994670454, 96.75086502738044, 0.45046301773002767, 0.41374192460843223, 198.10253624657935])


graph_sketch_2d_error([0.35059920243283577, 0.35059920243283577, 205.3385435469975, 0.35059920243283577, 0.35059920243283577, 102.6451424970296, 0.35059920243283577, 0.35059920243283577, 205.33857131217073])
graph_sketch_2d_error([0.38012486696243286, 0.41886645555496216, 0.025142056796751234, 0.3776308298110962, 0.3749050796031952, 113.44697663943987, 0.437418133020401, 0.4307648539543152, 0.03175388030942027])
graph_sketch_2d_error([0.38012485959943426, 0.41886644680893, 0.3141881686461768, 0.3776308385289864, 0.37490506631050735, 112.86423973288619, 0.4374181266734789, 0.43076486729276026, 0.31652149085070863])
graph_sketch_2d_error([0.38810536476135354, 0.3516347098325021, 87.70450190307753, 0.3499868705964847, 0.39135198812588134, 82.69880860217395, 0.3916704712593451, 0.3546043032653588, 87.73425625403289])
graph_sketch_2d_error([0.38012486696243286, 0.41886645555496216, 0.3169205896753077, 0.3776308298110962, 0.3749050796031952, 112.79595062763833, 0.437418133020401, 0.4307648539543152, 0.3141953334061201])

graph_sketch([0.38810536476135354, 0.3516347098325021, 87.70450190307753, 0.3499868705964847, 0.39135198812588134, 82.69880860217395, 0.3916704712593451, 0.3546043032653588, 87.73425625403289],deterministic=True)
graph_sketch([0.38810536476135354, 0.3516347098325021, 87.70450190307753, 0.3499868705964847, 0.39135198812588134, 82.69880860217395, 0.3916704712593451, 0.3546043032653588, 87.73425625403289],deterministic=False)


graph_sketch([0.3864108778578878, 0.3500235482342159, 88.00896853124556, 0.34998060168267814, 0.39122930071186185, 83.085492969265, 0.38639363738642074, 0.35002373286524124, 88.02569978124136],deterministic=False)

# graph_sketch([0.38480942369286364, 0.3499947773465851, 90.09078237421674, 0.3499915651978289, 0.3892539872120539, 85.20245999135912, 0.3847881167998742, 0.3499947461924299, 90.11443136804971],deterministic=True)
# graph_sketch([0.38480942369286364, 0.3499947773465851, 90.09078237421674, 0.3499915651978289, 0.3892539872120539, 85.20245999135912, 0.3847881167998742, 0.3499947461924299, 90.11443136804971],deterministic=False)

graph_sketch([0.3862615556030625, 0.3499939841276459, 88.00885592039084, 0.34998650046142143, 0.3913892787267462, 83.08536189343891, 0.38624411634884676, 0.34999400983925, 88.0255907015875],deterministic=True)
graph_sketch([0.3862615556030625, 0.3499939841276459, 88.00885592039084, 0.34998650046142143, 0.3913892787267462, 83.08536189343891, 0.38624411634884676, 0.34999400983925, 88.0255907015875],deterministic=False)

#

# X GATE

# graph_sketch([0.3497737289267662, 0.38660433888435364, 87.95792427281954, 0.389864981174469, 0.349604690257445, 83.16910603043563, 0.34977557666876713, 0.38658514618873596, 87.98620642779227])
#
# graph_sketch([0.4336734693877551, 0.39285714285714285, 87.49729480540553, 0.35, 0.386734693877551, 82.46640317936138, 0.43775510204081636, 0.39693877551020407, 87.53495302430657])

# graph_sketch([0.403861939907074, 0.44452139735221863, 87.49740155284205, 0.38988786935806274, 0.35053765773773193, 82.46629005290393, 0.4071420133113861, 0.44809266924858093, 87.53506021281085],deterministic=True)
#
# graph_sketch([0.40388432143712555, 0.44449128988778824, 87.49739798829086, 0.38991024034052724, 0.3505904739576645, 82.465240947693, 0.40715364211671945, 0.4480580263637626, 87.53526383050638],deterministic=True)
# graph_sketch([0.34977322562689356, 0.3866043432609896, 87.95792323757449, 0.38986497250151747, 0.3496041966486643, 83.16910897499586, 0.3497750659163064, 0.38658515350028577, 87.98620443976972],deterministic=True)
# graph_sketch([0.3497742281684655, 0.38660433888435364, 87.95792488910385, 0.389864981174469, 0.3496051893958919, 83.16910652978316, 0.34977607591245335, 0.38658514618873596, 87.98620705461],deterministic=True)
# graph_sketch([0.34979410153613133, 0.38660433888435364, 87.95794372265159, 0.389864981174469, 0.3496251184720355, 83.16912654312823, 0.3497959481838249, 0.38658514618873596, 87.98622658641716],deterministic=True)
#H

graph_sketch([0.3856911230095556, 0.34993062352397775, 88.01651302704578, 0.3498781146134757, 0.39023821771288725, 83.04698960151744, 0.38566904807329094, 0.3499308580022888, 88.0410089874454],deterministic=True)

graph_sketch([0.3863896018239778, 0.3499954599028283, 87.75391563033736, 0.3499877210987307, 0.39158270112900445, 82.7774118422982, 0.38636350805123454, 0.34999550778924926, 87.77890554614966],deterministic=True)
graph_sketch([0.3863896018239778, 0.3499954599028283, 87.75391563033736, 0.3499877210987307, 0.39158270112900445, 82.7774118422982, 0.38636350805123454, 0.34999550778924926, 87.77890554614966],deterministic=False)



graph_sketch([0.3986480483397012, 0.3611485099329048, 87.66117453687787, 0.3499893693808636, 0.39047746332905076, 82.62719771577761, 0.4044276052946167, 0.3660136210364151, 87.68905844284642],deterministic=True)
graph_sketch([0.3986480483397012, 0.3611485099329048, 87.66117453687787, 0.3499893693808636, 0.39047746332905076, 82.62719771577761, 0.4044276052946167, 0.3660136210364151, 87.68905844284642])

graph_sketch([0.43724958147327564, 0.39605000540894236, 87.51418156303991, 0.3499968023580319, 0.38791858907033666, 82.47351382279906, 0.4408203248719525, 0.39916910030032765, 87.54750285637637],deterministic=True)
graph_sketch([0.43724958147327564, 0.39605000540894236, 87.51418156303991, 0.3499968023580319, 0.38791858907033666, 82.47351382279906, 0.4408203248719525, 0.39916910030032765, 87.54750285637637])

# graph_sketch([0.4034164507197276, 0.4451940679609344, 87.4958809761127, 0.3892434555720146, 0.3516146533152023, 82.46383047895344, 0.4065899695708828, 0.44880486410112796, 87.53410034057288],deterministic=True)
# graph_sketch([0.4034164507197276, 0.4451940679609344, 87.4958809761127, 0.3892434555720146, 0.3516146533152023, 82.46383047895344, 0.4065899695708828, 0.44880486410112796, 87.53410034057288])
# graph_sketch([0.4395791583166333, 0.3980961923847695, 87.49729480540553, 0.3504008016032064, 0.3882765531062124, 82.46640317936138,  0.44278557114228456, 0.40070140280561123,87.53495302430657 ])
# graph_sketch([0.4395791583166333, 0.3980961923847695, 87.49729480540553, 0.3504008016032064, 0.3882765531062124, 82.46640317936138,  0.44278557114228456, 0.40070140280561123,87.53495302430657 ],deterministic=True)

graph_sketch([0.3945515561818523, 0.433309861121463, 88.26569912029086, 0.4213670822136001, 0.37932499491569543, 82.96193571534172, 0.39468565218829427, 0.43341486384402117, 88.31470256393993],deterministic=True)
graph_sketch([0.3945515561818523, 0.433309861121463, 88.26569912029086, 0.4213670822136001, 0.37932499491569543, 82.96193571534172, 0.39468565218829427, 0.43341486384402117, 88.31470256393993])

graph_sketch([0.40386194513588847, 0.444521398137139,  87.49729480540553, 0.3898878823274165, 0.3505376495874772,  82.46640317936138, 0.4071420248117633, 0.44809267830758526,  87.53495302430657],deterministic=True)
graph_sketch([0.40386194513588847, 0.444521398137139,  87.49729480540553, 0.3898878823274165, 0.3505376495874772,  82.46640317936138, 0.4071420248117633, 0.44809267830758526,  87.53495302430657])
graph_sketch([0.4135523788060193, 0.45306649510300917,  90.03344827802597, 0.3999719515696169, 0.36173017608995756,  84.85673370629938, 0.41674006187130613, 0.4565371758320344,  90.07219803950386])
graph_sketch([0.3844810777956269, 0.4274312042053985,  82.8307724157839, 0.3697197438430156, 0.32815259658251633,  78.06819500979545, 0.38794595069267773, 0.4312036832586868,  82.86642219634355],deterministic=True)

# H #
width = torch.tensor(0.4)

# H GATE - GAP = 1.2

graph_sketch([0.34997611539663565, 0.4500126276443256, 15.045592563840572, 0.3499771462671043, 0.3499897323231293, 26.436088088165096, 0.3499818334787786, 0.4500091669750361, 11.623244854559129],deterministic=False)

# graph_sketch([0.3749546378182842, 0.4057242892903664, 18.647989461963668, 0.37594900835217077, 0.3495584772812879, 19.423100989872836, 0.35185732642402445, 0.38164884131749965, 18.558383279242836])

graph_sketch([0.3499663205470181, 0.34997223063289834, 27.338976102389864, 0.34996342824628973, 0.3499727186136455, 28.407408456663077, 0.3499677062984449, 0.3499712680668102, 27.16523235167188])

graph_sketch([0.3495932329337753, 0.34959323093473493, 19.587575529946413, 0.3495782503535879, 0.3495782514182331, 20.30907880535339, 0.34959564719926745, 0.3495956453399209, 19.471455413135537])

graph_sketch([0.35000651123629534, 0.4500167618181761, 15.044419472421692, 0.35000213556196247, 0.3499825354426326, 26.637168702625576, 0.34999994262110085, 0.450018072906291, 11.631695605052235])

graph_sketch( [0.3500895870579666, 0.42959355163295765, 21.673695721384664, 0.3897923322439345, 0.3499571818904534, 14.598681000914013, 0.3501125279556983, 0.41639264488843014, 21.477101584709793])

graph_sketch([0.3500125475887855, 0.43039458765722416, 23.87623718430032, 0.4027533978698313, 0.35010942397531575, 15.313456638738812, 0.3499316872619565, 0.4322377139055034, 18.400399010213352])

graph_sketch([0.3499956785447807, 0.3858133284258017, 62.21112217235101])




# H GATE #

graph_sketch([0.3500000673141819, 0.44999992295270336, 8.244492010962203, 0.3500000398373937, 0.41032179604049324, 11.48138136472522, 0.35000007973656344, 0.38167972128082595, 10.975810202404265, 0.34999999971981355, 0.44999989907148624, 6.043852625586868])

graph_sketch([0.35000994047441775, 0.44999379576505566, 9.026199162097166, 0.3501923070771353, 0.42704383802190754, 5.749589851349145, 0.35007905161828445, 0.38064164309573517, 14.460740437502714, 0.3500089412476713, 0.4499889573471395, 7.052554084698665])

graph_sketch([0.3499725869831581, 0.45014737444751435, 14.250768475972787, 0.4360623243614979, 0.4500963146872402, 14.44717451061039, 0.3499884723461997, 0.45008657855374734, 8.83593165097425])

graph_sketch([0.3500236026490981, 0.45000056885330536, 14.49115153688338, 0.43922537812731655, 0.4500018395574066, 13.153278439292361, 0.4014668377165904, 0.4499192252700867, 1.8404770602620257, 0.35007318293074186, 0.4500009139135378, 8.396218041027446])

graph_sketch([0.35420536204176956, 0.44997578291386836, 14.03848218029598, 0.4311445510076908, 0.4498190127338668, 12.105503973485426, 0.4392450141770036, 0.4236478780602185, 1.128549588628337, 0.36357225721006203, 0.447473730603727, 11.068001761716976]
             )

graph_sketch([0.3622232440257242, 0.4482950068124242, 14.841289004341384, 0.41350436448678873, 0.4292736047050131, 8.172904419890989, 0.3862370544164804, 0.4457981000848923, 8.608302224468007, 0.35932860971355624, 0.4262905914642976, 7.396150378032533])

graph_sketch([0.3625805289834306, 0.449999416239135, 17.848228932979403, 0.44332002767111167, 0.4499993489269298, 12.495785828394983, 0.3500203999955872, 0.45000744884665067, 8.33856218515888])

graph_sketch([0.3628236218185769, 0.44995362298402775, 17.811664230429976, 0.4026501701994074, 0.4109419983306692, 12.054928540510891, 0.3500463173943722, 0.4499488754628536, 8.227209424268375])

graph_sketch([0.3502417873056916, 0.4499915631390665, 8.298833438956805, 0.4056963955675998, 0.4489310645470341, 16.87991707056952, 0.3903328903388734, 0.4499958837897362, 14.307951713629027])

graph_sketch([0.3500933129448907, 0.44996212437634153, 8.019287970579754, 0.375813218780459, 0.42173165895552017, 16.55394383693006, 0.3728774267816912, 0.4332120303980873, 14.028386391551223])

graph_sketch([0.3502126811858632, 0.43793500883045744, 17.596467772501338, 0.35049149414994357, 0.3625036739549172, 11.477452015238322, 0.349999953611743, 0.44997388674829836, 7.910514856626923])

graph_sketch([0.3690339346603942, 0.4395345475398205, 17.197067379322263, 0.3928695879196176, 0.4363029173711551, 12.847095165407591, 0.39251019302312207, 0.4452676356813456, 10.537576334346278])

graph_sketch([0.34982029442629825, 0.4501467896486531, 12.636319195212655, 0.3499829841158104, 0.4215542410392942, 3.266471921423181, 0.3499603555283653, 0.34997889201008453, 9.376500018422856, 0.349960059902685, 0.4428029130291205, 10.895082239168335])


covariance_error_graph([0.350105696,0.4088117,39.17195325])

covariance_error_graph([0.34982029442629825, 0.4501467896486531, 12.636319195212655, 0.3499829841158104, 0.4215542410392942, 3.266471921423181, 0.3499603555283653, 0.34997889201008453, 9.376500018422856, 0.349960059902685, 0.4428029130291205, 10.895082239168335])

covariance_error_graph([0.3502796695381757, 0.44999720368065993, 10.027796741739797, 0.3501888036200563, 0.4496404380739273, 5.1437846877418965, 0.35007020998201244, 0.3503891469753695, 10.435225211509534, 0.35018102922787675, 0.4469339994340576, 10.362323782524255])

covariance_error_graph([0.34995070135181994, 0.4500370280183894, 15.282302315834691, 0.34998235821019386, 0.3500012696634736, 10.693271942740726, 0.3499782530149288, 0.45000480557514677, 9.94204523954269])

covariance_error_graph([0.3496804499238524, 0.4502363153264771, 15.18997889437537, 0.34987891300092006, 0.3499999425798779, 10.740172519278609, 0.3498760629806855, 0.45006404197628685, 9.955369346395704])

covariance_error_graph([0.35004613644849236, 0.4499535114878301, 13.282142378964688, 0.37555347400309347, 0.3499885281476948, 7.550338400344818, 0.3500285799392844, 0.43047133850094627, 15.908535826163485])
covariance_error_graph([0.3500005278896713, 0.449995194982039, 11.302770187740538, 0.35000505491013334, 0.38946315776787477, 19.30740940499456, 0.3500662986221449, 0.4500017516117995, 5.945297767100885])


covariance_error_graph([0.3497103912016633, 0.45021488784278, 14.982108984317346, 0.35065045358513386, 0.3509886476632127, 9.400350746611617, 0.3512577370959024, 0.4454305221617572, 10.628190303338874])
covariance_error_graph([0.35123311276911773, 0.4495714481539855, 13.603099127510843, 0.3533472101182689, 0.35279752693717065, 6.934743729783495, 0.35086246817786065, 0.4258268054936425, 14.932456625462276])
covariance_error_graph([0.349932458957561, 0.4500450369510402, 15.20625137276606, 0.3499772151226439, 0.35013782044353087, 9.847601913878076, 0.34997694704989313, 0.45003197323959077, 9.906494690316256])
