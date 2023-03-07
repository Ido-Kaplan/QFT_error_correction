import numpy as np
import torch
import os
from math import pi, e
from matplotlib import pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# parameters
num_of_points_used = 51**2#101**2
num_of_matrices = 3
n=1
g=1.2
uncorrelation_factor = 0.05
total_cross_correlation = False

plot_loss_graph = False
fully_correlated = False
partial_correlated_between_waveguides = False
partial_correlated_between_segments = True
partial_correlated_both_rho_values = False

# matrices and factors
global_factors = [1j,1j,1j,1,1j**0.25]
global_factor = global_factors[n]

X_matrices = []
H = torch.tensor([[1.0 + 0j, 1.0 + 0j],[1.0 + 0j, -1.0 + 0j]])/(2**0.5)
X = torch.tensor([[0.0 + 0j, 1.0 + 0j],[1.0 + 0j, 0.0 + 0j]])
X_SQRT = torch.tensor([[0.5+0.5j,0.5-0.5j],[0.5-0.5j,0.5+0.5j]])
X_third = torch.sqrt(torch.tensor(1/3)) * torch.eye(2) - 1j * torch.sqrt(torch.tensor(2/3)) *  X
T_gate = torch.tensor([[1,0],[0,e**(1j*pi/4)]])
X_matrices.append(H) # no need to calculate X^0.
X_matrices.append(X)
X_matrices.append(X_SQRT)
X_matrices.append(X_third)
X_matrices.append(T_gate)
rho = X_matrices[n]

# Uc calculation - matrix which simulates the coupling interface
eta_c = torch.tensor(0.0528)
Uc = torch.sqrt(1-eta_c) * torch.eye(2) - 1j * torch.sqrt(eta_c) * X
Uc = Uc.type(torch.complex64)
Uc_dag = torch.transpose(torch.conj(Uc),0,1)


# width range and errors

min_width = 0.31#0.35
max_width = 0.49#0.45

width_error = 0.02/3
etch_error  = 0.003
num_of_sigma = 3

full_etch = False

if full_etch:
    width_error = 0.01/3
    min_width = 0.56
    max_width = 0.64
    g=0.8

# general math functions


def get_norm_dist_vector(mean,sd,num_of_points=num_of_points_used,num_of_sigma=num_of_sigma):
    if sd==0:
        sd=0.001
    x = np.linspace(mean-num_of_sigma*sd,mean+num_of_sigma*sd,num_of_points)
    norm_dist = np.exp(-0.5*((x-mean)/sd)**2) * (1/(sd*(np.pi*2)**0.5))
    norm_dist = norm_dist/sum(norm_dist)
    return norm_dist

# mat creation functions

def return_mat_generic(params,error_w1=0.0,error_w2=0.0,error_etching = 0.15,debug=False,uncorrelated_errors=False):
    mat = torch.eye(2)+0*1j
    for index in range(0,len(params),3):
        mat = mat@(get_R_mat(params,error_w1=error_w1,error_w2=error_w2,index=index,error_etching = error_etching,debug=debug))

    return mat*global_factor



def get_delta_deterministic(parameters,index,error_w1=0.0,error_w2=0.0,etch=0.15):
    if full_etch:
        return 54.77762688332884*(parameters[index]+error_w2) -108.23862451848406*((parameters[index]+error_w2)**2)+   100.573254921413*((parameters[index]+error_w2)**3) \
               -36.31459791924153*((parameters[index]+error_w2)**4)  -54.777590249118795*(parameters[index+1]+error_w1)+ \
               108.2384407372418*((parameters[index+1]+error_w1)**2) -100.57294803998917*((parameters[index+1]+error_w1)**3)+  36.31442735288298*((parameters[index+1]+error_w1)**4)


    if g==1.0:
        return 4.518902697716317*(parameters[index+1]+error_w2)-19.55034473213622*((parameters[index+1]+error_w2)**2)+ \
               29.83783108325303*((parameters[index+1]+error_w2)**3)-16.299611836082956*((parameters[index+1]+error_w2)**4) \
               -4.518902681127404*(parameters[index]+error_w1)+19.550344602641427*((parameters[index]+error_w1)**2) \
               -29.837830751123285*((parameters[index]+error_w1)**3)+16.299611556088976*((parameters[index]+error_w1)**4)



        # return (parameters[index]+error_w1-parameters[index+1]-error_w2)*0.966
    if g==0.8:
        return ((parameters[index]+error_w1-0.4)+(0.4-parameters[index+1]-error_w2))*0.874
    if g==1.2:
        if etch != 0.15:
            return (-0.56535098*(parameters[index]+error_w1)+17.70314054*((parameters[index]+error_w1)**2)-17.37052673*((parameters[index]+error_w1)**3)
                    +0.565351*(parameters[index+1]+error_w2)-17.70314065*((parameters[index+1]+error_w2)**2)+17.37052685*((parameters[index+1]+error_w2)**3))*(etch*(-5.38877444)+1)
        else:
            return 3.94502808*(parameters[index+1]+error_w1) -18.0203544*((parameters[index+1]+error_w1)**2)+   27.94843595*((parameters[index+1]+error_w1)**3) \
                -15.42295066*((parameters[index+1]+error_w1)**4)  -3.94502818*(parameters[index]+error_w2)+ \
                18.02035521*((parameters[index]+error_w2)**2) -27.94843797*((parameters[index]+error_w2)**3)+  15.42295233*((parameters[index]+error_w2)**4)


def get_omega_deterministic(parameters,index,error_w1=0.0,error_w2=0.0,etch=0.15):
    if full_etch:
        return 37.13264159015594 -133.4240683709093*((parameters[index]+parameters[index+1]+error_w1+error_w2)) \
               + 186.7310783954773*((parameters[index]+parameters[index+1]+error_w1+error_w2)**2)-125.57090875617125*((parameters[index]+parameters[index+1]+error_w1+error_w2)**3)+ \
               39.6619729751983*((parameters[index]+parameters[index+1]+error_w1+error_w2)**4)-4.458947519410447*((parameters[index]+parameters[index+1]+error_w1+error_w2)**5)

    if g==1.0:
        return -0.7815084452228636 + 5.752024688674808*((parameters[index]+parameters[index+1]+error_w1+error_w2)) \
                              -15.185988643025688*((parameters[index]+parameters[index+1]+error_w1+error_w2)**2)+ \
                              19.654587033031753*((parameters[index]+parameters[index+1]+error_w1+error_w2)**3) \
                              -12.602284532129548*((parameters[index]+parameters[index+1]+error_w1+error_w2)**4)+ \
                              3.213087927764657*((parameters[index]+parameters[index+1]+error_w1+error_w2)**5)
        # return 0.024255/(parameters[index+1]+parameters[index]+error_w1+error_w2)+0.02475
    if g==0.8:
        return (-((parameters[index]-parameters[index+1]-error_w1+error_w2)**2)*100*(1.0185-1.013)+(parameters[index]+parameters[index+1]+error_w1+error_w2)*0.015+1.0065)*0.1
    if g==1.2:
        if g==1.2:
            if etch!=0.15:
                return (-0.45973388*(parameters[index]+parameters[index+1]+error_w1+error_w2)+0.33273332*((parameters[index]+parameters[index+1]+error_w1+error_w2)**2)-0.08432326*((parameters[index]+parameters[index+1]+error_w1+error_w2)**3))*(-3.18606085*etch+1)+0.13539372
            else:
                return 0.38044405 -1.48138422*(parameters[index]+parameters[index+1]+error_w1+error_w2)+  2.51783632*((parameters[index]+parameters[index+1]+error_w1+error_w2)**2) \
            -1.9993113*((parameters[index]+parameters[index+1]+error_w1+error_w2)**3)+   0.60771393*((parameters[index]+parameters[index+1]+error_w1+error_w2)**4)


def get_R_mat(parameters,error_w1= 0.0,error_w2= 0.0, index=0,error_etching=None,debug=False):
    if error_etching==None:
        error_etching=0.15


    omega = get_omega_deterministic(parameters,index,error_w1=error_w1,error_w2=error_w2,etch=error_etching)
    delta = get_delta_deterministic(parameters,index,error_w1=error_w1,error_w2=error_w2,etch=error_etching)


    t = parameters[index+2]
    if debug:
        print("w0:",parameters[index]+error_w1,"w1:",parameters[index+1]+error_w2,"t:",t)
        print("omega:",float(omega),"delta:",float(delta),"length:",float(t))


    Omega_G=torch.sqrt((omega)**2+(delta)**2)
    A = t*Omega_G
    mat = torch.cos((A))*torch.eye(2)+ \
          torch.sin(A)*((delta)/Omega_G)*torch.tensor([[-1j,0],[0,1j]])+ \
          torch.sin(A)*((omega)/Omega_G)*torch.tensor([[0,-1j],[-1j,0]])
    mat = mat.type(torch.complex64)
    return mat






def return_mat_robust_generic(params,error_w1=0.0,error_w2=0.0,etch=0.15,num_of_points_used=num_of_points_used,total_cross_correlation=False):
    mat = torch.eye(2).repeat(num_of_points_used,1).reshape(num_of_points_used,2,2)
    mat = mat.type(torch.complex64)
    for index in range(0,len(params),3):
        if total_cross_correlation:
            mat = mat@(get_R_robust(params,error_w1=error_w1[:,index//3],error_w2=error_w2[:,index//3],etch=etch,index=index))
        else:
            mat = mat@(get_R_robust(params,error_w1=error_w1,error_w2=error_w2,etch=etch,index=index))
    return mat*global_factor




def get_R_robust(parameters,error_w1= 0.0,error_w2= 0.0,etch = 0.15, index=0):
    if type(error_w1)!=float:
        vec_len = len(error_w1)



    omega = get_omega_deterministic(parameters,index,error_w1=error_w1,error_w2=error_w2,etch=etch)
    delta = get_delta_deterministic(parameters,index,error_w1=error_w1,error_w2=error_w2,etch=etch)

    t = parameters[index+2]

    Omega_G=torch.sqrt((omega)**2+(delta)**2)
    A =t*Omega_G
    mat = torch.cos((A)).reshape(vec_len,1,1)*torch.eye(2).repeat(vec_len,1).reshape(vec_len,2,2)+ \
          (torch.sin(A)*((delta)/Omega_G)).reshape(vec_len,1,1)*torch.tensor([[-1j,0],[0,1j]]).repeat(vec_len,1).reshape(vec_len,2,2)+ \
          (torch.sin(A)*((omega)/Omega_G)).reshape(vec_len,1,1)*torch.tensor([[0,-1j],[-1j,0]]).repeat(vec_len,1).reshape(vec_len,2,2)
    return mat




# loss functions

def torch_trace(input, axis1=1, axis2=2):
    assert input.shape[axis1] == input.shape[axis2], input.shape
    shape = list(input.shape)
    strides = list(input.stride())
    strides[axis1] += strides[axis2]
    shape[axis2] = 1
    strides[axis2] = 0
    input = torch.as_strided(input, size=shape, stride=strides)
    return input.sum(dim=(axis1, axis2))

def loss_fn_Fidelity(sigma, rho=torch.tensor([[0+0j,1.0+0j],[1.0+0j,0+0j]])):
    if len(sigma.data.shape)==2:
        return 1-torch.abs((torch.trace(torch.transpose(torch.conj(rho),0,1)@sigma)/2))
    else:
        return 1 - torch.abs((torch.trace(torch.sum(((torch.conj(rho.T) @ sigma).T * norm_dist_vec_2d).T, axis=0)) / 2))
        # trace_vals = torch.tensor([2.0]*len(sigma))
        # return torch.sum(torch.abs((trace_vals - torch_trace(torch.conj(rho.T)@sigma)))* norm_dist_vec_2d)




def loss_F_norm(sigma, rho=torch.tensor([[0+0j,1.0+0j],[1.0+0j,0+0j]])): #Frobenius norm
    return torch.sqrt(torch.sum(torch.transpose(torch.conj(sigma-rho),0,1)*(sigma-rho)))/2

def delta_distance_loss(parameters):
    return ((parameters[1] - parameters[3])/(parameters[1] + parameters[3]))**2+((parameters[3] - parameters[5])/(parameters[3] + parameters[5]))**2

def loss_special(sigma, rho=torch.tensor([[0+0j,1.0+0j],[1.0+0j,0+0j]])):
    diff = sigma-rho
    if len(diff.data.shape)==2:
        return torch.sum((torch.real(diff))**2+(torch.imag(diff))**2)
    else:
        return torch.sum(torch.sum(torch.sum((torch.real(diff))**2+(torch.imag(diff))**2,axis=1),axis=1)*1)#*norm_dist_vec_2d)


def loss_fn_sum(sigma, rho=torch.tensor([[0+0j,1.0+0j],[1.0+0j,0+0j]])):
    return torch.sum(torch.abs(sigma - rho))

def negative_val_loss(parameters):
    return -torch.min(torch.min(parameters),torch.tensor(0))

def define_correct_range(params,min_val,max_val):
    error = torch.max(torch.tensor(0),torch.max((params[0]-max_val),(min_val-params[0])))**2
    for i in range(1,len(params)):
        error+=torch.max(torch.tensor(0),torch.max((params[i]-max_val),(min_val-params[i])))**2
    return error

def value_range_loss(parameters):
    ts_error = negative_val_loss(parameters[2::3])
    w1_error = define_correct_range(parameters[::3],min_width,max_width)
    w2_error = define_correct_range(parameters[1::3],min_width,max_width)
    return ts_error+w1_error+w2_error

def robustness_loss_generic(param, norm_dist_vec, low=-3*width_error,high=3*width_error,num_of_points = num_of_points_used,rho=torch.tensor([[0+0j,1.0+0j],[1.0+0j,0+0j]])):
    curr_loss = torch.tensor(0*1j)
    index_width_0 = 0
    for width_error in torch.linspace(low,high,num_of_points):
        curr_loss+=loss_fn_Fidelity(return_mat_generic(param,error_w1=width_error,error_w2=width_error),rho=rho)*norm_dist_vec[index_width_0]
        index_width_0+=1
    return curr_loss


if __name__ == "__main__":
    lr = 10**(-3)
    num_of_epochs = 100000
    lambda_factor = 0.5
    num_of_parameter_sets = 3000
    max_legal_loss_val = 10**(-4)
    write_enabled = True
    random_sample = False
    step_size = num_of_epochs//4
    gamma = 0.1

    # 2d normal distribution generation
    norm_dist_vec = torch.tensor(get_norm_dist_vector(0,0.1,num_of_points_used))
    norm_dist_vec_delta = get_norm_dist_vector(0,0.1,int(num_of_points_used**0.5),num_of_sigma=num_of_sigma)
    norm_dist_vec_omega = get_norm_dist_vector(0,0.1,int(num_of_points_used**0.5),num_of_sigma=num_of_sigma)
    norm_dist_vec_2d = np.array([1.0]*num_of_points_used)
    for i in range(len(norm_dist_vec_omega)):
        for j in range(len(norm_dist_vec_delta)):
            norm_dist_vec_2d[i*len(norm_dist_vec_omega)+j]*=norm_dist_vec_omega[i]*norm_dist_vec_delta[j]
    norm_dist_vec_2d[norm_dist_vec_2d==1.0] = np.min(norm_dist_vec_2d)
    norm_dist_vec_2d = torch.tensor(norm_dist_vec_2d)





    # main
    valid_values = 0
    for num_of_matrices in np.arange(3,20).repeat(5):
        f=open("new_values_for_n_approximation_bigger_n.txt", 'a')
        init_values=torch.tensor(np.random.random(num_of_matrices*3))*(max_width-min_width)+min_width
        init_values[1::3]=torch.tensor(np.random.random(num_of_matrices))*(max_width-min_width)+min_width
        max_t = pi/(max(1,n) * num_of_matrices * get_omega_deterministic(init_values,index=0))
        min_t = 0
        init_values[2::3]=torch.tensor(np.random.random(num_of_matrices))*(max_t-min_t)+min_t
        step_size = 1
        gamma = 0.1

        init_values = torch.tensor([0.375, 0.425, 24.627, 0.429, 0.363, 26.304, 0.391, 0.45, 23.315])

        init_values = init_values.type(torch.double)
        parameters = torch.nn.Parameter(init_values)


        # Fully correlated error
        if fully_correlated:
            norm_dist_vec_2d = norm_dist_vec
            error_w1 = torch.linspace(-width_error,width_error,num_of_points_used)*3
            error_w2 = torch.linspace(-width_error,width_error,num_of_points_used)*3


        # Partial uncorrelated error between waveguides
        elif partial_correlated_between_waveguides:

            cov = (width_error**2) * np.array([[1, 0.99], [0.99, 1]])
            all_errors = torch.tensor(np.random.multivariate_normal([0, 0], cov, num_of_points_used)).squeeze()
            error_w1 = all_errors[:, 0]
            error_w2 = all_errors[:, 1]
            error_w1 = error_w1.type(torch.complex64)
            error_w2 = error_w2.type(torch.complex64)

        # Partial uncorrelated error between waveguides
        elif partial_correlated_between_segments:

            rho_segments = 0.7
            cov = (width_error**2) * np.array([[1, rho_segments,rho_segments**2], [rho_segments, 1,rho_segments],[rho_segments**2,rho_segments,1]])
            all_errors = torch.tensor(np.random.multivariate_normal([0, 0,0], cov, num_of_points_used)).squeeze()
            error_w1 = all_errors
            error_w2 = all_errors
            error_w1 = error_w1.type(torch.complex64)
            error_w2 = error_w2.type(torch.complex64)
            total_cross_correlation = True



        elif partial_correlated_both_rho_values:
            num_of_segments = len(init_values)//3
            cross_cov = 0.7
            uncorrelation_factor = 0

            inside_segment_rho_mat = []
            between_segments_rho_mat = []
            for i in range(num_of_segments*2):
                inside_segment_rho_mat.append([])
                between_segments_rho_mat.append([])
                for j in range(num_of_segments*2):
                    if (i-j)%2==1:
                        inside_segment_rho_mat[-1].append(1-uncorrelation_factor)
                        between_segments_rho_mat[-1].append(0)
                    elif i==j:
                        inside_segment_rho_mat[-1].append(0)
                        between_segments_rho_mat[-1].append(0)
                    else:
                        inside_segment_rho_mat[-1].append(0)
                        between_segments_rho_mat[-1].append(cross_cov)


            inside_segment_rho_mat = np.array(inside_segment_rho_mat)
            between_segments_rho_mat = np.array(between_segments_rho_mat)

            cov = (width_error**2)*(np.eye(num_of_segments*2) + inside_segment_rho_mat + between_segments_rho_mat)
            all_errors = torch.tensor(np.random.multivariate_normal([0]*num_of_segments*2, cov, num_of_points_used)).squeeze()
            error_w1 = all_errors[:,0::2]
            error_w2 = all_errors[:,1::2]
            error_w1 = error_w1.type(torch.complex64)
            error_w2 = error_w2.type(torch.complex64)
            total_cross_correlation = True
            print(cov/(width_error**2))
            print(error_w1)
            print(error_w2)
            exit(0)
        # Fully uncorrelated error
        else:
            error_w1 = torch.linspace(-width_error,width_error,int(num_of_points_used**0.5)).repeat(int(num_of_points_used**0.5))
            error_w2 = torch.linspace(-width_error,width_error,int(num_of_points_used**0.5)).repeat_interleave(int(num_of_points_used**0.5))




        etch = 0.15
        loss_func = lambda x: 10*value_range_loss(parameters)+loss_fn_Fidelity(return_mat_robust_generic(parameters,error_w1=error_w1,error_w2=error_w2,num_of_points_used=num_of_points_used,total_cross_correlation=total_cross_correlation),rho=Uc_dag@rho@Uc_dag)#+10*loss_fn_Fidelity(return_mat_generic(parameters),rho=Uc_dag@rho@Uc_dag)

        mat= return_mat_generic(parameters)
        optimizer = torch.optim.Adam([parameters], lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        scheduler_flag = False
        destination_reached = False

        print("parameters=",parameters)
        print("BEFORE:\nmat=",mat,"\nloss func=",loss_func(parameters),"\nfidelity loss=",loss_fn_Fidelity(mat,rho=rho))
        loss_vec = []
        for step in range(num_of_epochs):
            loss = torch.tensor(0*1j)


            loss+=loss_func(parameters)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_vec += [ loss.item() ]

            # if not scheduler_flag:
            #     if abs(loss_vec[-1]) < 10 ** (-4):
            #         scheduler.step()
            #         scheduler_flag = True
            #         print("Made a single scheduler step")
            #
            # if abs(loss_vec[-1]) < 3*10**(-6):
            #     destination_reached = True
            #     break
            #
            # if step > 50000 and not scheduler_flag:
            #     break

            if step%100==0:
                cur_params = np.array(parameters.data)
                print("step number:",step,"current loss:",loss.item(),"current parameters:",list(cur_params))#list(np.hstack([np.array(parameters.data)[:-num_of_matrices],np.array(parameters.data)[-num_of_matrices:]/t_factor])))
                if step%1000==0:
                    print(Uc@return_mat_generic(parameters)@Uc)




        mat = return_mat_generic(parameters)
        print("parameters=",parameters)
        list_of_parameters = list(parameters.detach().numpy())
        if write_enabled and destination_reached:
            f.write(str(list_of_parameters)+"\n")
        f.close()
        if plot_loss_graph:
            plt.plot(loss_vec)
            plt.show()
