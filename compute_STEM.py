import numpy as np
from pySTEM import STEM
import netCDF4 as nc

#Default variable names. Can be changed with the set_variable_names function.
variable_names = {
"pressure":'lev',
"latitude":'lat',
"variance":'MSE2',
"eddy_covariance":'VMSE_p',
"mean_velocity":'V',
"mean_zeta":'MSE'
}

def set_variable_names(p_name=variable_names["pressure"],
                       lat_name=variable_names["latitude"],
                       variance_name=variable_names["variance"],
                       eddy_covar_name=variable_names["eddy_covariance"],
                       mean_velocity_name=variable_names["mean_velocity"],
                       mean_zeta_name=variable_names["mean_zeta"]):
    '''
    This function sets the name of the variables to read. zeta can be
    any thermodynamical coordinate that you want to have your stream function
    on.
    '''
    variable_names.update({
    "pressure":p_name,
    "latitude":lat_name,
    "variance":variance_name,
    "eddy_covariance":eddy_covar_name,
    "mean_velocity":mean_velocity_name,
    "mean_zeta":mean_zeta_name
    })
    return

def get_streamfunction(filename,T,var_isolated=None,filename_control=None,variable_names=variable_names):
    '''
    Computes the stream function on thermodynamical levels (the thermodynamical
    variable is chosen in the set_variable_names function.)
    Input : filename [string]
            T (plotting levels) [numpy 1D array]
            **kwargs:
            var_isolated (either 'mean_zeta', 'eddy_covariance' or 'variance') [string]
            filename_control (name of the control experiement to take all the data from except for the isolated variable is present)[string]
            variable_names [python dictionary]
    Output : stem_var (STEM variable including mass transport, stream function
             and the poleward heat transport.)
    '''
    if filename_control==None or var_isolated==None:
        filename_control=filename
        if var_isolated !=None:
            print('No control experiment file was given, the STEM calculations are done only with one file. Not isolating features.')
    #Read in variables from netCDF file.

    file_control=nc.Dataset(filename_control)
    P=file_control.variables[variable_names["pressure"]][:]
    if P[-1]//1000<=1:
        #converting to Pa
        P=P*100.


    L=file_control.variables[variable_names["latitude"]][:]
    om=file_control.variables[variable_names["mean_zeta"]][:,:]
    vm=file_control.variables[variable_names["mean_velocity"]][:,:]
    vo=file_control.variables[variable_names["eddy_covariance"]][:,:]
    o2=file_control.variables[variable_names["variance"]][:,:]
    file_control.close()

    if var_isolated == 'variance':
        file=nc.Dataset(filename)
        o2=file.variables[variable_names["variance"]][:,:]
        file.close()

    if var_isolated == 'eddy_covariance':
        file=nc.Dataset(filename)
        vo=file.variables[variable_names["eddy_covariance"]][:,:]
        file.close()


    if var_isolated == 'mean_zeta':
        file=nc.Dataset(filename)
        om=file.variables[variable_names["mean_zeta"]][:,:]
        file.close()

    #Compute the stem variables
    stem_var=STEM(np.squeeze(vm),np.squeeze(om),np.squeeze(vo),np.squeeze(o2),L,P,T)
    STEM.set_M(stem_var)
    STEM.set_S(stem_var)
    STEM.set_H(stem_var)
    return stem_var

def get_diff_streamfunction(psi_control,psi_perturbed):
    '''
    Computes the perturbation to the stream function between two different
    cases.
    Input : psi_control (STEM_var) [object]
            psi_perturbed (STEM_var) [object]
    Output : eddy_diff (difference in the eddy stream function)
             mean_diff (difference in the mean stream function)
             total_diff (difference in the total stream function)
    '''
    eddy_diff = psi_perturbed.SE - psi_control.SE
    mean_diff = psi_perturbed.SM - psi_control.SM
    total_diff = psi_perturbed.ST - psi_control.ST
    return eddy_diff, mean_diff, total_diff
