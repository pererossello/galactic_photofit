import numpy 
import matplotlib.pyplot as plt
import pyimfit
import astropy
from astropy.io import fits
import photutils as phu
import numpy as np
import xarray as xr
import os


def exp_model(x0, y0, PA_exp, ell_exp, I_0, h):
    p_exp = {'PA': [PA_exp, 0, 180], 'ell_exp': [ell_exp, 0, 1], 'I_0': [I_0, 0, 10*I_0],
             'h': [h, 0.0, 10*h]}
    exp_dict = {'name': "Exponential", 'label': "exp", 'parameters': p_exp}
    
    funcset_dict = {'X0': [x0, x0 - 10, x0 + 10], 'Y0': [y0, y0 - 10, y0 + 10],
                    'function_list': [exp_dict]}
    model_dict = {'function_sets': [funcset_dict]}
    
    model = pyimfit.ModelDescription.dict_to_ModelDescription(model_dict)
    return model


def sersic_model(x0, y0, PA_sersic, ell_sersic, n, I_e, r_e):
    p_sersic = {'PA': [PA_sersic, 0, 180], 'ell_sersic': [ell_sersic, 0, 1], 'n': [n, 0.5, 5],
                'I_e': [I_e, 0.0, 10*I_e], 'r_e': [r_e, 0.0, 10*r_e]}
    sersic_dict = {'name': "Sersic", 'label': "sersic", 'parameters': p_sersic}
    
    funcset_dict = {'X0': [x0, x0 - 10, x0 + 10], 'Y0': [y0, y0 - 10, y0 + 10],
                    'function_list': [sersic_dict]}
    model_dict = {'function_sets': [funcset_dict]}
    
    model = pyimfit.ModelDescription.dict_to_ModelDescription(model_dict)
    return model


def exp_sersic_model(x0, y0, PA_sersic, ell_sersic, n, I_e, r_e, 
                 PA_exp, ell_exp, I_0, h):
    # dict describing the bulge (first define the parameter dict, with initial values
    # and lower & upper limits for each parameter)
    p_sersic = {'PA': [PA_sersic, 0, 180], 'ell_sersic': [ell_sersic, 0, 1], 'n': [n, 0.5, 5],
                'I_e': [I_e, 0.0, 10*I_e], 'r_e': [r_e, 0.0, 10*r_e]}
    sersic_dict = {'name': "Sersic", 'label': "sersic", 'parameters': p_sersic}
    # do the same thing for the disk component
    p_exp = {'PA': [PA_exp, 0, 180], 'ell_exp': [ell_exp, 0, 1], 'I_0': [I_0, 0, 10*I_0],
                'h': [h, 0.0, 10*h]}
    exp_dict = {'name': "Exponential", 'label': "exp", 'parameters': p_exp}

    # make dict for the function set that combines the sersic and disk components
    # with a single shared center, and then a dict for the whole model
    funcset_dict = {'X0': [x0, x0 - 10, x0 + 10], 'Y0': [y0, y0 - 10, y0 + 10],
                    'function_list': [sersic_dict, exp_dict]}
    model_dict = {'function_sets': [funcset_dict]}

    model = pyimfit.ModelDescription.dict_to_ModelDescription(model_dict)
    return model

def exp_sersic_sersic_model(
    x0, y0, PA_sersic, ell_sersic, n, I_e, r_e,
    PA_exp, ell_exp, I_0, h,
    x02, y02, PA_sersic2, ell_sersic2, n2, I_e2, r_e2
):

    # dict describing the bulge (first define the parameter dict, with initial values
    # and lower & upper limits for each parameter)
    p_sersic = {'PA': [PA_sersic, 0, 180], 'ell_sersic': [ell_sersic, 0, 1], 'n': [n, 0.5, 5],
                'I_e': [I_e, 0.0, 10*I_e], 'r_e': [r_e, 0.0, 10*r_e]}
    sersic_dict = {'name': "Sersic", 'label': "sersic", 'parameters': p_sersic}
    # do the same thing for the disk component
    p_exp = {'PA': [PA_exp, 0, 180], 'ell_exp': [ell_exp, 0, 1], 'I_0': [I_0, 0, 10*I_0],
                'h': [h, 0.0, 10*h]}
    exp_dict = {'name': "Exponential", 'label': "exp", 'parameters': p_exp}

    # make dict for the function set that combines the sersic and disk components
    # with a single shared center, and then a dict for the whole model
    funcset_dict = {'X0': [x0, x0 - 10, x0 + 10], 'Y0': [y0, y0 - 10, y0 + 10],
                    'function_list': [sersic_dict, exp_dict]}

    # New parameters for the second Sersic component
    p_sersic_2 = {'PA': [PA_sersic2, 0, 180], 'ell_sersic': [ell_sersic2, 0, 1], 'n': [n2, 0.5, 5],
                  'I_e': [I_e2, 0.0, 10*I_e2], 'r_e': [r_e2, 0.0, 10*r_e2]}
    sersic_dict_2 = {'name': "Sersic", 'label': "sersic_2", 'parameters': p_sersic_2}

    # make dict for the function set for the new Sersic component
    funcset_dict_2 = {'X0': [x02, x02 - 10, x02 + 10], 'Y0': [y02, y02 - 10, y02 + 10],
                      'function_list': [sersic_dict_2]}

    # Updated model_dict to include the new function set
    model_dict = {'function_sets': [funcset_dict, funcset_dict_2]}

    model = pyimfit.ModelDescription.dict_to_ModelDescription(model_dict)
    return model



def get_dic_result(imfit_fitter, printit=False, savepath=None):

    param_names = imfit_fitter.numberedParameterNames
    chi2 = imfit_fitter.reducedFitStatistic
    param_errs = imfit_fitter.parameterErrors
    params = imfit_fitter.getFitResult().params

    # Initialize an empty dictionary
    result_dict = {}

    # Populate the dictionary
    for name, param, err in zip(param_names, params, param_errs):
        result_dict[name] = (param, err)

    # Add the chi^2 value
    result_dict['chi^2'] = chi2
    result_dict['AIC'] = imfit_fitter.getFitResult().aic
    result_dict['BIC'] = imfit_fitter.getFitResult().bic

    output_lines = []
    
    if printit or savepath is not None:
        # Header
        header = f"{'Parameter':<20} {'Value':<20} {'Error':<20}"
        output_lines.append(header)

        # Divider
        divider = "-" * 60
        output_lines.append(divider)

        # Prepare parameters and their values and errors
        for key, value in result_dict.items():
            if key in ['chi^2', 'AIC', 'BIC']:
                output_lines.append(f"{key:<20} {round(value, 2):<20}")
            else:
                val, err = value
                output_lines.append(f"{key:<20} {round(val, 2):<20} {round(err, 2):<20}")

    # Print to console
    if printit:
        print("\n".join(output_lines))

    # Save to file
    if savepath:
        folder_path = os.path.dirname(savepath)
        if folder_path and not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open(savepath, 'w') as f:
            f.write("\n".join(output_lines))
    
    return result_dict


def get_mean(res, mask=None):

    res = np.abs(res)
    if mask is not None:
        res_xr = xr.DataArray(res)
        res = res_xr.where(mask==0)
    mean = np.nanmean(res)

    return mean

def get_model_data(model_desc, shape=(256, 256), psf=None):
    if psf is None:
        fitter = pyimfit.Imfit(model_desc)
    else:
        fitter = pyimfit.Imfit(model_desc, psf=psf)
    new_params = model_desc.getRawParameters()

    model_data = fitter.getModelImage(shape=shape, newParameters=new_params)

    return model_data
    

def ADU_to_mag(data, sky=0, gain=0, zcal=0, arcsecpix=0.396):

    arcsec2pix = arcsecpix ** 2
    shape = data.shape
    sky_array = np.full(shape, sky)
    data = (data + sky_array) * gain
    data = data/arcsec2pix

    zcal_prime = np.abs(zcal) + 2.5 * np.log10(arcsec2pix)

    data_mag = zcal_prime - 2.5 * np.log10(data)
    return data_mag


# make a list flatten function
def flatten(l):
    return [item for sublist in l for item in sublist]
