import numpy as np

import btk
import btk.plot_utils
import btk.survey
import btk.draw_blends
import btk.catalog
import btk.sampling_functions
import astropy.table
import galsim

def BTK_blend_generator(catalog_name=None,
                        stamp_size=24.0,
                        max_number=2,
                        max_shift=0.0,
                        batch_size=10,
                        sampling_func=None,
                        survey="LSST",
                        add_noise="all"):
    
    """
    Helper function that puts the necessary pieces together to create
    a 'DrawBlendsGenerator' as shown in the diagram above. The "necessary
    pieces" are all the red boxes that flow into the first blue box.
    
    Args:
    -----
        catalog_name: str
            Path to the data file that you want to use to create the blends.
            Defaults to the OneDegSq.fits file.
        stamp_size: float
            Size of the stamp to be created, in arcseconds.
            Defaults to 24.0 arcseconds.
        max_number: int
            How many galaxies in the blend.
            Defaults to 2.
        max_shift: float
            Max shift of the galaxies from center, in arcseconds.
            Defaults to 0.0
        batch_size: int
            Number of samples you want to create from function call.
            Defaults to 10.
        sampling_func: btk.sampling_functions object
            Sampling function thatis used to produce blend tables.
            Defaults to btk.sampling_function.DefaultSampling()
        survey: str
            Name of the survey that you want to use. Returns specified
            surveys from galcheat extended to contain PSF information.
            Defaults to LSST.
        add_noise: str
            Add Poisson noise to the simulated survey image if requested.
            Defaults to "all".
        
    Returns:
    --------
        blend_images: np.ndarray
            Array containing the blended galaxy stamps.
            Array has shape: (N_samples, N_channels, H, W)
        isolated_images: np.ndarray
            Array containing the noise free individual galaxies
            that make up the blend image.
            Array has shape: (N_samples, max_number, N_channels, H, W)
    """
    
    
    # Catalog:
    #---------#
    if catalog_name == None:
        # Change this to be user agnostic
        catalog_name = "/Users/matt/Desktop/UCI_Research/LSST/btk_blending/DS_creation/Data/OneDegSq.fits"
        catalog = btk.catalog.CatsimCatalog.from_file(catalog_name)
    else:
        catalog_name = str(catalog_name)
        catalog = btk.catalog.CatsimCatalog.from_file(catalog_name)

    # Setting parameter values:
    #-------------------#
    stamp_size = stamp_size # Size of the stamp, in arcseconds
    max_number = max_number # Max number of galaxies in a blend
    max_shift = max_shift # Max shift of the galaxies from center, in arcseconds

    # Sampling Function:
    #-------------------#
    if sampling_func == None:
        sampling_func = btk.sampling_functions.DefaultSampling(max_number=max_number,
                                                              stamp_size=stamp_size,
                                                              maxshift=max_shift)
    else:
        sampling_func = sampling_func

    # Survey
    #-------#
    LSST = btk.survey.get_surveys(str(survey))

    # Draw Blends:
    #-------------#
    draw_generator = btk.draw_blends.CatsimGenerator(
        catalog=catalog,
        sampling_function=sampling_func,
        surveys=LSST,
        batch_size=batch_size,
        stamp_size=stamp_size,
        cpus=1,
        add_noise=str(add_noise), 
        seed=1)

    # Sample the blends from the blends generator:
    #----------------------------------------------#
    batch = next(draw_generator)
    blend_images = batch['blend_images']
    blend_list = batch['blend_list']
    isolated_images = batch['isolated_images']
    psf = batch['psf']
    
    return blend_images, isolated_images, psf


def create_PSF_image(psf_list, blend_imgs):
    """
    Function that retrieves and creates 6 PSF images for each
    LSST filter. 
    
    Args:
    -----
    psf_obj: list
        List of galsim.convolve.Convolution objects. It stores 
        pertinent information regarding the PSF for each LSST
        filter.
    blend_imgs: np.ndarray
        Array containing the blended galaxy image postage stamps.
        Has shape: (N_samples, N_channels, H, W).
        This array is used to create the PSF images with the same
        H and W.
        
    Returns:
    --------
    psf_arr: list
        A list of the 2D PSF np.ndarrays for each filter.
    """
    psf_arr = []
    for i in range(len(psf_list)):
        image_epsf = galsim.ImageF(blend_imgs.shape[-1], blend_imgs.shape[-1])
        psf_obj = psf_list[i].drawImage(image_epsf, scale=0.2)
        psf_img = getattr(psf_obj, '_array')
        psf_arr.append(psf_img)
        
    return psf_arr