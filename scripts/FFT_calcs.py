import pyfftw.interfaces.numpy_fft as fft
import numpy as np


def FFT_IFFT_calc(img, FFT=True):
    """
    Calculates the FFT or IFFT of an image. When computing the FT
    we return the power (ie. the complex valued FT), the magnitude 
    (ie. the L2-norm of the complex valued FT array), and we get the
    phase (ie. the angular offset the FT has).
    
    Args:
    -----
        img: np.ndarray
            2D image of galaxy's or PSFs
        FT: boolean
            Boolean to toggle between computing the FT or the IFT.
            The default is FT=True.
    Returns:
    --------
   
    """
    if FFT is True:        
        f = fft.fft2(img)
        fshift = fft.fftshift(f)
        magnitude_spectrum = np.log(1 + np.abs(fshift))
        return f, fshift, magnitude_spectrum
    
    else:
        f_ishift = fft.ifftshift(img)
        ift_img = fft.ifft2(f_ishift)
        ift_img = np.real(ift_img)
        return ift_img

def create_deconv_img(img, psf, 
                      sample_idx=0, gal_num_idx=0, filter_idx=0,
                      eps=0.0002):
    """
    Function that deconvolved galaxy image with its PSF image.
    The deconvolution calculation uses the convolution theorem, 
    which states that a deconvolution in Fourier Space is just
    the division of the FT of the image and the FT of the PSF ie:

    Deconvolved image = IFT(FT(img) / FT(psf_img))

    Args:
    -----
        img: np.ndarray
            An array of either a blended galaxy scene or an 
            isolated galaxy scene.
            NOTE: The number of galaxys in the blended scene
                  determines the number of isolated galaxy 
                  scenes. 
                  E.g. 3 galaxy blend = 3 individual isolated
                                        galaxy images
        psf: np.ndarray
            An array of the PSF image in a specific LSST band.
            Shape: (H, W)
        sample_idx: int
            The index of the sample that we want to deconvolve.
            Defaults to 0.
        gal_num_idx: int
            The index of which galaxy isolated galaxy image from
            the blended scene to use for deconvolution.
        filter_idx: int
            The specific LSST filter galaxy image to use. 
            There are 6 LSST filters. They correspond to:
                - u: 0
                - g: 1
                - r: 2
                - i: 3
                - z: 4
                - y: 5
        eps: float
            The constant value that replaces zero (or near-zero
            values) in the Fourier Transformed PSF image.

    Returns:
    --------
        deconv_img = np.ndarray
            An array containing the deconvolved galaxy image.
            Shape: (H, W)
    """
    
    # Isolated Images
    if img.ndim == 5:
    
        # Test to see how this does when we replace zero/near-zero FT PSF values to a constant value
        gal = img[sample_idx][gal_num_idx][filter_idx]
        psf_img = psf[filter_idx]

        GAL = fft.fft2(gal)
        PSF = fft.fft2(psf_img)
        PSF[np.abs(PSF) <= 0.0002]  = eps

        # Deconvolution/ inverse filtering
        deconv_hat = np.divide(GAL, PSF)

        deconv_img = fft.ifft2(deconv_hat).real
        deconv_img = fft.ifftshift(deconv_img)
    
        return deconv_img, gal, psf_img
    
    # Blended Images
    else:
        # Test to see how this does when we replace zero/near-zero FT PSF values to a constant value
        gal = img[sample_idx][filter_idx]
        psf_img = psf[filter_idx]

        GAL = fft.fft2(gal)
        PSF = fft.fft2(psf_img)
        PSF[np.abs(PSF) <= 0.0002]  = eps

        # Deconvolution/ inverse filtering
        deconv_hat = np.divide(GAL, PSF)

        deconv_img = fft.ifft2(deconv_hat).real
        deconv_img = fft.ifftshift(deconv_img)
    
        return deconv_img, gal, psf_img