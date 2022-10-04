import numpy as np
import pyfftw
import pyfftw.interfaces.numpy_fft as fft
import matplotlib.pyplot as plt

def gauss_2d(x0=0,
                y0=0,
                sigma_x=1.0,
                sigma_y=1.0,
                rho=0.5,
                A=1,
                N=128):
    '''
    General function to calculate 2D gaussian. Allows for circular, elliptical,
    rotated, and off-centered gaussians. The values of the Gaussian function are
    calculated by the given positions in the x and y meshgrid.
    
    Args:
    -----
        x0: int 
            The x-coordinate of the center of the Gaussian.
            Defaults to 0.
        y0: int
            The y-coordinate of the center of the Gaussian.
        rho: float
            The correlation coefficient. Measure of direction and magnitude of stretch/
            rotation. Values must be between [-1,1].
            Defaults to 0.5.
        sigma_x: float
            The width of the gaussian in the x-direction.
            Defaults to 1.0
        sigma_y: float
            The width of the gaussian in the y-direction.
            Defaults to 1.0
        A: int
            The amplitude or "height" of the Gaussian.
        N: int
            Number of samples to create.
            E.g. if N=128 we'll create a meshgrid that is (128,128) and thus return
            an (128,128) Gaussian image.
            Defaults to 128.
            
            
    Returns:
    --------
        gauss: np.ndarray
            The array containing the gaussian function evaluated at each point
            in the meshgrid. I.e. returns the image of the Gaussian.
    '''
    x = np.linspace(-10,10,N)
    y = np.linspace(-10,10,N)
    Xg, Yg = np.meshgrid(x, y)

    norm = 1 / (2*np.pi * sigma_x * sigma_y * np.sqrt(1-rho**2))
    exp_x = -(Xg-x0)**2 /(2*sigma_x**2 * (1-rho**2))
    exp_y = -(Yg-y0)**2 / (2*sigma_y**2 * (1-rho**2))
    exp_xy = (Xg-x0)*(Yg-y0)*rho / (sigma_x*sigma_y * (1-rho**2))
    
    rot_gauss = norm * np.exp(exp_x + exp_y + exp_xy)

    return rot_gauss


# For non-rotated gaussian analysis

def analytical_nonrot_gauss(sigma_x=1.0,
                            sigma_y=1.0,
                            A=1,
                            N=128,
                            d=0.1):
    """
    Function that calculates the FT of a non-rotated 2D Gaussian using the
    analytical solution I calculated.
    
    Args:
    -----
        sigma_x: float
            The width of the gaussian in the x-direction.
            Defaults to 1.0
        sigma_y: float
            The width of the gaussian in the y-direction.
            Defaults to 1.0
        A: int
            The amplitude or "height" of the Gaussian.
        N: int
            Number of samples to create.
            E.g. if N=128 we'll create a meshgrid that is (128,128) and thus return
            an (128,128) Gaussian image.
            Defaults to 128.
        d: float
            Sample spacing (inverse of the sampling rate).
            Defaults to 0.1
            
    Returns:
    --------
        gauss: np.ndarray
            Array of the original inputted 2D Gaussian
        FT: np.ndarray
            Complex array of the FT of the 2D Gaussian that was 
            computed via the analytical equation above.
    """
    # Get the gaussian and FT of the gaussian
    gauss = gauss_2d(sigma_x=sigma_x, sigma_y=sigma_y, rho=0, A=A, N=N)

     # Get the frequency coords of the FT ie. x/y -> u/v
    FreqCompRows = np.fft.fftfreq(gauss.shape[0],d=d) #u
    u = FreqCompRows
    FreqCompCols = np.fft.fftfreq(gauss.shape[1],d=d) #b
    v = FreqCompCols
    
    # Create frequency grid
    Ug, Vg = np.meshgrid(u, v)
    
    # Calc prefactor of the analytical FT
    prefactor= 1 # old eqn had 1 here
    expo = np.exp(-2*np.pi**2 * (Ug**2*sigma_x**2 + Vg**2*sigma_y**2))
    FT = prefactor*expo
    FT = fft.fftshift(FT)
    return gauss, FT

def visualize_nonrot_FT(sigma_x=1.0,
                        sigma_y=1.0,
                        A=1,
                        N=128,
                        d=0.1):
    """
    Function that visualizes a comparison between the analytical FT
    and the FFT output.
    
    Args:
    -----
        sigma_x: float
            The width of the gaussian in the x-direction.
            Defaults to 1.0
        sigma_y: float
            The width of the gaussian in the y-direction.
            Defaults to 1.0
        A: int
            The amplitude or "height" of the Gaussian.
        N: int
            Number of samples to create.
            E.g. if N=128 we'll create a meshgrid that is (128,128) and thus return
            an (128,128) Gaussian image.
            Defaults to 128.
        d: float
            Sample spacing (inverse of the sampling rate).
            Defaults to 0.1
            
    Returns:
    --------
        Plots of the analytical FT versus the FFT output
    """
    
    
    ingauss, analyFT = analytical_nonrot_gauss(sigma_x=sigma_x,
                                               sigma_y=sigma_y,
                                               N=N,
                                               d=d, 
                                               A=A)
    FFT = fft.fftshift(fft.fft2(ingauss))
    
#     vmax, vmin = np.percentile(np.abs(FFT), (99,1))
#     vmax, vmin = np.percentile(ingauss, (99,1))

    fig, axes = plt.subplots(2, 2, figsize=(30,26))
    axes[0,0].imshow(ingauss)
    axes[0,0].axis('off')
    axes[0,0].set_title('Input', fontsize=30)
    axes[0,1].imshow(np.abs(analyFT))
    axes[0,1].axis('off')
    axes[0,1].set_title('Analytical FT', fontsize=30)
    axes[1,0].imshow(ingauss)
    axes[1,0].axis('off')
    axes[1,0].set_title('Input', fontsize=30)
    axes[1,1].imshow(np.abs(FFT))
    axes[1,1].axis('off')
    axes[1,1].set_title('FFT FT', fontsize=30)


    plt.tight_layout()
    plt.show()


# For rotated gaussian analysis

def analytical_rotgauss_FT(sigma_x=1.0,
                            sigma_y=1.0,
                            rho=0.5,
                            A=1,
                            N=128,
                            d=0.1):

    """
    Function to calculate the FT of a rotated 2D gaussian via
    the analytical equation.
    
    Args:
    -----
        sigma_x: float
            The width of the gaussian in the x-direction.
            Defaults to 1.0
        sigma_y: float
            The width of the gaussian in the y-direction.
            Defaults to 1.0
        A: int
            The amplitude or "height" of the Gaussian.
        N: int
            Number of samples to create.
            E.g. if N=128 we'll create a meshgrid that is (128,128) and thus return
            an (128,128) Gaussian image.
            Defaults to 128.
        d: float
            Sample spacing (inverse of the sampling rate).
            Defaults to 0.1
            
    Returns:
    --------
        gauss: np.ndarray
            2D array containing the original inputted 2D Gaussian
        rot_FT: np.ndarray
            2D array containing the FT of the analytical equation
            above of the inputted 2D Gaussian.
    """
    
    # Get the gaussian 
    gauss = gauss_2d(sigma_x=sigma_x, sigma_y=sigma_y, rho=rho, N=N)

    # Get the frequency coords of the FT ie. x/y -> u/v
    FreqCompRows = np.fft.fftfreq(gauss.shape[0],d=d) #u
    u = FreqCompRows
    FreqCompCols = np.fft.fftfreq(gauss.shape[1],d=d) #v
    v = FreqCompCols
    
    # Create frequency grid
    Ug, Vg = np.meshgrid(u, v)
    
    exp_u = -2*np.pi**2 * sigma_x**2 * Ug**2
    exp_v = -2*np.pi**2 * sigma_y**2 * Vg**2
    exp_uv = -4*np.pi**2 * sigma_x * sigma_y * rho * Ug * Vg
    
    rot_FT = np.exp(exp_u + exp_v + exp_uv)
    rot_FT = fft.fftshift(rot_FT)
    
    return gauss, rot_FT

def visualize_rot_FT(sigma_x=1.0,
                     sigma_y=1.0,
                     rho=0.5,
                     A=1,
                     N=128,
                     d=0.1):
    """
    Function that visualizes a comparison between the analytical FT
    and the FFT output.
    
    Args:
    -----
        sigma_x: float
            The width of the gaussian in the x-direction.
            Defaults to 1.0
        sigma_y: float
            The width of the gaussian in the y-direction.
            Defaults to 1.0
        A: int
            The amplitude or "height" of the Gaussian.
        N: int
            Number of samples to create.
            E.g. if N=128 we'll create a meshgrid that is (128,128) and thus return
            an (128,128) Gaussian image.
            Defaults to 128.
        d: float
            Sample spacing (inverse of the sampling rate).
            Defaults to 0.1
            
    Returns:
    --------
        Plots of the analytical FT versus the FFT output
    """
    
    
    ingauss, analyFT = analytical_rotgauss_FT(sigma_x=sigma_x,
                                               sigma_y=sigma_y,
                                               rho=rho,
                                               N=N,
                                               d=d, 
                                               A=A)
    FFT = fft.fftshift(fft.fft2(ingauss))

    fig, axes = plt.subplots(2, 2, figsize=(30,26))
    axes[0,0].imshow(ingauss)
    axes[0,0].axis('off')
    axes[0,0].set_title('Input', fontsize=30)
    axes[0,1].imshow(np.abs(analyFT))
    axes[0,1].axis('off')
    axes[0,1].set_title('Analytical FT', fontsize=30)
    axes[1,0].imshow(ingauss)
    axes[1,0].axis('off')
    axes[1,0].set_title('Input', fontsize=30)
    axes[1,1].imshow(np.abs(FFT))
    axes[1,1].axis('off')
    axes[1,1].set_title('FFT FT', fontsize=30)


    plt.tight_layout()
    plt.show()
