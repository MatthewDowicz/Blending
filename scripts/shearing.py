import numpy as np
import matplotlib.pyplot as plt
import galsim
from scripts.analytical_FT import gauss_2d


def sheared_gauss(x0=0,
                  y0=0,
                  sigma_x=1.0,
                  sigma_y=1.0,
                  rho=0.0,
                  g1=0.1,
                  g2=0.1,
                  kappa=2,
                  A=1,
                  N=128):
    '''
    General function to calculate 2D gaussian. Allows for circular, elliptical,
    rotated, and off-centered gaussians. The values of the Gaussian function are
    calculated by the given positions in the x and y meshgrid.
    
    Args:
    -----
        x: np.ndarray
            Matrix of the x-coordinates that make up the x-coordinates of each point in 
            the meshgrid. 
            NOTE: In a meshgrid a single point has 2 coordinates ie. (x,y)
        y: np.ndarray
            Matrix of the y-coordinates that make up the y-coordinates of each point in 
            the meshgrid.
            NOTE: In a meshgrid a single point has 2 coordinates ie. (x,y)
        x0: int 
            The x-coordinate of the center of the Gaussian.
            Defaults to 0.
        y0: int
            The y-coordinate of the center of the Gaussian.
        rho: float
            The correlation coefficient. Measure of direction and magnitude of stretch/
            rotation. Values must be between [-1,1].
            Defaults to 0.5.
        g1: float
            The first reduced shear component that describes the elongation along 
            the coordinate axes.
            Defaults is 0.1. Values between [-1,1]
        g2: float
            The second reduced shear component that describes the elongation at 45°
            from the coordinate axes.
            Defaults is 0.1. Values between [-1,1]
        kappa: float
            Convergence of the shear. Describes the change in size and brightness
            of the sheared galaxy.
            Defaults to 2. Should be positive?
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

   # Create Gaussian
    non_sheared_gauss = gauss_2d(x0=x0,
                    y0=y0,
                    sigma_x=sigma_x,
                    sigma_y=sigma_y,
                    rho=rho,
                    A=A,
                    N=N)
    
    prefactor = -((1-kappa)**2/2)
    exp_x = (Xg**2 * (1-g1)**2 + Yg**2*g2**2 - 2*Xg*Yg*g2*(1-g1)) / (sigma_x**2 * (1-rho**2))
    exp_y = (Xg**2*g2**2 + Yg**2*(1+g1)**2 - 2*Xg*Yg*g2*(1+g1)) / (sigma_y**2*(1-rho**2))
    exp_xy = (2*rho*g2*(1-g1)*Xg**2 + 2*rho*g2*(1+g1)*Yg**2 
              - 2*Xg*Yg*rho*g2**2 - 2*Xg*Yg*rho*(1+g1)*(1-g1)) / (sigma_x*sigma_y*(1-rho**2))
    exp = prefactor * (exp_x + exp_y + exp_xy)
    
    sheared_gauss = A * np.exp(exp)
    return non_sheared_gauss, sheared_gauss

def visualize_shear(sigma_x=1.0,
                    sigma_y=1.0,
                    rho=0.0,
                    g1=0.0, # Stretching along coord axis
                    g2=0.0, # Stretching at 45 deg from coord axis
                    kappa=0.0):
    """
    Function to visualize shearing on a round/elliptical 
    gaussian galaxy profile.
    
    Args:
    -----
    
        sigma_x: float
            The width of the gaussian in the x-direction.
            Defaults to 1.0
        sigma_y: float
            The width of the gaussian in the y-direction.
            Defaults to 1.0
        rho: float
            The correlation coefficient. Measure of direction and magnitude of stretch/
            rotation. Values must be between [-1,1].
            Defaults to 0.0.
        g1: float
            The first reduced shear component that describes the elongation along 
            the coordinate axes.
            Defaults is 0.0. Values between [-1,1]
        g2: float
            The second reduced shear component that describes the elongation at 45°
            from the coordinate axes.
            Defaults is 0.0. Values between [-1,1]
        kappa: float
            Convergence of the shear. Describes the change in size and brightness
            of the sheared galaxy.
            Defaults to 0.0. Should be positive?
        
    
    Returns:
    --------
        Plot showing the sheared galaxy, the input galaxy, and the residual
        image between the sheared - input.
    """
    
    gauss, sheared = sheared_gauss(sigma_x=sigma_x,
                      sigma_y=sigma_x,
                      rho=rho,
                      g1=g1, # Stretching along coord axis
                      g2=g2, # Stretching at 45 deg from coord axis
                      kappa=kappa)
    
    # Get residual image to see the difference when changing params
    residual = sheared - gauss
    vmax_resid = np.percentile((residual), (99))
    vmin_resid = -vmax_resid

    # Sheared vmax/vmin
    vmax_shear = np.percentile(sheared, (99))
    vmin_shear = 0

    # Non-sheared vmax/vmin
    vmax = np.percentile(gauss, (99))
    vmin = 0

    fig, axes = plt.subplots(1, 3, figsize=(30,26))
    axes[0].imshow(sheared, vmax=vmax_shear, vmin=vmin_shear, origin='lower')
    axes[0].axis('off')
    axes[0].set_title('Sheared Gauss', fontsize=30)
    axes[1].imshow(gauss, vmin=vmin, vmax=vmax, origin='lower')
    axes[1].axis('off')
    axes[1].set_title('Non-sheared Gauss', fontsize=30)
    im = axes[2].imshow(residual, origin='lower', vmax=vmax_resid, vmin=vmin_resid, cmap='bwr')
    axes[2].axis('off')
    axes[2].set_title('Residual (Shear - No Shear)', fontsize=30)

    cax = plt.axes([0.93, 0.37, 0.01, 0.25])
    fig.colorbar(im, cax=cax, ax=axes[2])
    plt.show()