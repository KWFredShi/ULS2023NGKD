# Import necessary libraries
import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter
# import cv2  # Optional: Uncomment if you prefer using OpenCV for blurring

def apply_gaussian_blur(input_file, output_file, sigma=2):
    """
    Applies Gaussian blur to a NIfTI file.

    Parameters:
    - input_file (str): Path to the input NIfTI file.
    - output_file (str): Path to the output NIfTI file where the blurred image will be saved.
    - sigma (int or float): Standard deviation for Gaussian kernel. Higher values result in more blurring.

    Returns:
    - None
    """
    # Load the NIfTI file
    img = nib.load(input_file)
    img_data = img.get_fdata()
    
    # Apply Gaussian blur using scipy
    blurred_data = gaussian_filter(img_data, sigma=sigma)
    
    # Optional: Apply Gaussian blur using OpenCV (uncomment the following line)
    # Note: OpenCV might require conversion to 2D for each slice if working with 3D data.
    # blurred_data = cv2.GaussianBlur(img_data, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_DEFAULT)
    
    # Create a new NIfTI image with the blurred data
    new_img = nib.Nifti1Image(blurred_data, img.affine, img.header)
    
    # Save the new blurred image to a file
    nib.save(new_img, output_file)

# The following lines are for direct execution testing purposes, you can remove or comment them if using as a module
if __name__ == "__main__":
    input_nifti_file = 'path_to_your_input_file.nii'
    output_nifti_file = 'path_to_your_output_file.nii'
    apply_gaussian_blur(input_nifti_file, output_nifti_file, sigma=2)

