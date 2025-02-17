import cv2
import numpy as np
import glob

# Define checkerboard pattern size (number of inner corners, NOT squares)
CHECKERBOARD = (6, 8)  
SQUARE_SIZE = 25  # Size of a square in mm (change based on your pattern)

# Termination criteria for corner refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (3D points in real-world space)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE  # Scale according to real square size

# Arrays to store object points and image points
objpoints = []  # 3D points (real world)
imgpoints = []  # 2D points (image)

# Load all calibration images
images = glob.glob(r"C:\Users\Hassiba Info\OneDrive\Bureau\Fisheye\*.jpg")  

if not images:
    print("No images found! Check your file path.")
    exit()

for fname in images:
    img = cv2.imread(fname)
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    gray = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)  # Read directly as grayscale

    # Find the checkerboard corners
    #############
    # params can be : cv2.CALIB_CB_FAST_CHECK, cv2.CALIB_CB_ADAPTIVE_THRESH, 
    # cv2.CALIB_CB_NORMALIZE_IMAGE, cv2.CALIB_CB_FILTER_QUADS
    # None means no special detection param 
    #############
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    

    if ret:
        objpoints.append(objp)  # Store 3D points
        refined_corners = cv2.cornerSubPix(gray, corners, (8, 8), (-1, -1), criteria)
        
        imgpoints.append(refined_corners)  # Store 2D points

        # Draw and show the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, refined_corners, ret)
        imgr = cv2.resize(img, (1000,800), interpolation=cv2.INTER_AREA)
        cv2.imshow('Checkerboard Detection', imgr)
        cv2.waitKey()  # Display each for 500ms

cv2.destroyAllWindows()

# Perform camera calibration
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Compute the Field of View (FoV)
fov_x, fov_y, focal_length, principal_point, aspect_ratio = cv2.calibrationMatrixValues(
    camera_matrix, gray.shape[::-1], SQUARE_SIZE, SQUARE_SIZE
)

# Compute mean reprojection error
total_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    total_error += error

mean_error = total_error / len(objpoints)

# Print results
print("\nCamera Matrix:\n", camera_matrix)
print("\nDistortion Coefficients:\n", dist_coeffs.ravel())
print("\nField of View (X, Y):", fov_x, "°", fov_y, "°")
print("\nMean Reprojection Error:", mean_error)

# Save calibration data
np.savez("calibration_data.npz", camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)


# Open the txt file to save all parameters 
with open("calibration_data.txt", "w") as f:
    f.write("Camera Matrix:\n")
    np.savetxt(f, camera_matrix, fmt="%.6f")
    f.write("\nDistortion Coefficients:\n")
    np.savetxt(f, dist_coeffs, fmt="%.6f")
    
    # Save rotation vectors for each image (rvecs is a list of rotation vectors)
    f.write("\nRotation Vectors:\n")
    for rvec in rvecs:
        np.savetxt(f, rvec, fmt="%.6f")
    
    # Save translation vectors for each image (tvecs is a list of translation vectors)
    f.write("\nTranslation Vectors:\n")
    for tvec in tvecs:
        np.savetxt(f, tvec, fmt="%.6f")

    # Optionally, save field of view (FoV) or any other calibration parameters you want to include
    f.write("\nField of View (FoV) - X, Y:\n")
    f.write(f"fov_x: {fov_x:.6f}, fov_y: {fov_y:.6f}\n")
    
    f.write("\nPrincipal Point and Aspect Ratio:\n")
    f.write(f"Principal Point: {principal_point}\n")
    f.write(f"Aspect Ratio: {aspect_ratio:.6f}\n")



print("\nCalibration successful! Data saved to 'calibration_data.npz'\n")
