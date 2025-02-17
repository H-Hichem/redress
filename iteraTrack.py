import cv2
import numpy as np
import glob

# Define checkerboard pattern size
CHECKERBOARD = (6, 8)  
SQUARE_SIZE = 25  # Size of a square in mm

# Termination criteria
MAX_ITER = 30
EPSILON = 0.001

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, MAX_ITER, EPSILON)

# Prepare object points
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

# Arrays to store object points and image points
objpoints = []  
imgpoints = []  

# Load all calibration images
images = glob.glob(r"C:\Users\Hassiba Info\OneDrive\Bureau\Fisheye\*.jpg")  

if not images:
    print("No images found! Check your file path.")
    exit()

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)

    # Find the checkerboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    
    if ret:
        objpoints.append(objp)  # Store 3D points

        # Ensure corners are float32 (important for OpenCV functions)
        corners = np.array(corners, dtype=np.float32)

        # Initialize variables for manual tracking
        prev_corners = corners.copy()
        num_iters = 0

        while num_iters < MAX_ITER:
            refined_corners = cv2.cornerSubPix(gray, prev_corners, (8, 8), (-1, -1), criteria)
            movement = np.linalg.norm(refined_corners - prev_corners)

            print(f"Iteration {num_iters + 1}: movement = {movement}")  # Debug movement

            prev_corners = refined_corners.copy()
            num_iters += 1

            if movement < EPSILON:
                break  # Stop if convergence is reached

        imgpoints.append(refined_corners)  # Store 2D points

        print(f"Image {fname}: Converged in {num_iters} iterations")

        # Draw and show the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, refined_corners, ret)
        imgr = cv2.resize(img, (1000, 800), interpolation=cv2.INTER_AREA)
        cv2.imshow('Checkerboard Detection', imgr)
        cv2.waitKey()  

cv2.destroyAllWindows()
