import cv2
import numpy as np
import glob
import os

# modification done
images = glob.glob(r"C:\Users\Hassiba Info\OneDrive\Bureau\Fisheye\*.jpg") 

# Charger les paramètres de calibration
calib_data = np.load("calibration_data.npz")
camera_matrix = calib_data["camera_matrix"]
dist_coeffs = calib_data["dist_coeffs"]
idx=1
for idx,i in enumerate(images,start=1):
    img = cv2.imread(i)
    # Corriger la distorsion
    undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs)
    # Afficher et enregistrer le résultat
    cv2.imshow("Undistorted Image", undistorted_img)
    cv2.imwrite(f"undistorted_{idx}.jpg", undistorted_img)
    cv2.waitKey(0)
    
cv2.destroyAllWindows()