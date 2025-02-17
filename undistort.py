from defisheye import Defisheye

dtype = "linear" # linear, equalarea, orthographic, stereographic
format = "circular" # circular, fullframe
fov = 130
pfov = 100

img = "Calib1.jpg"
obj = Defisheye(img, dtype=dtype, format=format, fov=fov, pfov=pfov)
obj.convert(outfile="out.jpg")
