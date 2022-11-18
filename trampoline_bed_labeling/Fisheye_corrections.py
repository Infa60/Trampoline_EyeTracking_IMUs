"""
Credits : https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0
"""

import numpy as np
import cv2
from tqdm.notebook import tqdm
import pickle
# from .rectangle_labeler_video import load_video_frames

def load_video_frames(video_file, num_frames=None):
    video = cv2.VideoCapture(video_file)
    frames = []

    if num_frames is None:
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    num_frames_update = 0
    for i in tqdm(range(num_frames), desc='Loading video'):
        ret, frame = video.read()
        if type(frame) == np.ndarray:
            frames.append(frame)
            num_frames_update += 1

    video.release()

    return frames, num_frames_update

# You should replace these 3 lines with the output in calibration step
DIM = (1088, 1080)
K = np.array([[773.6719811071623, 0.0, 532.3446174857597], [0.0, 774.3187867828567, 565.9954169588382], [0.0, 0.0, 1.0]])
D = np.array([[0.007679273278292513], [-0.1766120836825416], [0.6417799798538761], [-0.7566634368371957]])

def undistort(img, balance=0.0, dim2=None, dim3=None):

	dim1 = img.shape[:2][::-1]
	#dim1 is the dimension of input image to un-distort
	assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
	if not dim2:
		dim2 = dim1
	if not dim3:
		dim3 = dim1

	scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
	scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
	# This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
	new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim1, np.eye(3), balance=balance)
	map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim1, cv2.CV_16SC2)
	undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

	return undistorted_img

	# if __name__ == '__main__':
	# 	for p in sys.argv[1:]:




def nothing(x):
	return
'e6e4181c_0_0-44_507.mp4
ratio_image = 1.5
movie_file_path = '../input/'
movie_file_name = '8ae28b5_0_0-25_798.mp4'
'
movie_file = movie_file_path + movie_file_name
frames, num_frames = load_video_frames(movie_file)
min_frames = 600
max_frames = 650
num_frames = max_frames - min_frames  # 30 ######################################################
frames_clone = frames.copy()

Image_name = "Video"
Trackbar_name = "Frames"
cv2.namedWindow(Image_name)
cv2.createTrackbar('Frames', Image_name, 0, num_frames, nothing)

playVideo = True
frame_counter = 0
image_clone = frames[0].copy()
width, height, rgb = np.shape(image_clone)

# small_images = [np.zeros((DIM[0], DIM[1], 3), dtype=np.int8) for i in range(num_frames)]
undistorted_images = [np.zeros((DIM[0], DIM[1], 3), dtype=np.int8) for i in range(num_frames)]
for i in range(min_frames, max_frames):
	# small_images[i] = cv2.resize(frames_clone[i], (int(round(width / ratio_image)), int(round(height / ratio_image))))
	undistorted_images[i - min_frames] = undistort(frames[i], balance=0.0)  # small_images[i]

with open(f'../output/{movie_file_name[:-4]}_undistorted_images.pkl', 'wb') as handle:
	pickle.dump(undistorted_images, handle)

while playVideo:

	key = cv2.waitKey(1) & 0xFF

	frame_counter = cv2.getTrackbarPos(Trackbar_name, Image_name)
	cv2.imshow(Image_name, undistorted_images[frame_counter])

	if key == ord(','):  # if `<` then go back
		frame_counter -= 1
		cv2.setTrackbarPos(Trackbar_name, Image_name, frame_counter)
		cv2.imshow(Image_name, undistorted_images[frame_counter])

	elif key == ord('.'):  # if `>` then advance
		print("frame_counter fleche +: ", frame_counter, ' -> ', frame_counter+1)
		frame_counter += 1
		cv2.setTrackbarPos(Trackbar_name, Image_name, frame_counter)
		cv2.imshow(Image_name, undistorted_images[frame_counter])

	elif key == ord('x'):  # if `x` then quit
		playVideo = False

cv2.destroyAllWindows()

