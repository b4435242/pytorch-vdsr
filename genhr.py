# Load the package you are going to use
import torch
from torch.autograd import Variable
from PIL import Image
import numpy as np
import time, math
import matplotlib.pyplot as plt
import os
#%matplotlib inline


# Define the colorization function
# We'll reuse the Cb and Cr channels from bicubic interpolation
def colorize(y, ycbcr): 
    img = np.zeros((y.shape[0], y.shape[1], 3), np.uint8)
    img[:,:,0] = y
    img[:,:,1] = ycbcr[:,:,1]
    img[:,:,2] = ycbcr[:,:,2]
    img = Image.fromarray(img, "YCbCr").convert("RGB")
    return img

def main():
	# Load the pretrained model
	model = torch.load("checkpoint/model_epoch_50.pth")["model"]

	test_dir = 'HW4/testing_lr_images'
	imgnames = os.listdir(test_dir)

	for imgname in imgnames:
		# Load the low-resolution image 
		imgpath = os.path.join(test_dir, imgname)
		im_b = Image.open(imgpath).convert("RGB")
		# Convert the images into YCbCr mode and extraction the Y channel (for PSNR calculation)
		im_b_ycbcr = np.array(im_b.convert("YCbCr"))
		im_b_y = im_b_ycbcr[:,:,0].astype(float)

		# Prepare for the input, a pytorch tensor
		im_input = im_b_y/255.
		im_input = Variable(torch.from_numpy(im_input).float()).\
		view(1, -1, im_input.shape[0], im_input.shape[1])

		# Now let's try the network feedforward in gpu mode
		model = model.cuda()
		im_input = im_input.cuda()

		# Let's see how long does it take for processing in gpu mode
		start_time = time.time()
		out = model(im_input)
		elapsed_time = time.time() - start_time
		print("It takes {}s for processing in gpu mode".format(elapsed_time))

		# Get the output image
		out = out.cpu()
		im_h_y = out.data[0].numpy().astype(np.float32)
		im_h_y = im_h_y * 255.
		im_h_y[im_h_y < 0] = 0
		im_h_y[im_h_y > 255.] = 255.
		im_h_y = im_h_y[0,:,:]

		# Colorize the grey-level image and convert into RGB mode
		im_h = colorize(im_h_y, im_b_ycbcr)

		save_dir = 'predicted_results'
		if os.path.isdir(save_dir):
			os.makedirs(save_dir)
		plt.imsave(os.path.join(save_dir, imgname), im_h)

if __name__ == "__main__":
	main()





