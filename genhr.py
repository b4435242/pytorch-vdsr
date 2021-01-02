# Load the package you are going to use
import torch
from torch.autograd import Variable
from PIL import Image
import numpy as np
import time, math
import matplotlib.pyplot as plt
import os
import argparse
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
	parser = argparse.ArgumentParser(description="PyTorch EDSR Test")
	parser.add_argument("--cuda", action="store_true", help="use cuda?")
	parser.add_argument("--model", default="/content/drive/MyDrive/Colab Notebooks/HW4/model/model_epoch_20.pth", type=str, help="model path")
	parser.add_argument("--dataset", default="HW4/testing_lr_images", type=str, help="image dataset")
	parser.add_argument("--scale_factor", default="3", type=int, help="scale factor")


	opt = parser.parse_args()
	cuda = opt.cuda
	scale_factor = opt.scale_factor

	if cuda and not torch.cuda.is_available():
		raise Exception("No GPU found, please run without --cuda")

	# Load the pretrained model
	model = torch.load(opt.model)["model"]

	imgnames = os.listdir(opt.dataset)

	for imgname in imgnames:
		# Load the low-resolution image 
		imgpath = os.path.join(opt.dataset, imgname)
		# load data
		img = Image.open(imgpath)
		img = img.convert('YCbCr')
		y, cb, cr = img.split()
		y = y.resize((y.size[0] * scale_factor, y.size[1] * scale_factor), Image.BICUBIC)

		input = Variable(ToTensor()(y)).view(1, -1, y.size[1], y.size[0])
		input = input.cuda()

		recon_img = model(input)

		# save result images
		#utils.save_img(recon_img.cpu().data, 1, save_dir=self.save_dir)

		out = recon_img.cpu()
		out_img_y = out.data[0]
		out_img_y = (((out_img_y - out_img_y.min()) * 255) / (out_img_y.max() - out_img_y.min())).numpy()
		# out_img_y *= 255.0
		# out_img_y = out_img_y.clip(0, 255)
		out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

		out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
		out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
		out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')

		'''im_b = Image.open(imgpath).convert("RGB")
		# Convert the images into YCbCr mode and extraction the Y channel (for PSNR calculation)
		im_b_ycbcr = np.array(im_b.convert("YCbCr"))
		im_b_y = im_b_ycbcr[:,:,0].astype(float)
		print(im_b_y.shape,im_b_y.shape[0], im_b_y.shape[1], scale_factor)
		im_b_y = im_b_y.resize((im_b_y.shape[0] * scale_factor, im_b_y.shape[1] * scale_factor), Image.BICUBIC)


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
		im_b_ycbcr[:,:,1] = im_b_ycbcr[:,:,1].resize((im_b_y.shape[0] * scale_factor, im_b_y.shape[1] * scale_factor), Image.BICUBIC)
		im_b_ycbcr[:,:,2] = im_b_ycbcr[:,:,2].resize((im_b_y.shape[0] * scale_factor, im_b_y.shape[1] * scale_factor), Image.BICUBIC)
		im_h = colorize(im_h_y, im_b_ycbcr)
		#w, h = im_h.size
		#im_h = im_h.resize((3*w, 3*h))'''

		save_dir = '/content/drive/MyDrive/Colab Notebooks/HW4/vdsr_results'
		if not os.path.isdir(save_dir):
			os.makedirs(save_dir)
		out_img.save(os.path.join(save_dir, imgname))

if __name__ == "__main__":
	main()





