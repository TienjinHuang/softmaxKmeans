import os
import sys
import numpy as np

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from models import *
from utils import progress_bar
from torch.autograd import Variable

from differential_evolution import differential_evolution

parser = argparse.ArgumentParser(description='One pixel attack with PyTorch')
parser.add_argument('--net', default='VGG', help='The target model')
parser.add_argument('--filename', default='ckpt', help='The target model')
parser.add_argument('--pixels', default=1, type=int, help='The number of pixels that can be perturbed.')
parser.add_argument('--maxiter', default=100, type=int, help='The maximum number of iteration in the DE algorithm.')
parser.add_argument('--popsize', default=400, type=int, help='The number of adverisal examples in each iteration.')
parser.add_argument('--samples', default=100, type=int, help='The number of image samples to attack.')
parser.add_argument('--targeted', action='store_true', help='Set this switch to test for targeted attacks.')
parser.add_argument('--save', default='./results/results.pkl', help='Save location for the results with pickle.')
parser.add_argument('--verbose', action='store_true', help='Print out additional information every iteration.')
parser.add_argument('--km', default=False, type=int, help='kmeans linear output')
parser.add_argument('--gc', default=False, type=int, help='garbage cluster integration')
parser.add_argument('--minconf', default=0.1, type=float, help='minimum confidence')

args = parser.parse_args()

torch.set_printoptions(precision=2,sci_mode=False)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def perturb_image(xs, img):
	if xs.ndim < 2:
		xs = np.array([xs])
	batch = len(xs)
	imgs = img.repeat(batch, 1, 1, 1)
	xs = xs.astype(int)

	count = 0
	for x in xs:
		pixels = np.split(x, len(x)/5)
		
		for pixel in pixels:
			x_pos, y_pos, r, g, b = pixel
			imgs[count, 0, x_pos, y_pos] = (r/255.0-0.4914)/0.2023
			imgs[count, 1, x_pos, y_pos] = (g/255.0-0.4822)/0.1994
			imgs[count, 2, x_pos, y_pos] = (b/255.0-0.4465)/0.2010
		count += 1

	return imgs

def confidence(net, inputs):
	W=net.module.classifier.weight.data.t()
	D = net.module.get_D(inputs)
	DW = D.mm(W)
	return torch.exp( -torch.sum(D**2,1).unsqueeze(1).expand_as(DW)+2*DW-torch.sum(W**2,0).unsqueeze(0).expand_as(DW) )

def predict_classes(xs, img, target_calss, net, minimize=True):
	with torch.no_grad():
		imgs_perturbed = perturb_image(xs, img.clone())
		#input = Variable(imgs_perturbed, volatile=True).cuda()
		input = imgs_perturbed.to(device)
		#predictions = F.softmax(net(input),1).data.cpu().numpy()[:, target_calss]
		#print("original shape:",F.softmax(net(input),1).data.cpu().numpy().shape, "target: ", target_calss)
		#print("predictions correct, shape:",predictions.shape)
		#predictions = F.softmax(net(input)).data.cpu().numpy()[:, target_calss]
		#print("predictions incorrect, shape:",predictions.shape)
		#predictions = confidence(net,input).data.cpu().numpy()[:, target_calss]
		predictions = torch.exp(net(input)).data.cpu().numpy()[:,target_calss]

	return predictions if minimize else 1 - predictions

def attack_success(x, img, target_calss, net, targeted_attack=False, verbose=False):
	with torch.no_grad():
		attack_image = perturb_image(x, img.clone())
		#input = Variable(attack_image, volatile=True).cuda()
		input = attack_image.to(device)
		conf = torch.exp(net(input)).data.cpu().numpy()[0]
		#conf = confidence(net,input).data.cpu().numpy()[0]
		predicted_class = np.argmax(conf)

	if (targeted_attack and predicted_class == target_calss) or (not targeted_attack and predicted_class != target_calss):
		return True


def attack(img, label, net, target=None, pixels=1, maxiter=75, popsize=400, verbose=False):
	# img: 1*3*W*H tensor
	# label: a number

	targeted_attack = target is not None
	target_calss = target if targeted_attack else label

	bounds = [(0,32), (0,32), (0,255), (0,255), (0,255)] * pixels

	popmul = max(1, popsize/len(bounds))

	predict_fn = lambda xs: predict_classes(
		xs, img, target_calss, net, target is None)
	callback_fn = lambda x, convergence: attack_success(
		x, img, target_calss, net, targeted_attack, verbose)

	inits = np.zeros([int(popmul)*len(bounds), len(bounds)])
	for init in inits:
		for i in range(pixels):
			init[i*5+0] = np.random.random()*32
			init[i*5+1] = np.random.random()*32
			init[i*5+2] = np.random.normal(128,127)
			init[i*5+3] = np.random.normal(128,127)
			init[i*5+4] = np.random.normal(128,127)

	attack_result = differential_evolution(predict_fn, bounds, maxiter=maxiter, popsize=popmul,
		recombination=1, atol=-1, callback=callback_fn, polish=False, init=inits)
	with torch.no_grad():
		attack_image = perturb_image(attack_result.x, img)
		#attack_var = Variable(attack_image, volatile=True).cuda()
		input = attack_image.to(device)
		predicted_probs = torch.exp(net(input)).data.cpu().numpy()[0]
		#predicted_probs = confidence(net,input).data.cpu().numpy()[0]
		predicted_class = np.argmax(predicted_probs)

	if ( (not targeted_attack and predicted_class != label) or
		(targeted_attack and predicted_class == target_calss) ) and predicted_probs[predicted_class] > args.minconf:
		print("confidence attack:",predicted_probs[predicted_class])
		return predicted_probs[predicted_class],1, attack_result.x.astype(int)
	return 0,0, [None]


def attack_all(net, loader, pixels=1, targeted=False, maxiter=75, popsize=400, verbose=False):

	correct = 0
	success = 0
	conf_sum =0
	success_rate=100

	with torch.no_grad():
		for batch_idx, (inputs, targets) in enumerate(loader):

			inputs = inputs.to(device)
			#img_var = Variable(input, volatile=True).cuda()
			#prior_probs = confidence(net,inputs)
			prior_probs = torch.exp(net(inputs))
			val, indices = torch.max(prior_probs, 1)
		
			if (targets[0] != indices.data.cpu()[0]) or (val < args.minconf):
				continue

			print("idx: ",correct)
			correct += 1
		
			targets = targets.numpy()

			target_cs = [None] if not targeted else range(10)

			for target_calss in target_cs:
				if (targeted):
					if (target_calss == targets[0]):
						continue
			
				conf, flag, x = attack(inputs, targets[0], net, target_calss, pixels=pixels, maxiter=maxiter, popsize=popsize, verbose=verbose)

				success += flag
				conf_sum += flag*conf
				if (targeted):
					success_rate = float(success)/(9*correct)
				else:
					success_rate = float(success)/correct
				print("confidence origin:",val)
				if flag == 1:
					print("success rate: %.4f (%d/%d), confidence rate: %.4f, [(x,y) = (%d,%d) and (R,G,B)=(%d,%d,%d)]"%(
						success_rate, success, correct, conf_sum/success, x[0],x[1],x[2],x[3],x[4]))
		
			if correct == args.samples:
				break

	return success_rate

def main():

	print("==> Loading data and model...")
	tranfrom_test = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])
	test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=tranfrom_test)
	testloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2)

	class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
	assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
	checkpoint = torch.load('./checkpoint/%s.t7'%args.filename)
	if args.net=='VGG':
    		net = VGGGauss('VGG16')
	elif args.net=='ResNet':
    		net = ResNet18Gauss()
	else : print('Net ',args.net,' is not known, choose between VGG and ResNet')
	#net = VGG(args.model,args.gc)
	#net = checkpoint['net']

	net = net.to(device)
	if device == 'cuda':
		net = torch.nn.DataParallel(net)
		cudnn.benchmark = True
	net.load_state_dict(checkpoint['net'])
	net.eval()
	#net.cuda()
	#cudnn.benchmark = True

	print("==> Starting attck...")

	results = attack_all(net, testloader, pixels=args.pixels, targeted=args.targeted, maxiter=args.maxiter, popsize=args.popsize, verbose=args.verbose)
	print("Final success rate: %.4f"%results)


if __name__ == '__main__':
	main()
