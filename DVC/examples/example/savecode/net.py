import numpy as np
import os
import torch
import torchvision.models as models
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import math
import torch.nn.init as init
import logging
from torch.nn.parameter import Parameter
from subnet import *
import torchac

import matplotlib.pyplot as plt
import matplotlib.image
from PIL import Image

SAVE_IMG = True
def save_image_as_plot(save_path, x, as_plot=False):

    if as_plot:
        fig = plt.figure(figsize=(3,3))
        plt.imshow(x)
        plt.title('{} ({:.2f}, {:.2f})\n{}'.format(x.shape, x.min(), x.max(), os.path.basename(save_path)))
        plt.tight_layout()
        plt.tick_params(left = False, right = False , labelleft = False , 
                        labelbottom = False, bottom = False) 

        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        x = (x-x.min()) / (x.max()-x.min())
        matplotlib.image.imsave(save_path, x)
    print("Saved:", save_path)

def save_model(model, iter):
    torch.save(model.state_dict(), "./snapshot/iter{}.model".format(iter))

def load_model(model, f):
    with open(f, 'rb') as f:
        pretrained_dict = torch.load(f)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    f = str(f)
    if f.find('iter') != -1 and f.find('.model') != -1:
        st = f.find('iter') + 4
        ed = f.find('.model', st)
        return int(f[st:ed])
    else:
        return 0

class VideoCompressor(nn.Module):
    def __init__(self):
        super(VideoCompressor, self).__init__()
        # self.imageCompressor = ImageCompressor()
        self.opticFlow = ME_Spynet()
        self.mvEncoder = Analysis_mv_net()
        self.Q = None
        self.mvDecoder = Synthesis_mv_net()
        self.warpnet = Warp_net()
        self.resEncoder = Analysis_net()
        self.resDecoder = Synthesis_net()
        self.respriorEncoder = Analysis_prior_net()
        self.respriorDecoder = Synthesis_prior_net()
        self.bitEstimator_z = BitEstimator(out_channel_N)
        self.bitEstimator_mv = BitEstimator(out_channel_mv)
        # self.flow_warp = Resample2d()
        # self.bitEstimator_feature = BitEstimator(out_channel_M)
        self.warp_weight = 0
        self.mxrange = 150
        self.calrealbits = False

    def forwardFirstFrame(self, x):
        output, bittrans = self.imageCompressor(x)
        cost = self.bitEstimator(bittrans)
        return output, cost

    def motioncompensation(self, ref, mv):
        warpframe = flow_warp(ref, mv)
        inputfeature = torch.cat((warpframe, ref), 1)
        warpnet = self.warpnet(inputfeature)
        prediction = self.warpnet(inputfeature) + warpframe
        return prediction, warpframe

    def forward(self, input_image, referframe, quant_noise_feature=None, quant_noise_z=None, quant_noise_mv=None):
        # print("-"*100)
        # print("FORWARD")
        estmv = self.opticFlow(input_image, referframe)

        if SAVE_IMG:
            save_path = "output/input_image.jpg"
            x = input_image.detach().cpu().squeeze().numpy()
            x = np.transpose(x, (1,2,0))
            save_image_as_plot(save_path, x)

            save_path = "output/estmv.jpg"
            x = estmv.detach().cpu().squeeze().numpy()
            x = np.transpose(np.vstack((x, np.zeros((1, x.shape[1], x.shape[2])))), (1, 2, 0))
            save_image_as_plot(save_path, x)

        mvfeature = self.mvEncoder(estmv)
        
        if self.training:
            quant_mv = mvfeature + quant_noise_mv
        else:
            quant_mv = torch.round(mvfeature)

        quant_mv_upsample = self.mvDecoder(quant_mv)

        if SAVE_IMG:
            save_path = "output/quant_mv_upsample.jpg"
            x = quant_mv_upsample.detach().cpu().squeeze().numpy()
            x = np.transpose(np.vstack((x, np.zeros((1, x.shape[1], x.shape[2])))), (1, 2, 0))
            save_image_as_plot(save_path, x)
        
        # if 1:
        #     save_path = "output/quant_mv_upsample.jpg"
        #     x = quant_mv_upsample.detach().cpu().squeeze().numpy()
        #     x = np.transpose(np.vstack((x, np.zeros((1, x.shape[1], x.shape[2])))), (1, 2, 0))
        #     save_image_as_plot(save_path, x)

        prediction, warpframe = self.motioncompensation(referframe, quant_mv_upsample)

        input_residual = input_image - prediction

        if SAVE_IMG:
            save_path = "output/warpframe.jpg"
            x = referframe.detach().cpu().squeeze().numpy()
            x = np.transpose(x, (1,2,0))
            save_image_as_plot(save_path, x)

            save_path = "output/referframe.jpg"
            x = referframe.detach().cpu().squeeze().numpy()
            x = np.transpose(x, (1,2,0))
            save_image_as_plot(save_path, x)

            save_path = "output/prediction.jpg"
            x = prediction.detach().cpu().squeeze().numpy()
            x = np.transpose(x, (1,2,0))
            save_image_as_plot(save_path, x)

            save_path = "output/warpframe.jpg"
            x = warpframe.detach().cpu().squeeze().numpy()
            x = np.transpose(x, (1,2,0))
            save_image_as_plot(save_path, x)

            save_path = "output/input_residual.jpg"
            x = input_residual.detach().cpu().squeeze().numpy()
            x = np.transpose(x, (1,2,0))
            save_image_as_plot(save_path, x)

        # x = np.transpose(input_residual.detach().cpu().squeeze().numpy(), (1, 2, 0))
        # save_image_as_plot("output/input_residual.jpg", x)

        feature = self.resEncoder(input_residual)

        batch_size = feature.size()[0]
        z = self.respriorEncoder(feature)

        if self.training:
            compressed_z = z + quant_noise_z
        else:
            compressed_z = torch.round(z)

        recon_sigma = self.respriorDecoder(compressed_z)
        print("compressed_z:", compressed_z.shape)
        print("recon_sigma:", recon_sigma.shape)
        feature_renorm = feature

        if self.training:
            compressed_feature_renorm = feature_renorm + quant_noise_feature
        else:
            compressed_feature_renorm = torch.round(feature_renorm)

        recon_res = self.resDecoder(compressed_feature_renorm)
        recon_image = prediction + recon_res

        if SAVE_IMG:
            save_path = "output/recon_res.jpg"
            x = recon_res.detach().cpu().squeeze().numpy()
            x = np.transpose(x, (1,2,0))
            save_image_as_plot(save_path, x)

            save_path = "output/recon_image.jpg"
            x = recon_image.detach().cpu().squeeze().numpy()
            x = np.transpose(x, (1,2,0))
            save_image_as_plot(save_path, x)

        # x = np.transpose(recon_image.detach().cpu().squeeze().numpy(), (1, 2, 0))
        # save_image_as_plot("output/recon_image.jpg", x)
        # assert 0
        clipped_recon_image = recon_image.clamp(0., 1.)

        # distortion
        mse_loss = torch.mean((recon_image - input_image).pow(2))

        # psnr = tf.cond(
        #     tf.equal(mse_loss, 0), lambda: tf.constant(100, dtype=tf.float32),
        #     lambda: 10 * (tf.log(1 * 1 / mse_loss) / np.log(10)))

        warploss = torch.mean((warpframe - input_image).pow(2))
        interloss = torch.mean((prediction - input_image).pow(2))
        

# bit per pixel

        def feature_probs_based_sigma(feature, sigma):
            def getrealbitsg(x, gaussian):
                # print("NIPS18noc : mn : ", torch.min(x), " - mx : ", torch.max(x), " range : ", self.mxrange)
                cdfs = []
                x = x + self.mxrange
                n,c,h,w = x.shape
                for i in range(-self.mxrange, self.mxrange):
                    cdfs.append(gaussian.cdf(i - 0.5).view(n,c,h,w,1))
                cdfs = torch.cat(cdfs, 4).cpu().detach()
                
                byte_stream = torchac.encode_float_cdf(cdfs, x.cpu().detach().to(torch.int16), check_input_bounds=True)

                real_bits = torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda()

                sym_out = torchac.decode_float_cdf(cdfs, byte_stream)

                return sym_out - self.mxrange, real_bits

            mu = torch.zeros_like(sigma)
            sigma = sigma.clamp(1e-5, 1e10)
            gaussian = torch.distributions.laplace.Laplace(mu, sigma)
            probs = gaussian.cdf(feature + 0.5) - gaussian.cdf(feature - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-5) / math.log(2.0), 0, 50))
            
            if self.calrealbits and not self.training:
                decodedx, real_bits = getrealbitsg(feature, gaussian)
                total_bits = real_bits

            return total_bits, probs

        def iclr18_estrate_bits_z(z):
            
            def getrealbits(x):
                cdfs = []
                x = x + self.mxrange
                n,c,h,w = x.shape
                for i in range(-self.mxrange, self.mxrange):
                    cdfs.append(self.bitEstimator_z(i - 0.5).view(1, c, 1, 1, 1).repeat(1, 1, h, w, 1))
                cdfs = torch.cat(cdfs, 4).cpu().detach()
                byte_stream = torchac.encode_float_cdf(cdfs, x.cpu().detach().to(torch.int16), check_input_bounds=True)

                real_bits = torch.sum(torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda())

                sym_out = torchac.decode_float_cdf(cdfs, byte_stream)

                return sym_out - self.mxrange, real_bits

            prob = self.bitEstimator_z(z + 0.5) - self.bitEstimator_z(z - 0.5)
            print("prob z:", prob.shape)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))


            if self.calrealbits and not self.training:
                decodedx, real_bits = getrealbits(z)
                total_bits = real_bits

            return total_bits, prob

        def iclr18_estrate_bits_mv(mv):
            def getrealbits(x):
                cdfs = []
                x = x + self.mxrange
                n,c,h,w = x.shape
                for i in range(-self.mxrange, self.mxrange):
                    cdfs.append(self.bitEstimator_mv(i - 0.5).view(1, c, 1, 1, 1).repeat(1, 1, h, w, 1))
                cdfs = torch.cat(cdfs, 4).cpu().detach()
                byte_stream = torchac.encode_float_cdf(cdfs, x.cpu().detach().to(torch.int16), check_input_bounds=True)

                real_bits = torch.sum(torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda())

                sym_out = torchac.decode_float_cdf(cdfs, byte_stream)
                return sym_out - self.mxrange, real_bits

            prob = self.bitEstimator_mv(mv + 0.5) - self.bitEstimator_mv(mv - 0.5)
            print("prob mv:", prob.shape)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))


            if self.calrealbits and not self.training:
                decodedx, real_bits = getrealbits(mv)
                total_bits = real_bits

            return total_bits, prob

        total_bits_feature, _ = feature_probs_based_sigma(compressed_feature_renorm, recon_sigma)
        # entropy_context = entropy_context_from_sigma(compressed_feature_renorm, recon_sigma)

        total_bits_z, _ = iclr18_estrate_bits_z(compressed_z)
        print("compressed_z:", compressed_z.shape)
        total_bits_mv, _ = iclr18_estrate_bits_mv(quant_mv)
        print("quant_mv:", quant_mv.shape)
        assert 0

        im_shape = input_image.size()

        bpp_feature = total_bits_feature / (batch_size * im_shape[2] * im_shape[3])
        bpp_z = total_bits_z / (batch_size * im_shape[2] * im_shape[3])
        bpp_mv = total_bits_mv / (batch_size * im_shape[2] * im_shape[3])
        bpp = bpp_feature + bpp_z + bpp_mv

        plot_data = {
            "input_image": input_image,
            "prediction": prediction,
            "input_residual": input_residual,
            "recon_res": recon_res,
            "recon_image": recon_image,
            "clipped_recon_image": clipped_recon_image,
            "referframe": referframe
        }        
        return plot_data, clipped_recon_image, mse_loss, warploss, interloss, bpp_feature, bpp_z, bpp_mv, bpp
        