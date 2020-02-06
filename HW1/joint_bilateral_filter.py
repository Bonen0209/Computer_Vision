# Server only
# -*- coding: future_fstrings -*- 

import argparse
import cv2
import numpy as np

from matplotlib import pyplot as plt

class Joint_bilateral_filter(object):
    
    def __init__(self, sigma_s, sigma_r, border_type='reflect'):
        self.border_type = border_type
        self.sigma_r = 255 * sigma_r
        self.sigma_s = sigma_s
        self.radius = 3 * sigma_s

    def joint_bilateral_filter(self, input, guidance):
        input_pad = cv2.copyMakeBorder(input, self.radius, self.radius, self.radius, self.radius, cv2.BORDER_REFLECT).astype(np.float64)
        guidance_pad = cv2.copyMakeBorder(guidance, self.radius, self.radius, self.radius, self.radius, cv2.BORDER_REFLECT).astype(np.float64)
        output = np.zeros(input.shape).astype(np.float64)

        h, w, _ = input.shape

        spacial_kernel = self.spacial_kernel_init()

        for x in range(self.radius, w + self.radius):
            for y in range(self.radius, h + self.radius):
                input_patch = input_pad[(y - self.radius):(y + self.radius + 1), (x - self.radius):(x + self.radius + 1), :]

                if len(guidance.shape) == 3:
                    guidance_patch = guidance_pad[(y - self.radius):(y + self.radius+1), (x - self.radius):(x + self.radius + 1), :]
                    pix_diff = np.sum(np.square(guidance_patch - guidance_pad[y][x]), axis=2)
                else:
                    guidance_patch = guidance_pad[(y - self.radius):(y + self.radius+1), (x - self.radius):(x + self.radius + 1)]
                    pix_diff = np.square(guidance_patch - guidance_pad[y][x])

                range_kernel = self.range_kernel_function(pix_diff)

                W = np.sum(np.multiply(spacial_kernel, range_kernel))

                out_y, out_x = y - self.radius, x - self.radius
                output[out_y][out_x][0] = np.sum(np.multiply(np.multiply(spacial_kernel, range_kernel), input_patch[:, :, 0])) / W
                output[out_y][out_x][1] = np.sum(np.multiply(np.multiply(spacial_kernel, range_kernel), input_patch[:, :, 1])) / W
                output[out_y][out_x][2] = np.sum(np.multiply(np.multiply(spacial_kernel, range_kernel), input_patch[:, :, 2])) / W

        output = output.astype(np.uint8)

        return output
    
    def spacial_kernel_init(self):
        base = np.tile(np.arange(-self.radius, self.radius + 1), (2 * self.radius + 1, 1))
        spacial_kernel = np.square(base) + np.square(base.T)
        return self.spacial_kernel_function(spacial_kernel)

    def spacial_kernel_function(self, input):
        return np.exp(-input / (2 * (self.sigma_s ** 2)))

    def range_kernel_function(self, input):
        return np.exp(-input / (2 * (self.sigma_r ** 2)))
    
def main():
    parser = argparse.ArgumentParser(description='JBF evaluation')
    parser.add_argument('--sigma_s', default=3, type=int, help='sigma of spatial kernel')
    parser.add_argument('--sigma_r', default=0.1, type=float, help='sigma of range kernel')
    parser.add_argument('--input_path', default='./HW1/testdata/ex.png', help='path of input image')
    parser.add_argument('--gt_bf_path', default='./HW1/testdata/ex_gt_bf.png', help='path of gt bf image')
    parser.add_argument('--gt_jbf_path', default='./HW1/testdata/ex_gt_jbf.png', help='path of gt jbf image')
    
    args = parser.parse_args()
    
    img = cv2.imread(args.input_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    guidance = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # create JBF class
    JBF = Joint_bilateral_filter(args.sigma_s, args.sigma_r, border_type='reflect')

    bf_out = JBF.joint_bilateral_filter(img_rgb, img_rgb).astype(np.uint8)
    jbf_out = JBF.joint_bilateral_filter(img_rgb, guidance).astype(np.uint8)

    bf_gt = cv2.cvtColor(cv2.imread(args.gt_bf_path), cv2.COLOR_BGR2RGB)
    jbf_gt = cv2.cvtColor(cv2.imread(args.gt_jbf_path), cv2.COLOR_BGR2RGB)

    bf_error = np.sum(np.abs(bf_out-bf_gt))
    jbf_error = np.sum(np.abs(jbf_out-jbf_gt))

    print('%d %d' %(bf_error, jbf_error))
    table = (bf_out == bf_gt)
    
    cv2.imwrite('./HW1/testdata/ex_b.png', cv2.cvtColor(bf_out, cv2.COLOR_RGB2BGR))
    cv2.imwrite('./HW1/testdata/ex_jb.png', cv2.cvtColor(bf_out, cv2.COLOR_RGB2BGR))

if __name__ == '__main__':
    main()