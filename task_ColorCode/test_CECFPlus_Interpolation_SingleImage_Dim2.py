import math
import sys
import time

import numpy as np

sys.path.append("..")
import torchvision.utils
from methods.MyCECFPlus.utils import get_config
from methods.MyCECFPlus.trainer_CECFPlus_SegDeepLabPretrain_MMD import UICC_Trainer
import argparse
import torch
import os
from torchvision import transforms
import torchvision.transforms.functional as ttf
from PIL import Image
# from train_SegmentModel import DeepLabv3
import matplotlib.pyplot as plt
import cv2


class Option():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--device", type=str)
        self.parser.add_argument('--config', type=str, help="net configuration")
        self.parser.add_argument('--input_folder', type=str, help="input image path")
        self.parser.add_argument('--output_folder', type=str, help="output image path")
        self.parser.add_argument('--checkpoint', type=str, help="checkpoint of autoencoders")
        self.parser.add_argument('--guide_path', type=str, default='', help="style image path")
        self.parser.add_argument('--output_path', type=str, default='.')
        self.parser.add_argument("--subfolder_prefix", type=str)
        self.parser.add_argument("--img_name", type=str)

        self.parser.add_argument("--have_mask", type=str)

    def start_parse(self):
        opts = self.parser.parse_args()
        return opts


def prepare():

    opts = Option().start_parse()
    config = get_config(opts.config)

    device = torch.device(opts.device)

    trainer = UICC_Trainer(config, device=device)

    state_dict = torch.load(opts.checkpoint, map_location=device)
    trainer.gen_a.load_state_dict(state_dict['a'])
    trainer.gen_b.load_state_dict(state_dict['b'])

    # trainer.gen_a.deeplab.load_state_dict(torch.load("../results/MyCECFPlus/DeepLabV3.pth"))
    print("loading pretrain success")

    new_size = config['new_size']

    imgs_files = os.listdir(opts.input_folder)
    imgs_files = [img for img in imgs_files if img.endswith(".jpg") or img.endswith(".png")]
    guide_files = os.listdir(opts.guide_path)
    guide_files = [img for img in guide_files if img.endswith(".jpg") or img.endswith(".png")]

    transform = transforms.Compose([transforms.Resize((new_size, new_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    return trainer, opts, imgs_files, guide_files, transform, new_size, device


def mk_cur_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def get_centroid(binary_mask):
    # calculate moments of binary image
    M = cv2.moments(binary_mask)

    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    print("centroid ", cX, cY)

    return cX, cY


def color_convert_hexadecimal2ten(sixteen):
    """

    :param sixteen: such as 312436 in RGB mode
    :return:
    """
    assert len(sixteen) == 6
    R = int("0x" + sixteen[:2], 16)
    G = int("0x" + sixteen[2:4], 16)
    B = int("0x" + sixteen[4:], 16)
    return R, G , B


def get_pur_color_background(image, mask):
    if len(mask.size()) == 3:
        mask = mask.unsqueeze(0)
    # 制作一个纯色背景
    pur_color_background = image.clone()
    # R, G, B = color_convert_hexadecimal2ten(sixteen="B9D3EE")
    R, G, B = [215, 230, 245]
    pur_color_background[:, 0, :, :] = R / 255
    pur_color_background[:, 1, :, :] = G / 255
    pur_color_background[:, 2, :, :] = B / 255
    pur_color_background[mask == 1] = 0

    return pur_color_background


if __name__ == "__main__":
    trainer, opts, imgs_files, guide_files, transform, new_size, device = prepare()
    trainer = trainer.to(device)
    trainer.eval()

    mk_cur_dir(opts.output_path)
    with torch.no_grad():
        # read the distorted image
        img_C = opts.img_name
        ori_img = Image.open(os.path.join(opts.input_folder, img_C)).convert('RGB')
        image = transform(ori_img).unsqueeze(0).to(device)

        if opts.have_mask == "Yes":
            mask_img = Image.open(os.path.join(opts.input_folder[:-1] + "_mask/", img_C))
            mask = ttf.to_tensor(mask_img)
            mask = ttf.resize(mask, size=[new_size, new_size])
            mask = mask.to(device)
            mask[mask > 0.2] = 1
            mask[mask <= 0.2] = 0

        else:
            mask = trainer.gen_a.get_segment_map(distortion=(image + 1) / 2)
            # mask =seg_net((image + 1) / 2)
            print("mask ranges ", mask.max(), mask.min(), mask.mean())
            mask[mask > 0.5] = 1
            mask[mask <= 0.5] = 0

        mask_numpy = mask[0, :, :].detach().cpu().numpy()
        cx, cy = get_centroid(mask_numpy)
        crop_w, crop_h = 40, 40

        #
        content, _, fix_color_code = trainer.gen_a.encode(image)
        print("fix color code size: ", fix_color_code.size())

        # save the mask
        original_output = trainer.gen_a.decode_fix(content, fix_color_code)

        # original enhanced results
        background = (1 - mask) * (original_output + 1) / 2

        # 根据二维的潜变量，设定边界
        LATENT_RANGE = 5
        # NUM_LATENT = 15
        # MIDDLE_POSITION = 7
        # NUM_LATENT_X = 15
        # NUM_LATENT_Y = 15
        NUM_LATENT_X = 9
        NUM_LATENT_Y = 15
        MIDDLE_POSITION_X = NUM_LATENT_X // 2
        MIDDLE_POSITION_Y = NUM_LATENT_Y // 2
        collected_visual_results = []
        collected_visual_results_with_index = []
        collected_visual_results_colormaps = []
        collected_visual_results_foreground_with_pure_background = []
        grid_x = np.linspace(-LATENT_RANGE, LATENT_RANGE, NUM_LATENT_X)
        grid_y = np.linspace(-LATENT_RANGE, LATENT_RANGE, NUM_LATENT_Y)[::-1]
        # assert NUM_LATENT_X % 2 == 1 and NUM_LATENT_Y % 2 == 1


        # 生成纯色背景
        white_background = get_pur_color_background(image=image, mask=mask)

        for x_index, x_coor in enumerate(grid_x):

            for y_index, y_coor in enumerate(grid_y):

                cur_output_folder = opts.output_folder + opts.subfolder_prefix + img_C[:-4] + "_GauRange" + str(LATENT_RANGE) + \
                    "_NX" + str(NUM_LATENT_X) + "_NY" + str(NUM_LATENT_Y)
                mk_cur_dir(dir=cur_output_folder)

                torchvision.utils.save_image(mask, os.path.join(cur_output_folder, "pred_mask_" + img_C))

                # generate color code
                guide_color_code = torch.tensor(data=[[x_coor, y_coor]], dtype=torch.float).to(fix_color_code.device)
                guide_color_code = guide_color_code.unsqueeze(2).unsqueeze(3)
                # print("guide color code size: ", guide_color_code.size())

                # Choose a fixed alpha
                alpha = 0.5

                # fuse the color code
                fuse_color_code = (1 - alpha) * fix_color_code + alpha * guide_color_code
                fuse_color_code = fuse_color_code / math.sqrt(alpha ** 2 + (1 - alpha) ** 2)

                outputs = trainer.gen_a.decode_fix(content, fuse_color_code)
                outputs = (outputs + 1) / 2.0

                foreground = mask * outputs
                fusion = foreground + background

                # 这张图为中心图像时，加上红色边框
                if x_index == MIDDLE_POSITION_X and y_index == MIDDLE_POSITION_Y:
                    fusion[:, 0, :, 0:10] = 1
                    fusion[:, 1, :, 0:10] = 0
                    fusion[:, 2, :, 0:10] = 0
                    fusion[:, 0, 0:10, :] = 1
                    fusion[:, 1, 0:10, :] = 0
                    fusion[:, 2, 0:10, :] = 0
                    fusion[:, 0, -10:, :] = 1
                    fusion[:, 1, -10:, :] = 0
                    fusion[:, 2, -10:, :] = 0
                    fusion[:, 0, :, -10:] = 1
                    fusion[:, 1, :, -10:] = 0
                    fusion[:, 2, :, -10:] = 0
                    print("set middle to red ")
                collected_visual_results.append(fusion)

                path = os.path.join(cur_output_folder, str(x_index) + "_" + str(y_index) + ".jpg")
                torchvision.utils.save_image(fusion, path)

                # 给fusion添加编号，便于后面索引
                fusion_with_index_numpy = fusion.clone().squeeze().detach().cpu().numpy().transpose(1, 2, 0) * 255
                fusion_with_index_numpy = fusion_with_index_numpy.astype(np.uint8).copy()
                cv2.putText(fusion_with_index_numpy, str(x_index) + "_" + str(y_index),
                            (10, int(new_size / 2)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
                # path = os.path.join(cur_output_folder, str(x_index) + "_" + str(y_index) + "_index.jpg")
                # cv2.cvtColor(fusion_with_index_numpy, cv2.COLOR_RGB2BGR, fusion_with_index_numpy)
                # cv2.imwrite(path, fusion_with_index_numpy)
                # cv2.cvtColor(fusion_with_index_numpy, cv2.COLOR_BGR2RGB, fusion_with_index_numpy)
                # fusion_with_index = torch.from_numpy(fusion_with_index_numpy)
                fusion_with_index = ttf.to_tensor(fusion_with_index_numpy).unsqueeze(0)
                collected_visual_results_with_index.append(fusion_with_index)


                # 保存具有纯色背景的图像
                # 保存前景的鱼
                pure_foreground = mask * outputs
                pure_foreground = pure_foreground + white_background
                # 进行crop，保存中心位置
                # pure_foreground = pure_foreground[:, :, 50:200, :]
                # pure_foreground = ttf.resize(pure_foreground, [170, 256])
                # print("size of pure_foreground: ", pure_foreground.size())
                # exit()
                path = os.path.join(cur_output_folder, str(x_index) + "_" + str(y_index) + "_foreground.jpg")
                torchvision.utils.save_image(pure_foreground, path)
                collected_visual_results_foreground_with_pure_background.append(pure_foreground)

                # -----------------------------------------------------------------------------
                # 根据前景提取主色调
                # plt.imshow(foreground.squeeze().cpu().numpy().transpose(1, 2, 0))
                # plt.show()
                foreground_numpy = foreground.detach().squeeze().cpu().numpy().transpose(1, 2, 0)
                foreground_numpy = (foreground_numpy * 255).astype(np.uint8).copy()
                # 计算质心

                foreground_crop = foreground[:, :, cy-crop_h: cy+crop_h, cx-crop_w: cx+crop_w]
                foreground_crop = ttf.resize(foreground_crop, [new_size, new_size])
                # cv2.imshow("foreground_crop", foreground_crop.squeeze().cpu().numpy().transpose(1, 2, 0))
                # cv2.waitKey(0)

                # 可视化所切的框
                # put text and highlight the center
                # cv2.circle(foreground_numpy, (cx, cy), 5, (255, 255, 255), -1)
                # cv2.putText(foreground_numpy, "centroid", (cx - 25, cy - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                # cv2.rectangle(foreground_numpy, (cx-crop_w, cy-crop_h), (cx+crop_w, cy+crop_h), (0, 255, 0), 2)
                # cv2.imshow("xx", foreground_numpy)
                # key = cv2.waitKey(0)

                color_component = foreground_crop.detach().clone()
                color_component = ttf.gaussian_blur(color_component, kernel_size=[251, 251], sigma=[171, 171])
                collected_visual_results_colormaps.append(color_component)
                # -----------------------------------------------------------------------------
                # 存储起来

        pad_pixel_num = 5

        # 保存生成的图像
        collected_visual_results = torch.cat(collected_visual_results)
        collected_visual_results = torchvision.utils.make_grid(collected_visual_results,
                                                               nrow=NUM_LATENT_Y, padding=pad_pixel_num, pad_value=1)
        # 将中轴白线1替换为其它颜色的线，确保NUM_LATENT为偶数
        # assert NUM_LATENT % 2 == 0
        # mid_index = (pad_pixel_num + new_size) * (NUM_LATENT // 2)
        # print("collected_visual_results size: ", collected_visual_results.size())
        # collected_visual_results[:, mid_index: mid_index+pad_pixel_num, :] = 128
        path = os.path.join(cur_output_folder, img_C)
        torchvision.utils.save_image(collected_visual_results, path)

        # 保存色彩图
        collected_visual_results_colormaps = torch.cat(collected_visual_results_colormaps)
        collected_visual_results_colormaps = torchvision.utils.make_grid(collected_visual_results_colormaps,
                                                               nrow=NUM_LATENT_Y, padding=pad_pixel_num, pad_value=1)
        path = os.path.join(cur_output_folder, img_C[:-4] + "_color.jpg")
        torchvision.utils.save_image(collected_visual_results_colormaps, path)

        # 保存具有纯色背景的图
        collected_visual_results_foreground_with_pure_background = torch.cat(collected_visual_results_foreground_with_pure_background)
        collected_visual_results_foreground_with_pure_background = torchvision.utils.make_grid(collected_visual_results_foreground_with_pure_background,
                                                               nrow=NUM_LATENT_Y, padding=0, pad_value=1)
        path = os.path.join(cur_output_folder, img_C[:-4] + "_pure_background.jpg")
        torchvision.utils.save_image(collected_visual_results_foreground_with_pure_background, path)

        # 保存生成的图像，带有索引的版本
        # 保存色彩图
        collected_visual_results_with_index = torch.cat(collected_visual_results_with_index)
        collected_visual_results_with_index = torchvision.utils.make_grid(collected_visual_results_with_index,
                                                               nrow=NUM_LATENT_Y, padding=pad_pixel_num, pad_value=1)
        path = os.path.join(cur_output_folder, img_C[:-4] + "_index.jpg")
        torchvision.utils.save_image(collected_visual_results_with_index, path)

        # 将色彩图作为生成图像的外边框
        print("collected_visual_results_colormaps size ", collected_visual_results_colormaps.size())
        collected_visual_results_colormaps[:, new_size + pad_pixel_num: (new_size + pad_pixel_num) * (NUM_LATENT_X -1),
        new_size + pad_pixel_num: (new_size + pad_pixel_num) * (NUM_LATENT_Y -1)] = collected_visual_results[:, new_size + pad_pixel_num: (new_size + pad_pixel_num) * (NUM_LATENT_X -1),
        new_size + pad_pixel_num: (new_size + pad_pixel_num) * (NUM_LATENT_Y -1)]
        path = os.path.join(cur_output_folder, img_C[:-4] + "_color_with_img.jpg")
        torchvision.utils.save_image(collected_visual_results_colormaps, path)

        # save the original images
        ori_img.save(os.path.join(cur_output_folder, "distorted_" + img_C))
