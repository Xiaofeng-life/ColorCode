import math
import sys
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


if __name__ == "__main__":
    trainer, opts, imgs_files, guide_files, transform, new_size, device = prepare()
    trainer = trainer.to(device)
    trainer.eval()

    # seg_net = DeepLabv3(nc=1).to(device)
    # seg_net.load_state_dict(torch.load("../results/MyCECFPlus/Seg/pth/498.pth"))
    # seg_net.load_state_dict(torch.load("../results/MyCECFPlus/DeepLabV3.pth"))
    # seg_net.eval()

    # alpha_weights = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    alpha_weights = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    # alpha_weights = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    # alpha_weights = [0.2]
    mk_cur_dir(opts.output_path)

    print("guide_files are ", guide_files)

    with torch.no_grad():
        for img_C in imgs_files:
            print("current underwater image is ", img_C)
            # img_C = opts.img_name
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

            #
            content, _, fix_color_code = trainer.gen_a.encode(image)

            # save the mask
            original_output = trainer.gen_a.decode_fix(content, fix_color_code)

            # original enhanced results
            background = (1 - mask) * (original_output + 1) / 2

            for img_S in guide_files:
                print("XXXXXXXXXXXXXXX: ", img_S)
                cur_output_folder = opts.output_folder + opts.subfolder_prefix + img_C[:-4] + "_guide_" + img_S[:-4]
                mk_cur_dir(dir=cur_output_folder)

                torchvision.utils.save_image(mask, os.path.join(cur_output_folder, "pred_mask_" + img_C))

                # read guide img
                ori_guide_img = Image.open(os.path.join(opts.guide_path, img_S)).convert('RGB')
                guide_image = transform(ori_guide_img).unsqueeze(0).to(device)

                # forward
                guide_content, _, guide_color_code = trainer.gen_a.encode(guide_image)

                # clip it if you want
                # guide_color_code = guide_color_code.clamp(min=-3, max=3)

                for j in range(len(alpha_weights)):
                    alpha = alpha_weights[j]

                    # fuse the color code
                    fuse_color_code = (1 - alpha) * fix_color_code + alpha * guide_color_code
                    fuse_color_code = fuse_color_code / math.sqrt(alpha ** 2 + (1 - alpha) ** 2)

                    outputs = trainer.gen_a.decode_fix(content, fuse_color_code)
                    outputs = (outputs + 1) / 2.0

                    foreground = mask * outputs
                    fusion = foreground + background
                    path = os.path.join(cur_output_folder, 'alpha_{}.jpg'.format(str(alpha)))
                    torchvision.utils.save_image(fusion, path)

                # save the original images
                ori_img.save(os.path.join(cur_output_folder, "distorted_" + img_C))
                ori_guide_img.save(os.path.join(cur_output_folder, "guide_" + img_S))

                # recon of guidance
                guide_recon = trainer.gen_a.decode_fix(guide_content, guide_color_code)
                guide_recon = (guide_recon + 1) / 2.0
                torchvision.utils.save_image(guide_recon, os.path.join(cur_output_folder, "guide_recon_" + img_S))

                # 按照原始MUNIT，将x的内容码和一个随机的风格码，放入到y的解码器中
                rand_val = torch.randn(size=fix_color_code.size()).cuda()
                x2y = trainer.gen_b.decode(content, rand_val)
                x2y = (x2y + 1) / 2.0
                torchvision.utils.save_image(x2y, os.path.join(cur_output_folder, "x2y_" + img_C))

                print("Saving !!!!!!!!!!!!!!!")
