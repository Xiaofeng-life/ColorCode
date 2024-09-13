import math
import sys
sys.path.append("..")
import torchvision.utils
from methods.CECF.utils import get_config
from methods.MyCECFPlus.trainer_CECFPlus_SegDeepLabPretrain_NoMMD import UICC_Trainer
import argparse
import torch
import os
from torchvision import transforms
from PIL import Image


class Option():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--device", type=str)
        self.parser.add_argument('--config', type=str, help="net configuration")
        self.parser.add_argument('--input_folder', type=str, help="input image path")
        self.parser.add_argument('--checkpoint', type=str, help="checkpoint of autoencoders")
        self.parser.add_argument('--output_folder', type=str, default='.')

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

    new_size = config['new_size']

    imgs_files = os.listdir(opts.input_folder)
    imgs_files = [img for img in imgs_files if img.endswith(".jpg") or img.endswith(".png")]

    transform = transforms.Compose([transforms.Resize(new_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    return trainer, opts, imgs_files, transform, new_size, device


def mk_cur_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


if __name__ == "__main__":
    trainer, opts, imgs_files, transform, new_size, device = prepare()
    trainer = trainer.to(device)
    trainer.eval()

    mk_cur_dir(opts.output_folder)
    with torch.no_grad():
        for idx, img_C in enumerate(imgs_files):

            # read the distorted image
            ori_img = Image.open(os.path.join(opts.input_folder, img_C)).convert('RGB')
            image = transform(ori_img).unsqueeze(0).to(device)

            #
            content, _, fix_color_code = trainer.gen_a.encode(image)

            outputs = trainer.gen_a.decode_fix(content, fix_color_code)
            outputs = (outputs + 1) / 2.0
            path = os.path.join(opts.output_folder, 'alpha_{}.jpg'.format(idx))
            torchvision.utils.save_image(outputs, path)
