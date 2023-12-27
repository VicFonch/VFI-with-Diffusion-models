from imports.common_imports import *

from datamodule.datamodule import TripletImagesDataset

if __name__ == "__main__":
    test_dir = "_data/test"
    test_dataset = TripletImagesDataset(test_dir)
    test_dataloader = DataLoader(test_dataset, batch_size = 8, shuffle=True)

    vqfigan_path = "_models/vqfigan/vqfigan.ckpt"
    denoiser_path = "_models/unet/denoising_unet.ckpt"

    #vqfigan = torch.load(vqfigan_path).eval().requires_grad_(False)
    denoiser_unet = torch.load(denoiser_path).eval().requires_grad_(False)   
