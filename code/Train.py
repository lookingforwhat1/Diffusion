import os
from typing import Dict

import torch
import torch.optim as optim
import torchvision
from torchvision.utils import save_image
from tqdm import tqdm

import Model
import DatasetWT
import Scheduler
import datetime
import Diffusion


def train(modelConfig: Dict):
    logFile = open('log.txt', mode='a+')
    for faultType in [1, 2, 3, 4]:
        logFile.write('faultType={}\n'.format(faultType))
        device = torch.device(modelConfig["device"])
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=0.5, std=0.5)
        ])
        dataset = DatasetWT.DatasetWTLabeled(rootPath='./dataTrain', labelType=faultType,
                                             fileList=list(range(40)), mode='L', transform=transform)
        dataloader = dataset.getLoader(10)
        # dataset = CIFAR10(
        #     root='./CIFAR10', train=True, download=True,
        #     transform=transforms.Compose([
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        #     ]))
        # dataloader = DataLoader(
        #     dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

        net_model = Model.UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                               attn=modelConfig["attn"],
                               num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
        if modelConfig["training_load_weight"] is not None:
            net_model.load_state_dict(torch.load(os.path.join(
                modelConfig["save_weight_dir"], modelConfig["training_load_weight"]), map_location=device))
        optimizer = torch.optim.AdamW(
            net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
        cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
        warmUpScheduler = Scheduler.GradualWarmupScheduler(
            optimizer=optimizer, multiplier=modelConfig["multiplier"], warm_epoch=modelConfig["epoch"] // 10,
            after_scheduler=cosineScheduler)
        trainer = Diffusion.GaussianDiffusionTrainer(
            net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

        startTime = datetime.datetime.now()
        logFile.write(str(startTime) + '\n')

        # start training
        for e in range(modelConfig["epoch"]):
            with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
                for images, labels in tqdmDataLoader:
                    # train
                    optimizer.zero_grad()
                    x_0 = images.to(device)
                    loss = trainer(x_0).sum() / 4000.
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        net_model.parameters(), modelConfig["grad_clip"])
                    optimizer.step()
                    tqdmDataLoader.set_postfix(ordered_dict={
                        "epoch": e,
                        "loss: ": loss.item(),
                        "img shape: ": x_0.shape,
                        "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                    })
            warmUpScheduler.step()
            if e % 1000 == 999:
                torch.save(net_model.state_dict(), os.path.join(
                    modelConfig["save_weight_dir"], str(faultType), 'ckpt_' + str(e) + "_.pt"))

        endTime = datetime.datetime.now()
        logFile.write(str(endTime) + '\n')
        logFile.write(str(endTime - startTime) + '\n')


def eval(modelConfig: Dict, faultType=2, start=0):
    with torch.no_grad():
        device = torch.device(modelConfig["device"])
        model = Model.UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                           attn=modelConfig["attn"],
                           num_res_blocks=modelConfig["num_res_blocks"], dropout=0.)
        ckpt = torch.load(os.path.join(
            modelConfig["save_weight_dir"], str(faultType), modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()
        sampler = Diffusion.GaussianDiffusionSampler(
            model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
        noisyImage = torch.randn(
            size=[180, 1, 64, 64], device=device)
        saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        save_image(saveNoisy, os.path.join(
            './results/', 'noise.png'), nrow=modelConfig["nrow"])
        sampledImgs = sampler(noisyImage)
        sampledImgs = sampledImgs * 0.5 + 0.5
        for i, img in enumerate(sampledImgs):
            save_image(img, os.path.join(
                './results/', str(faultType), str(i+start) + '.png'), nrow=modelConfig["nrow"])
