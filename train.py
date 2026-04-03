import os
import tqdm
import torch
import torch.nn as nn
from model import MERT_AES
from dataset import MOSdataset_moises
import utils as utils

def loss_fn(y, y_hat):
    return nn.BCELoss()(y, y_hat)

def train(args):
    # ======= enable cudnn benchmarking =======
    torch.backends.cudnn.benchmark = True
    
    # ======= model =======
    model = MERT_AES(
        proj_num_layer=args.model.proj_num_layer,
        proj_ln=args.model.proj_ln,
        proj_act_fn=args.model.proj_act_fn,
        proj_dropout=args.model.proj_dropout,
        output_dim=args.model.output_dim,
        binary_classification=args.model.binary_classification,
        freeze_encoder=args.model.freeze_encoder,
    )
    model.to(args.device)
    model.train()
    # print traning params
    for name, param in model.named_parameters():
        print(f"{name}: requires_grad = {param.requires_grad}")

    # ======= optimizer =======
    optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), 
    args.train.lr,
    weight_decay=args.train.weight_decay
    )

    # ======= scheduler =======
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # ======= dataset =======
    train_dataset = MOSdataset_moises(
        wav_root=args.data.train_data_path,
        duration_sec=args.data.duration_sec,
        sr=args.data.sr,
        split="train"
    )
    valid_dataset = MOSdataset_moises(
        wav_root=args.data.valid_data_path,
        duration_sec=args.data.duration_sec,
        sr=args.data.sr,
        split="valid"
    )

    # ======= loader =======
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.train.batch_size,
        shuffle=True,
        num_workers=args.train.num_workers,
        pin_memory=True,
        persistent_workers=args.train.num_workers > 0
    )
    print('> train dataset ready ...........')

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, 
        batch_size=args.train.batch_size,
        shuffle=False,
        num_workers=args.train.num_workers,
        pin_memory=True,
        persistent_workers=args.train.num_workers > 0
    )
    print('> valid dataset ready ...........')

    
    best_valid_loss = 9999
    best_valid_acc = 0

    log_interval = getattr(args.train, 'log_interval', None)
    if log_interval is None:
        log_interval = 50

    for epoch in range(args.train.epochs):
        train_loss, train_acc, valid_loss, valid_acc = train_one_epoch(
            model,
            optimizer,
            scheduler,
            train_loader,
            valid_loader,
            args.device,
            epoch=epoch,
            log_interval=log_interval,
            outdir=args.outdir,
        )

        utils.save_model(model=model, outdir=args.outdir, name='latest')
        utils.logging(epoch, train_loss, train_acc, valid_loss, valid_acc, args.outdir)

        if epoch == 0:
            utils.save_model(model=model, outdir=args.outdir, name='first_epoch')

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            utils.save_model(model=model, outdir=args.outdir, name='best_loss')

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            utils.save_model(model=model, outdir=args.outdir, name='best_acc')




def train_one_epoch(model, optimizer, scheduler, train_loader, valid_loader, device, epoch, log_interval, outdir):

    train_loss, train_acc = 0, 0
    for bidx, batch in enumerate(tqdm.tqdm(train_loader, desc=f"Epoch {epoch}"), start=1):

        model.train()
        optimizer.zero_grad()

        good_mix, bad_mix = batch
        B = good_mix.size(0)
        input = torch.cat([good_mix, bad_mix], dim=0)
        label = torch.cat([torch.ones(B, 1), torch.zeros(B, 1)])
        input, label = input.to(device), label.to(device)
        
        pred_label = model(input)

        loss = loss_fn(pred_label, label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += ((pred_label>0.5) == label).float().mean().item()

        if log_interval is not None and bidx % log_interval == 0:
            avg_train_loss = train_loss / bidx
            avg_train_acc = train_acc / bidx
            utils.logging(epoch, avg_train_loss, avg_train_acc, None, None, outdir, step=bidx)

    train_loss /= len(train_loader)
    train_acc /= len(train_loader)
    
    # Step scheduler after each epoch
    scheduler.step()

    model.eval()
    valid_loss, valid_acc = 0, 0
    with torch.no_grad():
        for bidx, batch in enumerate(valid_loader):

            good_mix, bad_mix = batch
            B = good_mix.size(0)
            input = torch.cat([good_mix, bad_mix], dim=0)
            label = torch.cat([torch.ones(B, 1), torch.zeros(B, 1)])
            input, label = input.to(device), label.to(device)
            
            pred_label = model(input)

            loss = loss_fn(pred_label, label)
            valid_loss += loss.item()
            valid_acc += ((pred_label>0.5) == label).float().mean().item()

        valid_loss /= len(valid_loader)
        valid_acc /= len(valid_loader)

    return train_loss, train_acc, valid_loss, valid_acc