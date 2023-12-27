from imports.common_imports import *

from encoder_decoder.trainer_encoder_decoder import TrainerEncoderDecoder

from datamodule.datamodule import TripletImagesDataset

if __name__ == "__main__":

    # callbacks = [
    #     ModelCheckpoint(
    #         monitor='val_loss',
    #         dirpath='_checkpoints/vqvae',
    #         filename='vqvae-{epoch:03d}-{val_loss:.4f}',
    #     )
    # ]
    trainer = pl.Trainer(
        #limit_train_batches=1,
        # limit_val_batches=1,
        # accumulate_grad_batches=2,
        max_epochs = 200, 
        num_nodes=1, 
        devices=2,
        accelerator='gpu',
        strategy="ddp_find_unused_parameters_true",
        #callbacks=callbacks,
    )   

    train_dir = "/home/est_posgrado_victor.fonte/Proyectos_y_tareas/Transformers/Proyecto Final/_data/train"
    val_dir = "/home/est_posgrado_victor.fonte/Proyectos_y_tareas/Transformers/Proyecto Final/_data/val"
    test_dir = "/home/est_posgrado_victor.fonte/Proyectos_y_tareas/Transformers/Proyecto Final/_data/test"
    train_dataset = TripletImagesDataset(train_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_dataset = TripletImagesDataset(val_dir)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    test_dataset = TripletImagesDataset(test_dir)
    test_dataloader = DataLoader(test_dataset, batch_size=6, shuffle=True)

    model = TrainerEncoderDecoder(test_dataloader, use_ema=False)

    print("start")
    trainer.fit(model, train_dataloader, val_dataloader)
    print("done")