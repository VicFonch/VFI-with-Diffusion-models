from imports.common_imports import *

from diffusion.trainer_diffusion import TrainerDiffusion

from datamodule.datamodule import TripletImagesDataset

if __name__ == "__main__":

    trainer = pl.Trainer(
        # limit_train_batches=1,
        # accumulate_grad_batches=2,
        max_epochs = 200, 
        num_nodes=1, 
        devices=2,
        accelerator='gpu',
        strategy="ddp_find_unused_parameters_true",
    )   

    train_dir = "/home/est_posgrado_victor.fonte/Proyectos_y_tareas/Transformers/Proyecto Final/_data/train"
    val_dir = "/home/est_posgrado_victor.fonte/Proyectos_y_tareas/Transformers/Proyecto Final/_data/val"
    test_dir = "/home/est_posgrado_victor.fonte/Proyectos_y_tareas/Transformers/Proyecto Final/_data/test"
    train_dataset = TripletImagesDataset(train_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataset = TripletImagesDataset(val_dir)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_dataset = TripletImagesDataset(test_dir)
    test_dataloader = DataLoader(test_dataset, batch_size = 8, shuffle=True)

    model = TrainerDiffusion(test_dataloader)

    print("start")
    trainer.fit(model, train_dataloader, test_dataloader)
    print("done")

