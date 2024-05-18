class Trainer():
    def __init__(self, model, dataloader, epoch, save_name, lr, device):
        self.epoch = epoch
        self.save_name = save_name
        self.lr = lr
        self.device = device
        self.model = model.to(device)
        self.dataloader = dataloader

