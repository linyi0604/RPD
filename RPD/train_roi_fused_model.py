import os
from torchvision import transforms
from albumentations import Compose, Resize
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

from utils import Logger, AverageMeter, evaluate
from dataset.dataset import BUSIDataset_image_roi as BUSIDataset
from networks.prior_fused_model import Prior_fused_model as Model


class Config(object):
    def __init__(self) -> None:
        self.log_path = "./logs/"
        self.image_path = "../../datasets/UDIAT/image/"
        self.mask_path = "../../datasets/UDIAT/mask/"
        self.train_mapping_path = "../../datasets/UDIAT/train_mapping.txt"
        self.test_mapping_path = "../../datasets/UDIAT/test_mapping.txt"
        self.model_state_path = "./model_state/"

        self.gpu_id = "0"

        self.class_num = 2
        self.network_input_size = (224, 224)    
        self.batch_size = 32
        self.num_workers = 32
        self.localizer_learning_rate = 0.001
        self.learning_rate = 0.001
        self.EPOCH = 120



def train_test(times):
    config = Config()
    logger = Logger(config.log_path + "roi_fused_model_%s.log"%times)
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id

    train_set = BUSIDataset(config.image_path, config.mask_path, 
                        config.train_mapping_path, 
                        transform=Compose([
                        Resize(config.network_input_size[0], config.network_input_size[1]),
                        ]))
    test_set = BUSIDataset(config.image_path, config.mask_path, 
                        config.test_mapping_path, 
                        transform=Compose([
                        Resize(config.network_input_size[0], config.network_input_size[1]),
                        ]))
    train_loader = DataLoader(train_set,
                            batch_size=config.batch_size,
                            shuffle=True,
                            pin_memory=True,
                            num_workers=config.num_workers)
    test_loader = DataLoader(test_set,
                            batch_size=config.batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=config.num_workers)
    

    model = Model(class_num=config.class_num, pretrained=True).cuda()

    optimizer = torch.optim.SGD(params=model.parameters(), lr=config.learning_rate, weight_decay=5e-4, momentum=0.9)
    loss_function = nn.CrossEntropyLoss()


    # training
    model.train()
    best_mean = 0
    best_log = ""
    for epoch in range(config.EPOCH):
        train_loss = AverageMeter()
        for step, (img, mask, label) in enumerate(train_loader):
            img = img.cuda()
            mask = mask.cuda()
            label = label.cuda()

            output = model(img, mask)
            loss = loss_function(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss.update(loss.item(), img.size(0))

        message = ""
        message += "epoch: %2d  \n" % epoch
        message += "    train:  loss: %.5f \n" % train_loss.avg
        # logger.write(message)

        # test after each epoch training
        with torch.no_grad():
            model.eval()
            test_loss = AverageMeter()
            labels = []
            predictions = []
            scores = []
            for step, (img, mask, label) in enumerate(test_loader):
                
                img = img.cuda()
                mask = mask.cuda()
                label = label.cuda()
                output = model(img, mask)
                loss = loss_function(output, label)

                test_loss.update(loss.item(), img.size(0))
                labels += label.cpu().tolist()
                predictions += torch.argmax(output, dim=1).cpu().tolist()
                scores += torch.softmax(output, dim=1).cpu().tolist()
             

            auc, accuracy, precision, specificity, sensitivity, f1, mean = evaluate(predictions, labels, scores)
            message += "    test:  loss: %.5f\n" % test_loss.avg
            message += "        auc \t accuracy \t precision \t specificity \t sensitivity \t F1 \n"
            message += "        %.2f%% \t %.2f%% \t %.2f%% \t %.2f%% \t %.2f%% \t %.2f%% \n" % (auc*100, accuracy*100, precision*100, specificity*100, sensitivity*100, f1*100)
            logger.write(message)

            if mean > best_mean:
                best_mean = mean
                best_log = "best test performance until now: \n" + message + "\n"
                torch.save(model.state_dict(), config.model_state_path + "prior_fused_model.pkl")
            logger.write(best_log)
                
            

if __name__ == "__main__":
    
    for i in range(5):
        train_test(i)
