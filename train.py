import time
import numpy as np
from tqdm.notebook import tqdm

def train(model, criterion, optimizer, train_dl, valid_dl, EPOCH, print_feq, device):

    for epoch in range(1, EPOCH + 1):
        model.train()
        train_loss = []

        for step, (imgs, boxes, labels) in enumerate(train_dl):
            time_1 = time.time()
            imgs = imgs.to(device)
            
            # boxes = torch.cat((boxes), dim=0)
            boxes = [box.to(device) for box in boxes]
            # labels = torch.cat((labels), dim=0)
            labels = [label.to(device) for label in labels]

            pred_loc, pred_sco = model(imgs)

            loss = criterion(pred_loc, pred_sco, boxes, labels)

            # Backward prop.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # losses.update(loss.item(), images.size(0))
            train_loss.append(loss.item())
            if step % print_feq == 0:
                print(
                    "epoch:",
                    epoch,
                    "\tstep:",
                    step + 1,
                    "/",
                    len(train_dl) + 1,
                    "\ttrain loss:",
                    "{:.4f}".format(loss.item()),
                    "\ttime:",
                    "{:.4f}".format((time.time() - time_1) * print_feq),
                    "s",
                )

        model.eval()
        valid_loss = []
        for step, (imgs, boxes, labels) in enumerate(tqdm(valid_dl)):
            imgs = imgs.to(device)
            boxes = [box.to(device) for box in boxes]
            labels = [label.to(device) for label in labels]
            pred_loc, pred_sco = model(imgs)
            loss = criterion(pred_loc, pred_sco, boxes, labels)
            valid_loss.append(loss.item())

        print(
            "epoch:",
            epoch,
            "/",
            EPOCH + 1,
            "\ttrain loss:",
            "{:.4f}".format(np.mean(train_loss)),
            "\tvalid loss:",
            "{:.4f}".format(np.mean(valid_loss)),
        )

    return np.mean(valid_loss)
