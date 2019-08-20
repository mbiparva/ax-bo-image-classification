from tv_abc import TVBase
import torch


class Validator(TVBase):
    def __init__(self, mode, meters, device):
        super().__init__(mode, meters, device)

    def set_net_mode(self, net):
        net.eval()

    def batch_main(self, net, image, annotation):
        with torch.no_grad():
            p = net.forward(image)

        a = self.generate_gt(annotation)

        loss = net.loss_update(p, a, step=False)

        acc = self.evaluate(p, a)

        return {'loss': loss,
                'label_accuracy': acc}
