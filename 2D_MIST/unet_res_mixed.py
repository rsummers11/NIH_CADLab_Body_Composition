import torch
from torch import nn

class batchnorm_relu(nn.Module):
    def __init__(self, in_c):
        super().__init__()

        self.bn = nn.BatchNorm2d(in_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.bn(inputs)
        x = self.relu(x)
        return x

    def froze_layers(self):
        for param in self.bn.parameters():
            param.requires_grad = False

        for param in self.relu.parameters():
            param.requires_grad = False

class res_block(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()

        #conv layer
        self.b1 = batchnorm_relu(in_c)
        self.c1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=stride)
        self.b2 = batchnorm_relu(out_c)
        self.c2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, stride=1)

        #Shortcut Connection (Identity Mapping)
        self.s = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0, stride=stride)

    def forward(self, inputs):
        x = self.b1(inputs)
        x = self.c1(x)
        x = self.b2(x)
        x = self.c2(x)
        s = self.s(inputs)

        skip = x + s
        return skip

    def froze_layers(self):
        for param in self.b1.parameters():
            param.requires_grad = False

        for param in self.c1.parameters():
            param.requires_grad = False

        for param in self.b2.parameters():
            param.requires_grad = False

        for param in self.c2.parameters():
            param.requires_grad = False

        for param in self.s.parameters():
            param.requires_grad = False

class decoder(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.r = res_block(in_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.upsample(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.r(x)
        return x

class resunet_mixed(nn.Module):
    def __init__(self, in_class, out_class):
        super().__init__()

        """ Encoder 1 """
        self.c11 = nn.Conv2d(in_class, 64, kernel_size=3, padding=1)
        self.br1 = batchnorm_relu(64)
        self.c12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.c13 = nn.Conv2d(1, 64, kernel_size=1, padding=0)

        """ Encoder 2 and 3 """
        self.r2 = res_block(64, 128, stride=2)
        self.r3 = res_block(128, 256, stride=2)

        """ Bridge """
        self.r4 = res_block(256, 512, stride=2)

        """ Decoder """
        self.d1 = decoder(512, 256)
        self.d2 = decoder(256, 128)
        self.d3 = decoder(128, 64)

        """ Output """
        self.output = nn.Conv2d(64, out_class, kernel_size=1, padding=0)

        if out_class == 1:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, inputs):
        """ Encoder 1 """
        x = self.c11(inputs)
        x = self.br1(x)
        x = self.c12(x)
        s = self.c13(inputs)
        skip1 = x + s

        """ Encoder 2 and 3 """
        skip2 = self.r2(skip1)
        skip3 = self.r3(skip2)

        """ Bridge """
        b = self.r4(skip3)

        """ Decoder """
        d1_A = self.d1(b, skip3)
        d2_A = self.d2(d1_A, skip2)
        d3_A = self.d3(d2_A, skip1)

        """ output """
        output_A = self.output(d3_A)
        # output = self.sigmoid(output)

        d1_B = self.d1(b, skip3)
        d2_B = self.d2(d1_B, skip2)
        d3_B = self.d3(d2_B, skip1)

        """ output """
        output_B = self.output(d3_B)

        return [output_A, output_B]

    def froze_network(self):
        for param in self.c11.parameters():
            param.requires_grad = False

        self.br1.froze_layers()

        for param in self.c12.parameters():
            param.requires_grad = False

        for param in self.c13.parameters():
            param.requires_grad = False

        self.r2.froze_layers()
        self.r3.froze_layers()
        self.r4.froze_layers()
