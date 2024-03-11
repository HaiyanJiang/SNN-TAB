# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

# __all__ = ['SeqToANNContainer', 'TAB_Layer', 'Conv_TAB_Layer', 'LIFNeuron', 'input_expand', 'TriangleSG']


class SeqToANNContainer(nn.Module):
    """ Altered form SpikingJelly. Input Shape is of [NTCHW] or [TNCHW] """
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1:
            self.module = args[0]
        else:
            self.module = nn.Sequential(*args)

    def forward(self, x_seq):
        """
        merge [TB,C,H,W], then reshape to [T,B,C,H,W]
        x_seq : torch.Tensor with ``shape=[T, batch_size, ...]``
        Returns: torch.Tensor with ``shape=[T, batch_size, ...]``
        """
        y_shape = [x_seq.shape[0], x_seq.shape[1]]
        y_seq = self.module(x_seq.flatten(0, 1).contiguous())
        y_shape.extend(y_seq.shape[1:])
        return y_seq.view(y_shape)


def _prob_check(p):
    p2 = p**2
    return p2


class TAB_Layer(nn.Module):
    def __init__(self, num_features, time_steps=4, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(TAB_Layer, self).__init__()
        self.time_steps = time_steps
        self.bn_list = nn.ModuleList([nn.BatchNorm3d(num_features, eps, momentum, affine, track_running_stats) for i in range(time_steps)])
        self.p = nn.Parameter(torch.ones(time_steps, 1, 1, 1, 1))

    def forward(self, x):
        # ## Shape of x is [NTCHW]
        self.p = (self.p).to(x.device)
        self.bn_list = nn.ModuleList([self.bn_list[i].to(x.device) for i in range(self.time_steps)])
        pt = _prob_check(self.p)

        assert x.shape[1] == self.time_steps, f"Time-steps not match input dimensions. x.shape: {x.shape}, x.shape[1]: {x.shape[1]} and self.time_steps: {self.time_steps}"
        # y_res = x.clone()
        y_res = []
        for t in range(self.time_steps):
            # xt = x[:,0:(t+1),...]
            y = x[:,0:(t+1),...].clone().transpose(1, 2).contiguous()  # [N,T,C,H,W] ==> [N,C,T,H,W], put C in dim1.
            y = self.bn_list[t](y)
            y = y.contiguous().transpose(1, 2).contiguous()  # [N,C,T,H,W] to [N,T,C,H,W]
            # y_res[:,t,...] = y[:,t,...].clone()  # Only slice the t-th
            y_res.append(y[:,t,...].clone())  # Only slice the t-th
        y_res = torch.stack(y_res, dim=1)
        # ### reshape the data and multipy the p[t] to each time-step [t]
        y = y_res.transpose(0, 1).contiguous()  # NTCHW  TNCHW
        # y = y * self.p
        y = y * pt
        y = y.contiguous().transpose(0, 1)  # TNCHW  NTCHW
        return y


class Conv_TAB_Layer(nn.Module):
    def __init__(self, time_steps, in_plane, out_plane, kernel_size, stride=1, padding=1):
        super(Conv_TAB_Layer, self).__init__()
        self.conv = SeqToANNContainer(
            nn.Conv2d(in_plane, out_plane, kernel_size, stride, padding, bias=False),
        )
        self.bn = TAB_Layer(out_plane, time_steps)

    def forward(self, input_):
        y = self.conv(input_)
        y = self.bn(y)
        return y



def input_expand(x, T):
    x = x.unsqueeze(1)
    x = x.repeat(1, T, 1, 1, 1)
    return x


class TriangleSG(torch.autograd.Function):
    """
    # Altered from code of Temporal Efficient Training, ICLR 2022 (https://openreview.net/forum?id=_XNtisL32jv)
    Triangular Surrogate Gradient.
    **Forward pass:** Heaviside step function shifted.
    The heaviside function is H(u-v_{th}), which is applied to mem-v_{th}, not mem directly.
       -- math:
           S=
            \begin{cases}
            1 & \text{if } U ≥ U_{\rm th} \\ \\
            0 & \text{if } U < U_{\rm th}
            \end{cases}
    **Backward pass:** Gradient of the triangular function.
    """

    @staticmethod
    def forward(ctx, input_, gamma):
        ctx.save_for_backward(input_)
        ctx.gamma = gamma
        out = (input_ > 0).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input_, ) = ctx.saved_tensors
        grad_input = grad_output.clone()
        tmp = (1 / ctx.gamma) * (1 / ctx.gamma) * ((ctx.gamma - input_.abs()).clamp(min=0))
        grad = grad_input * tmp
        return grad, None


class LIFNeuron(nn.Module):
    def __init__(self, tau, v_th, gamma=1.0, firing=True, mem_out=False):
        super(LIFNeuron, self).__init__()
        self.heaviside = TriangleSG.apply
        self.tau = tau
        self.v_th = v_th
        self.gamma = gamma  # For the surrogate gradient function
        self.firing = firing  # For the last layer, firing=False; other layers, firing=True
        self.mem_out = mem_out  # For the output of the last layer, output spiking (or/and) membrane potential

    def forward(self, input_):
        # ## Input has shape [NTCHW] or another notation [BTCHW] # input_.shape
        spk_rec = []
        mem_rec = []
        batch_size = input_.shape[0]
        time_steps = input_.shape[1]
        chw = input_.size()[2:]
        mem_potential = torch.zeros(batch_size, *chw).to(input_.device)
        # mem_potential.shape # [N,C,H,W]
        if self.firing:
            for t in range(time_steps):
                mem = self.tau * mem_potential + input_[:, t, ...]
                spike = self.heaviside(mem - self.v_th, self.gamma)
                mem_potential = mem * (1 - spike)
                spk_rec.append(spike)
                mem_rec.append(mem_potential)
        else:
            for t in range(time_steps):
                mem_potential = mem_potential + input_[:, t, ...]
                # spike = self.heaviside(mem_potential - self.v_th, self.gamma)
                # spk_rec.append(spike)
                mem_rec.append(mem_potential)
            return torch.stack(mem_rec, dim=1)
        if self.mem_out:
            return torch.stack(spk_rec, dim=1), torch.stack(mem_rec, dim=1)
        # # torch.stack(spk_rec, dim=1), dim=[BTCHW]; torch.stack(spk_rec, dim=0), dim=[TBCHW];
        # ## Very different with torch.cat(spk_rec, 1).shape, [B,TC,H,W]; torch.cat(spk_rec, 0).shape, [BT,C,H,W];
        else:
            return torch.stack(spk_rec, dim=1)



class BasicBlock4SNN(nn.Module):
    """BasicBlock4SNN for resnet 18 and resnet 34
    BasicBlock and BottleNeck block have different output size.
    we use class attribute expansion to distinct.
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, tau=0.9):
        super(BasicBlock4SNN, self).__init__()

        time_steps = 4
        self.stride = stride
        self.downsample = downsample

        # ## Both self.conv1 and self.downsample layers downsample the input imagezise when stride != 1
        self.conv1 = Conv_TAB_Layer(time_steps, inplanes, planes, 3, stride, 1)
        self.lif1 = LIFNeuron(tau=tau, firing=True)  # ## replace the relu with LIFNeuron() activation.
        self.conv2 = Conv_TAB_Layer(time_steps, planes, planes, 3, 1, 1)
        self.lif2 = LIFNeuron(tau=tau, firing=True)  # ## replace the relu with LIFNeuron() activation.
        self.downsample = downsample
        self.stride = stride
        self.bn = TAB_Layer(planes, time_steps)

    def forward(self, x):
        identity = x
        # ## conv1
        out = self.conv1(x)
        out = self.lif1(out)
        # ## conv2
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        # ## residual
        out += identity
        out = self.lif2(out)
        return out




class ResNet(nn.Module):
    def __init__(self, block=BasicBlock4SNN, layers=[3, 3, 2], tau=0.90):
        super(ResNet, self).__init__()
        self.tau = tau
        self.time_steps = 4
        self.in_ch = 128
        self.conv1 = Conv_TAB_Layer(self.time_steps, 3, self.in_ch, 3, 1, 1)

        self.sn1 = LIFNeuron(tau=self.tau)
        self.avgpool = SeqToANNContainer(nn.AvgPool2d(2))
        self.layer1 = self.make_layer(block, 128, layers[0])
        self.layer2 = self.make_layer(block, 256, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 512, layers[2], stride=2)

        self.fc1 =  SeqToANNContainer(nn.Dropout(0.25), nn.Linear(512 * 4 * 4, 256))
        self.fc2 =  SeqToANNContainer(nn.Dropout(0.25), nn.Linear(256, 100))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def make_layer(self, block, in_ch, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_ch != in_ch * block.expansion:
            downsample = TEBNLayer(self.in_ch, in_ch * block.expansion, 1, stride, 0)
        layers = []
        layers.append(block(self.in_ch, in_ch, stride, downsample, method=self.method, tau=self.tau))
        self.in_ch = in_ch * block.expansion
        for _ in range(1, block_num):
            layers.append(block(self.in_ch, in_ch, method=self.method, tau=self.tau))
        return nn.Sequential(*layers)

    def forward_imp(self, input):
        x = input_expand(input, self.time_steps)

        x = self.conv1(x)
        x = self.sn1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)

        x = torch.flatten(x, 2)
        x = self.fc2(self.fc1(x))
        return x

    def forward(self, input):
        return self.forward_imp(input)


class VGG9(nn.Module):
    def __init__(self, tau=0.9):
        super(VGG9, self).__init__()
        self.tau = tau
        self.time_steps = 4
        self.avgpool = SeqToANNContainer(nn.AvgPool2d(kernel_size=2, stride=2))
        self.lif1 = LIFNeuron(tau=self.tau, firing=True)
        self.lif2 = LIFNeuron(tau=self.tau, firing=False)

        self.features = nn.Sequential(
            SeqToANNContainer(nn.Conv2d(3, 64, 3, 1, 1, bias=False)),  # conv1
            TAB_Layer(64, self.time_steps),
            self.lif1,
            SeqToANNContainer(nn.Conv2d(64, 64, 3, 1, 1, bias=False)),  # conv2
            TAB_Layer(64, self.time_steps),
            self.lif1,
            self.avgpool,  # avgpool1
            SeqToANNContainer(nn.Conv2d(64, 128, 3, 1, 1, bias=False)),  # conv3
            TAB_Layer(128, self.time_steps),
            self.lif1,
            SeqToANNContainer(nn.Conv2d(128, 128, 3, 1, 1, bias=False)),  # conv4
            TAB_Layer(128, self.time_steps),
            self.lif1,
            self.avgpool,  # avgpool2
            SeqToANNContainer(nn.Conv2d(128, 256, 3, 1, 1, bias=False)),  # conv5
            TAB_Layer(256, self.time_steps),
            self.lif1,
            SeqToANNContainer(nn.Conv2d(256, 256, 3, 1, 1, bias=False)),  # conv6
            TAB_Layer(256, self.time_steps),
            self.lif1,
            SeqToANNContainer(nn.Conv2d(256, 256, 3, 1, 1, bias=False)),  # conv7
            TAB_Layer(256, self.time_steps),
            self.lif1,
            self.avgpool,  # avgpool3
            )

        self.fc1 =  SeqToANNContainer(nn.Dropout(0.25), nn.Linear(256 * 4 * 4, 1024))
        self.fc2 =  SeqToANNContainer(nn.Dropout(0.25), nn.Linear(1024, 10))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        input = input_expand(input, self.time_steps)
        x = self.features(input)
        x = torch.flatten(x, 2)
        x = self.fc2(self.fc1(x))
        return x
