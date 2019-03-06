
class Net(nn.Module):
    def __init__(self):
    super(Net, self).__init__()
    
    def conv_bn(inp, oup, stride, padding, dilation):
    return nn.Sequential(
    nn.Conv2d(inp, oup, 3, stride, padding, dilation, bias=False),
    nn.BatchNorm2d(oup),
    nn.ReLU(inplace=True)
    )
    
    def conv_dw(inp, oup, stride, padding, dilation):
    return nn.Sequential(
    nn.Conv2d(inp, inp, 3, stride, padding, dilation, groups=inp, bias=False),
    nn.BatchNorm2d(inp),
    nn.ReLU(inplace=True),
    
    nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
    nn.BatchNorm2d(oup),
    nn.ReLU(inplace=True),
    )
    
    def reduce_channels():
    return nn.Sequential(
    nn.Conv2d(512, 256, 3, 1, 1),
    nn.BatchNorm2d(256),
    nn.ReLU(inplace=True),
    nn.Conv2d(256, 128, 3, 1, 1),
    nn.BatchNorm2d(128),
    nn.ReLU(inplace=True)
    )
    
    def initial_block():
    return nn.Sequential(
    nn.Conv2d(128, 128, 3, 1, 1),
    nn.Conv2d(128, 128, 3, 1, 1),
    nn.Conv2d(128, 128, 3, 1, 1)
    )
    
    self.model = nn.Sequential(
    conv_bn(  3,  32, 2, 1, 1),
    conv_dw( 32,  64, 1, 1, 1),
    conv_dw( 64, 128, 2, 1, 1),
    conv_dw(128, 128, 1, 1, 1),
    conv_dw(128, 256, 1, 1, 1),
    conv_dw(256, 256, 1, 1, 1),
    conv_dw(256, 512, 2, 2, 2),
    conv_dw(512, 512, 1, 1, 1),
    conv_dw(512, 512, 1, 1, 1),
    conv_dw(512, 512, 1, 1, 1),
    conv_dw(512, 512, 1, 1, 1),
    reduce_channels()
    )
    
    def forward(self, x):
    x = self.model(x)
    return x

