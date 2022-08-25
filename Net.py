import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import matplotlib.pyplot as plt
##########################################################################
def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)
def conv1(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

##########################################################################
## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


##########################################################################

class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res

##########################################################################

class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1*x2
        x1 = x1+x
        return x1, img

##########################################################################
## U-Net

class Encoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff):
        super(Encoder, self).__init__()

        self.encoder_level1 = [CAB(n_feat,                     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level2 = [CAB(n_feat+scale_unetfeats,     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level3 = [CAB(n_feat+(scale_unetfeats*2), kernel_size, reduction, bias=bias, act=act) for _ in range(2)]

        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)

        self.down12  = DownSample(n_feat, scale_unetfeats)
        self.down23  = DownSample(n_feat+scale_unetfeats, scale_unetfeats)

        # Cross Stage Feature Fusion (CSFF)
        if csff:
            self.csff_enc1 = nn.Conv2d(n_feat,                     n_feat,                     kernel_size=1, bias=bias)
            self.csff_enc2 = nn.Conv2d(n_feat+scale_unetfeats,     n_feat+scale_unetfeats,     kernel_size=1, bias=bias)
            self.csff_enc3 = nn.Conv2d(n_feat+(scale_unetfeats*2), n_feat+(scale_unetfeats*2), kernel_size=1, bias=bias)

            self.csff_dec1 = nn.Conv2d(n_feat,                     n_feat,                     kernel_size=1, bias=bias)
            self.csff_dec2 = nn.Conv2d(n_feat+scale_unetfeats,     n_feat+scale_unetfeats,     kernel_size=1, bias=bias)
            self.csff_dec3 = nn.Conv2d(n_feat+(scale_unetfeats*2), n_feat+(scale_unetfeats*2), kernel_size=1, bias=bias)

    def forward(self, x, encoder_outs=None, decoder_outs=None):
        enc1 = self.encoder_level1(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc1 = enc1 + self.csff_enc1(encoder_outs[0]) + self.csff_dec1(decoder_outs[0])

        x = self.down12(enc1)

        enc2 = self.encoder_level2(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc2 = enc2 + self.csff_enc2(encoder_outs[1]) + self.csff_dec2(decoder_outs[1])

        x = self.down23(enc2)

        enc3 = self.encoder_level3(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc3 = enc3 + self.csff_enc3(encoder_outs[2]) + self.csff_dec3(decoder_outs[2])

        return [enc1, enc2, enc3]

class Decoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats):
        super(Decoder, self).__init__()

        self.decoder_level1 = [CAB(n_feat,                     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level2 = [CAB(n_feat+scale_unetfeats,     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level3 = [CAB(n_feat+(scale_unetfeats*2), kernel_size, reduction, bias=bias, act=act) for _ in range(2)]

        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)

        self.skip_attn1 = CAB(n_feat,                 kernel_size, reduction, bias=bias, act=act)
        self.skip_attn2 = CAB(n_feat+scale_unetfeats, kernel_size, reduction, bias=bias, act=act)

        self.up21  = SkipUpSample(n_feat, scale_unetfeats)
        self.up32  = SkipUpSample(n_feat+scale_unetfeats, scale_unetfeats)

    def forward(self, outs):
        enc1, enc2, enc3 = outs
        dec3 = self.decoder_level3(enc3)

        x = self.up32(dec3, self.skip_attn2(enc2))
        dec2 = self.decoder_level2(x)

        x = self.up21(dec2, self.skip_attn1(enc1))
        dec1 = self.decoder_level1(x)

        return [dec1, dec2, dec3]
class Decoder1(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats):
        super(Decoder1, self).__init__()
         #conv(n_feat, n_feat, kernel_size, bias=bias)
        self.decoder_level1 = [CAB(n_feat,                     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level2 = [conv1(n_feat,n_feat+scale_unetfeats,     kernel_size, bias=bias) for _ in range(2)]
        self.decoder_level3 = [conv1(n_feat,n_feat+(scale_unetfeats*2), kernel_size, bias=bias) for _ in range(2)]

        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)

        self.skip_attn1 = CAB(n_feat,                 kernel_size, reduction, bias=bias, act=act)
        self.skip_attn2 = CAB(n_feat+scale_unetfeats, kernel_size, reduction, bias=bias, act=act)

        self.up21  = SkipUpSample(n_feat, scale_unetfeats)
        self.up32  = SkipUpSample(n_feat+scale_unetfeats, scale_unetfeats)

    def forward(self, outs):
        enc1 = outs
        enc2 = outs
        enc3 = outs
        dec3 = self.decoder_level3(enc3)

        x = self.up32(dec3, self.skip_attn2(enc2))

        dec2 = self.decoder_level2(x)
        dec2_ad = dec2 + enc2
        x = self.up21(dec2_ad, self.skip_attn1(enc1))
        dec1 = self.decoder_level1(x)

        return [dec1,dec2,dec3]
##########################################################################
##---------- Resizing Modules ----------
class DownSample(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(DownSample, self).__init__()

        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels+s_factor, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        #print(x.size())
        x = self.down(x)
        #print(x.size())
        #print('*******************')
        return x

class UpSample(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels+s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x

class SkipUpSample(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels+s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x

##########################################################################

class ORB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_cab):
        super(ORB, self).__init__()
        modules_body = []
        modules_body = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(num_cab)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

##########################################################################
class ORSNet(nn.Module):
    def __init__(self, n_feat, scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats, num_cab):
        super(ORSNet, self).__init__()

        self.orb1 = ORB(n_feat+scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)
        self.orb2 = ORB(n_feat+scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)
        self.orb3 = ORB(n_feat+scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)

        self.up_enc1 = UpSample(n_feat, scale_unetfeats)
        self.up_dec1 = UpSample(n_feat, scale_unetfeats)

        self.up_enc2 = nn.Sequential(UpSample(n_feat+scale_unetfeats, scale_unetfeats), UpSample(n_feat, scale_unetfeats))
        self.up_dec2 = nn.Sequential(UpSample(n_feat+scale_unetfeats, scale_unetfeats), UpSample(n_feat, scale_unetfeats))

        self.conv_enc1 = nn.Conv2d(n_feat, n_feat+scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_enc2 = nn.Conv2d(n_feat, n_feat+scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_enc3 = nn.Conv2d(n_feat, n_feat+scale_orsnetfeats, kernel_size=1, bias=bias)

        self.conv_dec1 = nn.Conv2d(n_feat, n_feat+scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_dec2 = nn.Conv2d(n_feat, n_feat+scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_dec3 = nn.Conv2d(n_feat, n_feat+scale_orsnetfeats, kernel_size=1, bias=bias)

    def forward(self, x, encoder_outs, decoder_outs):
        x = self.orb1(x)
        x = x + self.conv_enc1(encoder_outs[0]) + self.conv_dec1(decoder_outs[0])

        x = self.orb2(x)
        x = x + self.conv_enc2(self.up_enc1(encoder_outs[1])) + self.conv_dec2(self.up_dec1(decoder_outs[1]))

        x = self.orb3(x)
        x = x + self.conv_enc3(self.up_enc2(encoder_outs[2])) + self.conv_dec3(self.up_dec2(decoder_outs[2]))

        return x


##########################################################################
class Net(nn.Module):
    def __init__(self, n_feat=96, scale_unetfeats=48, scale_orsnetfeats=32, num_cab=8, kernel_size=3, reduction=4, bias=False):
        super(Net, self).__init__()

        act=nn.PReLU()
        self.shallow_feat0 = nn.Sequential(conv1(3, n_feat, kernel_size, bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=torch.nn.SELU()))
        self.shallow_feat1 = nn.Sequential(conv(3, n_feat, kernel_size, bias=bias), CAB(n_feat,kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat2 = nn.Sequential(conv(3, n_feat, kernel_size, bias=bias), CAB(n_feat,kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat3 = nn.Sequential(conv(3, n_feat, kernel_size, bias=bias), CAB(n_feat,kernel_size, reduction, bias=bias, act=act))

        # Cross Stage Feature Fusion (CSFF)
        self.stage1_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.stage1_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)
        self.stage1_decoder1 =Decoder1(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)

        self.stage2_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=True)
        self.stage2_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)

        self.stage3_orsnet = ORSNet(n_feat, scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats, num_cab)

        self.sam12 = SAM(n_feat, kernel_size=1, bias=bias)
        self.sam23 = SAM(n_feat, kernel_size=1, bias=bias)

        self.concat12  = conv(n_feat*2, n_feat, kernel_size, bias=bias)
        self.concat23  = conv(n_feat*2, n_feat+scale_orsnetfeats, kernel_size, bias=bias)
        self.tail     = conv(n_feat+scale_orsnetfeats, 3, kernel_size, bias=bias)
        # self.laynorm =  nn.LayerNorm()
        self.attention_down = conv(96,48,1)
        self.attention_up = conv(48,96,1)
    def forward(self, x3_img):
        # Original-resolution Image for Stage 3
        H = x3_img.size(2)
        W = x3_img.size(3)

        # Multi-Patch Hierarchy: Split Image into four non-overlapping patches

        # Two Patches for Stage 2
        x2top_img  = x3_img[:,:,0:int(H/2),:]
        x2bot_img  = x3_img[:,:,int(H/2):H,:]
        #print("********************")
        # Four Patches for Stage 1
        x1ltop_img = x2top_img[:,:,:,0:int(W/2)]
        x1rtop_img = x2top_img[:,:,:,int(W/2):W]
        x1lbot_img = x2bot_img[:,:,:,0:int(W/2)]
        x1rbot_img = x2bot_img[:,:,:,int(W/2):W]
        # Eight Patches for Stage 0
        x0ltop_img1 = x1ltop_img[:, :, :, 0:int(W/4)]
        x0ltop_img2 = x1ltop_img[:, :,:, int(W/4):int(W/2)]
        x0rtop_img1 = x1rtop_img[:,:,:, 0:int(W/4)]
        x0rtop_img2 = x1rtop_img[:,:,:, int(W/4):int(W/2)]
        x0lbot_img1 = x1lbot_img[:,:,:, 0:int(W/4)]
        x0lbot_img2 = x1lbot_img[:,:,:, int(W/4):int(W/2)]
        x0rbot_img1 = x1rbot_img[:, :, :, 0:int(W/4)]
        x0rbot_img2 = x1rbot_img[:,:,:, int(W/4):int(W/2)]
        ##-------------------------------------------
        ##-------------- Stage 0---------------------
        ##-------------------------------------------
        x0ltop_img1 = self.shallow_feat0(x0ltop_img1)
        #print(" aaaaaaaaaaax0ltop_img1 shape{}".format(x0ltop_img1.size()))
        x0ltop_img2 = self.shallow_feat0(x0ltop_img2)
        # print(" aaaaaaaaaaaaax0ltop_img2 shape{}".format(x0ltop_img2.size()))
        x0rtop_img1 = self.shallow_feat0(x0rtop_img1)
        x0rtop_img2 = self.shallow_feat0(x0rtop_img2)
        x0lbot_img1 = self.shallow_feat0(x0lbot_img1)
        x0lbot_img2 = self.shallow_feat0(x0lbot_img2)
        x0rbot_img1 = self.shallow_feat0(x0rbot_img1)
        x0rbot_img2 = self.shallow_feat0(x0rbot_img2)
        ## Concat deep features--->4
        feat0_1 = torch.cat((x0ltop_img1, x0ltop_img2),3)#[torch.cat((k, v), 2) for k, v in zip(x0ltop_img1, x0ltop_img2)]
        feat0_2 = torch.cat((x0rtop_img1, x0rtop_img2),3)#[torch.cat((k, v), 2) for k, v in zip(x0rtop_img1, x0rtop_img2)]
        feat0_3 = torch.cat((x0lbot_img1, x0lbot_img2),3)#[torch.cat((k, v), 2) for k, v in zip(x0lbot_img1, x0lbot_img2)]
        feat0_4 = torch.cat((x0rbot_img1, x0rbot_img2),3)#[torch.cat((k, v), 2) for k, v in zip(x0rbot_img1, x0rbot_img2)]

        ## Concat deep features--->2
        feat0_1_0 = torch.cat((feat0_1, feat0_2),3)#[torch.cat((k, v), 2) for k, v in zip(feat0_1, feat0_2)]
        #print("concat feat 0_1 {}".format(feat0_1_0.size()))
        feat0_2_1 = torch.cat((feat0_3, feat0_4),3)#[torch.cat((k, v), 2) for k, v in zip(feat0_3, feat0_4)]
        ## Concat deep features--->1
        feat0_1_0_0 = torch.cat((feat0_1_0, feat0_2_1),3)#[torch.cat((k, v), 2) for k, v in zip(feat0_1_0, feat0_2_1)]
        #print("x1ltop_img {}".format(x1ltop_img.size()))
        #print("feat {}".format(feat0_1.size()))

        ##-------------- Stage 1---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        #a = x1ltop_img+feat0_1
        #x1ltop = self.shallow_feat1(a)
        x1ltop = self.shallow_feat1(x1ltop_img)
        x1rtop = self.shallow_feat1(x1rtop_img)
        x1lbot = self.shallow_feat1(x1lbot_img)
        x1rbot = self.shallow_feat1(x1rbot_img)

        ## Process features of all 4 patches with Encoder of Stage 1
        feat1_ltop = self.stage1_encoder(x1ltop)
        feat1_rtop = self.stage1_encoder(x1rtop)
        feat1_lbot = self.stage1_encoder(x1lbot)
        feat1_rbot = self.stage1_encoder(x1rbot)
        #feat1_ltop_m =feat1_ltop+ feat0_1
        ## Concat deep features
        feat1_top = [torch.cat((k,v), 3) for k,v in zip(feat1_ltop,feat1_rtop)]
        feat1_bot = [torch.cat((k,v), 3) for k,v in zip(feat1_lbot,feat1_rbot)]
        temp_feat1_top = feat1_top
        temp_feat1_bot = feat1_bot
        # print("sample   type fe   {}".format(feat1_top[0].size()))
        # print("sample   type fe   {}".format(feat0_1_0[0].size()))
        # print("sample   type fe   {}".format(type(feat0_1_0)))
        #print("this is len list {}".format(len(feat1_top)))
        temp_feat0_1_0 = feat0_1_0
        temp_feat0_2_1 = feat0_2_1
        a_temp = temp_feat0_1_0
        # print("this is a_temp size {}".format(a_temp.shape))
        # print("this is temp_feat1_top[0] {}".format(temp_feat1_top[0].shape))
        #a_temp = temp_feat0_1_0.reshape(1, 96, 128, 256)
        # a_temp_LN =self.laynorm(a_temp)

        lm = nn.LayerNorm(a_temp.size(),elementwise_affine=False)
        a_temp_LN = lm(a_temp)

        a_att_dw =self.attention_down(a_temp_LN)
        #plt.imshow(a_att_dw)
        #plt.show()
        a_att_up =self.attention_up(a_att_dw)
        #plt.imshow(a_att_up)
        #plt.show()
        a_ac = torch.sigmoid(a_att_up)
        a_last = a_ac*a_temp
        #b_temp = temp_feat0_2_1.reshape(1, 96, 128, 256)
        b_temp = temp_feat0_2_1
        #b_temp_LN = self.laynorm(b_temp)
        #print("ssssample   type fe   {}".format(a_temp.size()))
        #print("sample   type fe   {}".format(type(feat1_top[0])))
        #print("sample   type fe   {}".format(type(a_temp)))
        feat1_top1_ano = temp_feat1_top[0] + a_last
        #print("this is len list {}".format(len(feat0_1_0)))
        #print("this is len list {}".format(len(feat1_top)))
        feat1_bot1_ano = temp_feat1_bot[0] + b_temp
        #print("mymodift   type fe   {}".format(feat1_bot1.size()))
        #print("tttttttttthis is len list {}".format(len(feat1_top)))
        ## Pass features through Decoder of Stage 1
        temp_feat1_top[0] = feat1_top1_ano
        temp_feat1_bot[0] = feat1_bot1_ano
        #print("407")
        res1_top = self.stage1_decoder(temp_feat1_top)
        #print("409")
        res1_bot = self.stage1_decoder(temp_feat1_bot)

        #res1_top += feat0_1_0
        #res1_bot +=feat0_2_1
        #print("sample   type fe   {}".format(type(res1_top)))
        ## Apply Supervised Attention Module (SAM)
        x2top_samfeats, stage1_img_top = self.sam12(res1_top[0], x2top_img)
        #x2top_samfeats, stage1_img_top = res1_top[0], x2top_img
        #print("sample   type fe   {}".format(type(x2top_samfeats)))
        #print("sample   type fe   {}".format(type(stage1_img_top)))
        x2bot_samfeats, stage1_img_bot = self.sam12(res1_bot[0], x2bot_img)
        #x2bot_samfeats, stage1_img_bot = res1_bot[0], x2bot_img
        #print("420")
        ## Output image at Stage 1
        stage1_img = torch.cat([stage1_img_top, stage1_img_bot],2)
        #print("ooooooooooo  {}".format(type(stage1_img)))
        ##-------------------------------------------
        ##-------------- Stage 2---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        x2top  = self.shallow_feat2(x2top_img)
        x2bot  = self.shallow_feat2(x2bot_img)
        #print("430")
        ## Concatenate SAM features of Stage 1 with shallow features of Stage 2
        x2top_cat = self.concat12(torch.cat([x2top, x2top_samfeats], 1))
        x2bot_cat = self.concat12(torch.cat([x2bot, x2bot_samfeats], 1))

        ## Process features of both patches with Encoder of Stage 2
        feat2_top = self.stage2_encoder(x2top_cat, feat1_top, res1_top)
        feat2_bot = self.stage2_encoder(x2bot_cat, feat1_bot, res1_bot)

        ## Concat deep features
        feat2 = [torch.cat((k,v), 2) for k,v in zip(feat2_top,feat2_bot)]

        ## Pass features through Decoder of Stage 2
        res2 = self.stage2_decoder(feat2)
        #print("444")
        ## Apply SAM
        # print("ssssssssssssaaaaaaaaaaaaaaammmmmmmmmmm")
        # print(res2[0])
        # print("res20 type {}".format(type(res2)))
        x3_samfeats, stage2_img = self.sam23(res2[0], x3_img)
        #x3_samfeats, stage2_img = self.sam23(feat0_1_0_0[0], x3_img)
        #print("451")
        ##-------------------------------------------
        ##-------------- Stage 3---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        x3     = self.shallow_feat3(x3_img)
        #print("x3 {}".format(x3))
        #x3 +=feat0_1_0_0
        #print("x3 {}".format(feat0_1_0_0))
        ## Concatenate SAM features of Stage 2 with shallow features of Stage 3
        x3_cat = self.concat23(torch.cat([x3, x3_samfeats], 1))
        # print("5555555555555555555555555555555555555555555555555555")
        #print(type(x3_cat))
        #print(type(x3_cat.size()))

        #a= torch.tensor(feat0_1_0_0)
        #print("a tyep {}".format(a.size()))
        x3_cat = self.stage3_orsnet(x3_cat, feat2, res2)

        stage3_img = self.tail(x3_cat)
        #print("471")
        #a=torch.tensor(feat0_1_0_0)
        #print("this is typea {}".format(type(a)))
        #print("this type x3{}".format(x3_cat))
        return [stage3_img+x3_img, stage2_img, stage1_img]

if __name__ == '__main__':
    import os
    import time
    import torch
    from torchvision import transforms
    from PIL import Image

    image = Image.open('input.jpg')  # 这张图片假如是rgb三通道的，分辨率是180*180
    print(image.size)
    toTensor = transforms.ToTensor()  # 实例化一个toTensor
    image_tensor = toTensor(image)
    print(image_tensor.size())  # 这里就会输出[3,180,180]
    # 一般我们训练好的模型去detect需要把这个图像转成4维度
    input = image_tensor.reshape(1, 3, 720, 1280)
    #input = image_tensor
    #os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    #input = torch.rand([1,3,256,256])
    print(type(input))
    start = time.time()
    model = Net()
    _ = model(input)
    print(time.time() - start)
    print('verify success')
