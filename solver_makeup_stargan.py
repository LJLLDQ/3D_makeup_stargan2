
import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.nn.init as init
from torch.autograd import Variable
from torchvision.utils import save_image
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm as tqdm

# tools
from model.Deep3DFaceRecon_pytorch.models.bfm import ParametricFaceModel
from model.Deep3DFaceRecon_pytorch.models.networks import ReconNetWrapper
from model.arcface_torch.backbones.iresnet import iresnet100

# models
import net_stargan_org
import discriminator
from ops.histogram_matching import *
from ops.loss_added import GANLoss




class Solver_makeupGAN(object):
    def __init__(self, data_loaders, config,dataset_config):
        
        self.l1 = nn.L1Loss()
        self.log_writer = SummaryWriter()
        self.num_epochs = config.num_epochs
        self.data_loader_train = data_loaders[0]        
        self.criterionL1 = torch.nn.L1Loss()
        self.lambda_A = config.lambda_A
        self.norm = config.norm
        self.criterionGAN = GANLoss(use_lsgan=True, tensor =torch.cuda.FloatTensor)

        self.vgg=models.vgg16(pretrained=True)
        self.vgg.cuda()

        self.criterionL2 = torch.nn.MSELoss()
        self.batch_size = config.batch_size
        # lr from the config
        self.g_lr = config.G_LR
        self.d_lr = config.D_LR
        self.snapshot_step = config.snapshot_step
        self.vis_step = config.vis_step
        # self.MultiScaleGANLoss = MultiScaleGANLoss()

        self.task_name = config.task_name
        self.snapshot_path = config.snapshot_path + config.task_name
        self.lambda_idt = config.lambda_idt
        self.lambda_his_lip = config.lambda_his_lip
        self.lambda_his_eye = config.lambda_his_eye
        self.lambda_his_brow = config.lambda_his_brow
        self.lambda_his_skin = config.lambda_his_skin
        self.lambda_vgg = config.lambda_vgg
        # Model hyper-parameters
        self.img_size = config.img_size
        self.d_conv_dim = config.d_conv_dim
        self.d_repeat_num = config.d_repeat_num
        self.lips = config.lips
        self.skin = config.skin
        self.eye = config.eye
        self.brow = config.brow    

        if not os.path.exists(self.snapshot_path):
            os.makedirs(self.snapshot_path)       
        self.build_model()


        # sid loss
        # self.f_3d_checkpoint_path = '/home/jl/ECCV_fighting/model/Deep3DFaceRecon_pytorch/checkpoints/epoch_20.pth'
        # self.f_id_checkpoint_path = '/home/jl/ECCV_fighting/model/Deep3DFaceRecon_pytorch/ms1mv3_arcface_r100_fp16_backbone.pth'

        # self.f_3d = ReconNetWrapper(net_recon='resnet50', use_last_fc=False)
        # self.f_3d.load_state_dict(torch.load(self.f_3d_checkpoint_path, map_location='cpu')['net_recon'])
        # self.f_3d.eval()
        # self.f_3d.cuda()
        # self.face_model = ParametricFaceModel()

        # self.f_id = iresnet100(pretrained=False, fp16=False)
        # self.f_id.load_state_dict(torch.load(self.f_id_checkpoint_path, map_location='cpu'))
        # self.f_id.eval()
        # self.f_id.cuda()

    def weights_init_xavier(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            init.xavier_normal(m.weight.data, gain=1.0)
        elif classname.find('Linear') != -1:
            init.xavier_normal(m.weight.data, gain=1.0)

    def build_model(self):

        self.StyleEncoder = net_stargan_org.StyleEncoder().cuda()
        self.G = net_stargan_org.Generator().cuda()
        self.D_A = discriminator.Discriminator(self.img_size, self.d_conv_dim, self.d_repeat_num, self.norm).cuda()
        self.D_B = discriminator.Discriminator(self.img_size, self.d_conv_dim, self.d_repeat_num, self.norm).cuda()
        # initialization
        self.G.apply(self.weights_init_xavier)
        self.StyleEncoder.apply(self.weights_init_xavier)
        self.D_A.apply(self.weights_init_xavier)
        self.D_B.apply(self.weights_init_xavier)


        # self.load_checkpoint()
        self.criterionL1 = torch.nn.L1Loss()
        self.criterionL2 = torch.nn.MSELoss()
        self.criterionGAN = GANLoss(use_lsgan=True, tensor=torch.cuda.FloatTensor)

        # self.vgg = net_stargan_org.vgg16(pretrained=True)

        # Optimizers
        self.StyleEncoder_optimizer = torch.optim.Adam(self.StyleEncoder.parameters(), self.g_lr, [0.0, 0.999])        
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [0.0, 0.999])
        self.d_A_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D_A.parameters()), self.d_lr, [0.5, 0.999])
        self.d_B_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D_B.parameters()), self.d_lr, [0.5, 0.999])



        # if torch.cuda.is_available():
            # self.device = "cuda"
            # self.G.cuda()
            # self.vgg.cuda()
            # self.criterionHis.cuda()
            # self.criterionGAN.cuda()
            # self.criterionL1.cuda()
            # self.criterionL2.cuda()
            # self.D_A.cuda()
            # self.D_B.cuda()        




    # def save_models(self):
        # torch.save(self.Generator.state_dict(), os.path.join(self.snapshot_path, '{}_{}_G.pth'.format(self.e + 1, self.i + 1)))
        # torch.save(getattr(self, "D").state_dict(), os.path.join(self.snapshot_path, '{}_{}_D.pth'.format(self.e + 1, self.i + 1)))



    def save_models(self):
        # if not osp.exists(self.snapshot_path):
            # os.makedirs(self.snapshot_path)
        torch.save(self.G.state_dict(),os.path.join(self.snapshot_path, '{}_{}_G.pth'.format(self.e + 1, self.i + 1)))

        torch.save(self.D_A.state_dict(),os.path.join(self.snapshot_path, '{}_{}_D_A.pth'.format(self.e + 1, self.i + 1)))

        torch.save(self.D_B.state_dict(),os.path.join(self.snapshot_path, '{}_{}_D_B.pth'.format(self.e + 1, self.i + 1)))        


    def vgg_forward(self, model, x):
        for i in range(18):
            x=model.features[i](x)
        return x

    def to_var(self, x, requires_grad=True):
        if torch.cuda.is_available():
            x = x.cuda()
            # print(1)
        if not requires_grad:
            return Variable(x, requires_grad=requires_grad)
        else:
            # print(1)
            return Variable(x)        

    def mask_preprocess(self, mask_A, mask_B):
        index_tmp = mask_A.nonzero()
        x_A_index = index_tmp[:, 2]
        y_A_index = index_tmp[:, 3]
        index_tmp = mask_B.nonzero()
        x_B_index = index_tmp[:, 2]
        y_B_index = index_tmp[:, 3]
        mask_A = self.to_var(mask_A, requires_grad=False)
        mask_B = self.to_var(mask_B, requires_grad=False)
        index = [x_A_index, y_A_index, x_B_index, y_B_index]
        index_2 = [x_B_index, y_B_index, x_A_index, y_A_index]
        return mask_A, mask_B, index, index_2       

    def criterionHis(self, input_data, target_data, mask_src, mask_tar, index):
        input_data = (self.de_norm(input_data) * 255).squeeze()
        target_data = (self.de_norm(target_data) * 255).squeeze()
        mask_src = mask_src.expand(1, 3, mask_src.size(2), mask_src.size(2)).squeeze()
        mask_tar = mask_tar.expand(1, 3, mask_tar.size(2), mask_tar.size(2)).squeeze()
        # print(mask_src.shape)
        # exit()
        input_masked = input_data * mask_src
        target_masked = target_data * mask_tar

        input_match = histogram_matching(input_masked, target_masked, index) # 取下来的部分
        input_match = self.to_var(input_match, requires_grad=False)
        loss = self.criterionL1(input_masked, input_match)
        return loss             

    def de_norm(self, x):
        out = (x + 1) / 2
        return out.clamp(0, 1)

    def train(self):
        """Train StarGAN within a single dataset."""

        for self.e in tqdm(range(0, self.num_epochs)):
            for self.i, (img_A, img_B, mask_A, mask_B) in enumerate(tqdm(self.data_loader_train)): # img_A = non-makeup, img_B = makeup

                org_A = self.to_var(img_A, requires_grad=False)
                ref_B = self.to_var(img_B, requires_grad=False)
   
                # ======================================================================== Train D ======================================================================== #
                ref_B_feature = self.StyleEncoder(ref_B)
                fake_A = self.G(org_A, ref_B_feature)                    
                # Real
                out = self.D_A(ref_B)
                d_loss_real = self.criterionGAN(out, True)
                # Fake
                ref_B_feature = self.StyleEncoder(ref_B)
                fake_A = self.G(org_A, ref_B_feature)
                fake_A = Variable(fake_A.data).detach()
                out = self.D_A(fake_A)
                d_loss_fake =  self.criterionGAN(out, False)

                # Backward + Optimize
                d_loss = (d_loss_real + d_loss_fake) * 0.5
                self.d_A_optimizer.zero_grad()
                d_loss.backward(retain_graph=False)
                self.d_A_optimizer.step()


                # training D_B, D_B aims to distinguish class A
                # Real
                out = self.D_B(org_A)
                d_loss_real = self.criterionGAN(out, True)
                # Fake
                org_A_feature = self.StyleEncoder(org_A)
                fake_B = self.G(ref_B, org_A_feature)
                fake_B = Variable(fake_B.data).detach()
                out = self.D_B(fake_B)
                d_loss_fake =  self.criterionGAN(out, False)

                # Backward + Optimize
                d_loss = (d_loss_real + d_loss_fake) * 0.5
                self.d_B_optimizer.zero_grad()
                d_loss.backward(retain_graph=False)
                self.d_B_optimizer.step()                


                # ======================================================================== Train G ======================================================================== #

                # # ----------------------------------------------------------------------- identity loss -----------------------------------------------------------------------
                ref_B_feature = self.StyleEncoder(ref_B)
                org_A_feature = self.StyleEncoder(org_A)
                
                idt_org_A = self.G(org_A, org_A_feature) 
                idt_ref_B = self.G(ref_B, ref_B_feature)
                                          
                loss_idt_A1 = self.criterionL1(idt_org_A, org_A) 
                loss_idt_B1 = self.criterionL1(idt_ref_B, ref_B)

                # loss_idt
                id_loss = (loss_idt_A1 + loss_idt_B1) * 0.5

                # ----------------------------------------------------------------------- GAN loss -----------------------------------------------------------------------
                ref_B_feature = self.StyleEncoder(ref_B)
                fake_A = self.G(org_A, ref_B_feature)   
                pred_fake = self.D_A(fake_A)
                g_A_loss_adv = self.criterionGAN(pred_fake, True)

                org_A_feature = self.StyleEncoder(org_A)
                fake_B = self.G(ref_B, org_A_feature)

                pred_fake = self.D_B(fake_B)
                g_B_loss_adv = self.criterionGAN(pred_fake, True)

                gan_loss = (g_A_loss_adv + g_B_loss_adv) * 0.5

                # ----------------------------------------------------------------------- cycle loss -----------------------------------------------------------------------

                rec_A = self.G(fake_A, org_A_feature)
                rec_B = self.G(fake_B, ref_B_feature)

                g_loss_rec_A = self.criterionL1(rec_A, org_A)
                g_loss_rec_B = self.criterionL1(rec_B, ref_B)

                cycle_loss = (g_loss_rec_A + g_loss_rec_B) * 0.5

                # ----------------------------------------------------------------------- vgg loss -----------------------------------------------------------------------
                # # norm 一下
 
                vgg_org = self.vgg_forward(self.vgg,org_A)
                vgg_org = Variable(vgg_org.data).detach()
                vgg_fake_A = self.vgg_forward(self.vgg,fake_A)
                g_loss_A_vgg = self.criterionL2(vgg_fake_A, vgg_org)

                vgg_ref = self.vgg_forward(self.vgg, ref_B)
                vgg_ref = Variable(vgg_ref.data).detach()
                vgg_fake_B = self.vgg_forward(self.vgg,fake_B)
                g_loss_B_vgg = self.criterionL2(vgg_fake_B, vgg_ref)

                vgg_loss = (g_loss_A_vgg + g_loss_B_vgg) * 0.5

                # # # ----------------------------------------------------------------------------------------- color_histogram loss -----------------------------------------------------------------------------------------

                # # # # Convert tensor to variable
                # # # # 老位置
                # # # # mask attribute: 0:background 1:face 2:left-eyebrown 3:right-eyebrown 4:left-eye 5: right-eye 6: nose 
                # # # # 7: upper-lip 8: teeth 9: under-lip 10:hair 11: left-ear 12: right-ear 13: neck
                # # # # 新位置
                # # # # atts = [1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye', 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r',
                # # # #         10 'nose', 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']

                makeup_loss_A = 0
                makeup_loss_B = 0

                            
                # if self.lips==True:
                mask_A_lip = (mask_A==12).float() + (mask_A==13).float() # 上唇+下唇
                mask_B_lip = (mask_B==12).float() + (mask_B==13).float() # 上唇+下唇                     
                mask_A_lip, mask_B_lip, index_A_lip, index_B_lip = self.mask_preprocess(mask_A_lip, mask_B_lip)
                g_A_lip_loss_his = self.criterionHis(fake_A, ref_B, mask_A_lip, mask_B_lip, index_A_lip)
                g_B_lip_loss_his = self.criterionHis(fake_B, org_A, mask_B_lip, mask_A_lip, index_B_lip)
                makeup_loss_A += g_A_lip_loss_his
                makeup_loss_B += g_B_lip_loss_his

                # if self.brow==True:
                #     mask_A_brow = (mask_A==2).float() + (mask_A==3).float() # 左眉+右眉
                #     mask_B_brow = (mask_B==2).float() + (mask_B==3).float() # 左眉+右眉
                #     mask_A_brow, mask_B_brow, index_A_brow, index_B_brow = self.mask_preprocess(mask_A_brow, mask_B_brow)    
                #     g_A_brow_loss_his = self.criterionHis(fake_A, ref_B, mask_A_brow, mask_B_brow, index_A_brow) * 2
                #     makeup_loss += g_A_brow_loss_his

                # # skin loss 现在用这个，区别在于g_A_skin_loss_his的计算，fake_A的计算对象从ref_B变成了org_A
                # if self.skin==True:
                mask_A_skin = (mask_A==1).float() + (mask_A==10).float() + (mask_A==14).float() # 脸+鼻子+脖子
                mask_B_skin = (mask_B==1).float() + (mask_B==10).float() + (mask_B==14).float() # 脸+鼻子+脖子
                mask_A_skin, mask_B_skin, index_A_skin, index_B_skin = self.mask_preprocess(mask_A_skin, mask_B_skin)
                # 计算loss
                g_A_skin_loss_his = self.criterionHis(fake_A, ref_B, mask_A_skin, mask_B_skin, index_A_skin)
                g_B_skin_loss_his = self.criterionHis(fake_B, org_A, mask_B_skin, mask_A_skin, index_B_skin)

                makeup_loss_A += g_A_skin_loss_his
                makeup_loss_B += g_B_skin_loss_his

                # if self.eye==True:
                mask_A_eye_left = (mask_A==4).float() # 左眼
                mask_A_eye_right = (mask_A==5).float() # 右眼
                mask_B_eye_left = (mask_B==4).float() # 左眼
                mask_B_eye_right = (mask_B==5).float() # 右眼

                # 过mask_preprocess
                mask_A_eye_left, mask_B_eye_left, index_A_eye_left, index_B_eye_left = self.mask_preprocess(mask_A_eye_left, mask_B_eye_left) # 左眼
                mask_A_eye_right, mask_B_eye_right, index_A_eye_right, index_B_eye_right = self.mask_preprocess(mask_A_eye_right, mask_B_eye_right) # 右眼

                # 计算loss
                g_A_eye_left_loss_his = self.criterionHis(fake_A, ref_B, mask_A_eye_left, mask_B_eye_left, index_A_eye_left)
                g_A_eye_right_loss_his = self.criterionHis(fake_A, ref_B, mask_A_eye_right, mask_B_eye_right, index_A_eye_right)

                g_B_eye_left_loss_his = self.criterionHis(fake_B, org_A, mask_B_eye_left, mask_A_eye_left, index_B_eye_left)
                g_B_eye_right_loss_his = self.criterionHis(fake_B, org_A, mask_B_eye_right, mask_A_eye_right, index_B_eye_right)

                makeup_loss_A += g_A_eye_left_loss_his + g_A_eye_right_loss_his
                makeup_loss_B += g_B_eye_right_loss_his + g_B_eye_right_loss_his

                # ----------------------------------------------------------------------- FINAl loss -----------------------------------------------------------------------
 
                g_loss = 5 * id_loss + gan_loss + 5 * cycle_loss  + makeup_loss_A + makeup_loss_B + vgg_loss * 5e-2

                # tensorboard
                self.log_writer.add_scalar('normal/id_loss', float(id_loss), self.i)
                self.log_writer.add_scalar('normal/cycle_loss', float(cycle_loss), self.i)
                self.log_writer.add_scalar('normal/gan_loss', float(gan_loss), self.i)
                # self.log_writer.add_scalar('normal/vgg_loss', float(vgg_loss), self.i)
                self.log_writer.add_scalar('normal/makeup_loss_A', float(makeup_loss_A), self.i)
                self.log_writer.add_scalar('normal/makeup_loss_B', float(makeup_loss_B), self.i)
                

                # self.log_writer.add_scalar('/normal/g_A_loss_his_lip', float(g_A_loss_his_lip), self.i)
                # self.log_writer.add_scalar('/normal/g_A_loss_his_brow', float(g_A_loss_his_brow), self.i)
                # self.log_writer.add_scalar('/normal/g_A_loss_his_skin', float(g_A_loss_his_skin), self.i)
                # self.log_writer.add_scalar('/normal/g_A_loss_his_eye', float(g_A_loss_his_eye), self.i)

                # 优化

                self.g_optimizer.zero_grad()
                self.StyleEncoder_optimizer.zero_grad()
                g_loss.backward()            
                # clip step
                torch.nn.utils.clip_grad_norm_(self.G.parameters(), 5)                
                self.g_optimizer.step()
                self.StyleEncoder_optimizer.step()

                # save iamge
                if (self.i + 1) % self.vis_step == 0:
                    image_list = []
                    fid_list = []
                    image_list.append(org_A)
                    image_list.append(ref_B) 
                    self.StyleEncoder.eval()                      
                    self.G.eval()
                    with torch.no_grad():   
                        # org_A_feature = self.StyleEncoder(org_A)
                        ref_B_feature = self.StyleEncoder(ref_B)
                        fake_A = self.G(org_A, ref_B_feature) # fake_A是妆后图

                    # print(fake_A.shape)
                    image_list.append(fake_A)
                    fid_list.append(fake_A)

                    image_list = torch.cat(image_list, dim=3)
                    fid_list = torch.cat(fid_list, dim=1)

                    save_path = os.path.join('./stargan_result/{}.jpg'.format(self.i + 1))                    
                    save_path_fid = os.path.join('./stargan_result_fid/{}.jpg'.format(self.i + 1))

                    save_image(self.de_norm(image_list.data), save_path, nrow=1, padding=0, normalize=True)
                    save_image(self.de_norm(fid_list.data), save_path_fid, nrow=1, padding=0, normalize=True)



            # Save model checkpoints
            # if (self.e + 1) % self.snapshot_step == 0:
            self.save_models() # save model for each epoch






