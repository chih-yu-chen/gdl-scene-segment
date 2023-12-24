from pathlib import Path
import torch
from torch import nn
import torchsparse
import torchsparse.nn as spnn
from torchsparse import SparseTensor
from torch_scatter import scatter_max

import sys
pkg_path = Path(__file__).parents[2]/ "diffusion-net"/ "src"
sys.path.append(str(pkg_path))
import diffusion_net.layers as DiffNetLayers


class EuclideanBranch(nn.Module):

    def __init__(self,
                 c_in:int) -> None:
        
        super().__init__()
        
        # Unet-like structure
        #----- Input -----
        self.input_conv = nn.Sequential(
            spnn.Conv3d(c_in, 32, 3, 1),
            spnn.BatchNorm(32),
            spnn.ReLU(True)
        )

        #----- Encoder -----
        # Level 1
        self.enc_1 = nn.Sequential(
            spnn.Conv3d(32, 32, 3, 1),
            spnn.BatchNorm(32),
            spnn.ReLU(True),
            spnn.Conv3d(32, 32, 3, 1),
            spnn.BatchNorm(32),
            spnn.ReLU(True)
        )

        self.down_1 = nn.Sequential(
            spnn.Conv3d(32, 64, 2, 2),
            spnn.BatchNorm(64),
            spnn.ReLU(True)
        )

        # Level 2
        self.enc_2 = nn.Sequential(
            spnn.Conv3d(64, 64, 3, 1),
            spnn.BatchNorm(64),
            spnn.ReLU(True),
            spnn.Conv3d(64, 64, 3, 1),
            spnn.BatchNorm(64),
            spnn.ReLU(True)
        )

        self.down_2 = nn.Sequential(
            spnn.Conv3d(64, 96, 2, 2),
            spnn.BatchNorm(96),
            spnn.ReLU(True)
        )

        # Level 3
        self.enc_3 = nn.Sequential(
            spnn.Conv3d(96, 96, 3, 1),
            spnn.BatchNorm(96),
            spnn.ReLU(True),
            spnn.Conv3d(96, 96, 3, 1),
            spnn.BatchNorm(96),
            spnn.ReLU(True)
        )

        self.down_3 = nn.Sequential(
            spnn.Conv3d(96, 128, 2, 2),
            spnn.BatchNorm(128),
            spnn.ReLU(True)
        )

        #----- Middle -----
        self.mid = nn.Sequential(
            spnn.Conv3d(128, 128, 3, 1),
            spnn.BatchNorm(128),
            spnn.ReLU(True),
            spnn.Conv3d(128, 128, 3, 1),
            spnn.BatchNorm(128),
            spnn.ReLU(True)
        )

        #----- Decoder -----
        # Level 3
        self.up_3 = nn.Sequential(
            spnn.Conv3d(128, 96, 2, 2, transposed=True),
            spnn.BatchNorm(96),
            spnn.ReLU(True)
        )

        self.lin_net_3 = spnn.Conv3d(192, 96, kernel_size=1, stride=1, bias=False)
        self.dec_3 = nn.Sequential(
            spnn.Conv3d(192, 96, 3, 1),
            spnn.BatchNorm(96),
            spnn.ReLU(True),
            spnn.Conv3d(96, 96, 3, 1),
            spnn.BatchNorm(96),
            spnn.ReLU(True)
        )

        # Level 2
        self.up_2 = nn.Sequential(
            spnn.Conv3d(96, 64, 2, 2, transposed=True),
            spnn.BatchNorm(64),
            spnn.ReLU(True)
        )

        self.lin_net_2 = spnn.Conv3d(128, 64, kernel_size=1, stride=1, bias=False)
        self.dec_2 = nn.Sequential(
            spnn.Conv3d(128, 64, 3, 1),
            spnn.BatchNorm(64),
            spnn.ReLU(True),
            spnn.Conv3d(64, 64, 3, 1),
            spnn.BatchNorm(64),
            spnn.ReLU(True)
        )

        # Level 1
        self.up_1 = nn.Sequential(
            spnn.Conv3d(64, 32, 2, 2, transposed=True),
            spnn.BatchNorm(32),
            spnn.ReLU(True)
        )

        self.lin_net_1 = spnn.Conv3d(64, 32, kernel_size=1, stride=1, bias=False)
        self.dec_1 = nn.Sequential(
            spnn.Conv3d(64, 32, 3, 1),
            spnn.BatchNorm(32),
            spnn.ReLU(True),
            spnn.Conv3d(32, 32, 3, 1),
            spnn.BatchNorm(32),
            spnn.ReLU(True)
        )

        #----- Output -----
        self.output_layer = spnn.Conv3d(32, 20, kernel_size=1, stride=1, bias=True)

    def forward(self, feats, coords):

        x = SparseTensor(feats, coords)

        #----- Input -----
        x_1 = self.input_conv(x)

        #----- Encoder -----
        # Level 1
        x_enc_1 = self.enc_1(x_1)
        x_enc_1 = x_enc_1 + x_1
        x_2 = self.down_1(x_enc_1)
    
        # Level 2
        x_enc_2 = self.enc_2(x_2)
        x_enc_2 = x_enc_2 + x_2
        x_3 = self.down_2(x_enc_2)

        # Level 3
        x_enc_3 = self.enc_3(x_3)
        x_enc_3 = x_enc_3 + x_3
        x_m = self.down_3(x_enc_3)

        #----- Middle -----
        x_mid = self.mid(x_m)
        x_mid = x_mid + x_m

        #----- Decoder -----
        # Level 3
        y_3 = self.up_3(x_mid)
        y_3 = torchsparse.cat([y_3, x_enc_3])
        y_dec_3 = self.dec_3(y_3)
        y_dec_3 = y_dec_3 + self.lin_net_3(y_3)

        # Level 2
        y_2 = self.up_2(y_dec_3)
        y_2 = torchsparse.cat([y_2, x_enc_2])
        y_dec_2 = self.dec_2(y_2)
        y_dec_2 = y_dec_2 + self.lin_net_2(y_2)

        # Level 1
        y_1 = self.up_1(y_dec_2)
        y_1 = torchsparse.cat([y_1, x_enc_1])
        y_dec_1 = self.dec_1(y_1)
        y_dec_1 = y_dec_1 + self.lin_net_1(y_1)
            
        #----- Output -----
        output = self.output_layer(y_dec_1)

        return output.F



class GeodesicBranch(nn.Module):

    def __init__(self,
                 n_diffnet_blocks: int,
                 n_mlp_hidden: int,
                 dropout:bool,
                 c_in:int,
                 c_out:int,
                 c0:int,
                 c1:int,
                 c2:int,
                 c3:int,
                 c_m:int) -> None:

        super().__init__()

        # Unet-like structure
        #----- Input -----
        # self.input_lin = nn.Linear(c_in, c0)
        self.input_lin = nn.Linear(c_in, c1)

        #----- Encoder -----
        # # Level 0
        # self.enc_0 = [
        #     DiffNetLayers.DiffusionNetBlock(C_width=c0,
        #                                     mlp_hidden_dims=[c0]*n_mlp_hidden,
        #                                     dropout=dropout,
        #                                     diffusion_method='spectral',
        #                                     with_gradient_features=True, 
        #                                     with_gradient_rotations=True)
        # ] * n_diffnet_blocks
        # for i, block in enumerate(self.enc_0):
        #     self.add_module(f"Encoder_L0_DiffusionNetBlock_{i}", block)

        # Level 1
        # self.enc_lin_1 = nn.Linear(c0, c1)
        self.enc_1 = [
            DiffNetLayers.DiffusionNetBlock(C_width=c1,
                                            mlp_hidden_dims=[c1]*n_mlp_hidden,
                                            dropout=dropout,
                                            diffusion_method='spectral',
                                            with_gradient_features=True, 
                                            with_gradient_rotations=True)
        ] * n_diffnet_blocks
        for i, block in enumerate(self.enc_1):
            self.add_module(f"Encoder_L1_DiffusionNetBlock_{i}", block)

        # Level 2
        self.enc_lin_2 = nn.Linear(c1, c2)
        self.enc_2 = [
            DiffNetLayers.DiffusionNetBlock(C_width=c2,
                                            mlp_hidden_dims=[c2]*n_mlp_hidden,
                                            dropout=dropout,
                                            diffusion_method='spectral',
                                            with_gradient_features=True, 
                                            with_gradient_rotations=True)
        ] * n_diffnet_blocks
        for i, block in enumerate(self.enc_2):
            self.add_module(f"Encoder_L2_DiffusionNetBlock_{i}", block)

        # Level 3
        self.enc_lin_3 = nn.Linear(c2, c3)
        self.enc_3 = [
            DiffNetLayers.DiffusionNetBlock(C_width=c3,
                                            mlp_hidden_dims=[c3]*n_mlp_hidden,
                                            dropout=dropout,
                                            diffusion_method='spectral',
                                            with_gradient_features=True, 
                                            with_gradient_rotations=True)
        ] * n_diffnet_blocks
        for i, block in enumerate(self.enc_3):
            self.add_module(f"Encoder_L3_DiffusionNetBlock_{i}", block)

        #----- Middle -----
        self.enc_lin_m = nn.Linear(c3, c_m)
        self.mid = [
            DiffNetLayers.DiffusionNetBlock(C_width=c_m,
                                            mlp_hidden_dims=[c_m]*n_mlp_hidden,
                                            dropout=dropout,
                                            diffusion_method='spectral',
                                            with_gradient_features=True, 
                                            with_gradient_rotations=True)
        ] * n_diffnet_blocks
        for i, block in enumerate(self.mid):
            self.add_module(f"Middle_DiffusionNetBlock_{i}", block)

        #----- Decoder -----
        # Level 3
        self.dec_lin_30 = nn.Linear(c_m, c3)
        self.dec_lin_31 = nn.Linear(c3*2, c3)
        self.dec_3 = [
            DiffNetLayers.DiffusionNetBlock(C_width=c3,
                                            mlp_hidden_dims=[c3]*n_mlp_hidden,
                                            dropout=dropout,
                                            diffusion_method='spectral',
                                            with_gradient_features=True, 
                                            with_gradient_rotations=True)
        ] * n_diffnet_blocks
        for i, block in enumerate(self.dec_3):
            self.add_module(f"Decoder_L3_DiffusionNetBlock_{i}", block)

        # Level 2
        self.dec_lin_20 = nn.Linear(c3, c2)
        self.dec_lin_21 = nn.Linear(c2*2, c2)
        self.dec_2 = [
            DiffNetLayers.DiffusionNetBlock(C_width=c2,
                                            mlp_hidden_dims=[c2]*n_mlp_hidden,
                                            dropout=dropout,
                                            diffusion_method='spectral',
                                            with_gradient_features=True, 
                                            with_gradient_rotations=True)
        ] * n_diffnet_blocks
        for i, block in enumerate(self.dec_2):
            self.add_module(f"Decoder_L2_DiffusionNetBlock_{i}", block)

       # Level 1
        self.dec_lin_10 = nn.Linear(c2, c1)
        self.dec_lin_11 = nn.Linear(c1*2, c1)
        self.dec_1 = [
            DiffNetLayers.DiffusionNetBlock(C_width=c1,
                                            mlp_hidden_dims=[c1]*n_mlp_hidden,
                                            dropout=dropout,
                                            diffusion_method='spectral',
                                            with_gradient_features=True, 
                                            with_gradient_rotations=True)
        ] * n_diffnet_blocks
        for i, block in enumerate(self.dec_1):
            self.add_module(f"Decoder_L1_DiffusionNetBlock_{i}", block)

       # # Level 0
       #  self.dec_lin_00 = nn.Linear(c1, c0)
       #  self.dec_lin_01 = nn.Linear(c0*2, c0)
       #  self.dec_0 = [
       #      DiffNetLayers.DiffusionNetBlock(C_width=c0,
       #                                      mlp_hidden_dims=[c0]*n_mlp_hidden,
       #                                      dropout=dropout,
       #                                      diffusion_method='spectral',
       #                                      with_gradient_features=True, 
       #                                      with_gradient_rotations=True)
       #  ] * n_diffnet_blocks
       #  for i, block in enumerate(self.dec_0):
       #      self.add_module(f"Decoder_L0_DiffusionNetBlock_{i}", block)

        #----- Output -----
        # self.output_lin = nn.Linear(c0, c_out)
        self.output_lin = nn.Linear(c1, c_out)

    def forward(self,
                x_in,
                # mass_0, L_0, evals_0, evecs_0, gradX_0, gradY_0,
                mass_1, L_1, evals_1, evecs_1, gradX_1, gradY_1,
                mass_2, L_2, evals_2, evecs_2, gradX_2, gradY_2,
                mass_3, L_3, evals_3, evecs_3, gradX_3, gradY_3,
                mass_m, L_m, evals_m, evecs_m, gradX_m, gradY_m,
                traces01, traces12, traces23, traces34):

        #----- Input -----
        # x_0 = self.input_lin(x_in)
        x_1 = self.input_lin(x_in)

        #----- Encoder -----
        # # Level 0
        # x_enc_0 = self.enc_0[0](x_0, mass_0, L_0, evals_0, evecs_0, gradX_0, gradY_0)
        # for block in self.enc_0[1:]:
        #     x_enc_0 = block(x_enc_0, mass_0, L_0, evals_0, evecs_0, gradX_0, gradY_0)
        # x_enc_0 = x_enc_0 + x_0
        # x_1, _ = scatter_max(x_enc_0, traces01, dim=-2)
        
        # Level 1
        # x_1 = self.enc_lin_1(x_1)
        x_enc_1 = self.enc_1[0](x_1, mass_1, L_1, evals_1, evecs_1, gradX_1, gradY_1)
        for block in self.enc_1[1:]:
            x_enc_1 = block(x_enc_1, mass_1, L_1, evals_1, evecs_1, gradX_1, gradY_1)
        x_enc_1 = x_enc_1 + x_1
        x_2, _ = scatter_max(x_enc_1, traces12, dim=-2)

        # Level 2
        x_2 = self.enc_lin_2(x_2)
        x_enc_2 = self.enc_2[0](x_2, mass_2, L_2, evals_2, evecs_2, gradX_2, gradY_2)
        for block in self.enc_2[1:]:
            x_enc_2 = block(x_enc_2, mass_2, L_2, evals_2, evecs_2, gradX_2, gradY_2)
        x_enc_2 = x_enc_2 + x_2
        x_3, _ = scatter_max(x_enc_2, traces23, dim=-2)

        # Level 3
        x_3 = self.enc_lin_3(x_3)
        x_enc_3 = self.enc_3[0](x_3, mass_3, L_3, evals_3, evecs_3, gradX_3, gradY_3)
        for block in self.enc_3[1:]:
            x_enc_3 = block(x_enc_3, mass_3, L_3, evals_3, evecs_3, gradX_3, gradY_3)
        x_enc_3 = x_enc_3 + x_3
        x_m, _ = scatter_max(x_enc_3, traces34, dim=-2)

        #----- Middle -----
        x_m = self.enc_lin_m(x_m)
        x_mid = self.mid[0](x_m, mass_m, L_m, evals_m, evecs_m, gradX_m, gradY_m)
        for block in self.mid[1:]:
            x_mid = block(x_mid, mass_m, L_m, evals_m, evecs_m, gradX_m, gradY_m)
        x_mid = x_mid + x_m

        #----- Decoder -----
        # Level 3
        y_3 = x_mid[:,traces34,:]
        y_3 = self.dec_lin_30(y_3)
        y_3 = torch.cat((y_3, x_enc_3), dim=-1)
        y_3 = self.dec_lin_31(y_3)
        y_dec_3 = self.dec_3[0](y_3, mass_3, L_3, evals_3, evecs_3, gradX_3, gradY_3)
        for block in self.dec_3[1:]:
            y_dec_3 = block(y_dec_3, mass_3, L_3, evals_3, evecs_3, gradX_3, gradY_3)
        y_dec_3 = y_dec_3 + y_3

        # Level 2
        y_2 = y_dec_3[:,traces23,:]
        y_2 = self.dec_lin_20(y_2)
        y_2 = torch.cat((y_2, x_enc_2), dim=-1)
        y_2 = self.dec_lin_21(y_2)
        y_dec_2 = self.dec_2[0](y_2, mass_2, L_2, evals_2, evecs_2, gradX_2, gradY_2)
        for block in self.dec_2[1:]:
            y_dec_2 = block(y_dec_2, mass_2, L_2, evals_2, evecs_2, gradX_2, gradY_2)
        y_dec_2 = y_dec_2 + y_2

        # Level 1
        y_1 = y_dec_2[:,traces12,:]
        y_1 = self.dec_lin_10(y_1)
        y_1 = torch.cat((y_1, x_enc_1), dim=-1)
        y_1 = self.dec_lin_11(y_1)
        y_dec_1 = self.dec_1[0](y_1, mass_1, L_1, evals_1, evecs_1, gradX_1, gradY_1)
        for block in self.dec_1[1:]:
            y_dec_1 = block(y_dec_1, mass_1, L_1, evals_1, evecs_1, gradX_1, gradY_1)
        y_dec_1 = y_dec_1 + y_1

        # # Level 0
        # y_0 = y_dec_1[:,traces01,:]
        # y_0 = self.dec_lin_00(y_0)
        # y_0 = torch.cat((y_0, x_enc_0), dim=-1)
        # y_0 = self.dec_lin_01(y_0)
        # y_dec_0 = self.dec_0[0](y_0, mass_0, L_0, evals_0, evecs_0, gradX_0, gradY_0)
        # for block in self.dec_0[1:]:
        #     y_dec_0 = block(y_dec_0, mass_0, L_0, evals_0, evecs_0, gradX_0, gradY_0)
        # y_dec_0 = y_dec_0 + y_0

        #----- Output -----
        # y = self.output_lin(y_dec_0)
        y = self.output_lin(y_dec_1)

        return y

class DiffusionVoxelNet(nn.Module):

    def __init__(self,
                 n_diffnet_blocks,
                 n_mlp_hidden, dropout,
                 c_in, c_out, c0, c1, c2, c3, c_m
        ) -> None:

        super().__init__()

        self.EuclideanBranch = EuclideanBranch(c_in-3)
        self.GeodesicBranch = GeodesicBranch(
            n_diffnet_blocks,
            n_mlp_hidden, dropout,
            c_in, c_out, c0, c1, c2, c3, c_m
        )

        self.c_in = c_in

    def forward(self,
                x_in, vox_coords, vox_feats,
                # mass_0, L_0, evals_0, evecs_0, gradX_0, gradY_0,
                mass_1, L_1, evals_1, evecs_1, gradX_1, gradY_1,
                mass_2, L_2, evals_2, evecs_2, gradX_2, gradY_2,
                mass_3, L_3, evals_3, evecs_3, gradX_3, gradY_3,
                mass_m, L_m, evals_m, evecs_m, gradX_m, gradY_m,
                traces01, traces12, traces23, traces34):

        """
        x_in:   (B,N,C) or (N,C)
        x_1:    (B,N1,C) or (N1,C)
        traces: (N,)
        """

        ## Check dimensions, and append batch dimension if not given
        if x_in.shape[-1] != self.c_in:
            raise ValueError(f"Channel mismatch: c_in set at {self.c_in}, got x_in with shape {x_in.shape}")
        N = x_in.shape[-2]

        if len(x_in.shape) == 2:
            appended_batch_dim = True

            # add a batch dim to all inputs
            x_in = x_in.unsqueeze(0)
            vox_coords = torch.cat((torch.zeros_like(vox_coords[:,:1], dtype=torch.int),
                                    vox_coords), dim=-1)
            # mass_0 = mass_0.unsqueeze(0)
            mass_1 = mass_1.unsqueeze(0)
            mass_2 = mass_2.unsqueeze(0)
            mass_3 = mass_3.unsqueeze(0)
            mass_m = mass_m.unsqueeze(0)
            # L_0 = L_0.unsqueeze(0)
            L_1 = L_1.unsqueeze(0)
            L_2 = L_2.unsqueeze(0)
            L_3 = L_3.unsqueeze(0)
            L_m = L_m.unsqueeze(0)
            # evals_0 = evals_0.unsqueeze(0)
            evals_1 = evals_1.unsqueeze(0)
            evals_2 = evals_2.unsqueeze(0)
            evals_3 = evals_3.unsqueeze(0)
            evals_m = evals_m.unsqueeze(0)
            # evecs_0 = evecs_0.unsqueeze(0)
            evecs_1 = evecs_1.unsqueeze(0)
            evecs_2 = evecs_2.unsqueeze(0)
            evecs_3 = evecs_3.unsqueeze(0)
            evecs_m = evecs_m.unsqueeze(0)
            # gradX_0 = gradX_0.unsqueeze(0)
            gradX_1 = gradX_1.unsqueeze(0)
            gradX_2 = gradX_2.unsqueeze(0)
            gradX_3 = gradX_3.unsqueeze(0)
            gradX_m = gradX_m.unsqueeze(0)
            # gradY_0 = gradY_0.unsqueeze(0)
            gradY_1 = gradY_1.unsqueeze(0)
            gradY_2 = gradY_2.unsqueeze(0)
            gradY_3 = gradY_3.unsqueeze(0)
            gradY_m = gradY_m.unsqueeze(0)

        elif len(x_in.shape) == 3:
            appended_batch_dim = False
        
        else:
            raise ValueError(f"x_in should be tensor with shape [N,C] or [B,N,C], got {x_in.shape}")

        euc_out = self.EuclideanBranch(vox_feats, vox_coords)
        geo_out = self.GeodesicBranch(x_in,
            # mass_0, L_0, evals_0, evecs_0, gradX_0, gradY_0,
            mass_1, L_1, evals_1, evecs_1, gradX_1, gradY_1,
            mass_2, L_2, evals_2, evecs_2, gradX_2, gradY_2,
            mass_3, L_3, evals_3, evecs_3, gradX_3, gradY_3,
            mass_m, L_m, evals_m, evecs_m, gradX_m, gradY_m,
            traces01, traces12, traces23, traces34
        )

        if appended_batch_dim:
            geo_out = geo_out.squeeze(0)

        return euc_out, geo_out
