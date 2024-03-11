from pathlib import Path
import torch
from torch import nn
from torch_scatter import scatter_max

import sys
pkg_path = Path(__file__).parents[2]/ "diffusion-net"/ "src"
sys.path.append(str(pkg_path))
import diffusion_net.layers as DiffNetLayers



class GeodesicBranch(nn.Module):

    def __init__(self,
                 n_diffnet_blocks: int,
                 n_mlp_hidden: int,
                 dropout:bool,
                 c_in:int,
                 c_out:int,
                #  c1:int,
                #  c2:int,
                 c3:int,
                 c_m:int) -> None:

        super().__init__()

        # Unet-like structure
        #----- Input -----
        self.input_linear = nn.Sequential(
            nn.Linear(c_in, c3),
            nn.ReLU()
        )

        #----- Encoder -----
        # # Level 1
        # self.enc_diffusion_1 = []
        # for i in range(n_diffnet_blocks):
        #     block = DiffNetLayers.DiffusionNetBlock(C_width=c1,
        #                                             mlp_hidden_dims=[c1]*n_mlp_hidden,
        #                                             dropout=dropout,
        #                                             diffusion_method='spectral',
        #                                             with_gradient_features=True,
        #                                             with_gradient_rotations=True)
        #     self.enc_diffusion_1.append(block)
        #     self.add_module(f"Encoder_L1_DiffusionNetBlock_{i}", self.enc_diffusion_1[-1])
        # self.enc_widen_1 = nn.Sequential(
        #     nn.Linear(c1, c2),
        #     nn.ReLU()
        # )

        # # Level 2
        # self.enc_diffusion_2 = []
        # for i in range(n_diffnet_blocks):
        #     block = DiffNetLayers.DiffusionNetBlock(C_width=c2,
        #                                             mlp_hidden_dims=[c2]*n_mlp_hidden,
        #                                             dropout=dropout,
        #                                             diffusion_method='spectral',
        #                                             with_gradient_features=True,
        #                                             with_gradient_rotations=True)
        #     self.enc_diffusion_2.append(block)
        #     self.add_module(f"Encoder_L2_DiffusionNetBlock_{i}", self.enc_diffusion_2[-1])
        # self.enc_widen_2 = nn.Sequential(
        #     nn.Linear(c2, c3),
        #     nn.ReLU()
        # )

        # Level 3
        self.enc_diffusion_3 = []
        for i in range(n_diffnet_blocks):
            block = DiffNetLayers.DiffusionNetBlock(C_width=c3,
                                                    mlp_hidden_dims=[c3]*n_mlp_hidden,
                                                    dropout=dropout,
                                                    diffusion_method='spectral',
                                                    with_gradient_features=True,
                                                    with_gradient_rotations=True)
            self.enc_diffusion_3.append(block)
            self.add_module(f"Encoder_L3_DiffusionNetBlock_{i}", self.enc_diffusion_3[-1])
        self.enc_widen_3 = nn.Sequential(
            nn.Linear(c3, c_m),
            nn.ReLU()
        )

        #----- Middle -----
        self.mid_diffusion = []
        for i in range(n_diffnet_blocks):
            block = DiffNetLayers.DiffusionNetBlock(C_width=c_m,
                                                    mlp_hidden_dims=[c_m]*n_mlp_hidden,
                                                    dropout=dropout,
                                                    diffusion_method='spectral',
                                                    with_gradient_features=True,
                                                    with_gradient_rotations=True)
            self.mid_diffusion.append(block)
            self.add_module(f"Middle_DiffusionNetBlock_{i}", self.mid_diffusion[-1])

        #----- Decoder -----
        # Level 3
        self.dec_narrow_3 = nn.Sequential(
            nn.Linear(c_m, c3),
            nn.ReLU()
        )
        self.dec_halve_3 = nn.Sequential(
            nn.Linear(c3*2, c3),
            nn.ReLU()
        )
        self.dec_diffusion_3 = []
        for i in range(n_diffnet_blocks):
            block = DiffNetLayers.DiffusionNetBlock(C_width=c3,
                                                    mlp_hidden_dims=[c3]*n_mlp_hidden,
                                                    dropout=dropout,
                                                    diffusion_method='spectral',
                                                    with_gradient_features=True,
                                                    with_gradient_rotations=True)
            self.dec_diffusion_3.append(block)
            self.add_module(f"Decoder_L3_DiffusionNetBlock_{i}", self.dec_diffusion_3[-1])

        # # Level 2
        # self.dec_narrow_2 = nn.Sequential(
        #     nn.Linear(c3, c2),
        #     nn.ReLU()
        # )
        # self.dec_halve_2 = nn.Sequential(
        #     nn.Linear(c2*2, c2),
        #     nn.ReLU()
        # )
        # self.dec_diffusion_2 = []
        # for i in range(n_diffnet_blocks):
        #     block = DiffNetLayers.DiffusionNetBlock(C_width=c2,
        #                                             mlp_hidden_dims=[c2]*n_mlp_hidden,
        #                                             dropout=dropout,
        #                                             diffusion_method='spectral',
        #                                             with_gradient_features=True,
        #                                             with_gradient_rotations=True)
        #     self.dec_diffusion_2.append(block)
        #     self.add_module(f"Decoder_L2_DiffusionNetBlock_{i}", self.dec_diffusion_2[-1])

        # # Level 1
        # self.dec_narrow_1 = nn.Sequential(
        #     nn.Linear(c2, c1),
        #     nn.ReLU()
        # )
        # self.dec_halve_1 = nn.Sequential(
        #     nn.Linear(c1*2, c1),
        #     nn.ReLU()
        # )
        # self.dec_diffusion_1 = []
        # for i in range(n_diffnet_blocks):
        #     block = DiffNetLayers.DiffusionNetBlock(C_width=c1,
        #                                             mlp_hidden_dims=[c1]*n_mlp_hidden,
        #                                             dropout=dropout,
        #                                             diffusion_method='spectral',
        #                                             with_gradient_features=True,
        #                                             with_gradient_rotations=True)
        #     self.dec_diffusion_1.append(block)
        #     self.add_module(f"Decoder_L1_DiffusionNetBlock_{i}", self.dec_diffusion_1[-1])

        #----- Output -----
        self.output_linear = nn.Linear(c3, c_out)

    def forward(self,
                x_in,
                # mass_1, L_1, evals_1, evecs_1, gradX_1, gradY_1,
                # mass_2, L_2, evals_2, evecs_2, gradX_2, gradY_2,
                mass_3, L_3, evals_3, evecs_3, gradX_3, gradY_3,
                mass_m, L_m, evals_m, evecs_m, gradX_m, gradY_m,
                # traces12,
                # traces23,
                traces34
                ):

        #----- Input -----
        x_3 = self.input_linear(x_in)

        #----- Encoder -----
        # # Level 1
        # x_enc1 = self.enc_diffusion_1[0](x_1, mass_1, L_1, evals_1, evecs_1, gradX_1, gradY_1)
        # for block in self.enc_diffusion_1[1:]:
        #     x_enc1 = block(x_enc1, mass_1, L_1, evals_1, evecs_1, gradX_1, gradY_1)
        # x_2, _ = scatter_max(x_enc1, traces12, dim=-2)
        # x_2 = self.enc_widen_1(x_2)

        # # Level 2
        # x_enc2 = self.enc_diffusion_2[0](x_2, mass_2, L_2, evals_2, evecs_2, gradX_2, gradY_2)
        # for block in self.enc_diffusion_2[1:]:
        #     x_enc2 = block(x_enc2, mass_2, L_2, evals_2, evecs_2, gradX_2, gradY_2)
        # x_3, _ = scatter_max(x_enc2, traces23, dim=-2)
        # x_3 = self.enc_widen_2(x_3)

        # Level 3
        x_enc3 = self.enc_diffusion_3[0](x_3, mass_3, L_3, evals_3, evecs_3, gradX_3, gradY_3)
        for block in self.enc_diffusion_3[1:]:
            x_enc3 = block(x_enc3, mass_3, L_3, evals_3, evecs_3, gradX_3, gradY_3)
        x_m, _ = scatter_max(x_enc3, traces34, dim=-2)
        x_m = self.enc_widen_3(x_m)

        #----- Middle -----
        x_mid = self.mid_diffusion[0](x_m, mass_m, L_m, evals_m, evecs_m, gradX_m, gradY_m)
        for block in self.mid_diffusion[1:]:
            x_mid = block(x_mid, mass_m, L_m, evals_m, evecs_m, gradX_m, gradY_m)

        #----- Decoder -----
        # Level 3
        y_3 = x_mid[:,traces34,:]
        y_3 = self.dec_narrow_3(y_3)
        y_3 = torch.cat([y_3, x_enc3], dim=-1)
        y_3 = self.dec_halve_3(y_3)
        y_dec3 = self.dec_diffusion_3[0](y_3, mass_3, L_3, evals_3, evecs_3, gradX_3, gradY_3)
        for block in self.dec_diffusion_3[1:]:
            y_dec3 = block(y_dec3, mass_3, L_3, evals_3, evecs_3, gradX_3, gradY_3)

        # # Level 2
        # y_2 = y_dec3[:,traces23,:]
        # y_2 = self.dec_narrow_2(y_2)
        # y_2 = torch.cat([y_2, x_enc2], dim=-1)
        # y_2 = self.dec_halve_2(y_2)
        # y_dec2 = self.dec_diffusion_2[0](y_2, mass_2, L_2, evals_2, evecs_2, gradX_2, gradY_2)
        # for block in self.dec_diffusion_2[1:]:
        #     y_dec2 = block(y_dec2, mass_2, L_2, evals_2, evecs_2, gradX_2, gradY_2)

        # # Level 1
        # y_1 = y_dec2[:,traces12,:]
        # y_1 = self.dec_narrow_1(y_1)
        # y_1 = torch.cat([y_1, x_enc1], dim=-1)
        # y_1 = self.dec_halve_1(y_1)
        # y_dec1 = self.dec_diffusion_1[0](y_1, mass_1, L_1, evals_1, evecs_1, gradX_1, gradY_1)
        # for block in self.dec_diffusion_1[1:]:
        #     y_dec1 = block(y_dec1, mass_1, L_1, evals_1, evecs_1, gradX_1, gradY_1)

        #----- Output -----
        y = self.output_linear(y_dec3)

        return y

class DiffusionVoxelNet(nn.Module):

    def __init__(self,
                 n_diffnet_blocks,
                 n_mlp_hidden, dropout,
                 c_in,
                 c_out,
                #  c1,
                #  c2,
                 c3,
                 c_m
        ) -> None:

        super().__init__()

        self.c_in = c_in
        self.GeodesicBranch = GeodesicBranch(
            n_diffnet_blocks,
            n_mlp_hidden, dropout,
            c_in, c_out, c3, c_m
        )

    def forward(self,
                x_in,
                # mass_1, L_1, evals_1, evecs_1, gradX_1, gradY_1,
                # mass_2, L_2, evals_2, evecs_2, gradX_2, gradY_2,
                mass_3, L_3, evals_3, evecs_3, gradX_3, gradY_3,
                mass_m, L_m, evals_m, evecs_m, gradX_m, gradY_m,
                # traces12,
                # traces23,
                traces34
                ):

        """
        x_in:   (B,N,C) or (N,C)
        x_1:    (B,N1,C) or (N1,C)
        traces: (N,)
        """

        ## Check dimensions, and append batch dimension if not given
        if x_in.shape[-1] != self.c_in:
            raise ValueError(f"Channel mismatch: c_in set at {self.c_in}, got x_in with shape {x_in.shape}")

        if len(x_in.shape) == 2:
            appended_batch_dim = True

            # add a batch dim to all inputs
            x_in = x_in.unsqueeze(0)
            # mass_0 = mass_0.unsqueeze(0)
            # mass_1 = mass_1.unsqueeze(0)
            # mass_2 = mass_2.unsqueeze(0)
            mass_3 = mass_3.unsqueeze(0)
            mass_m = mass_m.unsqueeze(0)
            # L_0 = L_0.unsqueeze(0)
            # L_1 = L_1.unsqueeze(0)
            # L_2 = L_2.unsqueeze(0)
            L_3 = L_3.unsqueeze(0)
            L_m = L_m.unsqueeze(0)
            # evals_0 = evals_0.unsqueeze(0)
            # evals_1 = evals_1.unsqueeze(0)
            # evals_2 = evals_2.unsqueeze(0)
            evals_3 = evals_3.unsqueeze(0)
            evals_m = evals_m.unsqueeze(0)
            # evecs_0 = evecs_0.unsqueeze(0)
            # evecs_1 = evecs_1.unsqueeze(0)
            # evecs_2 = evecs_2.unsqueeze(0)
            evecs_3 = evecs_3.unsqueeze(0)
            evecs_m = evecs_m.unsqueeze(0)
            # gradX_0 = gradX_0.unsqueeze(0)
            # gradX_1 = gradX_1.unsqueeze(0)
            # gradX_2 = gradX_2.unsqueeze(0)
            gradX_3 = gradX_3.unsqueeze(0)
            gradX_m = gradX_m.unsqueeze(0)
            # gradY_0 = gradY_0.unsqueeze(0)
            # gradY_1 = gradY_1.unsqueeze(0)
            # gradY_2 = gradY_2.unsqueeze(0)
            gradY_3 = gradY_3.unsqueeze(0)
            gradY_m = gradY_m.unsqueeze(0)

        elif len(x_in.shape) == 3:
            appended_batch_dim = False

        else:
            raise ValueError(f"x_in should be tensor with shape [N,C] or [B,N,C], got {x_in.shape}")

        out = self.GeodesicBranch(x_in,
            # mass_1, L_1, evals_1, evecs_1, gradX_1, gradY_1,
            # mass_2, L_2, evals_2, evecs_2, gradX_2, gradY_2,
            mass_3, L_3, evals_3, evecs_3, gradX_3, gradY_3,
            mass_m, L_m, evals_m, evecs_m, gradX_m, gradY_m,
            # traces12,
            # traces23,
            traces34
        )

        if appended_batch_dim:
            out = out.squeeze(0)

        return out
