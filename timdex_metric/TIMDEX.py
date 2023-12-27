from imports.common_imports import *

# Have in mind that the tvl1 flow estimation is slower than farneback, but it is more accurate
# I use farneback for training and tvl1 for testing
# Be free to pruebe other flow estimation methods
class TIMDEXPosEncoding(nn.Module):
    def __init__(self, time_dim, flow_estimation = 'farneback'):
        super(TIMDEXPosEncoding, self).__init__()

        assert flow_estimation in ['farneback', 'tvl1'], 'flow_estimation must be farneback or tvl1'
        if flow_estimation == 'farneback': 
            self.flow_estimation = self.OptFlow_Farneback
        elif flow_estimation == 'tvl1':
            self.flow_estimation = self.OptFlow_DualTVL1

        self.time_dim = time_dim

    def OptFlow_Farneback(self, I0, I1):
        device = I0.device
        
        I0 = I0.cpu().clamp(0, 1) * 255
        I1 = I1.cpu().clamp(0, 1) * 255

        batch_size = I0.shape[0]
        for i in range(batch_size):
            I0_np = I0[i].permute(1, 2, 0).numpy().astype(np.uint8)
            I1_np = I1[i].permute(1, 2, 0).numpy().astype(np.uint8)

            I0_gray = cv2.cvtColor(I0_np, cv2.COLOR_BGR2GRAY)
            I1_gray = cv2.cvtColor(I1_np, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(I0_gray, I1_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow = torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(0).float()
            if i == 0:
                flows = flow
            else:
                flows = torch.cat((flows, flow), dim = 0)

        return flows.to(device)

    def OptFlow_DualTVL1(self, I0, I1,
        tau: float = 0.25,
        lambda_: float = 0.15,
        theta: float = 0.3,
        scales_number: int = 5,
        warps: int = 5,
        epsilon: float = 0.01,
        inner_iterations: int = 30,
        outer_iterations: int = 10,
        scale_step: float = 0.8,
        gamma: float = 0.0
    ):
        optical_flow = cv2.optflow.createOptFlow_DualTVL1()
        optical_flow.setTau(tau)
        optical_flow.setLambda(lambda_)
        optical_flow.setTheta(theta)
        optical_flow.setScalesNumber(scales_number)
        optical_flow.setWarpingsNumber(warps)
        optical_flow.setEpsilon(epsilon)
        optical_flow.setInnerIterations(inner_iterations)
        optical_flow.setOuterIterations(outer_iterations)
        optical_flow.setScaleStep(scale_step)
        optical_flow.setGamma(gamma)

        device = I0.device
        
        I0 = I0.cpu().clamp(0, 1) * 255
        I1 = I1.cpu().clamp(0, 1) * 255

        batch_size = I0.shape[0]
        for i in range(batch_size):
            I0_np = I0[i].permute(1, 2, 0).numpy().astype(np.uint8)
            I1_np = I1[i].permute(1, 2, 0).numpy().astype(np.uint8)

            I0_gray = cv2.cvtColor(I0_np, cv2.COLOR_BGR2GRAY)
            I1_gray = cv2.cvtColor(I1_np, cv2.COLOR_BGR2GRAY)

            flow = optical_flow.calc(I0_gray, I1_gray, None)
            flow = torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(0).float()
            if i == 0:
                flows = flow
            else:
                flows = torch.cat((flows, flow), dim = 0)

        return flows.to(device)

    def morph_open(self, x, k):
        if k==0:
            return x
        else:
            with torch.no_grad():
                return kornia.morphology.opening(x, torch.ones(k,k,device=x.device))

    def TIMDEX(self, I0, It, I1, k = 5, threshold = 2e-2):
        flow0tot = self.flow_estimation(I0, It)
        flow1tot = self.flow_estimation(It, I1)
    
        transform = transforms.Grayscale()
        I0_gray = transform(I0)
        It_gray = transform(It)
        I1_gray = transform(I1)

        mask0tot = self.morph_open(It_gray - I0_gray, k=k)
        mask1tot = self.morph_open(I1_gray - It_gray, k=k)

        mask0tot = (abs(mask0tot) > threshold).to(torch.uint8)
        mask1tot = (abs(mask1tot) > threshold).to(torch.uint8)
        
        d0tot = torch.sum(torch.norm(flow0tot*mask0tot, dim=1), dim = (1, 2)) 
        d1tot = torch.sum(torch.norm(flow1tot*mask1tot, dim=1), dim = (1, 2))
        
        return d0tot / (d0tot + d1tot + 1e-8)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, I0, It, I1):
        t = self.TIMDEX(I0, It, I1)
        t = t.unsqueeze(1)
        t = self.pos_encoding(t, self.time_dim)
        return t