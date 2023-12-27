from imports.common_imports import *

class Diffusion:
    def __init__(self, noise_steps,  beta_start = 0.00085, beta_end = 0.012, beta_scheduler = "scaled_linear", img_size=8):
        assert beta_scheduler in ["linear", "scaled_linear"], "beta_scheduler must be either 'linear' or 'scaled_linear'"
        
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size

        if beta_scheduler == "linear":
            self.beta = torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
        else:
            self.beta = torch.linspace(self.beta_start**(0.5), self.beta_end**(0.5), self.noise_steps)**2
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def noise_schedule_to_device(self, device):
        self.beta = self.beta.to(device)
        self.alpha = self.alpha.to(device)
        self.alpha_hat = self.alpha_hat.to(device)

    def noise_images(self, x, t): 
        self.noise_schedule_to_device(t.device)
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        gauss_noise = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * gauss_noise, gauss_noise

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def ddpm_sample(self, model, z0, z1):
        n = z0.shape[0]
        x = torch.randn((n, 3, self.img_size, self.img_size)).to(z0.device)
        self.noise_schedule_to_device(z0.device)
        
        #for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
        for i in reversed(range(1, self.noise_steps)):
            t = (torch.ones(n) * i).long().to(z0.device)
            predicted_noise = model(x, z0, z1, t)
            
            alpha = self.alpha[t][:, None, None, None]
            alpha_hat = self.alpha_hat[t][:, None, None, None]
            beta = self.beta[t][:, None, None, None]
            
            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
        
            mean_eps = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise)
            x = mean_eps + torch.sqrt(beta) * noise
    
        return x

    def ddim_sample(self, model, z0, z1):
        n = z0.shape[0]
        x = torch.randn((n, 3, self.img_size, self.img_size)).to(z0.device)
        self.alpha_hat = self.alpha_hat.to(z0.device)

        for i in reversed(range(1, self.noise_steps)):
            t = (torch.ones(n) * i).long().to(z0.device)
            predicted_noise = model(x, z0, z1, t)
            
            alpha_hat_t = self.alpha_hat[t][:, None, None, None]
            alpha_hat_tm1 = self.alpha_hat[t - 1][:, None, None, None]

            z0 = (1/torch.sqrt(alpha_hat_t))*(x - torch.sqrt(1 - alpha_hat_t) * predicted_noise)
            x = torch.sqrt(alpha_hat_tm1) * z0 + torch.sqrt(1 - alpha_hat_tm1) * predicted_noise
        return x


        
