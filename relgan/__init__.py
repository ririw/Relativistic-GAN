import torch
from fs import osfs

from torchvision import datasets
import fs.appfs
from tqdm import tqdm
import matplotlib.pyplot as plt

from relgan import networks, trainer
from relgan.experiment import ex

img_loc = osfs.OSFS('./runs', create=True)

@ex.main
def my_main():
    img_loc.removetree('.')
    samples = img_loc.makedir('samples')
    weights = img_loc.makedir('weights')

    train_data = datasets.MNIST(fs.appfs.UserCacheFS('torchvision').getsyspath('MNIST'))
    tr = trainer.RelGAN(
        networks.Generator(),
        networks.Classifier(),
        train_data.train_data.float() / 256
    )
    for i in tqdm(range(50000), miniters=50):
        tr.step()

        if i % 50 == 0:
            g = tr.generator(10)
            v = torch.cat([g[j] for j in range(10)], 1)

            plt.imshow(v.detach().numpy())
            plt.savefig(samples.getsyspath('step_{:04d}.png'.format(i)))
            plt.close('all')

            with weights.open('weights-{:04d}.pkl'.format(i), 'wb') as f:
                torch.save(tr.generator, f)
                torch.save(tr.critic, f)
