import torch
from fs import osfs

from torchvision import datasets
import fs.appfs
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from relgan import networks, trainer
from relgan.experiment import ex


@ex.config
def my_config():
    dataset_name = 'FashionMNIST'

@ex.main
def my_main(dataset_name):
    img_loc = osfs.OSFS('./runs', create=True)
    img_loc.removetree('.')
    samples = img_loc.makedir('samples')
    weights = img_loc.makedir('weights')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    assert hasattr(datasets, dataset_name)
    ds = getattr(datasets, dataset_name)
    data_loc = fs.appfs.UserCacheFS('torchvision').getsyspath(dataset_name)
    train_data = ds(data_loc, download=True)
    tr = trainer.RelGAN(
        networks.Generator(),
        networks.Classifier(),
        train_data.train_data.float() / 256,
        device
    )
    for i in tqdm(range(50000), miniters=50):
        tr.step()

        if i % 100 == 0:
            with torch.no_grad():
                g = tr.generator(10)
                v = torch.cat([g[j] for j in range(10)], 1).detach()
                if v.device != 'cpu':
                    v = v.cpu()

                plt.imshow(v.numpy())
                plt.savefig(samples.getsyspath('step_{:06d}.png'.format(i)))
                plt.close('all')

                with weights.open('weights-{:06d}.pkl'.format(i), 'wb') as f:
                    torch.save(tr.generator, f)
                    torch.save(tr.critic, f)
