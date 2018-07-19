from fs import osfs

from sacred import Experiment
from torchvision import datasets
import fs.appfs
from tqdm import tqdm
import matplotlib.pyplot as plt

from relgan import networks, trainer


ex = Experiment()
img_loc = osfs.OSFS('./runs', create=True)

@ex.main
def my_main():
    img_loc.removetree('.')
    train_data = datasets.FashionMNIST(fs.appfs.UserCacheFS('torchvision').getsyspath('FashionMNIST'))
    tr = trainer.RelGAN(
        networks.Generator(),
        networks.Classifier(),
        train_data.train_data.float() / 256
    )
    for i in tqdm(range(1000)):
        tr.step()

        if i % 10 == 0:
            v = tr.generator(1)[0]

            plt.imshow(v.detach().numpy())
            plt.savefig(img_loc.getsyspath('step_{:04d}.png'.format(i)))
            plt.close('all')
