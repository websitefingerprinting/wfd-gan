import argparse
import os
from os.path import join

import numpy as np
from sklearn import preprocessing
import joblib
from torch.utils.data import DataLoader
import torch.utils.data as Data
from torch.autograd import Variable
import torch.autograd as autograd
from torchsummaryX import summary

import common as cm
from model import *
import utils

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

k = 2
p = 6


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", required=True, help="Dataset directory.")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=2, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
    parser.add_argument("--freq", type=int, default=20, help="Checkpoint every freq epochs")
    args = parser.parse_args()
    logger = utils.init_logger('gan')
    return args, logger


def init_directory(dir, tag=""):
    basedir = os.path.split(os.path.split(dir)[0])[0]
    modeldir = join(basedir, 'model_' + tag)
    checkpointdir = join(basedir, 'checkpoint_' + tag)
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)
    if not os.path.exists(checkpointdir):
        os.makedirs(checkpointdir)
    return modeldir, checkpointdir


if __name__ == '__main__':
    # argumments
    args, logger = parse_args()

    # create folder
    modeldir, checkpointdir = init_directory(args.dir)

    # Configure data loader
    X, y = utils.load_dataset(args.dir)
    logger.info("Loaded dataset:{}, min burst:{} max burst:{}".format(X.shape, X.min(), X.max()))
    scaler = preprocessing.MinMaxScaler()
    X = scaler.fit_transform(X)
    X = torch.from_numpy(X)
    y = torch.from_numpy(y)
    class_dim = y.max() + 1  # index start from 0
    seq_len = X.size(1)
    assert seq_len > 1
    assert class_dim > 1
    logger.info("X shape {}, y shape {}, class num: {}".format(X.shape, y.shape, class_dim))
    dataset = Data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        num_workers=args.n_cpu,
        batch_size=args.batch_size,
        shuffle=True,
    )

    # Initialize generator and discriminator
    generator = Generator(seq_len, class_dim, args.latent_dim)
    discriminator = Discriminator(seq_len, class_dim)
    if torch.cuda.is_available():
        generator.cuda()
        discriminator.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    loss_checkpoints = {'generator': [], 'discriminator': [], 'dist': []}
    for epoch in range(args.n_epochs):
        total_real = []
        total_fake = []
        total_c = []
        generator_loss_epoch = 0
        discriminator_loss_epoch = 0
        w_dist_epoch = 0
        for i, (traces, c) in enumerate(dataloader):
            # Configure input
            real_traces = Variable(traces.type(Tensor), requires_grad=True)
            c = Variable(Tensor(np.eye(class_dim)[c]))

            # ---------------------
            #  Train Discriminator
            # ---------------------]

            optimizer_D.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (traces.shape[0], args.latent_dim))))
            # Generate a batch of traces
            fake_traces = generator(z, c)
            # Real traces
            real_validity = discriminator(real_traces, c)
            # Fake traces
            fake_validity = discriminator(fake_traces, c)

            # Compute W-div gradient penalty
            real_grad_out = Variable(Tensor(real_traces.size(0), 1).fill_(1.0), requires_grad=False)
            real_grad = autograd.grad(
                real_validity, real_traces, real_grad_out, create_graph=True, retain_graph=True, only_inputs=True
            )[0]
            real_grad_norm = real_grad.view(real_grad.size(0), -1).pow(2).sum(1) ** (p / 2)

            fake_grad_out = Variable(Tensor(fake_traces.size(0), 1).fill_(1.0), requires_grad=False)
            fake_grad = autograd.grad(
                fake_validity, fake_traces, fake_grad_out, create_graph=True, retain_graph=True, only_inputs=True
            )[0]
            fake_grad_norm = fake_grad.view(fake_grad.size(0), -1).pow(2).sum(1) ** (p / 2)

            div_gp = torch.mean(real_grad_norm + fake_grad_norm) * k / 2

            # Adversarial loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + div_gp

            # # Gradient penalty
            # gradient_penalty = utils.compute_gradient_penalty(discriminator, real_traces.data, fake_traces.data, c,
            #                                                   Tensor)
            # # Adversarial loss
            # d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + cm.lambda_gp * gradient_penalty

            w_dist_epoch += (torch.mean(real_validity) - torch.mean(fake_validity)).item() / len(dataloader)
            discriminator_loss_epoch += d_loss.item() / len(dataloader)

            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()
            # Train the generator every n_critic steps
            if i % args.n_critic == 0:
                # -----------------
                #  Train Generator
                # -----------------

                # Generate a batch of traces
                fake_traces = generator(z, c)
                total_real.extend(traces.cpu().numpy())
                total_fake.extend(fake_traces.detach().cpu().numpy())
                total_c.extend(c.detach().cpu().numpy())
                # Loss measures generator's ability to fool the discriminator
                # Train on fake traces
                fake_validity = discriminator(fake_traces, c)
                g_loss = -torch.mean(fake_validity)

                generator_loss_epoch += g_loss.item() / (len(dataloader) // args.n_critic)
                g_loss.backward()

                optimizer_G.step()

        logger.info(
            "[Epoch %2d/%2d] [D loss: %.4f] [G loss: %.4f] [w dist: %.4f]"
            % (epoch + 1, args.n_epochs, discriminator_loss_epoch, generator_loss_epoch, w_dist_epoch)
        )
        if (epoch == 0) or (epoch + 1) % args.freq == 0:
            # every args.freq epoch, checkpoint
            total_real = np.array(total_real)
            total_fake = np.array(total_fake)
            total_c = np.array(total_c).argmax(axis=1)
            total_real = scaler.inverse_transform(total_real)
            total_fake = scaler.inverse_transform(total_fake)
            logger.debug("Get {} samples, min burst:{}, max burst: {}".format(total_fake.shape[0], int(total_fake.min()),
                                                                             int(total_fake.max())))
            np.savez_compressed(join(checkpointdir, "epoch_{}.npy".format(epoch + 1)),
                                x=total_real, recon_x=total_fake, label=total_c)
        loss_checkpoints['generator'].append(generator_loss_epoch)
        loss_checkpoints['discriminator'].append(discriminator_loss_epoch)
        loss_checkpoints['dist'].append(w_dist_epoch)
    np.savez_compressed(join(checkpointdir, "loss.npz".format(loss_checkpoints)),
                        generator=loss_checkpoints['generator'], discriminator=loss_checkpoints['discriminator'],
                        dist=loss_checkpoints['dist'])
    torch.save(generator.state_dict(),
               join(modeldir, 'generator_seqlen{}_cls{}_latentdim{}.ckpt'.format(seq_len, class_dim, args.latent_dim)))
    torch.save(discriminator.state_dict(), join(modeldir, 'discriminator.ckpt'))
    logger.info("Model saved at {}".format(modeldir))
    joblib.dump(scaler, join(modeldir, 'scaler.gz'))
    logger.info("Scaler saved at {}".format(modeldir))
    # generator.eval()
    # with torch.no_grad():
    #     batch_size = args.batch_size
    #     # generate some examples for each batch
    #     samples = []
    #     labels = []
    #     for cls in range(class_dim):
    #         z = Variable(Tensor(np.random.normal(0, 1, (batch_size, args.latent_dim))))
    #         tmp = np.zeros((batch_size, class_dim))
    #         tmp[:,cls] = 1
    #         c = Variable(Tensor(tmp))
    #         sample = generator(z, c).cpu().numpy()
    #         sample = scaler.inverse_transform(sample)
    #         samples.extend(sample)
    #         labels.extend([cls]*batch_size)
    #     samples = np.round(np.array(samples)).astype(int)
    #     labels = np.array(labels)
    #     fake_dict = {"feature":samples, "label":labels}
    #     logger.info("fake shape:{} labels:{}".format(samples.shape, labels.shape))
    #     np.save(join(checkpointdir, "fake.npy"), fake_dict)
