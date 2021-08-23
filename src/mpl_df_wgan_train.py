import argparse
import os
from os.path import join
from time import strftime
import json

import numpy as np
import torch.optim
from sklearn import preprocessing
import joblib
from torch.utils.data import DataLoader
import torch.utils.data as Data
from torch.autograd import Variable
import torch.autograd as autograd
from torch.optim import lr_scheduler
from torchsummaryX import summary

import common as cm
from model import *
import utils

k = 2
p = 6
w_dist_threshold = 0.05

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

MAX_BURST_LENGTH = 1382
MAX_OUTGOING_SIZE = 75
MAX_INCOMING_SIZE = 179


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", required=True, help="Dataset directory.")
    parser.add_argument("--clip", action="store_true", default=False,
                        help="Whether to clip the burst size of the dataset before training")
    parser.add_argument("--f_model", type=str, required=True, help="The directory of the pre-trained DF.")
    parser.add_argument("--n_epochs", type=int, default=300, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="learning rate")
    parser.add_argument("--n_cpu", type=int, default=2, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=50, help="dimensionality of the latent space")
    parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
    parser.add_argument("--alpha_max", type=float, default=0.01, help="Max ratio of f loss")
    parser.add_argument("--alpha_step", type=float, default=0.0004, help="alpha growth step size")
    parser.add_argument("--alpha_freq", type=int, default=20, help="alpha value update frequency")
    parser.add_argument("--freq", type=int, default=20, help="Checkpoint every freq epochs")
    parser.add_argument("--cuda_id", type=int, default=0, help="GPU ID")
    args = parser.parse_args()
    return args


def test_DF_acc_epoch(model, X, y):
    X = torch.from_numpy(np.array(X)[:, 1:]).float()
    X = X[:, np.newaxis, :]
    y = torch.from_numpy(np.array(y))
    dataset = Data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        num_workers=1,
        batch_size=128,
        shuffle=False,
    )

    model.eval()
    total = 0.0
    correct = 0
    for _, (batch_x, batch_y) in enumerate(dataloader):
        batch_x = Variable(batch_x.type(Tensor), requires_grad=False)
        batch_y = Variable(batch_y.type(LongTensor), requires_grad=False)
        outputs = model(batch_x)
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
    return correct / total


def my_clip(arr):
    if arr[0] > MAX_BURST_LENGTH:
        arr[0] = MAX_BURST_LENGTH
    for i in range(1, len(arr), 2):
        if arr[i] > MAX_OUTGOING_SIZE:
            arr[i] = MAX_OUTGOING_SIZE
    for i in range(2, len(arr), 2):
        if arr[i] > MAX_INCOMING_SIZE:
            arr[i] = MAX_INCOMING_SIZE
    return arr


if __name__ == '__main__':
    # argumments
    args = parse_args()

    device = torch.device("cuda:{}".format(args.cuda_id) if (torch.cuda.is_available()) else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.cuda_id)  # id=0, 1, 2

    cf = utils.read_conf(cm.confdir)
    # create folder
    pardir, modeldir, checkpointdir = utils.init_directory(args.dir, tag='training_{}'.format(strftime('%m%d_%H%M%S')))
    logger = utils.init_logger('gan', join(pardir, 'log.txt'))

    logger.info(args)
    logger.debug("Output to {}".format(pardir))

    # save parameters to file
    with open(join(pardir, 'args.json'), 'wt') as f:
        json.dump(vars(args), f, indent=4)

    # Configure data loader
    X, y = utils.load_dataset(args.dir)
    logger.info("Loaded dataset:{}, min burst:{} max burst:{}, min label:{}, max label:{}"
                .format(X.shape, X[:, 1:].min(), X[:, 1:].max(), y.min(), y.max()))
    # reset max for each feature
    # The 95 percentile of burst len, outgoing, incoming are 1382, 75, 179, respectively
    # this is computed over 100,000 traces in rimmer_top877 (each trace compute the max incoming, max outgoing)
    if args.clip:
        X = np.apply_along_axis(my_clip, 1, X)
        assert X[:, 0].max() == MAX_BURST_LENGTH
        assert X[:, 1::2].max() == MAX_OUTGOING_SIZE
        assert X[:, 2::2].max() == MAX_INCOMING_SIZE
        np.savez_compressed(args.dir.split('.npz')[0] + '_clip.npz', features=X, labels=y)
        logger.info("Save back  the clipped dataset to {}".format(args.dir.split('.npz')[0] + '_clip.npz'))

    # reindex label starting from 0
    y -= y.min()

    scaler = preprocessing.MinMaxScaler()
    X = scaler.fit_transform(X)
    X = torch.from_numpy(X)
    y = torch.from_numpy(y)
    class_dim = len(np.unique(y))  # index start from 0
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
    f_model = DF(seq_len - 1, class_dim)  # remove the number_of_burst feature, so #feature - 1
    f_model.load_state_dict(torch.load(args.f_model, map_location=device))
    f_model.eval()

    if torch.cuda.is_available():
        generator.cuda()
        discriminator.cuda()
        f_model.cuda()

    # Optimizers
    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=args.lr)

    # F model criterion
    criterion = nn.CrossEntropyLoss()

    # alpha initialization
    alpha = args.alpha_max

    loss_checkpoints = {'generator': [], 'discriminator': [], 'dist': []}
    for epoch in range(args.n_epochs):
        total_real = []
        total_fake = []
        total_c = []
        generator_g_loss_epoch = 0
        generator_f_loss_epoch = 0
        generator_loss_combined_epoch = 0
        discriminator_loss_epoch = 0
        w_dist_epoch = 0
        for i, (traces, c) in enumerate(dataloader):
            # Configure input
            real_traces = Variable(traces.type(Tensor), requires_grad=True)
            c_onehot = Variable(Tensor(np.eye(class_dim)[c]))
            c = Variable(c.type(LongTensor))

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (traces.shape[0], args.latent_dim))))
            # Generate a batch of traces
            fake_traces = generator(z, c_onehot)
            # Real traces
            real_validity = discriminator(real_traces, c_onehot)
            # Fake traces
            fake_validity = discriminator(fake_traces, c_onehot)

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
                fake_traces = generator(z, c_onehot)
                total_real.extend(traces.cpu().numpy())
                total_fake.extend(fake_traces.detach().cpu().numpy())
                total_c.extend(c.detach().cpu().numpy())
                # Loss measures generator's ability to fool the discriminator
                # Train on fake traces
                fake_validity = discriminator(fake_traces, c_onehot)
                g_loss = -torch.mean(fake_validity)

                # We pick out those traces that the discriminator considered as real ones
                # to ask f model whether they look like real ones from label c
                fake_traces_seems_real = fake_traces[fake_validity[:, 0] > 0]
                labels_seems_real = c[fake_validity[:, 0] > 0]
                if len(fake_traces_seems_real) > 0:
                    # logger.debug("We have {}/{} traces considered to be real in epoch {} step {}".format(
                    # len(labels_seems_real), len(c), epoch, i))
                    # Convert to CNN format, remove the first element which is the burst length
                    fake_traces_converted = fake_traces_seems_real[:, 1:]
                    fake_traces_converted = fake_traces_converted.reshape(fake_traces_converted.size(0), 1,
                                                                          fake_traces_converted.size(1))
                    # compute f_loss
                    outputs = f_model(fake_traces_converted)
                    f_loss = criterion(outputs, labels_seems_real)
                    g_loss_combined = g_loss + alpha * f_loss
                    f_loss_item = f_loss.item()
                else:
                    g_loss_combined = g_loss
                    f_loss_item = 0

                generator_g_loss_epoch += g_loss.item() / (len(dataloader) // args.n_critic)
                generator_f_loss_epoch += f_loss_item / (len(dataloader) // args.n_critic)
                generator_loss_combined_epoch += g_loss_combined.item() / (len(dataloader) // args.n_critic)

                g_loss_combined.backward()

                optimizer_G.step()

        # Test DF accuracy
        df_acc = test_DF_acc_epoch(f_model, total_fake, total_c)

        logger.info(
            "[Epoch %2d/%2d] [D loss: %.4f] [G loss: %.4f + %.4f * %.4f = %.4f] [DF acc: %.4f] [w dist: %.4f]"
            % (epoch + 1, args.n_epochs, discriminator_loss_epoch, generator_g_loss_epoch, alpha,
               generator_f_loss_epoch, generator_loss_combined_epoch, df_acc, w_dist_epoch)
        )

        # if epoch % args.alpha_freq == 0:
        #     alpha = min(alpha + args.alpha_step, args.alpha_max)

        if (epoch == 0) or (epoch + 1) % args.freq == 0 or (w_dist_epoch <= w_dist_threshold and df_acc >= 0.9):
            # every args.freq epoch, checkpoint
            total_real = np.array(total_real)
            total_fake = np.array(total_fake)
            total_c = np.array(total_c)
            total_real = scaler.inverse_transform(total_real)
            total_fake = scaler.inverse_transform(total_fake)
            logger.debug(
                "Get {} samples, min burst:{}, max burst: {}".format(total_fake.shape[0], int(total_fake[:, 1:].min()),
                                                                     int(total_fake[:, 1:].max())))
            np.savez_compressed(join(checkpointdir, "epoch_{}".format(epoch + 1)),
                                x=total_real, recon_x=total_fake, label=total_c)
        loss_checkpoints['generator'].append(generator_g_loss_epoch)
        loss_checkpoints['discriminator'].append(discriminator_loss_epoch)
        loss_checkpoints['dist'].append(w_dist_epoch)
        if w_dist_epoch <= w_dist_threshold and df_acc >= 0.9:
            logger.info('Early stopping since the w-dist = {}, df acc = {}.'.format(w_dist_epoch, df_acc))
            break

    np.savez_compressed(join(checkpointdir, "loss.npz".format(loss_checkpoints)),
                        generator=loss_checkpoints['generator'], discriminator=loss_checkpoints['discriminator'],
                        dist=loss_checkpoints['dist'])
    torch.save(generator.state_dict(),
               join(modeldir, 'generator_seqlen{}_cls{}_latentdim{}.ckpt'.format(seq_len, class_dim, args.latent_dim)))
    torch.save(discriminator.state_dict(), join(modeldir, 'discriminator.ckpt'))
    logger.info("Model saved at {}".format(modeldir))
    joblib.dump(scaler, join(modeldir, 'scaler.gz'))
    logger.info("Scaler saved at {}".format(modeldir))
