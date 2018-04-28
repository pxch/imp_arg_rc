import argparse
from pathlib import Path

import torch
from torch import optim

from data.seq_dataset import load_seq_dataset
from model.pointer_net import PointerNet, train_epoch, validate
from utils import load_vocab, log, add_file_handler

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('training_path', help='Path to training dataset')
    parser.add_argument('validation_path', help='Path to validation dataset')
    parser.add_argument('word2vec_dir', help='Directory for word2vec files')
    parser.add_argument('word2vec_name', help='Name of word2vec files')
    parser.add_argument('output_path', help='Path to save trained models')
    parser.add_argument('--device', help='index of CUDA device to use',
                        type=int, default=0)
    parser.add_argument('--max_len', help='maximum length of document allowed',
                        type=int, default=100)
    parser.add_argument('--batch_size', help='Number of examples in minibatch',
                        type=int, default=32)
    parser.add_argument('--use_bucket', help='Use BucketIterator',
                        action='store_true')
    parser.add_argument('--sort_query', action='store_true',
                        help='Sort examples by both document length and '
                             'query length')
    parser.add_argument('--hidden_size', help='Size of hidden state in GRU',
                        type=int, default=100)
    parser.add_argument('--num_layers', help='Number of layers in GRU',
                        type=int, default=1)
    parser.add_argument('--dropout', help='Dropout rate for encoder',
                        type=float, default=0.2)
    parser.add_argument('--optimizer', help='Optimization method',
                        default='adam')
    parser.add_argument('--lr', help='Learning rate for optimizer',
                        type=float, default=0.001)
    parser.add_argument('--regularization', help='Regularization rate',
                        type=float, default=0.0)
    parser.add_argument('--log_every',
                        help='Log training loss after number of batches',
                        type=int, default=1000)
    parser.add_argument('--val_every',
                        help='Validate after number of batches',
                        type=int, default=50000)
    parser.add_argument('--num_epochs', help='Number of epochs to train',
                        type=int, default=10)
    parser.add_argument('--num_jobs', type=int, default=1,
                        help='Number of parallel jobs in loading data')

    args = parser.parse_args()

    # check CUDA device is available
    assert torch.cuda.is_available(), 'No CUDA device found!'

    assert 0 <= args.device < torch.cuda.device_count(), \
        'Invalid CUDA device {}'.format(args.device)

    # make sure output path exists
    output_path = Path(args.output_path)
    if not output_path.exists():
        output_path.mkdir(parents=True)

    # only log to a file named {output_path}/log
    add_file_handler(log, file_path=output_path / 'log', exclusive=True)

    # make sure the directory to save model checkpoints exists
    checkpoint_path = output_path / 'checkpoints'
    if not checkpoint_path.exists():
        checkpoint_path.mkdir(parents=True)

    with torch.cuda.device(args.device):
        training_iter = load_seq_dataset(
            args.training_path,
            n_jobs=args.num_jobs,
            max_len=args.max_len,
            use_bucket=args.use_bucket,
            batch_size=args.batch_size,
            sort_query=args.sort_query,
            train=True,
            sort_within_batch=True,
            repeat=False
        )

        validation_iter = load_seq_dataset(
            args.validation_path,
            n_jobs=args.num_jobs,
            max_len=args.max_len,
            use_bucket=args.use_bucket,
            batch_size=args.batch_size,
            sort_query=args.sort_query,
            train=False,
            sort_within_batch=True
        )

        word2vec_dir = Path(args.word2vec_dir)
        fname = word2vec_dir / (args.word2vec_name + '.bin')
        fvocab = word2vec_dir / (args.word2vec_name + '.vocab')
        assert fname.exists() and fvocab.exists()

        vocab = load_vocab(fname=str(fname), fvocab=str(fvocab))
        vocab_size, input_size = vocab.vectors.shape

        log.info(
            'Initializing pointer network with vocab_size = {}, '
            'input_size = {}, hidden_size = {}, num_layers = {}, '
            'dropout = {}'.format(
                vocab_size, input_size, args.hidden_size, args.num_layers,
                args.dropout))
        pointer_net = PointerNet(
            vocab_size=vocab_size,
            input_size=input_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            bidirectional=True,
            dropout_p=args.dropout,
            attn_method='general'
        )
        log.info('Initializing embedding layer with pretrained word vectors.')
        pointer_net.init_embedding(vocab.vectors)

        log.info('Moving all model parameters to GPU.')
        pointer_net.cuda()

        log.info('Model specifications:\n{}'.format(pointer_net))

        optim_dict = {
            'adadelta': optim.Adadelta,
            'adagrad': optim.Adagrad,
            'adam': optim.Adam,
            'sparseadam': optim.SparseAdam,
            'adamax': optim.Adamax,
            'rmsprop': optim.RMSprop,
        }
        if args.optimizer == 'sgd':
            optimizer = optim.SGD(pointer_net.parameters(), lr=args.lr)
        elif args.optimizer in optim_dict:
            optimizer = optim_dict[args.optimizer](pointer_net.parameters())
        else:
            raise NotImplementedError()

        log.info('Initializing {} optimizer with lr = {}'.format(
            args.optimizer, optimizer.defaults['lr']))

        log.info(
            'Training with regularization = {}, log_every = {}, '
            'val_every = {}'.format(
                args.regularization, args.log_every, args.val_every))

        validate(pointer_net, validation_iter, 'Before training')

        for epoch in range(args.num_epochs):
            log.info('Start training epoch {:3d}/{:3d}'.format(
                epoch + 1, args.num_epochs))
            train_epoch(
                pointer_net=pointer_net,
                training_iter=training_iter,
                validation_iter=validation_iter,
                optimizer=optimizer,
                regularization=args.regularization,
                log_every=args.log_every,
                val_every=args.val_every
            )

            validate(
                pointer_net, validation_iter,
                'After {:3d}/{:3d} epochs'.format(epoch + 1, args.num_epochs))

            model_output_path = \
                checkpoint_path / 'epoch-{}-model'.format(epoch + 1)
            log.info('Save model states to {}'.format(model_output_path))
            torch.save(pointer_net.state_dict(), model_output_path)

            optim_output_path = \
                checkpoint_path / 'epoch-{}-optim'.format(epoch + 1)
            log.info('Save optimizer states to {}'.format(optim_output_path))
            torch.save(optimizer.state_dict(), optim_output_path)
