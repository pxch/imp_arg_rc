from utils import log
from torch.nn.utils import clip_grad_norm
import torch


def compute_batch_loss(pointer_net, batch, regularization=0.1, predict=False):
    # check each column of batch.softmax_mask contains at least one nonzero
    # assert batch.softmax_mask.sum(dim=0).nonzero().size(0) == batch.batch_size

    attn = pointer_net(
        doc_input_seqs=batch.doc_input[0],
        doc_input_lengths=batch.doc_input[1],
        query_input_seqs=batch.query_input[0],
        query_input_lengths=batch.query_input[1],
        doc_entity_ids=batch.doc_entity_ids
    )

    target_mask = (batch.doc_entity_ids == batch.target_entity_id.unsqueeze(0))

    masked_attn = attn * target_mask.float()
    logit_attn, _ = masked_attn.max(dim=0)

    loss = -torch.log(logit_attn).sum() / batch.batch_size

    if regularization > 0:
        reg_loss = 0.
        for param in pointer_net.parameters():
            reg_loss += param.norm() ** 2
        reg_loss = reg_loss ** 0.5
        # print('{:.2f}+{:.2f}'.format(loss.data[0], reg_loss.data[0]), end=' ')
        loss += regularization * reg_loss

    # else:
    #     print('{:.2f}'.format(loss.data[0]), end=' ')

    if predict:
        max_indices = attn.max(dim=0)[1].unsqueeze(0)
        predicted = batch.doc_entity_ids.gather(index=max_indices, dim=0)
        predicted = predicted.squeeze()
        return loss, predicted
    else:
        return loss


def validate(pointer_net, validation_iter, msg=''):
    pointer_net.train(False)

    val_loss = 0.0

    num_correct = 0
    num_total = 0

    for batch in validation_iter:
        loss, predicted = compute_batch_loss(
            pointer_net, batch, regularization=0, predict=True)
        val_loss += loss.data[0]
        num_correct += (predicted == batch.target_entity_id).sum().data[0]
        num_total += batch.batch_size

    val_loss /= len(validation_iter)

    pointer_net.train(True)

    # print()
    log.info(
        '{}: validation loss = {:.8f}, '
        'accuracy = {:8d}/{:8d} = {:.2f}%'.format(
            msg, val_loss, num_correct, num_total,
            num_correct / num_total * 100))


def evaluate(pointer_net, evaluation_iter):
    pointer_net.train(False)

    val_loss = 0.0

    num_correct = {'all': 0, 'subj': 0, 'dobj': 0, 'pobj': 0}
    num_total = {'all': 0, 'subj': 0, 'dobj': 0, 'pobj': 0}

    for batch in evaluation_iter:
        loss, predicted = compute_batch_loss(
            pointer_net, batch, regularization=0, predict=True)
        val_loss += loss.data[0]
        # num_correct['all'] += \
        #     (predicted == batch.target_entity_id).sum().data[0]
        # num_total['all'] += batch.batch_size

        for idx in range(batch.batch_size):
            query = batch.query_input[0][:, idx]
            pos = ''
            for token_id in query:
                if token_id.data[0] == 1:
                    pos = 'subj'
                    break
                elif token_id.data[0] == 2:
                    pos = 'dobj'
                    break
                elif token_id.data[0] == 3:
                    pos = 'pobj'
                    break

            num_total['all'] += 1
            num_total[pos] += 1

            if predicted[idx].data[0] == batch.target_entity_id[idx].data[0]:
                num_correct['all'] += 1
                num_correct[pos] += 1

    val_loss /= len(evaluation_iter)

    pointer_net.train(True)

    # print()
    log.info('validation loss = {:.8f}'.format(val_loss))
    for key in num_correct.keys():
        log.info('{} accuracy = {:8d}/{:8d} = {:.2f}%'.format(
            key, num_correct[key], num_total[key],
            num_correct[key] / num_total[key] * 100))


def train_epoch(pointer_net, training_iter, validation_iter, optimizer,
                regularization=0.1, log_every=1000, val_every=50000):
    training_loss = 0.0

    for batch in training_iter:
        loss = compute_batch_loss(
            pointer_net, batch, regularization=regularization, predict=False)

        optimizer.zero_grad()
        loss.backward()

        clip_grad_norm(pointer_net.parameters(), max_norm=1.0)

        optimizer.step()

        training_loss += loss.data[0]

        if training_iter.iterations % log_every == 0:
            # print()
            log.info(
                'Finished {:8d}/{:8d} batches, training loss in '
                'last {} batches = {:.8f}'.format(
                    training_iter.iterations, len(training_iter),
                    log_every, training_loss / log_every))
            training_loss = 0.0

        if training_iter.iterations % val_every == 0:
            validate(pointer_net, validation_iter,
                     'After {:8d} batches'.format(training_iter.iterations))

    num_remaining_batch = len(training_iter) % log_every
    if num_remaining_batch > 0:
        # print()
        log.info(
            'Finished {:8d}/{:8d} batches, training loss in '
            'last {} batches = {:.8f}'.format(
                len(training_iter), len(training_iter),
                num_remaining_batch, training_loss / num_remaining_batch))
