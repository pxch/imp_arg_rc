from collections import Counter
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from model.pointer_net.encoder import SelfAttentiveEncoder
from utils import log


def get_self_attn_target(batch, target_for_pred='none'):
    assert target_for_pred in ['none', 'self', 'self_attn']

    # doc_entity_ids need to be transposed from batch input, i.e.,
    # it should have shape of B * Ld
    doc_entity_ids = batch.doc_entity_ids.transpose(0, 1)
    input_lengths = batch.doc_input[1]

    batch_size, max_len = doc_entity_ids.shape

    # get target based on same entity_ids
    self_attn_target = \
        (doc_entity_ids.unsqueeze(2) == doc_entity_ids.unsqueeze(1))

    # remove entries with -1 entity_id (either predicates or padding tokens)
    self_attn_target *= (doc_entity_ids != -1).unsqueeze(2)

    # if target_for_pred in 'self_attn', add information from
    # corefered predicates (predicates with shared arguments)
    if target_for_pred == 'self_attn':
        coref_pred_indices = torch.cat(
            [batch.coref_pred_1, batch.coref_pred_2], dim=0)
        col_indices = (coref_pred_indices[1, :] != -1).nonzero()
        if col_indices.numel() > 0:
            coref_pred_indices = coref_pred_indices[:, col_indices.squeeze(1)]
            sparse_target = torch.cuda.sparse.ByteTensor(
                coref_pred_indices,
                torch.ones(coref_pred_indices.size(1)).to(self_attn_target),
                torch.Size([batch_size, max_len, max_len]))
            self_attn_target += sparse_target + sparse_target.transpose(1, 2)

        '''
        for ex_idx in range(batch_size):
            coref_pred_index_1 = batch.coref_pred_1[0][ex_idx, :]
            coref_pred_index_2 = batch.coref_pred_2[0][ex_idx, :]
            coref_pred_count = batch.coref_pred_1[1][ex_idx].item()
            if coref_pred_index_1[0] != -1:
                # compute indices for sparse tensor
                coref_pred_indices = torch.stack(
                    [coref_pred_index_1, coref_pred_index_2], dim=0)
                coref_pred_indices = coref_pred_indices[:, :coref_pred_count]
                # create sparse tensor with 1s
                sparse_target = torch.cuda.sparse.ByteTensor(
                    coref_pred_indices,
                    torch.ones(coref_pred_count).to(self_attn_target),
                    torch.Size([max_len, max_len]))
                # add the sparse tensor to self_attn_target
                # (after making it symmetric)
                self_attn_target[ex_idx, :, :] += \
                    (sparse_target + sparse_target.t()).to_dense()
        '''

    # restore entries of predicates by adding a diagonal matrix
    # (also prevents dividing-by-zero in the next step)
    self_attn_target += torch.eye(max_len).to(self_attn_target).unsqueeze(0)
    self_attn_target = torch.clamp(self_attn_target, max=1)

    # any changes to the attention targets of predicates should happen before
    # normalization (currently only attend to itself)

    # normalization
    self_attn_target = self_attn_target.float()
    self_attn_target /= self_attn_target.sum(dim=2, keepdim=True)

    # if target_for_pred is none, then mask out all rows of predicates
    if target_for_pred == 'none':
        target_mask = (doc_entity_ids != -1).unsqueeze(2).float()
    # otherwise, apply self_attn_mask
    else:
        target_mask = SelfAttentiveEncoder.get_mask_for_self_attention(
            input_lengths, max_len).float()

    self_attn_target *= target_mask

    return self_attn_target


def compute_batch_loss(pointer_net, batch, neg_loss_type='none',
                       regularization=0.1, predict=False, use_sum=False,
                       use_sigmoid=False, self_attn_target_for_pred='none'):
    assert neg_loss_type in ['none', 'multi_arg', 'multi_slot', 'max_margin']

    # check each column of batch.softmax_mask contains at least one nonzero
    # assert batch.softmax_mask.sum(dim=0).nonzero().size(0) == batch.batch_size

    softmax_mask = (batch.doc_entity_ids != -1)

    kwargs = {}
    if pointer_net.use_salience:
        if pointer_net.salience_vocab_size:
            max_num_mentions = pointer_net.salience_vocab_size - 1
            kwargs['num_mentions_total'] = torch.clamp(
                batch.num_mentions_total, min=0, max=max_num_mentions)
            kwargs['num_mentions_named'] = torch.clamp(
                batch.num_mentions_named, min=0, max=max_num_mentions)
            kwargs['num_mentions_nominal'] = torch.clamp(
                batch.num_mentions_nominal, min=0, max=max_num_mentions)
            kwargs['num_mentions_pronominal'] = torch.clamp(
                batch.num_mentions_pronominal, min=0, max=max_num_mentions)
        else:
            kwargs['num_mentions_total'] = batch.num_mentions_total
            kwargs['num_mentions_named'] = batch.num_mentions_named
            kwargs['num_mentions_nominal'] = batch.num_mentions_nominal
            kwargs['num_mentions_pronominal'] = batch.num_mentions_pronominal

    if neg_loss_type == 'multi_slot':
        kwargs['neg_query_input_seqs'] = batch.neg_query_input[0]
        kwargs['neg_query_input_lengths'] = batch.neg_query_input[1]

    attn, self_attn, first_hop_attn, neg_query_attn = pointer_net(
        doc_input_seqs=batch.doc_input[0],
        doc_input_lengths=batch.doc_input[1],
        query_input_seqs=batch.query_input[0],
        query_input_lengths=batch.query_input[1],
        softmax_mask=softmax_mask,
        return_energy=use_sigmoid,
        **kwargs
    )

    if use_sigmoid:
        attn = torch.sigmoid(attn) * softmax_mask.float()

    target_mask = batch.doc_entity_ids.eq(batch.target_entity_id.unsqueeze(0))

    masked_attn = attn * target_mask.float()

    if use_sum:
        logit_attn = masked_attn.sum(dim=0)
    else:
        logit_attn, _ = masked_attn.max(dim=0)

    # loss term for main objective
    losses = {'main': -torch.log(logit_attn).sum() / batch.batch_size}

    if self_attn is not None:
        self_attn_target = get_self_attn_target(
            batch, target_for_pred=self_attn_target_for_pred)
        self_attn_loss = F.kl_div(torch.log(self_attn), self_attn_target)
        # loss term for self attention
        losses['self_attn'] = self_attn_loss

    if first_hop_attn is not None:
        first_hop_attn_target = batch.argument_mask.float()
        first_hop_attn_target /= first_hop_attn_target.sum(dim=0).unsqueeze(0)

        first_hop_attn_loss = F.kl_div(
            torch.log(first_hop_attn), first_hop_attn_target)
        # loss term for first hop attention (in multi_hop setting)
        losses['first_hop_attn'] = first_hop_attn_loss

    if neg_loss_type != 'none':
        # additional loss for negative target entity (in multi_arg setting)
        if neg_loss_type == 'multi_arg':
            neg_target_mask = \
                batch.doc_entity_ids.eq(batch.neg_target_entity_id.unsqueeze(0))
            neg_masked_attn = attn * neg_target_mask.float()

        # additional loss for negative query (in multi_slot setting)
        elif neg_loss_type == 'multi_slot':
            assert neg_query_attn is not None
            if use_sigmoid:
                neg_query_attn = \
                    torch.sigmoid(neg_query_attn) * softmax_mask.float()

            neg_masked_attn = neg_query_attn * target_mask.float()

        # additional loss for negative attention weight (in max_margin setting)
        else:  # neg_loss_type == 'max_margin'
            neg_target_mask = \
                batch.doc_entity_ids.ne(batch.target_entity_id.unsqueeze(0))
            neg_target_mask = neg_target_mask * softmax_mask

            neg_masked_attn = attn * neg_target_mask.float()

        if use_sum:
            neg_logit_attn = neg_masked_attn.sum(dim=0)
        else:
            neg_logit_attn, _ = neg_masked_attn.max(dim=0)

        # add additional loss to the loss term for main objective
        losses['main'] -= torch.log(1 - neg_logit_attn).sum() / batch.batch_size

    # add regularization loss to the loss term for main objective
    if regularization > 0:
        reg_loss = 0.
        for param in pointer_net.parameters():
            reg_loss += param.norm() ** 2
        reg_loss = reg_loss ** 0.5
        losses['main'] += regularization * reg_loss

    if predict:
        max_indices = attn.max(dim=0)[1].unsqueeze(0)
        predicted = batch.doc_entity_ids.gather(index=max_indices, dim=0)
        predicted = predicted.squeeze(0)
        return losses, predicted
    else:
        return losses


def validate(pointer_net, validation_iter, self_attn_target_for_pred='none',
             msg=''):
    pointer_net.train(False)

    val_loss = 0.0
    val_loss_self_attn = 0.0
    val_loss_first_hop_attn = 0.0

    num_correct = 0
    num_total = 0

    for batch in validation_iter:
        losses, predicted = compute_batch_loss(
            pointer_net, batch, regularization=0.0, predict=True,
            self_attn_target_for_pred=self_attn_target_for_pred)

        val_loss += losses['main'].item()
        if pointer_net.use_self_attn:
            assert 'self_attn' in losses
            val_loss_self_attn += losses['self_attn'].item()
        if pointer_net.multi_hop:
            assert 'first_hop_attn' in losses
            val_loss_first_hop_attn += losses['first_hop_attn'].item()

        num_correct += \
            int(predicted.eq(batch.target_entity_id).float().sum().item())
        num_total += batch.batch_size

    val_loss /= len(validation_iter)
    val_loss_self_attn /= len(validation_iter)
    val_loss_first_hop_attn /= len(validation_iter)

    pointer_net.train(True)

    log.info(
        '{}: validation loss = {:.8f}{}{}, '
        'accuracy = {:8d}/{:8d} = {:.2f}%'.format(
            msg, val_loss,
            ', self_attn loss = {:.8f}'.format(val_loss_self_attn)
            if pointer_net.use_self_attn else '',
            ', first_hop_attn loss = {:.8f}'.format(val_loss_first_hop_attn)
            if pointer_net.multi_hop else '',
            num_correct, num_total,
            num_correct / num_total * 100))

    return val_loss, val_loss_self_attn, val_loss_first_hop_attn


def evaluate(pointer_net, evaluation_iter, report_entity_freq=False,
             return_results=False):
    pointer_net.train(False)

    results = []

    eval_loss = 0.0

    num_correct = {'all': 0, 'subj': 0, 'dobj': 0, 'pobj': 0}
    num_total = {'all': 0, 'subj': 0, 'dobj': 0, 'pobj': 0}

    if report_entity_freq:
        for entity_freq in range(1, 11):
            num_correct[entity_freq] = 0
            num_total[entity_freq] = 0

    for batch in evaluation_iter:
        losses, predicted = compute_batch_loss(
            pointer_net, batch, regularization=0, predict=True)

        eval_loss += losses['main'].item()

        # num_correct['all'] += \
        #     (predicted == batch.target_entity_id).sum().item()
        # num_total['all'] += batch.batch_size

        for idx in range(batch.batch_size):
            query = batch.query_input[0][:, idx]
            pos = ''
            if pointer_net.multi_hop:
                for token_id in query:
                    if token_id.item() == 4:
                        pos = 'subj'
                        break
                    elif token_id.item() == 5:
                        pos = 'dobj'
                        break
                    elif token_id.item() == 6:
                        pos = 'pobj'
                        break
            else:
                for token_id in query:
                    if token_id.item() == 1:
                        pos = 'subj'
                        break
                    elif token_id.item() == 2:
                        pos = 'dobj'
                        break
                    elif token_id.item() == 3:
                        pos = 'pobj'
                        break

            num_total['all'] += 1
            num_total[pos] += 1

            correct = (predicted[idx].item() ==
                       batch.target_entity_id[idx].item())
            results.append(int(correct))

            if correct:
                num_correct['all'] += 1
                num_correct[pos] += 1

            if report_entity_freq:
                entity_freq = \
                    Counter(batch.doc_entity_ids[:, idx].cpu().numpy())[
                        batch.target_entity_id[idx].item()]

                if entity_freq > 10:
                    entity_freq = 10

                num_total[entity_freq] += 1
                if correct:
                    num_correct[entity_freq] += 1

    eval_loss /= len(evaluation_iter)

    pointer_net.train(True)

    # print()
    log.info('evaluation loss = {:.8f}'.format(eval_loss))
    for key in num_correct.keys():
        log.info('{} accuracy = {:8d}/{:8d} = {:.2f}%'.format(
            key, num_correct[key], num_total[key],
            num_correct[key] / num_total[key] * 100))

    if return_results:
        return deepcopy(results)


def train_epoch(pointer_net, training_iter, validation_iter, optimizer,
                neg_loss_type='none', regularization=0.1, use_sum=False,
                use_sigmoid=False, self_attn_target_for_pred='none',
                backward_self_attn_loss=False,
                backward_first_hop_attn_loss=False, max_grad_norm=1.0,
                log_every=1000, val_every=50000):
    training_loss = 0.0
    training_loss_self_attn = 0.0
    training_loss_first_hop_attn = 0.0

    for batch in training_iter:
        losses = compute_batch_loss(
            pointer_net, batch, neg_loss_type=neg_loss_type,
            regularization=regularization, predict=False, use_sum=use_sum,
            use_sigmoid=use_sigmoid,
            self_attn_target_for_pred=self_attn_target_for_pred)

        training_loss += losses['main'].item()
        if pointer_net.use_self_attn:
            assert 'self_attn' in losses
            training_loss_self_attn += losses['self_attn'].item()
        if pointer_net.multi_hop:
            assert 'first_hop_attn' in losses
            training_loss_first_hop_attn += losses['first_hop_attn'].item()

        optimizer.zero_grad()

        loss = losses['main']
        if backward_self_attn_loss and pointer_net.use_self_attn:
            loss += losses['self_attn']
        if backward_first_hop_attn_loss and pointer_net.multi_hop:
            loss += losses['first_hop_attn']

        loss.backward()

        clip_grad_norm_(pointer_net.parameters(), max_norm=max_grad_norm)

        optimizer.step()

        if training_iter.iterations % log_every == 0:
            log.info(
                'Finished {:8d}/{:8d} batches, training loss in '
                'last {} batches = {:.8f}{}{}'.format(
                    training_iter.iterations, len(training_iter),
                    log_every, training_loss / log_every,
                    ', self_attn loss = {:.8f}'.format(
                        training_loss_self_attn / log_every)
                    if pointer_net.use_self_attn else '',
                    ', first_hop_attn loss = {:.8f}'.format(
                        training_loss_first_hop_attn / log_every)
                    if pointer_net.multi_hop else ''
                ))
            training_loss = 0.0
            training_loss_self_attn = 0.0
            training_loss_first_hop_attn = 0.0

        if training_iter.iterations % val_every == 0:
            _ = validate(
                pointer_net, validation_iter,
                self_attn_target_for_pred=self_attn_target_for_pred,
                msg='After {:8d} batches'.format(training_iter.iterations))

    num_remaining_batch = len(training_iter) % log_every
    if num_remaining_batch > 0:
        log.info(
            'Finished {:8d}/{:8d} batches, training loss in '
            'last {} batches = {:.8f}{}{}'.format(
                len(training_iter), len(training_iter),
                num_remaining_batch, training_loss / num_remaining_batch,
                ', self_attn loss = {:.8f}'.format(
                    training_loss_self_attn / num_remaining_batch)
                if pointer_net.use_self_attn else '',
                ', first_hop_attn loss = {:.8f}'.format(
                    training_loss_first_hop_attn / num_remaining_batch)
                if pointer_net.multi_hop else ''
            ))
