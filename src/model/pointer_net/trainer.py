from collections import Counter
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from model.pointer_net.encoder import SelfAttentiveEncoder
from utils import log


# note that doc_entity_ids need to be transposed from batch input, i.e.,
# it should have shape of B * Ld
def get_self_attn_target(doc_entity_ids, input_lengths, zero_predicates=True):
    batch_size, max_len = doc_entity_ids.shape

    # get target based on same entity_ids
    self_attn_target = \
        (doc_entity_ids.unsqueeze(2) == doc_entity_ids.unsqueeze(1))

    # remove entries with -1 entity_id (either predicates or padding tokens)
    self_attn_target *= (doc_entity_ids != -1).unsqueeze(2)

    # restore entries of predicates by adding a diagonal matrix
    # (also prevents dividing-by-zero in the next step)
    self_attn_target += torch.eye(max_len).to(self_attn_target).unsqueeze(0)
    self_attn_target = torch.clamp(self_attn_target, max=1)

    # any changes to the attention targets of predicates should happen before
    # normalization (currently only attend to itself)

    # normalization
    self_attn_target = self_attn_target.float()
    self_attn_target /= self_attn_target.sum(dim=2, keepdim=True)

    # if zero_predicates is True, then mask out all rows of predicates
    if zero_predicates:
        target_mask = (doc_entity_ids != -1).unsqueeze(2).float()
    # otherwise, apply self_attn_mask
    else:
        target_mask = SelfAttentiveEncoder.get_mask_for_self_attention(
            input_lengths, max_len).float()

    self_attn_target *= target_mask

    return self_attn_target


def compute_batch_loss(pointer_net, batch, objective_type='normal',
                       regularization=0.1, predict=False, use_sum=False,
                       use_sigmoid=False):
    assert objective_type in [
        'normal', 'multi_arg', 'multi_slot', 'multi_hop', 'max_margin',
        'self_attn']

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

    if objective_type == 'self_attn':
        attn, self_attn = pointer_net(
            doc_input_seqs=batch.doc_input[0],
            doc_input_lengths=batch.doc_input[1],
            query_input_seqs=batch.query_input[0],
            query_input_lengths=batch.query_input[1],
            softmax_mask=softmax_mask,
            return_energy=use_sigmoid,
            **kwargs
        )

    elif objective_type == 'multi_hop':
        attn_1, attn = pointer_net(
            doc_input_seqs=batch.doc_input[0],
            doc_input_lengths=batch.doc_input[1],
            query_input_seqs=batch.query_input[0],
            query_input_lengths=batch.query_input[1],
            softmax_mask=softmax_mask,
            multi_hop=True,
            return_energy=use_sigmoid,
            **kwargs
        )

    elif objective_type == 'multi_slot':
        kwargs['neg_query_input_seqs'] = batch.neg_query_input[0]
        kwargs['neg_query_input_lengths'] = batch.neg_query_input[1]

        attn, neg_attn = pointer_net(
            doc_input_seqs=batch.doc_input[0],
            doc_input_lengths=batch.doc_input[1],
            query_input_seqs=batch.query_input[0],
            query_input_lengths=batch.query_input[1],
            softmax_mask=softmax_mask,
            return_energy=use_sigmoid,
            **kwargs
        )

    else:
        attn = pointer_net(
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

    loss = -torch.log(logit_attn).sum() / batch.batch_size

    if regularization > 0:
        reg_loss = 0.
        for param in pointer_net.parameters():
            reg_loss += param.norm() ** 2
        reg_loss = reg_loss ** 0.5
        loss += regularization * reg_loss

    if objective_type == 'self_attn':
        self_attn_target = get_self_attn_target(
            doc_entity_ids=batch.doc_entity_ids.transpose(0, 1),
            input_lengths=batch.doc_input[1],
            zero_predicates=True)
        self_attn_loss = F.kl_div(torch.log(self_attn), self_attn_target)
        loss = (loss, self_attn_loss)

    if objective_type == 'multi_hop':
        attn_1_target = batch.argument_mask.float()
        attn_1_target /= attn_1_target.sum(dim=0).unsqueeze(0)

        # attn_1_loss = F.kl_div(
        #     torch.log(attn_1), attn_1_target,
        #     size_average=False) / batch.batch_size
        attn_1_loss = F.kl_div(torch.log(attn_1), attn_1_target)
        loss = (loss, attn_1_loss)

    if objective_type == 'multi_arg':
        neg_target_mask = \
            batch.doc_entity_ids.eq(batch.neg_target_entity_id.unsqueeze(0))
        neg_masked_attn = attn * neg_target_mask.float()

        if use_sum:
            neg_logit_attn = neg_masked_attn.sum(dim=0)
        else:
            neg_logit_attn, _ = neg_masked_attn.max(dim=0)

        loss -= torch.log(1 - neg_logit_attn).sum() / batch.batch_size

    if objective_type == 'multi_slot':
        if use_sigmoid:
            neg_attn = torch.sigmoid(neg_attn) * softmax_mask.float()

        neg_masked_attn = neg_attn * target_mask.float()

        if use_sum:
            neg_logit_attn = neg_masked_attn.sum(dim=0)
        else:
            neg_logit_attn, _ = neg_masked_attn.max(dim=0)

        loss -= torch.log(1 - neg_logit_attn).sum() / batch.batch_size

    if objective_type == 'max_margin':
        neg_target_mask = \
            batch.doc_entity_ids.ne(batch.target_entity_id.unsqueeze(0))
        neg_target_mask = neg_target_mask * softmax_mask

        neg_masked_attn = attn * neg_target_mask.float()

        if use_sum:
            neg_logit_attn = neg_masked_attn.sum(dim=0)
        else:
            neg_logit_attn, _ = neg_masked_attn.max(dim=0)

        loss -= torch.log(1 - neg_logit_attn).sum() / batch.batch_size

    if predict:
        max_indices = attn.max(dim=0)[1].unsqueeze(0)
        predicted = batch.doc_entity_ids.gather(index=max_indices, dim=0)
        predicted = predicted.squeeze(0)
        return loss, predicted
    else:
        return loss


def validate(pointer_net, validation_iter, objective_type='normal', msg=''):
    pointer_net.train(False)

    val_loss = 0.0
    val_attn_loss = 0.0

    num_correct = 0
    num_total = 0

    for batch in validation_iter:
        loss, predicted = compute_batch_loss(
            pointer_net, batch, objective_type=objective_type,
            regularization=0, predict=True)
        if objective_type in ['multi_hop', 'self_attn']:
            val_loss += loss[0].item()
            val_attn_loss += loss[1].item()
        else:
            val_loss += loss.item()

        num_correct += \
            int(predicted.eq(batch.target_entity_id).float().sum().item())
        num_total += batch.batch_size

    val_loss /= len(validation_iter)
    if objective_type in ['multi_hop', 'self_attn']:
        val_attn_loss /= len(validation_iter)

    pointer_net.train(True)

    log.info(
        '{}: validation loss = {:.8f}{}, '
        'accuracy = {:8d}/{:8d} = {:.2f}%'.format(
            msg, val_loss,
            ', attention loss = {:.8f}'.format(val_attn_loss)
            if objective_type in ['multi_hop', 'self_attn'] else '',
            num_correct, num_total,
            num_correct / num_total * 100))


def evaluate(pointer_net, evaluation_iter, report_entity_freq=False,
             multi_hop=False, self_attn=False, return_results=False):
    pointer_net.train(False)

    results = []

    val_loss = 0.0

    num_correct = {'all': 0, 'subj': 0, 'dobj': 0, 'pobj': 0}
    num_total = {'all': 0, 'subj': 0, 'dobj': 0, 'pobj': 0}

    if report_entity_freq:
        for entity_freq in range(1, 11):
            num_correct[entity_freq] = 0
            num_total[entity_freq] = 0

    if multi_hop:
        objective_type = 'multi_hop'
    elif self_attn:
        objective_type = 'self_attn'
    else:
        objective_type = 'normal'

    for batch in evaluation_iter:
        loss, predicted = compute_batch_loss(
            pointer_net, batch, regularization=0, predict=True,
            objective_type=objective_type)

        if multi_hop or self_attn:
            val_loss += loss[0].item()
        else:
            val_loss += loss.item()

        # num_correct['all'] += \
        #     (predicted == batch.target_entity_id).sum().item()
        # num_total['all'] += batch.batch_size

        for idx in range(batch.batch_size):
            query = batch.query_input[0][:, idx]
            pos = ''
            if multi_hop:
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

    val_loss /= len(evaluation_iter)

    pointer_net.train(True)

    # print()
    log.info('validation loss = {:.8f}'.format(val_loss))
    for key in num_correct.keys():
        log.info('{} accuracy = {:8d}/{:8d} = {:.2f}%'.format(
            key, num_correct[key], num_total[key],
            num_correct[key] / num_total[key] * 100))

    if return_results:
        return deepcopy(results)


def train_epoch(pointer_net, training_iter, validation_iter, optimizer,
                objective_type='normal', backward_with_attn_loss=False,
                regularization=0.1, max_grad_norm=1.0, log_every=1000,
                val_every=50000, use_sum=False, use_sigmoid=False):
    training_loss = 0.0
    training_attn_loss = 0.0

    for batch in training_iter:
        loss = compute_batch_loss(
            pointer_net, batch, objective_type=objective_type,
            regularization=regularization, predict=False, use_sum=use_sum,
            use_sigmoid=use_sigmoid)

        if objective_type in ['multi_hop', 'self_attn']:
            training_loss += loss[0].item()
            training_attn_loss += loss[1].item()
        else:
            training_loss += loss.item()

        optimizer.zero_grad()

        if objective_type in ['multi_hop', 'self_attn']:
            if backward_with_attn_loss:
                loss = loss[0] + loss[1]
            else:
                loss = loss[0]
        loss.backward()

        clip_grad_norm_(pointer_net.parameters(), max_norm=max_grad_norm)

        optimizer.step()

        if training_iter.iterations % log_every == 0:
            log.info(
                'Finished {:8d}/{:8d} batches, training loss in '
                'last {} batches = {:.8f}{}'.format(
                    training_iter.iterations, len(training_iter),
                    log_every, training_loss / log_every,
                    ', attention loss = {:.8f}'.format(
                        training_attn_loss / log_every)
                    if objective_type in ['multi_hop', 'self_attn'] else ''
                ))
            training_loss = 0.0
            training_attn_loss = 0.0

        if training_iter.iterations % val_every == 0:
            validate(
                pointer_net, validation_iter, objective_type=objective_type,
                msg='After {:8d} batches'.format(training_iter.iterations))

    num_remaining_batch = len(training_iter) % log_every
    if num_remaining_batch > 0:
        log.info(
            'Finished {:8d}/{:8d} batches, training loss in '
            'last {} batches = {:.8f}{}'.format(
                len(training_iter), len(training_iter),
                num_remaining_batch, training_loss / num_remaining_batch,
                ', attention loss = {:.8f}'.format(
                    training_attn_loss / log_every)
                if objective_type in ['multi_hop', 'self_attn'] else ''
            ))
