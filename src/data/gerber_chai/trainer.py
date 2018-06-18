from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm

from utils import log
from .helper import compute_f1

optim_dict = {
    'adadelta': optim.Adadelta,
    'adagrad': optim.Adagrad,
    'adam': optim.Adam,
    'adamax': optim.Adamax,
    'rmsprop': optim.RMSprop,
    'sgd': optim.SGD
}


def print_msg(results, prefix, multi_hop=False):
    msg = '{} loss = {:.2f}{}, {} score = {:.2f}'.format(
        prefix,
        results[0],
        ', attention loss = {:.2f}'.format(results[1]) if multi_hop else '',
        prefix,
        results[-1])
    return msg


def compute_batch_loss(pointer_net, batch, regularization=0.0, predict=False,
                       multi_hop=False, predict_entity=False, max_margin=False,
                       use_sum=False):
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

    if multi_hop:
        attn_1, attn = pointer_net(
            doc_input_seqs=batch.doc_input[0],
            doc_input_lengths=batch.doc_input[1],
            query_input_seqs=batch.query_input[0],
            query_input_lengths=batch.query_input[1],
            softmax_mask=batch.candidate_mask,
            multi_hop=True,
            **kwargs
        )
    else:
        attn = pointer_net(
            doc_input_seqs=batch.doc_input[0],
            doc_input_lengths=batch.doc_input[1],
            query_input_seqs=batch.query_input[0],
            query_input_lengths=batch.query_input[1],
            softmax_mask=batch.candidate_mask,
            **kwargs
        )

    max_dice_scores = batch.dice_scores.max(dim=0)[0].unsqueeze(0)
    target_mask = (batch.dice_scores == max_dice_scores)

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

    if max_margin:
        neg_target_mask = (batch.dice_scores != max_dice_scores)
        neg_target_mask = neg_target_mask * batch.candidate_mask

        neg_masked_attn = attn * neg_target_mask.float()

        if use_sum:
            neg_logit_attn = neg_masked_attn.sum(dim=0)
        else:
            neg_logit_attn, _ = neg_masked_attn.max(dim=0)

        loss -= torch.log(1 - neg_logit_attn).sum() / batch.batch_size

    if multi_hop:
        attn_1_target = batch.argument_mask.float()
        attn_1_target /= attn_1_target.sum(dim=0).unsqueeze(0)

        attn_1_loss = F.kl_div(torch.log(attn_1), attn_1_target)
        loss = (loss, attn_1_loss)

    if predict:
        if predict_entity:
            predicted_entity_ids = \
                np.zeros(batch.batch_size, dtype=np.dtype(int))

            for ex_idx in range(batch.batch_size):
                doc_entity_ids = batch.doc_entity_ids[:, ex_idx]
                unique_entity_ids = torch.from_numpy(
                    np.unique(doc_entity_ids.data.cpu().numpy()))

                entity_attn_score_mapping = []
                for entity_id in unique_entity_ids:
                    entity_id_mask = (doc_entity_ids == entity_id).float()
                    entity_attn_score = \
                        (attn[:, ex_idx] * entity_id_mask).sum().data[0]
                    entity_attn_score_mapping.append(
                        (entity_attn_score, entity_id))

                predicted_entity_ids[ex_idx] = \
                    max(entity_attn_score_mapping)[1]

            predicted_entity_ids = Variable(
                torch.from_numpy(predicted_entity_ids).cuda().unsqueeze(0))

            entity_mask = (batch.doc_entity_ids == predicted_entity_ids)
            predicted_dice_scores = \
                (batch.dice_scores * entity_mask.float()).max(dim=0)[0]

        else:
            predicted_indices = attn.max(dim=0)[1].unsqueeze(0)
            predicted_dice_scores = \
                batch.dice_scores.gather(index=predicted_indices, dim=0)
            predicted_dice_scores = predicted_dice_scores.squeeze()

        return loss, predicted_dice_scores
    else:
        return loss


def validate(pointer_net, val_iter, multi_hop=False, predict_entity=False):
    pointer_net.train(False)

    val_loss = 0.0
    val_attn_loss = 0.0
    val_score = 0.0

    for batch in val_iter:
        loss, predicted_dice_scores = compute_batch_loss(
            pointer_net, batch, regularization=0.0, predict=True,
            multi_hop=multi_hop, predict_entity=predict_entity)
        if multi_hop:
            val_loss += loss[0].data[0]
            val_attn_loss += loss[1].data[0]
        else:
            val_loss += loss.data[0]

        val_score += predicted_dice_scores.sum().data[0]

    val_loss /= len(val_iter)
    if multi_hop:
        val_attn_loss /= len(val_iter)

    pointer_net.train(True)

    if multi_hop:
        return val_loss, val_attn_loss, val_score
    else:
        return val_loss, val_score


def train_epoch(pointer_net, train_iter, optimizer, regularization=0.0,
                max_grad_norm=None, multi_hop=False,
                backward_with_attn_loss=False, max_margin=False, use_sum=False):
    training_loss = 0.0
    training_attn_loss = 0.0

    for batch in train_iter:
        loss = compute_batch_loss(
            pointer_net, batch, regularization=regularization, predict=False,
            multi_hop=multi_hop, max_margin=max_margin, use_sum=use_sum)

        if multi_hop:
            training_loss += loss[0].data[0]
            training_attn_loss += loss[1].data[0]
        else:
            training_loss += loss.data[0]

        optimizer.zero_grad()

        if multi_hop:
            if backward_with_attn_loss:
                loss = loss[0] + loss[1]
            else:
                loss = loss[0]
        loss.backward()

        if max_grad_norm:
            clip_grad_norm(pointer_net.parameters(), max_norm=max_grad_norm)
        optimizer.step()

    training_loss /= len(train_iter)

    if multi_hop:
        training_attn_loss /= len(train_iter)
        return training_loss, training_attn_loss
    else:
        return training_loss


def cross_val(pointer_net, model_state_dict_path, iterators_by_fold, param_grid,
              verbose=0, num_gt=970, num_epochs=20, regularization=0.0,
              max_grad_norm=1.0, val_on_dice_score=True, multi_hop=False,
              predict_entity=False, backward_with_attn_loss=False,
              max_margin=False, use_sum=False,
              fix_embedding=False, fix_doc_encoder=False,
              fix_query_encoder=False, keep_models=False):

    model_state_dict_list = []

    for fold in range(len(iterators_by_fold)):
        log.info(
            '----------Training model for fold #{}'.format(fold))

        train_iter = iterators_by_fold[fold][0]
        val_iter = iterators_by_fold[fold][1]
        test_iter = iterators_by_fold[fold][2]

        best_state_dict = None
        best_param = None

        pointer_net.load_state_dict(torch.load(
            model_state_dict_path,
            map_location=lambda storage, loc: storage.cuda(0)))

        val_results = validate(
            pointer_net, val_iter, multi_hop=multi_hop,
            predict_entity=predict_entity)
        test_results = validate(
            pointer_net, test_iter, multi_hop=multi_hop,
            predict_entity=predict_entity)

        best_val_loss = val_results[0]
        best_val_score = val_results[-1]

        if verbose > 0:
            log.info('Before training, {}, {}'.format(
                print_msg(val_results, 'val', multi_hop=multi_hop),
                print_msg(test_results, 'test', multi_hop=multi_hop)))

        for param in param_grid:
            log.info('Train {} epochs using {} optimizer with lr = {}'.format(
                num_epochs, param['optimizer'], param['lr']))

            pointer_net.load_state_dict(torch.load(
                model_state_dict_path,
                map_location=lambda storage, loc: storage.cuda(0)))

            if fix_embedding:
                pointer_net.embedding.weight.requires_grad = False
            if fix_doc_encoder:
                for p in pointer_net.doc_encoder.parameters():
                    p.requires_grad = False
            if fix_query_encoder:
                for p in pointer_net.query_encoder.parameters():
                    p.requires_grad = False

            optimizer = optim_dict[param['optimizer']](
                filter(lambda p: p.requires_grad, pointer_net.parameters()),
                lr=param['lr'])

            for epoch in range(num_epochs):
                train_epoch(
                    pointer_net,
                    train_iter=train_iter,
                    optimizer=optimizer,
                    regularization=regularization,
                    max_grad_norm=max_grad_norm,
                    multi_hop=multi_hop,
                    backward_with_attn_loss=backward_with_attn_loss,
                    max_margin=max_margin,
                    use_sum=use_sum
                )

                if verbose > 1:
                    val_results = validate(
                        pointer_net, val_iter, multi_hop=multi_hop,
                        predict_entity=predict_entity)
                    test_results = validate(
                        pointer_net, test_iter, multi_hop=multi_hop,
                        predict_entity=predict_entity)

                    log.info('After {} epochs, {}, {}'.format(
                        epoch + 1,
                        print_msg(val_results, 'val', multi_hop=multi_hop),
                        print_msg(test_results, 'test', multi_hop=multi_hop)))

            val_results = validate(
                pointer_net, val_iter, multi_hop=multi_hop,
                predict_entity=predict_entity)
            test_results = validate(
                pointer_net, test_iter, multi_hop=multi_hop,
                predict_entity=predict_entity)

            log_msg = 'Finished, {}, {}'.format(
                print_msg(val_results, 'val', multi_hop=multi_hop),
                print_msg(test_results, 'test', multi_hop=multi_hop))

            if (val_on_dice_score and val_results[-1] > best_val_score) or \
                    (not val_on_dice_score and val_results[0] < best_val_loss):
                log_msg += ' (NEW BEST)'
                best_val_loss = val_results[0]
                best_val_score = val_results[-1]
                best_state_dict = deepcopy(pointer_net.state_dict())
                best_param = param

            log.info(log_msg)

        model_state_dict_list.append(best_state_dict)

    log.info('----------')

    num_dice = 0.0
    num_model = 0

    for fold in range(len(iterators_by_fold)):
        pointer_net.load_state_dict(model_state_dict_list[fold])

        test_iter = iterators_by_fold[fold][2]

        test_results = validate(
            pointer_net, test_iter, multi_hop=multi_hop,
            predict_entity=predict_entity)

        log.info('Fold #{} {}'.format(
            fold, print_msg(test_results, 'test', multi_hop=multi_hop)))

        num_dice += test_results[-1]

        num_model += len(test_iter.dataset.examples)

    log.info('----------')

    log.info('num_dice = {:.2f}, num_gt = {}, num_model = {}'.format(
        num_dice, num_gt, num_model))
    precision, recall, f1 = compute_f1(num_dice, num_gt, num_model)
    log.info('precision = {:.2f}, recall = {:.2f}, f1 = {:.2f}'.format(
        precision * 100, recall * 100, f1 * 100))

    if keep_models:
        return model_state_dict_list
    else:
        del model_state_dict_list
