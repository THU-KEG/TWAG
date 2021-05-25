import torch
from torch import nn
from torch.nn import functional as F
from src.c_generate_soft.data import SOS, EOS


class Beam(object):
    def __init__(self, tokens, log_probs, state, context, coverage):
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state
        self.context = context
        self.coverage = coverage

    def extend(self, token, log_prob, state, context, coverage):
        return Beam(tokens=self.tokens + [token],
                    log_probs=self.log_probs + [log_prob],
                    state=state,
                    context=context,
                    coverage=coverage)

    def has_repeat_token(self, repeat_num):
        token_dic = {}
        for token in self.tokens:
            if token in token_dic.keys():
                token_dic[token] += 1
                if token_dic[token] > repeat_num:
                    return True
            else:
                token_dic[token] = 1
        return False

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def avg_log_prob(self):
        if (len(self.tokens) == 0):
            return 1e-6
        return sum(self.log_probs) / len(self.tokens)


class PointerGenerator(nn.Module):

    def __init__(self, word_lookup, dim_word, dim_h, num_layers, num_vocab, dropout, min_dec_len, max_dec_len,
                 beam_size, is_coverage, repeat_token_tres):
        super().__init__()
        self.word_lookup = word_lookup
        self.dim_h = dim_h
        self.min_dec_len = min_dec_len
        self.max_dec_len = max_dec_len
        self.beam_size = beam_size
        self.is_coverage = is_coverage
        self.repeat_token_tres = repeat_token_tres
        self.dropout = nn.Dropout(dropout)
        input_size = dim_word + dim_h
        self.decoder = nn.GRU(
            input_size=input_size,
            hidden_size=dim_h,
            num_layers=2,
            dropout=dropout,
            batch_first=True
        )
        self.out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * dim_h, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_vocab)
        )
        self.p_gen_linear = nn.Linear(2 * dim_h, 1)
        self.linear_enc = nn.Linear(dim_h, 256, bias=False)
        self.linear_dec = nn.Linear(dim_h, 256, bias=False)
        self.linear_att = nn.Linear(256, 1, bias=False)
        if is_coverage:
            self.linear_cov = nn.Linear(1, 256, bias=False)

    def attend_enc_hidden(self, enc_hiddens, enc_masks, dec_hiddens, coverage=None):
        """
        Args:
            - enc_hiddens (m, len_doc, dim_h)
            - enc_masks (m, len_doc)
            - dec_hiddens (m, len_gt, dim_h)
            - coverage (m, len_doc)
        """

        enc_hiddens = enc_hiddens.unsqueeze(1)  # (m, 1, len_doc, dim_h)
        dec_hiddens = dec_hiddens.unsqueeze(2)  # (m, len_gt, 1, dim_h)

        m, len_doc = enc_masks.size()
        len_gt = dec_hiddens.size(1)
        enc_masks = enc_masks.unsqueeze(1).expand(-1, len_gt, -1)  # (m, len_gt, len_doc)

        if self.is_coverage:
            # assert len_gt == 1, "Must decode step-by-step if coverage is used"

            coverage_input = coverage.view(m, 1, -1, 1)  # (m, 1, len_doc, 1)
            coverage_feat = self.linear_cov(coverage_input)  # (m, 1, len_doc, dim_h)
            att_feat = self.linear_enc(enc_hiddens) + self.linear_dec(dec_hiddens) + coverage_feat
        else:
            att_feat = self.linear_enc(enc_hiddens) + self.linear_dec(dec_hiddens)

        A = F.softmax(
            self.linear_att(torch.tanh(att_feat)),
            dim=2
        ).squeeze(-1)  # (m, len_gt, len_doc)
        A = A * enc_masks.float()
        norm_factor = A.sum(dim=2, keepdim=True)
        norm_factor[norm_factor < 1e-5] = 1.0  # avoid zero division
        A = A / norm_factor
        enc_weighted = torch.bmm(A, enc_hiddens.squeeze(1))  # (m, len_gt, dim_h)

        if self.is_coverage:
            coverage = coverage.view(m, 1, len_doc)
            __sent_coverage = A.sum(dim=1, keepdim=True)
            next_coverage = coverage + __sent_coverage
            min_ = torch.min(torch.cat([__sent_coverage, coverage], dim=1), dim=1)[0]
            step_loss = torch.sum(min_, dim=1).mean()
        else:
            next_coverage = None
            step_loss = torch.tensor(0.0).to(enc_hiddens.device)
        return A, enc_weighted, next_coverage, step_loss

    def decode(self, input_index, enc_hiddens, enc_masks, dec_inputs, dec_hiddens=None, coverage=None):
        """
        Args:
            - input_index [Tensor] (m, len_doc)
            - enc_hidden [Tensor] (m, len_doc, dim_h)
            - enc_masks (m, len_doc)
            - dec_inputs [Tensor] (m, len_dec, dim_w+dim_h)
            - dec_hiddens [Tensor] (1, m, dim_h)
            - coverage (m, len_doc)
        """
        if dec_hiddens is None:
            dec_outputs, dec_hiddens = self.decoder(dec_inputs)
        else:
            dec_outputs, dec_hiddens = self.decoder(dec_inputs, dec_hiddens)
            # dec_outputs (m, len_gt, dim_h)
        attn_dist, enc_weighted, next_coverage, step_loss = self.attend_enc_hidden(enc_hiddens, enc_masks, dec_outputs,
                                                                                   coverage)
        dec_outputs = torch.cat([enc_weighted, dec_outputs], dim=2)  # (m, len_gt, 2*dim_h)

        p_gen = torch.sigmoid(self.p_gen_linear(dec_outputs))  # (m, len_gt, 1)
        vocab_dist = F.softmax(self.out(dec_outputs), dim=2)  # (m, len_gt, num_vocab)
        vocab_dist = p_gen * vocab_dist
        attn_dist = (1 - p_gen) * attn_dist
        final_dist = vocab_dist.scatter_add_(2, input_index.unsqueeze(1).expand_as(attn_dist), attn_dist)

        # (m, len_dec_inputs, num_vocab), (1, m, dim_h)
        return final_dist, dec_hiddens, next_coverage, step_loss

    def sort_beams(self, beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)

    def forward(self, sent_hidden, input_index, enc_hiddens, enc_masks, dec_labels=None, dec_masks=None,
                prev_coverage=None):
        """
        Args:
            - sent_hidden [Tensor] (m, dim_h) : the encoded sentence hidden state
            - input_index [Tensor] (m, len_doc) : word index of input document
            - enc_hidden [Tensor] (m, len_doc, dim_h) : Concatenate sentences of one group, feed the document into a sentence encoder 
            to get hidden states of each word, for copying
            - enc_masks [Tensor] (m, len_doc)
            - dec_labels [Tensor] (m, len_gt) : word index of ground truth summary
            - dec_masks [Tensor] (m, len_gt)
            - prev_coverage [Tensor] (m, len_doc)
        Return:
            - pred_all [list of list] (m, (len_group_summary, ))
            - final_coverage [Tensor] (m, len_doc)
        """
        device = input_index.device
        len_doc = input_index.shape[1]
        if self.training:
            dec_inputs = self.dropout(
                self.word_lookup(dec_labels))  # (m, len_gt, dim_word) word embedding of ground truth summary

            m, len_gt = dec_inputs.size(0), dec_inputs.size(1)
            sent_hidden = sent_hidden.unsqueeze(1).expand(-1, len_gt, -1)  # (m, len_gt, dim_h)
            dec_inputs = torch.cat([sent_hidden, dec_inputs], dim=2)  # (m, len_gt, dim_w+dim_h)
            dec_hiddens = None

            loss_coverage = torch.tensor(0.0).to(device)
            if self.is_coverage:
                coverage = prev_coverage

                final_dist, dec_hiddens, next_coverage, step_loss = self.decode(input_index, enc_hiddens, enc_masks,
                                                                                dec_inputs, dec_hiddens, coverage)
                loss_coverage += step_loss

                loss_coverage = loss_coverage / len_gt
                final_coverage = next_coverage.squeeze(1)
            else:
                final_dist = self.decode(input_index, enc_hiddens, enc_masks, dec_inputs, dec_hiddens)[0]
                final_coverage = None

            criterion = nn.NLLLoss()
            dec_masks = dec_masks[:, :-1]  # (m, len_gt-1)
            final_dist = final_dist[:, :-1][dec_masks]  # (?, num_vocab)
            dec_labels = dec_labels[:, 1:][dec_masks]  # (?)
            loss = criterion(torch.log(final_dist + 1e-9), dec_labels)
            # return loss, loss_coverage
            return loss, loss_coverage, final_coverage
        else:
            m = len(input_index)
            d = self.dim_h
            chosen_beams = []
            sent_hidden = sent_hidden.view(m, 1, 1, d).expand(-1, self.beam_size, -1, -1).contiguous().view(
                m * self.beam_size, 1, d)  # (m*b, 1, dim_h)
            input_index = input_index.view(m, 1, -1).expand(-1, self.beam_size, -1).contiguous().view(
                m * self.beam_size, -1)  # (m*b, len_doc)
            enc_hiddens = enc_hiddens.view(m, 1, -1, d).expand(-1, self.beam_size, -1, -1).contiguous().view(
                m * self.beam_size, -1, d)  # (m*b, len_doc, dim_h)
            enc_masks = enc_masks.view(m, 1, -1).expand(-1, self.beam_size, -1).contiguous().view(m * self.beam_size,
                                                                                                  -1)  # (m*b, len_doc)

            state = None
            beams = [Beam(tokens=[SOS],
                          log_probs=[0.0],
                          state=None if state is None else state[i // m],
                          context=None,
                          coverage=prev_coverage.to(device))
                     for i in range(m * self.beam_size)]
            steps = 0
            while steps < self.max_dec_len:
                latest_tokens = torch.LongTensor([h.latest_token for h in beams]).to(device)
                dec_inputs = self.word_lookup(latest_tokens).unsqueeze(1)  # (m*b, 1, dim_w)
                dec_inputs = torch.cat([sent_hidden, dec_inputs], dim=2)  # (m*b, 1, dim_w+dim_h)
                if beams[0].state is None:
                    dec_hiddens = None
                else:
                    dec_hiddens = torch.stack([h.state for h in beams], dim=1)  # (num_layer, m*b, dim_h)

                if self.is_coverage:
                    coverage = [h.coverage for h in beams]
                    coverage = torch.cat(coverage, dim=0)  # (m*b, len_doc)
                    final_dist, dec_hiddens, coverage, _ = self.decode(input_index, enc_hiddens, enc_masks, dec_inputs,
                                                                       dec_hiddens, coverage)
                else:
                    final_dist, dec_hiddens, _, _ = self.decode(input_index, enc_hiddens, enc_masks, dec_inputs,
                                                                dec_hiddens)

                topk_log_probs, topk_ids = torch.topk(torch.log(final_dist + 1e-9), self.beam_size * 2)  # (m*b, 1, 2b)
                new_beams = []

                all_beams = []
                num_orig_beams = 1 if steps == 0 else self.beam_size
                for i in range(num_orig_beams):
                    h = beams[i]
                    state_i = dec_hiddens[:, i]  # (num_layer, dim_h)
                    context_i = None
                    coverage_i = coverage[i] if self.is_coverage else None
                    for j in range(self.beam_size * 2):  # for each of the top 2*beam_size hyps:
                        new_beam = h.extend(token=topk_ids[i, 0, j].item(),
                                            log_prob=topk_log_probs[i, 0, j].item(),
                                            state=state_i,
                                            context=context_i,
                                            coverage=coverage_i)
                        all_beams.append(new_beam)
                # prune new beams to number b, and pop results when reaching <end>
                for h in self.sort_beams(all_beams):
                    # remove repeat <unk>
                    if h.tokens.count(1) > self.repeat_token_tres:
                        all_beams.remove(h)
                        continue
                    if (h.latest_token == EOS):
                        if steps >= self.min_dec_len and len(chosen_beams) < self.beam_size:
                            chosen_beams.append(h)
                    else:
                        new_beams.append(h)
                    if len(new_beams) == self.beam_size:
                        break

                steps += 1
                beams = new_beams

            if len(chosen_beams) == 0:
                chosen_beams = beams[:self.beam_size]
            good_beams_index = []
            for index in range(len(chosen_beams)):
                chosen_beam = chosen_beams[index]
                if not chosen_beam.has_repeat_token(self.repeat_token_tres * 2):
                    good_beams_index.append(index)
            good_chosen_beams = [chosen_beams[k] for k in good_beams_index]
            if len(good_chosen_beams) > 0:
                final_beam = self.sort_beams(good_chosen_beams)[0]
            else:
                final_beam = self.sort_beams(chosen_beams)[0]
            results = final_beam.tokens
            final_coverage = final_beam.coverage

            return results, final_coverage
