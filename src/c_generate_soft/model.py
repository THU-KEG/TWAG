# coding: utf-8

import torch
import torch.nn as nn
from transformers import AlbertTokenizer, AlbertModel
from src.c_generate_soft.pointer_generator import PointerGenerator


class AlbertClassifierModel(nn.Module):
    def __init__(self, num_topics=5, dropout=0.1,
                 albert_model_dir="/data1/tsq/TWAG/data/pretrained_models/albert"):
        super(AlbertClassifierModel, self).__init__()
        self.ntopic = num_topics
        # self.albert_model = AlbertModel.from_pretrained('albert-base-v2')
        try:
            self.albert_model = AlbertModel.from_pretrained(albert_model_dir)
        except OSError:
            model = AlbertModel.from_pretrained('albert-base-v2')
            model.save_pretrained(albert_model_dir)
            self.albert_model = model

        self.fc_layer = nn.Linear(768, self.ntopic)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, use_cls=False):
        if (use_cls):
            cls_hidden = self.albert_model(input_ids=input_ids, attention_mask=attention_mask)[1]
            y = self.fc_layer(cls_hidden)
        else:
            all_hidden = self.albert_model(input_ids=input_ids, attention_mask=attention_mask)[0]
            mean_hidden = torch.mean(all_hidden, dim=1, keepdim=False)
            y = self.fc_layer(mean_hidden)
        return y

    def predict(self, input_ids, attention_mask, use_cls=False, include_noise=True):
        if (use_cls):
            cls_hidden = self.albert_model(input_ids=input_ids, attention_mask=attention_mask)[1]
            y = self.fc_layer(cls_hidden)
        else:
            all_hidden = self.albert_model(input_ids=input_ids, attention_mask=attention_mask)[0]
            mean_hidden = torch.mean(all_hidden, dim=1, keepdim=False)
            y = self.fc_layer(mean_hidden)

        if not (include_noise):
            y[:, -1] = -10

        predict_y = torch.softmax(y, dim=1)

        return predict_y


class TopicGenerator():
    def __init__(self, topic_num, device):
        self.topic_num = topic_num
        self.device = device

    def gather_topics(self, doc, doc_len, doc_label):
        """
        Args:
            - doc (n, len)
            - doc_len (n)
            - doc_label (n)
        Return:
            - topic_docs (topic_num, t_len) sentences grouped with topic, concated in each topic
            - topic_len (topic_num)
            - topic_masks (topic_num, max_topic_len)
            returns 'pt' tensors
        """

        assert len(doc) == len(doc_label)

        n, max_len = doc.size()
        topic_members = [[] for i in range(self.topic_num)]
        for i, label in enumerate(doc_label):
            topic_members[label].append(i)

        input_mask = torch.arange(max_len).view(1, max_len).to(self.device).lt(doc_len.view(n, 1))  # (n, max_sent_len)
        topic_docs = []
        for i in range(self.topic_num):
            indices = topic_members[i]
            topic_masks = input_mask[indices]
            topic_docs.append(doc[indices][topic_masks])  # (num_sents,)

        topic_len = torch.LongTensor([len(g) for g in topic_docs]).to(self.device)
        topic_docs = nn.utils.rnn.pad_sequence(topic_docs, batch_first=True)  # (m, max_topic_len)
        topic_masks = torch.arange(max(topic_len)).view(1, -1).to(self.device).lt(topic_len.view(self.topic_num, 1)).to(
            self.device)  # (m, max_len_of_group)

        return topic_docs, topic_len, topic_masks


class BiGRUEncoder(nn.Module):
    def __init__(self, dim_word, dim_h, num_layers, num_vocab, dropout):
        super(BiGRUEncoder, self).__init__()
        self.encoder = nn.GRU(input_size=dim_word,
                              hidden_size=dim_h // 2,
                              num_layers=num_layers,
                              batch_first=True,
                              bidirectional=True)

        self.word_lookup = nn.Embedding(num_vocab, dim_word)

    def forward(self, input, length):
        """
        Args:
            - input (bsz, len)
            - length (bsz, )
        Return:
            - hidden (num_topic, len, h_dim) : hidden state of each word
            - output (num_topic, h_dim) : sentence embedding
            - h_n (num_layers * 2, bsz, h_dim//2)
        """
        bsz, max_len = input.size(0), input.size(1)
        input = self.word_lookup(input)
        lengths_clamped = length.clamp(min=1, max=max_len)

        sorted_seq_lengths, indices = torch.sort(lengths_clamped, descending=True)
        _, desorted_indices = torch.sort(indices, descending=False)
        input = input[indices]
        packed_input = nn.utils.rnn.pack_padded_sequence(input, sorted_seq_lengths, batch_first=True)
        hidden, h_n = self.encoder(packed_input)
        # h_n is (num_layers * num_directions, bsz, h_dim//2)
        hidden = nn.utils.rnn.pad_packed_sequence(hidden, batch_first=True, total_length=max_len)[
            0]  # (bsz, max_len, h_dim)

        output = h_n[-2:, :, :]  # (2, bsz, h_dim//2), take the last layer's state
        output = output.permute(1, 0, 2).contiguous().view(bsz, -1)  # (bsz, h_dim), merge forward and backward h_n

        # recover order
        hidden = hidden[desorted_indices]
        output = output[desorted_indices]
        h_n = h_n[:, desorted_indices]

        # MASKING HERE
        # mask everything that had seq_length as 0 in input as 0
        output.masked_fill_((length == 0).view(-1, 1), 0)
        h_n.masked_fill_((length == 0).view(-1, 1), 0)

        return hidden, output, h_n


class DocumentDecoder(nn.Module):
    def __init__(self, dim_h, num_topics):
        super(DocumentDecoder, self).__init__()
        self.decoder = nn.GRUCell(input_size=dim_h, hidden_size=dim_h)
        self.out_linear = nn.Linear(dim_h, num_topics)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, hidden, avail_topic_mask=None):
        """
        Args:
            - input (bsz, dim_h)
            - hidden (bsz, dim_h)
            - avail_topic_mask (bsz, num_topics)
        Return:
            - hidden_out (bsz, dim_h) : hidden state of this step
            - topic_dist (bsz, num_topics) : probablity distribution of next sentence on topics
        """
        hidden_out = self.decoder(input, hidden)
        topic_dist = self.out_linear(hidden_out)
        topic_dist = self.softmax(topic_dist)
        # if not (avail_topic_mask is None):
        #     masked_dist = self.softmax(topic_dist * avail_topic_mask)
        # else:
        masked_dist = None

        return hidden_out, topic_dist, masked_dist


class TopicDecodeModel(nn.Module):
    def __init__(self, num_topics, num_vocab, dim_word, dim_h, max_dec_sent, num_layers, dropout, min_dec_len,
                 max_dec_len, beam_size, is_coverage, repeat_token_tres):
        super(TopicDecodeModel, self).__init__()
        self.encoder = BiGRUEncoder(dim_word, dim_h, num_layers, num_vocab, dropout)
        self.doc_decoder = \
            DocumentDecoder(dim_h, num_topics)  
        self.sent_decoder = PointerGenerator(word_lookup=nn.Embedding(num_vocab, dim_word),
                                             dim_word=dim_word,
                                             dim_h=dim_h,
                                             num_layers=num_layers,
                                             num_vocab=num_vocab,
                                             dropout=dropout,
                                             min_dec_len=min_dec_len,
                                             max_dec_len=max_dec_len,
                                             beam_size=beam_size,
                                             is_coverage=is_coverage,
                                             repeat_token_tres=repeat_token_tres)
        self.stop_fc_layer = nn.Linear(dim_h, 2)
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.num_topics = num_topics
        self.num_vocab = num_vocab
        self.dim_word = dim_word
        self.dim_h = dim_h
        self.max_dec_sent = max_dec_sent
        self.word_lookup = nn.Embedding(num_vocab, dim_word)
        self.is_coverage = is_coverage

    def forward(self, topic, topic_len, avail_topic_mask, abst=None, abs_len=None, abs_labels=None):
        device = topic.device
        max_topic_len = topic.shape[1]
        if not (abst is None):
            max_abs_len = abst.shape[1]

        assert topic.shape[0] == self.num_topics

        # print("max_topic_len", max_topic_len)
        # print("topic_len", topic_len)
        # print("topic", topic)
        # print("device", device)
        # print("topic_len.cpu() device", topic_len.cpu().device)

        t_hidden, topic_outputs, h_n = self.encoder(topic, topic_len)

        len_doc = t_hidden.shape[1]
        dim_h = t_hidden.shape[2]
        all_topic_corpus = topic.view(1, -1)  # (1,max_topic_len*num_topic)
        all_topic_len = all_topic_corpus.shape[1]
        topic_mask = torch.arange(all_topic_len).to(device).lt(all_topic_len).unsqueeze(0)  # (1, all_topic_len)
        all_topic_hidden = t_hidden.view(-1, dim_h).unsqueeze(0)  # (1,all_topic_len*num_topic,dim_h)
        input_doc = torch.mean(topic_outputs, 0).unsqueeze(0)
        hidden_doc = torch.zeros((1, self.dim_h)).to(device)

        all_sentence_losses = []
        all_coverage_losses = []
        all_stop_losses = []
        all_tokens = []
        all_coverage = [torch.zeros((all_topic_len)).to(device)]

        if (self.training):
            sent_num = abs_len.shape[0]  # no 'STOP' topic for the last step

        else:
            sent_num = self.max_dec_sent

        for step in range(sent_num):
            coverage = all_coverage[0].unsqueeze(0)
            # perform token-level decode
            if (self.training):
                abs_mask = torch.arange(max_abs_len).to(device).lt(abs_len[step]).unsqueeze(0)  # (1, max_abs_len)
                __abs = abst[step].unsqueeze(0)
                
                doc_hidden, topic_dist, masked_dist = self.doc_decoder(input_doc, hidden_doc, avail_topic_mask)
                topic_weight = topic_dist.view(-1, 1)
                sent_hiddens = torch.mul(topic_outputs, topic_weight)
                avg_sent_hidden = (torch.sum(sent_hiddens, dim=0)).unsqueeze(0)  # ( bsz, h_dim)
                h_0_sent = avg_sent_hidden + doc_hidden
                loss, loss_coverage, final_coverage = self.sent_decoder(h_0_sent, all_topic_corpus,
                                                                        all_topic_hidden,
                                                                        topic_mask, __abs,
                                                                        abs_mask, coverage
                                                                        )
                # stop for last sentence
                stop_probs = self.stop_fc_layer(avg_sent_hidden)  # (bsz, 2)
                if step < sent_num - 1:
                    should_stop_tensor = torch.tensor([[0] for bs in range(stop_probs.shape[0])]).view(-1).to(device)
                else:
                    should_stop_tensor = torch.tensor([[1] for bs in range(stop_probs.shape[0])]).view(-1).to(device)
                stop_loss = self.loss_func(stop_probs, should_stop_tensor)

                all_sentence_losses.append(loss)
                all_stop_losses.append(stop_loss)
                if (self.is_coverage):
                    all_coverage_losses.append(loss_coverage)

                # update decoder states
                input_doc = avg_sent_hidden
                hidden_doc = doc_hidden

                if (self.is_coverage):
                    all_coverage[0] = final_coverage.squeeze(0)

            else:
                doc_hidden, topic_dist, masked_dist = self.doc_decoder(input_doc, hidden_doc, avail_topic_mask)
                topic_weight = topic_dist.view(-1, 1)
                sent_hiddens = torch.mul(topic_outputs, topic_weight)
                avg_sent_hidden = (torch.sum(sent_hiddens, dim=0)).unsqueeze(0)  # ( bsz, h_dim)
                h_0_sent = avg_sent_hidden + doc_hidden
                predicted_tokens, final_coverage = self.sent_decoder(
                    h_0_sent, all_topic_corpus,
                    all_topic_hidden,
                    topic_mask, None, None, coverage)
                all_tokens.append(predicted_tokens)

                stop_probs = self.stop_fc_layer(avg_sent_hidden)  # (bsz, 2)
                stop_choice = torch.argmax(stop_probs).cpu().numpy()
                if (stop_choice == 1):
                    break
                # update decoder states
                input_doc = avg_sent_hidden
                hidden_doc = doc_hidden

                if (self.is_coverage):
                    all_coverage[0] = final_coverage.squeeze(0)

        if (self.training):
            sentence_loss = torch.mean(torch.stack(all_sentence_losses))
            coverage_loss = torch.mean(torch.stack(all_coverage_losses))
            stop_loss = torch.mean(torch.stack(all_stop_losses))

            return sentence_loss, coverage_loss, stop_loss
        else:
            pred_len = [len(p) for p in all_tokens]
            return all_tokens, pred_len
