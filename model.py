import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from utils import to_var

class SentenceVAE(nn.Module):

    def __init__(self, vocab_size, embedding_size, rnn_type, hidden_size, word_dropout, embedding_dropout, latent_size,
                sos_idx, eos_idx, pad_idx, unk_idx, max_sequence_length, num_layers=1, bidirectional=False):

        super().__init__()
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

        self.max_sequence_length = max_sequence_length
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx

        self.latent_size = latent_size

        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.word_dropout_rate = word_dropout
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        if rnn_type == 'rnn':
            rnn = nn.RNN
        elif rnn_type == 'gru':
            rnn = nn.GRU
        # elif rnn_type == 'lstm':
        #     rnn = nn.LSTM
        else:
            raise ValueError()
        
        # MY FIX: Added a paraphrase encoder
        #self.encoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional, batch_first=True)
        self.original_encoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional, batch_first=True)
        self.paraphrase_encoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional, batch_first=True)
        self.decoder_encoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional, batch_first=True)
        self.decoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional, batch_first=True)

        self.hidden_factor = (2 if bidirectional else 1) * num_layers

        self.hidden2mean = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.hidden2logv = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.latent2hidden = nn.Linear(latent_size, hidden_size * self.hidden_factor)
        self.outputs2vocab = nn.Linear(hidden_size * (2 if bidirectional else 1), vocab_size)

    # Added a second argument for paraphrase
    #def forward(self, input_sequence, length):
    def forward(self, original, original_length, paraphrase, paraphrase_length):
        batch_size = original.size(0) #batch sizes are the same for original and paraphrase
        
        # sort inputs
        original_sorted_lengths, original_sorted_idx = torch.sort(original_length, descending=True)
        original = original[original_sorted_idx]
        paraphrase_sorted_lengths, paraphrase_sorted_idx = torch.sort(paraphrase_length, descending=True)
        paraphrase = paraphrase[paraphrase_sorted_idx]

        # FIRST ENCODER
        original_embedding = self.embedding(original)
        original_packed_input = rnn_utils.pack_padded_sequence(original_embedding, original_sorted_lengths.data.tolist(), batch_first=True)
        _, original_hidden = self.original_encoder_rnn(original_packed_input)

        # SECOND ENCODER
        paraphrase_embedding = self.embedding(paraphrase)
        paraphrase_packed_input = rnn_utils.pack_padded_sequence(paraphrase_embedding, paraphrase_sorted_lengths.data.tolist(), batch_first=True)
        _, paraphrase_hidden = self.paraphrase_encoder_rnn(paraphrase_packed_input, original_hidden) 

        if self.bidirectional or self.num_layers > 1:
            # flatten hidden state
            paraphrase_hidden = paraphrase_hidden.view(batch_size, self.hidden_size*self.hidden_factor)
        else:
            paraphrase_hidden = paraphrase_hidden.squeeze()

        # REPARAMETERIZATION
        mean = self.hidden2mean(paraphrase_hidden)
        logv = self.hidden2logv(paraphrase_hidden)
        std = torch.exp(0.5 * logv)

        z = to_var(torch.randn([batch_size, self.latent_size]))
        z = z * std + mean

        # DECODER
        z_hidden = self.latent2hidden(z)
        z_hidden = z_hidden.unsqueeze(0)
        decoder_original_embedding = self.embedding(original)
        decoder_original_packed_input = rnn_utils.pack_padded_sequence(decoder_original_embedding, original_sorted_lengths.data.tolist(), batch_first=True)
        _, hidden = self.decoder_encoder_rnn(decoder_original_packed_input, z_hidden)
        
        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, 2 * self.hidden_size)        

        # decoder input
        if self.word_dropout_rate > 0:
            # randomly replace decoder input with <unk>
            prob = torch.rand(paraphrase.size())
            if torch.cuda.is_available():
                prob=prob.cuda()
            prob[(paraphrase.data - self.sos_idx) * (paraphrase.data - self.pad_idx) == 0] = 1
            decoder_input_sequence = paraphrase.clone()
            decoder_input_sequence[prob < self.word_dropout_rate] = self.unk_idx
            paraphrase_embedding = self.embedding(decoder_input_sequence)
        paraphrase_embedding = self.embedding_dropout(paraphrase_embedding)
        packed_input = rnn_utils.pack_padded_sequence(paraphrase_embedding, paraphrase_sorted_lengths.data.tolist(), batch_first=True)

        # decoder forward pass
        outputs, _ = self.decoder_rnn(packed_input, hidden)

        # process outputs
        padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True)[0]
        padded_outputs = padded_outputs.contiguous()
        _,reversed_idx = torch.sort(paraphrase_sorted_idx)
        padded_outputs = padded_outputs[reversed_idx]
        b,s,_ = padded_outputs.size()

        # project outputs to vocab
        reshaped_padded_outputs = padded_outputs.view(-1, padded_outputs.size(2))
        vocab = self.outputs2vocab(reshaped_padded_outputs)
        logp = nn.functional.log_softmax(vocab, dim=-1)
        logp = logp.view(b, s, self.embedding.num_embeddings)


        return logp, mean, logv, z


    # our new inference scheme takes in an original to paraphrase
    #def inference(self, n=4, z=None):
    def inference(self, original, z=None):
        batch_size = original.size(0) #batch sizes are the same for original and paraphrase
        
        # sample z
        z = to_var(torch.randn([batch_size, self.latent_size]))
        
        # Create initial hidden
        z_hidden = self.latent2hidden(z)
        z_hidden = z_hidden.unsqueeze(0)
        decoder_original_embedding = self.embedding(original)
        _, hidden = self.decoder_encoder_rnn(decoder_original_embedding, z_hidden)
        
        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)

        # required for dynamic stopping of sentence generation
        sequence_idx = torch.arange(0, batch_size, out=self.tensor()).long() # all idx of batch
        sequence_running = torch.arange(0, batch_size, out=self.tensor()).long() # all idx of batch which are still generating
        sequence_mask = torch.ones(batch_size, out=self.tensor()).bool()

        running_seqs = torch.arange(0, batch_size, out=self.tensor()).long() # idx of still generating sequences with respect to current loop

        generations = self.tensor(batch_size, self.max_sequence_length).fill_(self.pad_idx).long()

        t=0
        while(t<self.max_sequence_length and len(running_seqs)>0):

            if t == 0:
                input_sequence = to_var(torch.Tensor(batch_size).fill_(self.sos_idx).long())

            input_sequence = input_sequence.unsqueeze(1)

            input_embedding = self.embedding(input_sequence)

            output, hidden = self.decoder_rnn(input_embedding, hidden)

            logits = self.outputs2vocab(output)

            input_sequence = self._sample(logits)

            # save next input
            generations = self._save_sample(generations, input_sequence, sequence_running, t)

            # update gloabl running sequence
            sequence_mask[sequence_running] = (input_sequence != self.eos_idx).data
            sequence_running = sequence_idx.masked_select(sequence_mask)

            # update local running sequences
            running_mask = (input_sequence != self.eos_idx).data
            running_seqs = running_seqs.masked_select(running_mask)

            # prune input and hidden state according to local update
            if len(running_seqs) > 0:
                if len(input_sequence.shape) == 0:
                    input_sequence = input_sequence.unsqueeze(0)
                input_sequence = input_sequence[running_seqs]
                hidden = hidden[:, running_seqs]

                running_seqs = torch.arange(0, len(running_seqs), out=self.tensor()).long()

            t += 1

        return generations, z

    def _sample(self, dist, mode='greedy'):

        if mode == 'greedy':
            _, sample = torch.topk(dist, 1, dim=-1)
        sample = sample.squeeze()

        return sample

    def _save_sample(self, save_to, sample, running_seqs, t):
        # select only still running
        running_latest = save_to[running_seqs]
        # update token at position t
        running_latest[:,t] = sample.data
        # save back
        save_to[running_seqs] = running_latest

        return save_to
