import os
import json
import torch
import argparse

from model import SentenceVAE
from utils import to_var, idx2word, interpolate
from torch.utils.data import DataLoader

from nltk.tokenize import TweetTokenizer
        
from paranmt import ParaNMT

#from coco import Coco

def main(args):
    
    paranmt = ParaNMT(
        data_dir=args.data_dir,
        split='test',
        create_data=args.create_data,
        max_sequence_length=args.max_sequence_length,
        min_occ=args.min_occ
    )

    data_loader = DataLoader(
        dataset=paranmt,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    #with open(args.data_dir+'/ptb.vocab.json', 'r') as file:
    with open(args.data_dir+'/paranmt_vocab.json', 'r') as file:
        vocab = json.load(file)

    w2i, i2w = vocab['w2i'], vocab['i2w']

    model = SentenceVAE(
        vocab_size=len(w2i),
        sos_idx=w2i['<sos>'],
        eos_idx=w2i['<eos>'],
        pad_idx=w2i['<pad>'],
        unk_idx=w2i['<unk>'],
        max_sequence_length=args.max_sequence_length,
        embedding_size=args.embedding_size,
        rnn_type=args.rnn_type,
        hidden_size=args.hidden_size,
        word_dropout=args.word_dropout,
        embedding_dropout=args.embedding_dropout,
        latent_size=args.latent_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional
        )

    if not os.path.exists(args.load_checkpoint):
        raise FileNotFoundError(args.load_checkpoint)

    model.load_state_dict(torch.load(args.load_checkpoint))
    print("Model loaded from %s"%(args.load_checkpoint))

    if torch.cuda.is_available():
        model = model.cuda()
    
    model.eval()
    correct = 0
    total = 0

    for iteration, batch in enumerate(data_loader):
        batch_size = batch['original']['input'].size(0) # batch sizes are the same for original and paraphrase

        original = batch['original']['input'].long().cuda()
        paraphrase = batch['paraphrase']['target'].long().cuda()
        paraphrase_length = batch['paraphrase']['length']

        # Inference Network
        samples, z = model.inference(original)
        
        # Accuracy Measure
        mask = torch.arange(0, args.max_sequence_length).repeat(batch_size, 1)
        mask = mask < paraphrase_length.unsqueeze(1).repeat(1, args.max_sequence_length)
        result = (paraphrase.cpu()==samples.cpu())*mask
        correct += result.sum().item()
        total += paraphrase_length.sum().item()
        if iteration % 100 == 0:
            print("Batch {:5d} Accuracy : {:4d} / {:7d} = {:3.1f}%".format(iteration, correct, total, (100*correct/total)) )
        if iteration == 0:
            print(*idx2word(paraphrase[:5], i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')
            print(*idx2word(samples[:5], i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')
        #print('----------BATCH {:4d}----------'.format(iteration))
        #print(*idx2word(original, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')
        #print('-------------------------------')
        #print(*idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')
        #input() # just to pause for a bit

        # No Interpolation for now
        #z1 = torch.randn([args.latent_size]).numpy()
        #z2 = torch.randn([args.latent_size]).numpy()
        #z = to_var(torch.from_numpy(interpolate(start=z1, end=z2, steps=8)).float())
        
        # Temporary Input Sequence
        #samples, _ = model.inference(SAMPLE_EMBEDDING, z=z)
        #print('-------INTERPOLATION-------')
        #print(*idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--load_checkpoint', type=str)
    parser.add_argument('-n', '--num_samples', type=int, default=10)

    parser.add_argument('-dd', '--data_dir', type=str, default='data')
    parser.add_argument('-ms', '--max_sequence_length', type=int, default=60)
    parser.add_argument('-eb', '--embedding_size', type=int, default=300)
    parser.add_argument('-rnn', '--rnn_type', type=str, default='gru')
    parser.add_argument('-hs', '--hidden_size', type=int, default=256)
    parser.add_argument('-wd', '--word_dropout', type=float, default=0)
    parser.add_argument('-ed', '--embedding_dropout', type=float, default=0.5)
    parser.add_argument('-ls', '--latent_size', type=int, default=16)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-bi', '--bidirectional', action='store_true')
    
    # data loading args
    parser.add_argument('--create_data', action='store_true')
    parser.add_argument('--min_occ', type=int, default=1)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)

    args = parser.parse_args()

    args.rnn_type = args.rnn_type.lower()

    assert args.rnn_type in ['rnn', 'lstm', 'gru']
    assert 0 <= args.word_dropout <= 1

    main(args)
