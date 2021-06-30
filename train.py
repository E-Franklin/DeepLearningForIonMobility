import torch
from torch import nn

import wandb
from Collate import pad_sort_collate
from ConvNet import ConvNet
from DataUtils import load_training_data, get_vocab
from LSTM_Models import LSTMOnehot
from evaluate import evaluate_model_quick, evaluate_model
from transformer import SequenceTransformer


def train():
    config = wandb.config

    # load the training data
    collate_fn = pad_sort_collate
    train_loader, scaler = load_training_data(collate_fn)

    # Define the model
    if config.model_type == 'LSTM':
        print('Building LSTM model')
        model = LSTMOnehot(config.embedding_dim, config.num_lstm_units, config.num_layers,
                           get_vocab(), config.bidirectional, config.use_charge).to(config.device)
    elif config.model_type == 'Conv':
        print('Building convolutional model')
        model = ConvNet(config.kernel, config.use_charge, embedding_dim=config.embedding_dim).to(config.device)
    elif config.model_type == 'Transformer':
        print('Building Transformer model')
        model = SequenceTransformer(get_vocab(), config.embedding_dim, config.num_attn_heads, config.dim_feed_fwd,
                                    config.num_layers, config.use_charge, config.dropout).to(config.device)
    else:
        print('Unsupported model type')
        return

    # Loss and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    sum_loss = 0

    for epoch in range(config.num_epochs):
        for i, (seqs, charges, targets, lengths) in enumerate(train_loader):
            seqs = seqs.to(config.device).long()
            targets = targets.view(config.batch_size, 1).to(config.device).float()
            charges = charges.to(config.device)

            # Forward pass
            outputs = model(seqs, charges, lengths)
            loss = loss_fn(outputs, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()

            if (i + 1) % 100 == 0:
                wandb.log({'epoch': epoch, 'loss': sum_loss / 100})
                sum_loss = 0

    print('Training complete. Saving model ' + config.model_name)
    torch.save(model.state_dict(), 'models/' + config.model_name + '.pt')
    print('Model saved.')

    if config.plot_eval:
        evaluate_model(model, config.model_name, collate_fn, config, scaler)
    else:
        evaluate_model_quick(model, collate_fn, config, scaler)
