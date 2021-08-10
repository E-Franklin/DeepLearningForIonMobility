import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

import wandb
from Collate import pad_sort_collate
from ConvNet import ConvNet
from DataUtils import load_training_data, get_vocab, load_validation_data
from LSTM_Models import LSTMOnehot
from evaluate import evaluate_model
from transformer import SequenceTransformer


def train_model():
    config = wandb.config

    # load the training data
    collate_fn = pad_sort_collate
    train_loader, scaler = load_training_data(collate_fn)
    val_loader = load_validation_data(collate_fn, scaler)

    model = build_model(config)

    # Loss and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # divide the learning rate by 10 the learning rate every 5 epochs
    s1_scheduler = StepLR(optimizer, step_size=1, gamma=0.1, verbose=True)
    plat_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1,
                                       threshold=0.0001, threshold_mode='abs', verbose=True)

    for epoch in range(config.num_epochs):
        print(f'Begin epoch {epoch}')
        train(model, config, train_loader, loss_fn, optimizer, epoch)
        med_error = evaluate_model(model, config.model_name, train_loader, config, 'training', scaler)
        evaluate_model(model, config.model_name, val_loader, config, 'validation', scaler)
        if epoch == 0:
            s1_scheduler.step()
        else:
            plat_scheduler.step(med_error)

    evaluate_model(model, config.model_name, val_loader, config, 'validation', scaler, plot_eval=True)
    print('Training complete. Saving model ' + config.model_name)
    torch.save(model.state_dict(), 'models/' + config.model_name + '.pt')
    print('Model saved.')


def build_model(config):
    # Define the model
    if config.model_type == 'LSTM':
        print('Building LSTM model')
        model = LSTMOnehot(config.embedding_dim, config.num_lstm_units, config.num_layers,
                           get_vocab(), config.bidirectional, config.use_charge, config.dropout).to(config.device)
    elif config.model_type == 'Conv':
        print('Building convolutional model')
        model = ConvNet(config.kernel, config.use_charge, embedding_dim=config.embedding_dim).to(config.device)
    elif config.model_type == 'Transformer':
        print('Building Transformer model')
        model = SequenceTransformer(get_vocab(), config.embedding_dim, config.num_attn_heads, config.dim_feed_fwd,
                                    config.num_layers, config.use_charge, dropout=config.dropout).to(config.device)
    else:
        print('Unsupported model type')
        model = None

    return model


def train(model, config, train_loader, loss_fn, optimizer, epoch):
    model.train()
    sum_loss = 0
    for i, (seqs, charges, targets, lengths) in enumerate(train_loader):
        seqs = seqs.to(config.device).long()
        targets = targets.view(config.batch_size, 1).to(config.device).float()
        charges = charges.to(config.device)

        # Forward pass
        if config.model_type == 'LSTM':
            outputs = model(seqs, charges, lengths)
        else:
            outputs = model(seqs, charges)

        loss = loss_fn(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sum_loss += loss.item()

        if (i + 1) % 100 == 0:
            wandb.log({'epoch': epoch, 'loss': sum_loss / 100})
            sum_loss = 0
