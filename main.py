# models.py

import torch
from torch import nn
import torch.nn.functional as F
from sentiment_data import read_sentiment_examples, get_indexer_and_embedding
from torch.utils.data import DataLoader
import time
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from BOWmodels import SentimentDatasetBOW, NN2BOW, NN3BOW
from DANmodels import SentimentDatasetDAN, DANModel
from tokenizer import BasicTokenizer
from utils import plot_accuracy


# Training function
def train_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(data_loader):
        # X = X.float()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_train_loss = train_loss / num_batches
    accuracy = correct / size
    return accuracy, average_train_loss


# Evaluation function
def eval_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.eval()
    eval_loss = 0
    correct = 0
    for batch, (X, y) in enumerate(data_loader):
        # X = X.float()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        eval_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    average_eval_loss = eval_loss / num_batches
    accuracy = correct / size
    return accuracy, average_eval_loss


# Experiment function to run training and evaluation for multiple epochs
def experiment(model, train_loader, test_loader, num_epoch=100):
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    all_train_accuracy = []
    all_test_accuracy = []
    for epoch in range(num_epoch):
        train_accuracy, train_loss = train_epoch(train_loader, model, loss_fn, optimizer)
        all_train_accuracy.append(train_accuracy)

        test_accuracy, test_loss = eval_epoch(test_loader, model, loss_fn, optimizer)
        all_test_accuracy.append(test_accuracy)

        if epoch % 10 == 9:
            print(f'Epoch #{epoch + 1}: train accuracy {train_accuracy:.3f}, dev accuracy {test_accuracy:.3f}')
    
    return all_train_accuracy, all_test_accuracy


def main():

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run model training based on specified model type')
    parser.add_argument('--model', type=str, required=True, help='Model type to train (e.g., BOW)')
    parser.add_argument('--embedding', type=str, default=None, help="path of embedding file, if not specified random embeddings will be initialized")
    parser.add_argument('--tokenizer', type=str, default=None, help="specify the tokenizer path without the extension. E.g. specify /tokenizer/model/cpe1000. DO NOT include extension")
    
    # Parse the command-line arguments
    args = parser.parse_args()

    if args.embedding != None and args.tokenizer != None:
        raise argparse.ArgumentTypeError("You cannot use Glove embedding and tokenizer together...")

    # Load dataset
    start_time = time.time()

    if args.model == "BOW":
        train_data = SentimentDatasetBOW("data/train.txt")
        dev_data = SentimentDatasetBOW("data/dev.txt")
    elif args.model == "DAN":
        embedding_file = args.embedding
        tokenizer = None
        if args.tokenizer != None:
            tokenizer = BasicTokenizer()
            tokenizer.load(args.tokenizer)
        indexer, embedding, sentiment_ex = get_indexer_and_embedding(embeddings_file=embedding_file, embedding_dim=100, infile="data/train.txt", tokenizer=tokenizer, frozen=True)
        
        train_data = SentimentDatasetDAN(sentiment_ex, indexer=indexer, tokenizer=tokenizer)
        dev_sentiment_ex = read_sentiment_examples("data/dev.txt")
        dev_data = SentimentDatasetDAN(dev_sentiment_ex, indexer=indexer, tokenizer=tokenizer)

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(dev_data, batch_size=16, shuffle=False)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Data loaded in : {elapsed_time} seconds")


    # Check if the model type is "BOW"
    if args.model == "BOW":
        # Train and evaluate NN2
        start_time = time.time()
        print('\n2 layers:')
        nn2_train_accuracy, nn2_test_accuracy = experiment(NN2BOW(input_size=512, hidden_size=100), train_loader, test_loader)

        # Train and evaluate NN3
        print('\n3 layers:')
        nn3_train_accuracy, nn3_test_accuracy = experiment(NN3BOW(input_size=512, hidden_size=100), train_loader, test_loader)
        end_time = time.time()

        plot_accuracy([nn2_train_accuracy, nn3_train_accuracy], ['2 layers', '3 layers'], 'Training Accuracy', title='Training Accuracy for 2 and 3 Layer Networks', filename='train_accuracy_bow.png')
        plot_accuracy([nn2_test_accuracy, nn3_test_accuracy], ['2 layers', '3 layers'], 'Dev Accuracy', title='Dev Accuracy for 2 and 3 Layer Networks', filename='dev_accuracy_bow.png')
        
        print(f"Training done in : {end_time-start_time} seconds")
    elif args.model == "DAN":
        #TODO:  Train and evaluate your DAN
       
         # Train and evaluate DAN
        start_time = time.time()
        print('\nDeep Averaging Network:')
        dan_train_accuracy, dan_test_accuracy = experiment(DANModel(embedding_layer=embedding, hidden_size=256), train_loader, test_loader, num_epoch=100)
        end_time = time.time()
        
        plot_accuracy([dan_train_accuracy], ['DAN'], 'Training Accuracy', title='Training Accuracy for Deep Averaging Network', filename='train_accuracy_dan.png')
        plot_accuracy([dan_test_accuracy], ['DAN'], 'Dev Accuracy', title='Dev Accuracy for Deep Averaging Network', filename='dev_accuracy_dan.png')
        
        print(f"Training done in : {end_time-start_time} seconds")

if __name__ == "__main__":
    main()
