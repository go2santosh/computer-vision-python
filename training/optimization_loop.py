import torch

class OptimizationLoop:
    def __init__(self, model, loss_fn, device):
        self.model = model
        self.loss_fn = loss_fn
        self.device = device

    # Train loop
    def train_loop(self, dataloader, optimizer):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        batch_size = dataloader.batch_size
        train_loss = 0
        # Set the model to training mode
        self.model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)
            # Compute prediction and loss
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Update training loss
            train_loss += loss.item()
            if batch % 100 == 0:
                current = batch * batch_size + len(X)
                print(f"loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")
        
        # Return the final loss for the epoch
        average_loss = train_loss / num_batches
        return average_loss

    # Evaluate loop
    def evaluate_loop(self, dataloader):
        # Set the model to evaluation mode
        self.model.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        eval_loss, correct = 0, 0

        # Evaluating the model with torch.no_grad() 
        # ensures that no gradients are computed during evaluation mode
        # also serves to reduce unnecessary gradient computations
        # and memory usage for tensors with requires_grad=True
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                eval_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        eval_loss /= num_batches
        correct /= size
        print(f"Evaluation Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {eval_loss:>8f} \n")
        return eval_loss, correct
