import data
import util
import transformer
import torch
import os
import sys

# Create model_snapshot directory if it doesn't exist
os.makedirs('model_snapshot', exist_ok=True)

device = util.device
print(device)

def train_model(text_filename, epochs=100, lr=0.0004, weight_decay=0.1, seed=123):
    """
    Train a GPT model on a given text file.
    
    Args:
        text_filename (str): Path to the text file to train on
        epochs (int): Number of training epochs
        lr (float): Learning rate for the optimizer
        weight_decay (float): Weight decay for the optimizer
        seed (int): Random seed for reproducibility
    
    Returns:
        tuple: Trained model and list of losses
    """
    # Extract just the filename without any directory path (e.g., "data/alice.txt" -> "alice_txt")
    # First get the base filename without directory
    base_filename = os.path.basename(text_filename)
    # Then replace dots with underscores to avoid confusion with file extensions
    base_filename = base_filename.replace('.', '_')
    
    print(f"Model snapshots will be saved with prefix: {base_filename}")
    
    torch.manual_seed(seed)
    model = transformer.GPTModel()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    tokens_seen, global_step = 0, -1
    
    losses = []
    
    train_loader = data.get_train_loader(text_filename)
    
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        
        epoch_loss = 0
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # Reset loss gradients from previous batch iteration
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    
            logits = model(input_batch)
            loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
            epoch_loss += loss.item()
            loss.backward() # Calculate loss gradients
            optimizer.step() # Update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1
    
            if global_step % 1000 == 0:
                # Save your models, log metrics, etc.
                print(f"Step {global_step}, Loss: {loss.item():.4f}, Tokens seen: {tokens_seen}")
                # Include text filename in the checkpoint name
                torch.save(model.state_dict(), f"model_snapshot/{base_filename}_model_step_{global_step}.pt")
        
        # Log epoch results
        avg_epoch_loss = epoch_loss / len(train_loader)
        losses.append(avg_epoch_loss)
        print(f"Epoch {epoch}, Average Loss: {avg_epoch_loss:.4f}")
    
    # Save final model with text filename included
    torch.save(model.state_dict(), f"model_snapshot/{base_filename}_model_final.pt")
    return model, losses

# Run training with command line arguments
if __name__ == "__main__":
    # Check if filename is provided as command line argument
    if len(sys.argv) > 1:
        # Use the first command line argument as the filename
        filename = sys.argv[1]
    else:
        # Default to Alice in Wonderland text if no filename is provided
        filename = "data/alice.txt_cleaned.txt"
        print(f"No filename provided, using default: {filename}")
    
    print(f"Training on: {filename}")
    trained_model, training_losses = train_model(filename)

