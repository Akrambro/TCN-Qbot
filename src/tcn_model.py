"""
PyTorch-based Temporal Convolutional Network (TCN) for Forex Forecasting

Based on the research paper methodology described in AboutBot.pdf:
- Dilated causal convolutions for long-range dependencies
- Stacked residual blocks with exponentially increasing dilation
- Binary classification for up/down direction prediction
- Multivariate input (OHLC + technical indicators)

Reference: Bai et al. (2018) - Temporal Convolutional Networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in binary classification.
    
    Focal loss down-weights easy examples and focuses on hard, misclassified examples.
    This prevents the model from collapsing to always predicting the majority class.
    
    Formula: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: Weighting factor in [0, 1] to balance positive/negative examples
        gamma: Focusing parameter (gamma >= 0). Higher gamma = more focus on hard examples
        reduction: 'mean', 'sum', or 'none'
    
    Reference: Lin et al. (2017) - Focal Loss for Dense Object Detection
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits from model (before sigmoid) [batch_size, 1]
            targets: Ground truth binary labels [batch_size]
        """
        # Convert logits to probabilities
        probs = torch.sigmoid(inputs).squeeze()
        
        # Ensure targets are float
        targets = targets.float()
        
        # Calculate focal loss components
        # p_t = p if y==1, else (1-p)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # alpha_t = alpha if y==1, else (1-alpha)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Focal loss: -alpha_t * (1 - p_t)^gamma * log(p_t)
        focal_weight = alpha_t * (1 - p_t).pow(self.gamma)
        
        # BCE loss: -log(p_t)
        bce_loss = -torch.log(p_t + 1e-8)  # Add epsilon for numerical stability
        
        # Combine
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CausalConv1d(nn.Module):
    """
    Causal 1D convolution to ensure output at time t only depends on inputs <= t
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size,
            padding=self.padding, 
            dilation=dilation,
            **kwargs
        )
    
    def forward(self, x):
        # Apply causal padding by cropping the output
        x = self.conv(x)
        if self.padding > 0:
            return x[:, :, :-self.padding]
        return x


class TemporalBlock(nn.Module):
    """
    Residual block with two causal convolutions and gating
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        # First causal conv layer
        self.conv1 = CausalConv1d(
            n_inputs, n_outputs, kernel_size, 
            dilation=dilation, stride=stride
        )
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        # Second causal conv layer
        self.conv2 = CausalConv1d(
            n_outputs, n_outputs, kernel_size,
            dilation=dilation, stride=stride
        )
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Residual connection (1x1 conv if dimensions don't match)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Forward through main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        # Residual connection
        res = x if self.downsample is None else self.downsample(x)
        
        return self.relu(out + res)


class TCNForex(nn.Module):
    """
    Temporal Convolutional Network for Forex Direction Prediction
    
    Architecture as described in AboutBot.pdf:
    - Stacked temporal blocks with exponentially increasing dilation
    - Multivariate input channels (OHLC + indicators)
    - Binary classification output (up/down)
    
    Args:
        input_channels: Number of input features (e.g., 4 for OHLC, more with indicators)
        num_channels: List of hidden channel sizes for each temporal block
        kernel_size: Convolution kernel size (typically 3)
        dropout: Dropout rate for regularization
    """
    def __init__(
        self, 
        input_channels: int = 4,
        num_channels: List[int] = [16, 16, 8],
        kernel_size: int = 3,
        dropout: float = 0.1
    ):
        super(TCNForex, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation = 2 ** i  # Exponential dilation: 1, 2, 4, 8, 16, 32...
            in_channels = input_channels if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(
                TemporalBlock(
                    in_channels, 
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation,
                    dropout=dropout
                )
            )
        
        self.network = nn.Sequential(*layers)
        
        # Final classification layer
        self.fc = nn.Linear(num_channels[-1], 1)
        
        # Calculate receptive field
        self.receptive_field = self._calculate_receptive_field(kernel_size, num_levels)
    
    def _calculate_receptive_field(self, kernel_size: int, num_levels: int) -> int:
        """
        Calculate effective receptive field of the TCN
        For exponential dilation [1, 2, 4, 8...], receptive field = 2^(num_levels+1) - 1
        """
        return (kernel_size - 1) * (2 ** num_levels - 1) + 1
    
    def forward(self, x, return_logits=False):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, input_channels, sequence_length)
            return_logits: If True, return raw logits instead of probabilities
        
        Returns:
            predictions: Tensor of shape (batch_size,) with probabilities [0, 1] or logits
        """
        # Pass through TCN layers
        y = self.network(x)
        
        # Use last time step for prediction
        y = y[:, :, -1]  # Shape: (batch_size, num_channels[-1])
        
        # Final classification
        y = self.fc(y)  # Shape: (batch_size, 1)
        y = y.squeeze(1)  # Shape: (batch_size,)
        
        if return_logits:
            return y
        else:
            return torch.sigmoid(y)
    
    def get_receptive_field(self) -> int:
        """Return the receptive field size"""
        return self.receptive_field


class TCNTrainer:
    """
    Training utilities for TCN model
    """
    def __init__(
        self, 
        model: TCNForex,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    def train_epoch(
        self, 
        train_loader,
        criterion,
        optimizer,
        use_logits=False
    ) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            if use_logits:
                logits = self.model(X_batch, return_logits=True)
                loss = criterion(logits, y_batch.float())
                predictions = torch.sigmoid(logits)
            else:
                predictions = self.model(X_batch, return_logits=False)
                loss = criterion(predictions, y_batch.float())
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Metrics
            total_loss += loss.item() * X_batch.size(0)
            predicted_labels = (predictions > 0.5).long()
            correct += (predicted_labels == y_batch).sum().item()
            total += y_batch.size(0)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(
        self,
        val_loader,
        criterion,
        use_logits=False
    ) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # Forward pass
                if use_logits:
                    logits = self.model(X_batch, return_logits=True)
                    loss = criterion(logits, y_batch.float())
                    predictions = torch.sigmoid(logits)
                else:
                    predictions = self.model(X_batch, return_logits=False)
                    loss = criterion(predictions, y_batch.float())
                
                # Metrics
                total_loss += loss.item() * X_batch.size(0)
                predicted_labels = (predictions > 0.5).long()
                correct += (predicted_labels == y_batch).sum().item()
                total += y_batch.size(0)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def fit(
        self,
        train_loader,
        val_loader,
        epochs: int = 50,
        learning_rate: float = 0.001,
        early_stopping_patience: int = 10,
        verbose: bool = True,
        pos_weight: float = None,
        use_focal_loss: bool = False,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        weight_decay: float = 1e-4,
        use_lr_scheduler: bool = True
    ):
        """
        Train the model with early stopping
        
        Args:
            pos_weight: Weight for positive class to handle imbalance (None = no weighting)
            use_focal_loss: If True, use Focal Loss instead of BCE (recommended for imbalance)
            focal_alpha: Alpha parameter for Focal Loss
            focal_gamma: Gamma parameter for Focal Loss
            weight_decay: L2 regularization weight decay
            use_lr_scheduler: If True, use ReduceLROnPlateau scheduler
        """
        # Choose loss function
        if use_focal_loss:
            criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
            use_logits = True
            if verbose:
                print(f"Using Focal Loss (alpha={focal_alpha}, gamma={focal_gamma})")
        elif pos_weight is not None:
            pos_weight_tensor = torch.tensor([pos_weight]).to(self.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
            use_logits = True
        else:
            criterion = nn.BCELoss()
            use_logits = False
        
        # Add weight decay (L2 regularization)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Learning rate scheduler
        scheduler = None
        if use_lr_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5
            )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer, use_logits)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader, criterion, use_logits)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Learning rate scheduling
            if scheduler is not None:
                scheduler.step(val_loss)
            
            if verbose:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} - "
                      f"LR: {current_lr:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'models/best_tcn_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load('models/best_tcn_model.pt'))
        
        return self.history
    
    def predict(self, X: torch.Tensor) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Input tensor of shape (batch_size, input_channels, sequence_length)
        
        Returns:
            predictions: Array of probabilities
        """
        self.model.eval()
        with torch.no_grad():
            X = X.to(self.device)
            predictions = self.model(X)
            return predictions.cpu().numpy()
    
    def save_model(self, path: str):
        """Save model weights"""
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path: str):
        """Load model weights"""
        self.model.load_state_dict(torch.load(path))
        self.model.eval()


# Example usage
if __name__ == "__main__":
    # Example configuration as per AboutBot.pdf
    model = TCNForex(
        input_channels=4,  # OHLC
        num_channels=[16, 16, 8],  # 3 temporal blocks
        kernel_size=3,
        dropout=0.1
    )
    
    print(model)
    print(f"\nReceptive field: {model.get_receptive_field()} time steps")
    
    # Test forward pass
    batch_size = 64
    sequence_length = 60  # 60 minutes as recommended
    x = torch.randn(batch_size, 4, sequence_length)
    
    output = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
