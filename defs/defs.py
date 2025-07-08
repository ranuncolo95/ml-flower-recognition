'''
How Callbacks Work During Training
1. Integration with model.fit()
When you add a callback to model.fit(), TensorFlow automatically calls its methods at specific training stages:

model.fit(..., callbacks=[TrainingMonitor()])


2. Training Loop Execution
For each epoch, TensorFlow executes callback methods in this order:

on_epoch_begin() → on_batch_begin() → on_batch_end() → on_epoch_end()

######
When Each Method is Called:

on_train_begin()
Called once at training start
Perfect for initializing variables

on_epoch_end()
Called after each epoch completes
Receives all metrics through logs dict
Best place for your diagnostics

'''

from tensorflow.keras.callbacks import Callback
import numpy as np
import pandas as pd


class TrainingMonitor(Callback):
    def __init__(self, patience=3):
        super().__init__()
        self.patience = patience
        self.best_weights = None
    
    def on_train_begin(self, logs=None):
        # Initialize tracking variables
        self.val_losses = []
        self.train_losses = []
        self.wait = 0  # For early stopping
        
    def on_epoch_end(self, epoch, logs=None):
        # 1. Record metrics
        current_val_loss = logs.get('val_loss', np.inf)
        current_train_loss = logs.get('loss', np.inf)
        self.val_losses.append(current_val_loss)
        self.train_losses.append(current_train_loss)
        
        # 2. Overfitting detection
        if epoch > 0:
            if (current_val_loss > self.val_losses[-2] and 
                current_train_loss < self.train_losses[-2]):
                print(f"\n⚠️ Overfitting alert! (Epoch {epoch+1})")
                self.wait += 1
                if self.wait >= self.patience:
                    self.model.stop_training = True
                    print("Stopping training due to persistent overfitting")
            else:
                self.wait = 0
        
        # 3. Underfitting check
        if logs.get('val_accuracy', 0) < 0.6 and epoch > 5:
            print("\n⚠️ Underfitting detected - consider model changes")
        
        # 4. Save best weights
        if current_val_loss == np.min(self.val_losses):
            self.best_weights = self.model.get_weights()



def analyze_training(history):
    hist = pd.DataFrame(history.history)
    best_epoch = hist['val_loss'].idxmin() + 1
    
    print("\n🔍 Training Analysis:")
    print(f"Optimal stopping epoch: {best_epoch}")
    print(f"Best val accuracy: {hist['val_accuracy'].max():.2f}")
    
    if hist['val_loss'].iloc[-1] > hist['val_loss'].iloc[-2]:
        print("⚠️ Model was still overfitting at training end")
    
    if hist['val_accuracy'].iloc[-1] < 0.7:
        print("⚠️ Potential underfitting - consider model capacity")