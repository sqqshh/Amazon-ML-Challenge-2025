import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from sklearn.model_selection import KFold
from tqdm import tqdm
import re
import warnings
warnings.filterwarnings('ignore')

# ============================================================================ 
# 1. CONFIGURATION
# ============================================================================
class Config:
    TRAIN_CSV = 'train.csv'
    TEST_CSV = 'test.csv'
    OUTPUT_FILE = 'test_out_stella.csv'
    
    TEXT_MODEL = 'Marqo/dunzhang-stella_en_400M_v5'
    MAX_LENGTH = 448
    EMBEDDING_DIM = 1024
    HIDDEN_DIMS = [512, 256, 128]
    DROPOUT = 0.15
    
    BATCH_SIZE = 64
    EPOCHS = 20
    LR = 2e-5
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    N_FOLDS = 5
    USE_KFOLD = False
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SEED = 42
    
    GRADIENT_ACCUMULATION = 2
    GRAD_CLIP = 1.0
    LABEL_SMOOTHING = 0.0
    USE_PSEUDO_LABELING = False

# ============================================================================ 
# 2. TEXT PREPROCESSING
# ============================================================================
def extract_features(text):
    text = str(text).lower()
    features = {}
    ipq_match = re.search(r'ipq[:\s]*(\d+)', text)
    features['ipq'] = float(ipq_match.group(1)) if ipq_match else 1.0
    numbers = re.findall(r'\d+\.?\d*', text)
    features['max_number'] = max([float(n) for n in numbers]) if numbers else 0.0
    features['num_count'] = len(numbers)
    features['text_length'] = len(text)
    features['word_count'] = len(text.split())
    premium_brands = ['sony', 'apple', 'samsung', 'nike', 'adidas', 'lg', 'dell', 'hp']
    features['is_premium'] = 1.0 if any(brand in text for brand in premium_brands) else 0.0
    return features

# ============================================================================ 
# 3. DATASET CLASS
# ============================================================================
class ProductDataset(Dataset):
    def __init__(self, df, tokenizer, is_test=False):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.is_test = is_test
        self.features = [extract_features(t) for t in df['catalog_content']]

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = re.sub(r'[^\w\s.,]', ' ', str(row['catalog_content']))
        text = re.sub(r'\s+', ' ', text).strip()
        encoding = self.tokenizer(
            text,
            max_length=Config.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        feat = self.features[idx]
        additional_features = torch.tensor([
            feat['ipq'], feat['max_number'], feat['num_count'],
            feat['text_length'], feat['word_count'], feat['is_premium']
        ], dtype=torch.float32)
        additional_features = torch.log1p(additional_features)
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'features': additional_features,
            'sample_id': row['sample_id']
        }
        if not self.is_test:
            price = np.log1p(row['price'])
            item['price'] = torch.tensor(price, dtype=torch.float32)
            item['raw_price'] = torch.tensor(row['price'], dtype=torch.float32)
        return item

# ============================================================================ 
# 4. MODEL
# ============================================================================
class PricePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(Config.TEXT_MODEL)
        for param in self.text_encoder.embeddings.parameters():
            param.requires_grad = False
        for layer in self.text_encoder.encoder.layer[:-8]:
            for param in layer.parameters():
                param.requires_grad = False
        self.feature_projection = nn.Sequential(
            nn.Linear(6, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(Config.DROPOUT)
        )
        self.attention = nn.Sequential(nn.Linear(Config.EMBEDDING_DIM, 1), nn.Softmax(dim=1))
        input_dim = Config.EMBEDDING_DIM + 64
        layers = []
        for hidden_dim in Config.HIDDEN_DIMS:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(Config.DROPOUT)
            ])
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.regressor = nn.Sequential(*layers)
        
    def forward(self, input_ids, attention_mask, features):
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        sequence_output = outputs.last_hidden_state
        attention_weights = self.attention(sequence_output)
        text_features = torch.sum(attention_weights * sequence_output, dim=1)
        feat_proj = self.feature_projection(features)
        combined = torch.cat([text_features, feat_proj], dim=1)
        log_price = self.regressor(combined)
        return log_price.squeeze(-1)

# ============================================================================ 
# 5. LOSS
# ============================================================================
class HuberSMAPELoss(nn.Module):
    def __init__(self, delta=1.0, smape_weight=0.5):
        super().__init__()
        self.delta = delta
        self.smape_weight = smape_weight
        self.huber = nn.HuberLoss(delta=delta)
    def forward(self, log_pred, log_target, raw_target):
        huber_loss = self.huber(log_pred, log_target)
        pred = torch.expm1(log_pred)
        target = raw_target
        epsilon = 1e-8
        smape = torch.mean(200 * torch.abs(pred - target) / (torch.abs(target) + torch.abs(pred) + epsilon))
        return (1 - self.smape_weight) * huber_loss + self.smape_weight * smape

# ============================================================================ 
# 6. TRAINING FUNCTIONS
# ============================================================================
def train_epoch(model, dataloader, optimizer, scheduler, criterion, device, accumulation_steps):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    progress_bar = tqdm(dataloader, desc='Training')
    for idx, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        features = batch['features'].to(device)
        prices = batch['price'].to(device)
        raw_prices = batch['raw_price'].to(device)

        predictions = model(input_ids, attention_mask, features)
        loss = criterion(predictions, prices, raw_prices)
        loss = loss / accumulation_steps

        loss.backward()

        if (idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        total_loss += loss.item() * accumulation_steps
        progress_bar.set_postfix({'loss': loss.item() * accumulation_steps})

    return total_loss / len(dataloader)

def validate(model, dataloader, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            features = batch['features'].to(device)
            raw_prices = batch['raw_price'].to(device)
            log_predictions = model(input_ids, attention_mask, features)
            predictions = torch.expm1(log_predictions)
            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(raw_prices.cpu().numpy())
    all_preds, all_targets = np.array(all_preds), np.array(all_targets)
    epsilon = 1e-8
    smape = np.mean(200 * np.abs(all_preds - all_targets) / (np.abs(all_targets) + np.abs(all_preds) + epsilon))
    return smape

# ============================================================================ 
# 7–10 SAME (unchanged, continue your script)
# ============================================================================
# keep the rest of your code (train_kfold, train_single, predict, main)


# ============================================================================ 
# 7. K-FOLD TRAINING
# ============================================================================
def train_kfold():
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    
    print(f"Using device: {Config.DEVICE}")
    
    # Load data
    print("Loading training data...")
    train_df = pd.read_csv(Config.TRAIN_CSV)
    print(f"Training samples: {len(train_df)}")
    print(f"Price range: ${train_df['price'].min():.2f} - ${train_df['price'].max():.2f}")
    print(f"Price median: ${train_df['price'].median():.2f}")
    
    tokenizer = AutoTokenizer.from_pretrained(Config.TEXT_MODEL)
    
    kfold = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_df)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold + 1}/{Config.N_FOLDS}")
        print(f"{'='*60}")
        
        train_fold = train_df.iloc[train_idx]
        val_fold = train_df.iloc[val_idx]
        
        train_dataset = ProductDataset(train_fold, tokenizer)
        val_dataset = ProductDataset(val_fold, tokenizer)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            num_workers=4,
            pin_memory=True
        )
        
        # Initialize model
        model = PricePredictor().to(Config.DEVICE)
        
        # Optimizer and scheduler
        num_training_steps = len(train_loader) * Config.EPOCHS // Config.GRADIENT_ACCUMULATION
        num_warmup_steps = int(num_training_steps * Config.WARMUP_RATIO)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.LR,
            weight_decay=Config.WEIGHT_DECAY,
            eps=1e-8
        )
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        criterion = HuberSMAPELoss(delta=1.0, smape_weight=0.6)
        
        best_smape = float('inf')
        patience = 3
        patience_counter = 0
        
        for epoch in range(Config.EPOCHS):
            print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
            
            train_loss = train_epoch(
                model, train_loader, optimizer, scheduler, 
                criterion, Config.DEVICE, Config.GRADIENT_ACCUMULATION
            )
            val_smape = validate(model, val_loader, Config.DEVICE)
            
            print(f"Train Loss: {train_loss:.4f}, Val SMAPE: {val_smape:.2f}%")
            
            if val_smape < best_smape:
                best_smape = val_smape
                torch.save(model.state_dict(), f'model_fold_{fold}.pth')
                print(f"✓ Model saved! Best SMAPE: {best_smape:.2f}%")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
        
        fold_scores.append(best_smape)
        print(f"\nFold {fold+1} Best SMAPE: {best_smape:.2f}%")
    
    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    for i, score in enumerate(fold_scores):
        print(f"Fold {i+1}: {score:.2f}%")
    print(f"Mean SMAPE: {np.mean(fold_scores):.2f}% ± {np.std(fold_scores):.2f}%")
    print(f"{'='*60}")

# ============================================================================ 
# 8. SINGLE TRAIN/VAL SPLIT
# ============================================================================
def train_single():
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    
    print(f"Using device: {Config.DEVICE}")
    
    # Load data
    print("Loading training data...")
    train_df = pd.read_csv(Config.TRAIN_CSV)
    
    # Split
    from sklearn.model_selection import train_test_split
    train_fold, val_fold = train_test_split(
        train_df, test_size=0.10, random_state=Config.SEED
    )
    
    print(f"Train: {len(train_fold)}, Val: {len(val_fold)}")
    
    tokenizer = AutoTokenizer.from_pretrained(Config.TEXT_MODEL)
    
    train_dataset = ProductDataset(train_fold, tokenizer)
    val_dataset = ProductDataset(val_fold, tokenizer)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        num_workers=4,
        pin_memory=True
    )
    
    model = PricePredictor().to(Config.DEVICE)
    
    num_training_steps = len(train_loader) * Config.EPOCHS // Config.GRADIENT_ACCUMULATION
    num_warmup_steps = int(num_training_steps * Config.WARMUP_RATIO)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LR,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    criterion = HuberSMAPELoss(delta=1.0, smape_weight=0.6)
    
    best_smape = float('inf')
    patience = 3
    patience_counter = 0
    
    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
        
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler,
            criterion, Config.DEVICE, Config.GRADIENT_ACCUMULATION
        )
        val_smape = validate(model, val_loader, Config.DEVICE)
        
        print(f"Train Loss: {train_loss:.4f}, Val SMAPE: {val_smape:.2f}%")
        
        if val_smape < best_smape:
            best_smape = val_smape
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"✓ Model saved! Best SMAPE: {best_smape:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    print(f"\nTraining complete! Best SMAPE: {best_smape:.2f}%")

# ============================================================================ 
# 9. INFERENCE WITH ENSEMBLE
# ============================================================================
def predict():
    print("Loading test data...")
    test_df = pd.read_csv(Config.TEST_CSV)
    print(f"Test samples: {len(test_df)}")
    
    tokenizer = AutoTokenizer.from_pretrained(Config.TEXT_MODEL)
    test_dataset = ProductDataset(test_df, tokenizer, is_test=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        num_workers=4,
        pin_memory=True
    )
    
    # Ensemble predictions
    all_predictions = []
    
    if Config.USE_KFOLD:
        model_files = [f'model_fold_{i}.pth' for i in range(Config.N_FOLDS)]
    else:
        model_files = ['best_model.pth']
    
    for model_file in model_files:
        print(f"Loading {model_file}...")
        model = PricePredictor().to(Config.DEVICE)
        model.load_state_dict(torch.load(model_file))
        model.eval()
        
        fold_preds = []
        sample_ids = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f'Predicting {model_file}'):
                input_ids = batch['input_ids'].to(Config.DEVICE)
                attention_mask = batch['attention_mask'].to(Config.DEVICE)
                features = batch['features'].to(Config.DEVICE)
                
                log_preds = model(input_ids, attention_mask, features)
                preds = torch.expm1(log_preds)
                
                fold_preds.extend(preds.cpu().numpy())
                #if len(sample_ids) == 0:
                sample_ids.extend(batch['sample_id'])
        
        all_predictions.append(fold_preds)
    
    # Average ensemble
    final_predictions = np.mean(all_predictions, axis=0)
    final_predictions = np.maximum(final_predictions, 0)  # Ensure positive
    
    # Create output
    output_df = pd.DataFrame({
        'sample_id': sample_ids,
        'price': final_predictions
    })
    
    def extract_int(x):
        if isinstance(x, str):
            m = re.search(r'\d+', x)
            return int(m.group()) if m else None
        return int(x)

    if 'sample_id' in output_df.columns:
        output_df['sample_id'] = output_df['sample_id'].apply(extract_int)
    
    output_df.to_csv(Config.OUTPUT_FILE, index=False)
    print(f"\n✓ Predictions saved to {Config.OUTPUT_FILE}")
    print(f"Total predictions: {len(output_df)}")
    print(f"Price range: ${output_df['price'].min():.2f} - ${output_df['price'].max():.2f}")

# ============================================================================ 
# 10. MAIN EXECUTION
# ============================================================================
if __name__ == '__main__':
    print("="*60)
    print("PRODUCT PRICE PREDICTION - BGE FINE-TUNING")
    print("="*60)
    
    # Training
    if Config.USE_KFOLD:
        train_kfold()
    else:
        train_single()
    
    # Inference
    predict()
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED!")
    print("="*60)
