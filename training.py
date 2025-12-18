"""
Video Captioning Model using CNN + Transformer
Generates captions for silent videos using visual features only.

Requirements:
pip install torch torchvision transformers opencv-python pillow
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from transformers import GPT2Tokenizer
import cv2
import json
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np


class VideoFeatureExtractor(nn.Module):
    """CNN-based feature extractor for video frames."""
    
    def __init__(self, embed_dim=512):
        super().__init__()
        # Use pretrained ResNet as backbone
        resnet = models.resnet50(pretrained=True)
        # Remove final classification layer
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        # Project to desired embedding dimension
        self.projection = nn.Linear(2048, embed_dim)
        
    def forward(self, frames):
        """
        Args:
            frames: (batch, num_frames, channels, height, width)
        Returns:
            features: (batch, num_frames, embed_dim)
        """
        batch_size, num_frames, c, h, w = frames.shape
        # Reshape to process all frames at once
        frames = frames.view(batch_size * num_frames, c, h, w)
        
        # Extract features
        with torch.no_grad():
            features = self.cnn(frames)  # (batch*num_frames, 2048, 1, 1)
        
        features = features.view(batch_size, num_frames, -1)  # (batch, num_frames, 2048)
        features = self.projection(features)  # (batch, num_frames, embed_dim)
        
        return features


class VideoCaptioningModel(nn.Module):
    """Complete video captioning model with CNN encoder and Transformer decoder."""
    
    def __init__(self, vocab_size, embed_dim=512, num_heads=8, num_layers=6, max_len=50):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.max_len = max_len
        
        # Feature extractor for video frames
        self.feature_extractor = VideoFeatureExtractor(embed_dim)
        
        # Token embedding for captions
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, max_len, embed_dim))
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Output projection to vocabulary
        self.output_projection = nn.Linear(embed_dim, vocab_size)
        
        # Learnable query tokens for cross-attention
        self.frame_query = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, frames, captions, caption_mask=None):
        """
        Args:
            frames: (batch, num_frames, channels, height, width)
            captions: (batch, seq_len) - tokenized captions
            caption_mask: (batch, seq_len) - attention mask
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        # Extract visual features
        visual_features = self.feature_extractor(frames)  # (batch, num_frames, embed_dim)
        memory = self.frame_query(visual_features)  # (batch, num_frames, embed_dim)
        
        # Embed caption tokens
        caption_embeds = self.token_embedding(captions)  # (batch, seq_len, embed_dim)
        
        # Add positional encoding
        seq_len = captions.shape[1]
        caption_embeds = caption_embeds + self.positional_encoding[:, :seq_len, :]
        
        # Create causal mask for decoder (prevent attending to future tokens)
        causal_mask = self.generate_square_subsequent_mask(seq_len).to(captions.device)
        
        # Transformer decoder
        decoder_output = self.transformer_decoder(
            tgt=caption_embeds,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=caption_mask
        )
        
        # Project to vocabulary
        logits = self.output_projection(decoder_output)
        
        return logits
    
    def generate_square_subsequent_mask(self, sz):
        """Generate causal mask for autoregressive generation."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    @torch.no_grad()
    def generate_caption(self, frames, tokenizer, max_length=50, device='cuda'):
        """Generate caption for video frames using greedy decoding."""
        self.eval()
        frames = frames.to(device)
        
        # Extract visual features
        visual_features = self.feature_extractor(frames)
        memory = self.frame_query(visual_features)
        
        # Start with BOS token
        generated = torch.tensor([[tokenizer.bos_token_id]], device=device)
        
        for _ in range(max_length):
            # Embed current sequence
            caption_embeds = self.token_embedding(generated)
            seq_len = generated.shape[1]
            caption_embeds = caption_embeds + self.positional_encoding[:, :seq_len, :]
            
            # Create causal mask
            causal_mask = self.generate_square_subsequent_mask(seq_len).to(device)
            
            # Decode
            decoder_output = self.transformer_decoder(
                tgt=caption_embeds,
                memory=memory,
                tgt_mask=causal_mask
            )
            
            # Get next token prediction
            logits = self.output_projection(decoder_output[:, -1, :])
            next_token = logits.argmax(dim=-1, keepdim=True)
            
            # Stop if EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)
        
        # Decode to text
        caption = tokenizer.decode(generated[0], skip_special_tokens=True)
        return caption


class VideoDataset(Dataset):
    """Dataset for video-caption pairs."""
    
    def __init__(self, video_folder, caption_folder, tokenizer, num_frames=16, max_length=50):
        self.video_folder = Path(video_folder)
        self.caption_folder = Path(caption_folder)
        self.tokenizer = tokenizer
        self.num_frames = num_frames
        self.max_length = max_length
        
        # Transform for video frames
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Get all video-caption pairs
        self.samples = self._load_samples()
    
    def _load_samples(self):
        """Load video and caption file pairs."""
        samples = []
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        
        for video_file in self.video_folder.iterdir():
            if video_file.suffix.lower() in video_extensions:
                # Find corresponding caption file
                caption_file = self.caption_folder / f"{video_file.stem}.txt"
                if caption_file.exists():
                    samples.append((video_file, caption_file))
        
        return samples
    
    def _extract_frames(self, video_path):
        """Extract evenly-spaced frames from video."""
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame indices to extract
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame = self.transform(frame)
                frames.append(frame)
        
        cap.release()
        
        # Stack frames
        frames = torch.stack(frames)  # (num_frames, 3, 224, 224)
        return frames
    
    def _load_caption(self, caption_path):
        """Load and tokenize caption."""
        with open(caption_path, 'r', encoding='utf-8') as f:
            caption = f.read().strip()
        
        # Tokenize
        tokens = self.tokenizer.encode(
            caption,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return tokens.squeeze(0)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        video_path, caption_path = self.samples[idx]
        
        frames = self._extract_frames(video_path)
        caption = self._load_caption(caption_path)
        
        return frames, caption


def train_model(model, dataloader, optimizer, criterion, device, epoch):
    """Training loop for one epoch."""
    model.train()
    total_loss = 0
    
    for batch_idx, (frames, captions) in enumerate(dataloader):
        frames = frames.to(device)
        captions = captions.to(device)
        
        # Prepare input and target
        input_captions = captions[:, :-1]  # All but last token
        target_captions = captions[:, 1:]  # All but first token
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(frames, input_captions)
        
        # Calculate loss
        loss = criterion(logits.reshape(-1, logits.shape[-1]), target_captions.reshape(-1))
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch [{epoch}] Batch [{batch_idx+1}/{len(dataloader)}] Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def main():
    # Configuration
    VIDEO_FOLDER = "videos"
    CAPTION_FOLDER = "captions"
    NUM_EPOCHS = 10
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    NUM_FRAMES = 16
    EMBED_DIM = 512
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset and dataloader
    dataset = VideoDataset(VIDEO_FOLDER, CAPTION_FOLDER, tokenizer, num_frames=NUM_FRAMES)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    print(f"Loaded {len(dataset)} video-caption pairs")
    
    # Initialize model
    model = VideoCaptioningModel(
        vocab_size=len(tokenizer),
        embed_dim=EMBED_DIM,
        num_heads=8,
        num_layers=6
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(1, NUM_EPOCHS + 1):
        avg_loss = train_model(model, dataloader, optimizer, criterion, device, epoch)
        print(f"Epoch [{epoch}/{NUM_EPOCHS}] Average Loss: {avg_loss:.4f}\n")
        
        # Save checkpoint
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'checkpoint_epoch_{epoch}.pth')
    
    print("Training complete!")
    
    # Save final model
    torch.save(model.state_dict(), 'video_captioning_model.pth')


if __name__ == "__main__":
    main()