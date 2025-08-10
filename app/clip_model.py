from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np

class CLIPWrapper:
    def __init__(self):
        """
        Initialize the CLIP model and processor on available device (GPU if available).
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    def encode_text(self, texts):
        """
        Encode a single text or a list of texts into CLIP text embeddings.
        
        Args:
            texts (str or List[str]): Text or list of texts to encode.
        
        Returns:
            np.ndarray: Normalized text embeddings of shape (batch_size, embedding_dim)
        """
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k,v in inputs.items()}
        with torch.no_grad():
            embeddings = self.model.get_text_features(**inputs)
        embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
        return embeddings.cpu().numpy()

    def encode_image(self, images):
        """
        Encode a single image or a list of images into CLIP image embeddings.
        
        Args:
            images (PIL.Image.Image or List[PIL.Image.Image]): Image(s) to encode.
        
        Returns:
            np.ndarray: Normalized image embeddings of shape (batch_size, embedding_dim)
        """
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k,v in inputs.items()}
        with torch.no_grad():
            embeddings = self.model.get_image_features(**inputs)
        embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
        return embeddings.cpu().numpy()
    
    def cosine_similarity(self, emb1, emb2):
        """
        Compute cosine similarity between two sets of embeddings.
        
        Args:
            emb1 (np.ndarray): Embeddings of shape (N, D)
            emb2 (np.ndarray): Embeddings of shape (M, D)
        
        Returns:
            np.ndarray: Similarity matrix of shape (N, M)
        """
        return np.dot(emb1, emb2.T)
