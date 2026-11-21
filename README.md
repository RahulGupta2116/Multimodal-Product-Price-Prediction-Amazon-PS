## Problem Statement
### Smart Product Pricing Challenge
In e-commerce, determining the optimal price point for products is crucial for marketplace success and customer satisfaction.
The challenge is to develop an ML solution that analyzes product details and predicts the price of the product. The relationship between product attributes and pricing is complex - factors like brand, specifications and product quantity directly influence pricing. The task is to build a model that can analyze these product details holistically and suggest an optimal price.

Data Description:

The dataset consists of the following columns:
1. sample_id: A unique identifier for the input sample
2. catalog_content: Text field containing title, product description and an Item Pack Quantity (IPQ) concatenated.
3. image_link: Public URL where the product image is available for download. Example link - https://m.media-amazon.com/images/I/71XfHPR36-L.jpg
4. price: Price of the product (Target variable - only available in training data)

# Approach

TL;DR I use embeddings for each modality, pass them through modality-wise networks, and feed the result to a final regressor network.

Using pretrained embeddings without fine-tuning, combined with careful selection and training of the subsequent layers, gave strong results without the cost of fine-tuning, reaching an average 5-fold cross-validation SMAPE of 41.67%. Using pretrained frozen encoders might seem less cool, but it worked.

It turns out that LLMs distilled to smaller versions have weaker representation capability, and the quality of their embeddings drops drastically. Fine-tuning them can only take you so far, so bigger models sometimes beat fine-tuned smaller models.

It's tempting to pick a single modality, like text or image, without experimenting. Bringing in the image modality from the very beginning helped me realize its importance. It's easy to assume images don't matter for predicting price, but drawing signal from multiple modalities improves performance.

These points come from a benchmark setting, but they loosely extend to production settings, where getting a better score matters a lot.

## Evaluation Metric
Symmetric Mean Absolute Percentage Error (SMAPE)

$$
\text{SMAPE} = \frac{100\%}{n} \sum_{i=1}^{n} 
\frac{|y_i - \hat{y}_i|}{\frac{|y_i| + |\hat{y}_i|}{2}}
$$

- $$\( y_i \)$$: true value  
- $$\( \hat{y}_i \)$$: predicted value  
- Range: $$\( [0, 200] \)$$  
- Lower is better 

# Handling Dataset
<img width="738" height="463" alt="Image" src="https://github.com/user-attachments/assets/9eeafda9-d168-42c1-8849-b313f185b5b2" />


- Right-skewed dataset -> Applied log1p transformation
- Same image + different catalogs and prices, same catalog + different images -> Need for a multimodal approach
- Parsed the catalog content -> Product Description, Bullet Points, Quantity, Unit
- Inconsistent unit representation -> Mapped units to a consistent representation (Ex: grams, gm, g -> grams)
- Converted numbers to words -> Gives better embeddings (Ex: 100 -> One Hundred)
- Large catalog content -> If a Product Description is available, use only that; otherwise use the top 5 bullet points

# Embedding Generation

## Models Used:

After experimenting with multiple models, I used the following in my final pipeline:

1. Qwen-3-4B Text Only [HF](https://huggingface.co/Qwen/Qwen3-4B)
2. Siglip2 Giant - 2B - Text + Image [HF](https://huggingface.co/google/siglip2-giant-opt-patch16-256)
3. DinoV3 - 0.8B - Image [HF](https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m)

Instead of fine-tuning a smaller version of Qwen3, I used the 4B version frozen, and its embeddings were good enough to give a solid improvement on their own. Lesson: try bigger models before fine-tuning smaller ones.

# Modality Specific Networks
<img width="722" height="812" alt="AMLC2025-Final drawio" src="https://github.com/user-attachments/assets/c5792a0a-15f5-4093-bb53-a5949cb21647" />

Embeddings from each model live in different spaces, and it is hard for a model to learn from them by simply concatenating them.
Applying modality-specific transformations made training easier and helped it converge quickly. I believe this design choice was a key driver of the performance.

# Loss function

## Log-based MSE Loss

$$
\mathcal{L} = \frac{1}{n} \sum_{i=1}^{n} 
\left( \log(y_i + 1) - \hat{y}_i \right)^2
$$

where  
- $$\( y_i \)$$: true target value  
- $$\( \hat{y}_i \)$$: predicted value (already in log scale)  
- $$\( n \)$$: number of samples
