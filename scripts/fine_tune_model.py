from sentence_transformers import SentenceTransformer, InputExample, losses
from datasets import load_dataset
from torch.utils.data import DataLoader

# Load SNLI dataset
dataset_snli = load_dataset('snli')

# Preprocessing: remove ambiguous examples
dataset_snli = dataset_snli.filter(lambda x: x['label'] != -1)

# Prepare training examples
train_examples = [
    InputExample(texts=[x['premise'], x['hypothesis']], label=x['label'])
    for x in dataset_snli['train']
]

# Initialize model
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
model = SentenceTransformer(model_name)

# DataLoader setup
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)

# Define loss function (SoftmaxLoss for classification tasks)
train_loss = losses.SoftmaxLoss(
    model=model,
    sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
    num_labels=3
)

# Fine-tune model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=4,
    warmup_steps=1000,
    output_path='../fine_tuned_model'
)
