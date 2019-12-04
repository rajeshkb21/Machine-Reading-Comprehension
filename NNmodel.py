import torch
from torch import nn


class model(nn.Module):

    def __init__(self, vocab_size, embed_dim=128, context_dim=50, out_dim=2, dropout=0):
        super(model, self).__init__()

        self.embeds = nn.Embedding(vocab_size, embed_dim)

        self.question_encoder = nn.GRU(embed_dim, context_dim, bidirectional=True)
        self.context_encoder = nn.GRU(embed_dim, context_dim, bidirectional=True)

        self.loss = nn.CrossEntropyLoss()
        self.out = nn.Linear(2 * context_dim, out_dim)

    def compute_loss(self, pred_vec, gold_seq):
        return self.loss(pred_vec, gold_seq)

    def forward(self, context, question):
        context_vectors = self.embeds(context)
        context_vectors = context_vectors.unsqueeze(1)
        question_vectors = self.embeds(question)
        question_vectors = question_vectors.unsqueeze(1)

        _, context_encoder_output = self.context_encoder(context_vectors)
        _, question_encoder_output = self.question_encoder(question_vectors)
        concatenation = torch.cat([context_encoder_output, question_encoder_output], 2)
        output = self.out(concatenation)
        predictions = output.squeeze()
        val, prediction = torch.max(predictions, 0)
        return predictions, prediction.item()
