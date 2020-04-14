from summarizer import Summarizer

body = '''
We introduce an attention-based Bi-LSTM for Chinese implicit discourse relations and demonstrate that modeling argument pairs as a joint sequence can outperform word order-agnostic approaches. 
Our model benefits from a partial sampling scheme and is conceptually simple, yet achieves state-of-the-art performance on the Chinese Discourse Treebank. 
We also visualize its attention activity to illustrate the model’s ability to selectively focus on the relevant parts of an input sequence.
'''

model = Summarizer()
result = model(body, min_length=60)
full = ''.join(result)
print(full)
"""
We introduce an attention-based Bi-LSTM for Chinese implicit discourse relations and demonstrate that modeling argument pairs as a joint sequence can outperform word order-agnostic approaches. We also visualize its attention activity to illustrate the model’s ability to selectively focus on the relevant parts of an input sequence.
"""