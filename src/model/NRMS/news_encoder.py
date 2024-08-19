# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from model.general.attention.multihead_self import MultiHeadSelfAttention
# from model.general.attention.additive import AdditiveAttention

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# class NewsEncoder(torch.nn.Module):
#     def __init__(self, config, pretrained_word_embedding):
#         super(NewsEncoder, self).__init__()
#         self.config = config
        
#         if pretrained_word_embedding is None:
#             self.word_embedding = nn.Embedding(config.num_words,
#                                                config.word_embedding_dim,
#                                                padding_idx=0)
#         else:
#             self.word_embedding = nn.Embedding.from_pretrained(
#                 pretrained_word_embedding, freeze=False, padding_idx=0)
        
#         self.num_words_title=config.num_words_title
#         self.category_emb = nn.Embedding(config.num_categories ,config.category_embedding_dim, padding_idx=0)
#         self.category_dense = nn.Linear(config.category_embedding_dim, config.word_embedding_dim)
#         self.category_attn = MultiHeadSelfAttention(config.category_embedding_dim, config.num_attention_heads) 
    
#         self.multihead_self_attention = MultiHeadSelfAttention(
#             config.word_embedding_dim, config.num_attention_heads)
#         self.additive_attention = AdditiveAttention(config.query_vector_dim,
#                                                     config.word_embedding_dim)
#         # Đánh dấu cho x: gồm các chỉ số là batch size và num_category
#         self.x=torch.zeros(config.batch_size, config.num_categories).to(device)
#     def forward(self, news,x,mask=None):
#         """
#         Args:
#             news:
#                 {
#                     "title": batch_size * num_words_title
#                 }
#         Returns:
#             (shape) batch_size, word_embedding_dim
#         """
#         # batch_size, num_words_title, word_embedding_dim
#         news_vector = F.dropout(self.word_embedding(news["title"].to(device)),
#                                 p=self.config.dropout_probability,
#                                 training=self.training)
        
        
#         # batch_size, num_words_title, word_embedding_dim
#         multihead_news_vector = self.multihead_self_attention(news_vector)
#         multihead_news_vector = F.dropout(multihead_news_vector,
#                                           p=self.config.dropout_probability,
#                                           training=self.training)
        
#         # Bắt đầu đoạn xử lý thằng category
#         start = self.num_words_title
        
#         # batch_size, word_embedding_dim
#         final_news_vector = self.additive_attention(multihead_news_vector)
#         all_vecs=[final_news_vector]
#         category = torch.narrow(self.x, -1, start, 1).squeeze(dim=-1).long()
#         # category = news['category'].squeeze(dim=-1).long()
#         category_vecs = self.category_dense(self.category_emb(category))
#         all_vecs.append(category_vecs)
#         start += 1
#         if len(all_vecs) == 1:
#             news_vecs = all_vecs[0]
#         else:
#             all_vecs = torch.stack(all_vecs, dim=1)
#             news_vecs = self.additive_attention(all_vecs)
#         return final_news_vector

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.general.attention.multihead_self import MultiHeadSelfAttention
from model.general.attention.additive import AdditiveAttention

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class NewsEncoder(torch.nn.Module):
    def __init__(self, config, pretrained_word_embedding):
        super(NewsEncoder, self).__init__()
        self.config = config
        if pretrained_word_embedding is None:
            self.word_embedding = nn.Embedding(config.num_words,
                                               config.word_embedding_dim,
                                               padding_idx=0)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(
                pretrained_word_embedding, freeze=False, padding_idx=0)

        self.multihead_self_attention = MultiHeadSelfAttention(
            config.word_embedding_dim, config.num_attention_heads)
        self.additive_attention = AdditiveAttention(config.query_vector_dim,
                                                    config.word_embedding_dim)

    def forward(self, news):
        """
        Args:
            news:
                {
                    "title": batch_size * num_words_title
                }
        Returns:
            (shape) batch_size, word_embedding_dim
        """
        # batch_size, num_words_title, word_embedding_dim
        news_vector = F.dropout(self.word_embedding(news["title"].to(device)),
                                p=self.config.dropout_probability,
                                training=self.training)
        # batch_size, num_words_title, word_embedding_dim
        multihead_news_vector = self.multihead_self_attention(news_vector)
        multihead_news_vector = F.dropout(multihead_news_vector,
                                          p=self.config.dropout_probability,
                                          training=self.training)
        # batch_size, word_embedding_dim
        final_news_vector = self.additive_attention(multihead_news_vector)
        return final_news_vector