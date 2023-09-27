import torch
import torch.nn as nn

from .embed import DataEmbedding
from .decoder import Decoder, DecoderLayer
from .attn import FullAttention, ProbAttention, AttentionLayer
from .encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from .xformer import data_transformation_4_xformer


class Informer(nn.Module):
    """
    Paper: Informer: Beyond Efﬁcient Transformer for Long Sequence Time-Series Forecasting
    Link: https://arxiv.org/abs/2012.07436
    Ref Official Code: https://github.com/zhouhaoyi/Informer2020
    """

    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                time_of_day_size, day_of_week_size, day_of_month_size=None, day_of_year_size=None,
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
                output_attention = False, distil=True, mix=True, num_time_features=-1):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.label_len = int(label_len)
        self.attn = attn
        self.output_attention = output_attention

        self.time_of_day_size =time_of_day_size
        self.day_of_week_size = day_of_week_size
        self.day_of_month_size = day_of_month_size
        self.day_of_year_size = day_of_year_size
        self.embed = embed
        self.num_time_features = num_time_features

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, time_of_day_size, day_of_week_size, day_of_month_size, day_of_year_size, embed, num_time_features, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, time_of_day_size, day_of_week_size, day_of_month_size, day_of_year_size, embed, num_time_features, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)

        self.history_emb = nn.Parameter(torch.empty(seq_len,enc_in))
        nn.init.xavier_uniform_(self.history_emb)

        self.future_emb = nn.Parameter(torch.empty(out_len,enc_in))
        nn.init.xavier_uniform_(self.future_emb)

        self.avg_pool = nn.AvgPool2d(kernel_size=(3, 1), stride=1, padding=(1, 0))

        
    def forward_xformer(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor, x_dec: torch.Tensor, x_mark_dec: torch.Tensor,
                enc_self_mask: torch.Tensor=None, dec_self_mask: torch.Tensor=None, dec_enc_mask: torch.Tensor=None) -> torch.Tensor:
        """Feed forward of Informer. Kindly note that `enc_self_mask`, `dec_self_mask`, and `dec_enc_mask` are not actually used in Informer.

        Args:
            x_enc (torch.Tensor): input data of encoder (without the time features). Shape: [B, L1, N]
            x_mark_enc (torch.Tensor): time features input of encoder w.r.t. x_enc. Shape: [B, L1, C-1]
            x_dec (torch.Tensor): input data of decoder. Shape: [B, start_token_length + L2, N]
            x_mark_dec (torch.Tensor): time features input to decoder w.r.t. x_dec. Shape: [B, start_token_length + L2, C-1]
            enc_self_mask (torch.Tensor, optional): encoder self attention masks. Defaults to None.
            dec_self_mask (torch.Tensor, optional): decoder self attention masks. Defaults to None.
            dec_enc_mask (torch.Tensor, optional): decoder encoder self attention masks. Defaults to None.

        Returns:
            torch.Tensor: outputs with shape [B, L2, N, 1]
        """

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        
        return dec_out[:, -self.pred_len:, :].unsqueeze(-1)  # [B, L, N, C]
    
    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor) -> torch.Tensor:
        """

        Args:
            history_data (Tensor): Input data with shape: [B, L1, N, C]
            future_data (Tensor): Future data with shape: [B, L2, N, C]

        Returns:
            torch.Tensor: outputs with shape [B, L2, N, 1]
        """

        batch_size, _, _  = history_data.shape
        node_emb1 = self.history_emb.unsqueeze(0).expand(batch_size, -1, -1)
        node_emb2 = self.future_emb.unsqueeze(0).expand(batch_size, -1, -1)

        history_data = history_data.transpose(-2, -1)
        history_data = history_data.unsqueeze(-1)

        future_data = future_data.transpose(-2, -1)
        future_data = future_data.unsqueeze(-1)

        for i in range(self.num_time_features):
            history_data = torch.cat([history_data,node_emb1.unsqueeze(-1)],dim=-1)
            future_data = torch.cat([future_data, node_emb2.unsqueeze(-1)], dim=-1)
            node_emb1 = self.avg_pool(node_emb1)
            node_emb2 = self.avg_pool(node_emb2)

        x_enc, x_mark_enc, x_dec, x_mark_dec = data_transformation_4_xformer(history_data=history_data, future_data=future_data, start_token_len=self.label_len)
        prediction = self.forward_xformer(x_enc=x_enc, x_mark_enc=x_mark_enc, x_dec=x_dec, x_mark_dec=x_mark_dec)
        prediction = prediction.squeeze(-1)
        prediction = prediction.transpose(-2, -1)

        return prediction


class InformerStack(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 time_of_day_size, day_of_week_size, day_of_month_size=None, day_of_year_size=None,
                factor=5, d_model=512, n_heads=8, e_layers=[3,2,1], d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                output_attention = False, distil=True, mix=True, num_time_features=-1):
        super(InformerStack, self).__init__()
        self.pred_len = out_len
        self.label_len = int(label_len)
        self.attn = attn
        self.output_attention = output_attention
        self.num_time_features = num_time_features

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, time_of_day_size, day_of_week_size, day_of_month_size,
                                           day_of_year_size, embed, num_time_features, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, time_of_day_size, day_of_week_size, day_of_month_size,
                                           day_of_year_size, embed, num_time_features, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder

        inp_lens = list(range(len(e_layers))) # [0,1,2,...] you can customize here
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                    d_model, n_heads, mix=False),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(el)
                ],
                [
                    ConvLayer(
                        d_model
                    ) for l in range(el-1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            ) for el in e_layers]
        self.encoder = EncoderStack(encoders, inp_lens)
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)

        self.history_emb = nn.Parameter(torch.empty(seq_len,enc_in))
        nn.init.xavier_uniform_(self.history_emb)

        self.future_emb = nn.Parameter(torch.empty(out_len,enc_in))
        nn.init.xavier_uniform_(self.future_emb)

        self.avg_pool = nn.AvgPool2d(kernel_size=(3, 1), stride=1, padding=(1, 0))
        
    def forward_xformer(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor, x_dec: torch.Tensor, x_mark_dec: torch.Tensor,
                enc_self_mask: torch.Tensor=None, dec_self_mask: torch.Tensor=None, dec_enc_mask: torch.Tensor=None) -> torch.Tensor:
        """Feed forward of Informer. Kindly note that `enc_self_mask`, `dec_self_mask`, and `dec_enc_mask` are not actually used in Informer.

        Args:
            x_enc (torch.Tensor): input data of encoder (without the time features). Shape: [B, L1, N]
            x_mark_enc (torch.Tensor): time features input of encoder w.r.t. x_enc. Shape: [B, L1, C-1]
            x_dec (torch.Tensor): input data of decoder. Shape: [B, start_token_length + L2, N]
            x_mark_dec (torch.Tensor): time features input to decoder w.r.t. x_dec. Shape: [B, start_token_length + L2, C-1]
            enc_self_mask (torch.Tensor, optional): encoder self attention masks. Defaults to None.
            dec_self_mask (torch.Tensor, optional): decoder self attention masks. Defaults to None.
            dec_enc_mask (torch.Tensor, optional): decoder encoder self attention masks. Defaults to None.

        Returns:
            torch.Tensor: outputs with shape [B, L2, N, 1]
        """

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        return dec_out[:, -self.pred_len:, :].unsqueeze(-1)  # [B, L, N, C]

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor) -> torch.Tensor:
        """

        Args:
            history_data (Tensor): Input data with shape: [B, L1, N, C]
            future_data (Tensor): Future data with shape: [B, L2, N, C]

        Returns:
            torch.Tensor: outputs with shape [B, L2, N, 1]
        """
        batch_size, _, _  = history_data.shape
        node_emb1 = self.history_emb.unsqueeze(0).expand(batch_size, -1, -1)
        node_emb2 = self.future_emb.unsqueeze(0).expand(batch_size, -1, -1)

        history_data = history_data.transpose(-2, -1)
        history_data = history_data.unsqueeze(-1)

        future_data = future_data.transpose(-2, -1)
        future_data = future_data.unsqueeze(-1)

        for i in range(self.num_time_features):
            history_data = torch.cat([history_data,node_emb1.unsqueeze(-1)],dim=-1)
            future_data = torch.cat([future_data, node_emb2.unsqueeze(-1)], dim=-1)
            node_emb1 = self.avg_pool(node_emb1)
            node_emb2 = self.avg_pool(node_emb2)

        x_enc, x_mark_enc, x_dec, x_mark_dec = data_transformation_4_xformer(history_data=history_data, future_data=future_data, start_token_len=self.label_len)
        prediction = self.forward_xformer(x_enc=x_enc, x_mark_enc=x_mark_enc, x_dec=x_dec, x_mark_dec=x_mark_dec)

        prediction = prediction.squeeze(-1)
        prediction = prediction.transpose(-2, -1)
        return prediction
