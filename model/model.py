import torch
from torch import nn
import transformers
if int(transformers.__version__[0]) <=3:
    from transformers.modeling_roberta import RobertaPreTrainedModelm
else:
    from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel

from transformers import RobertaModel
import torch.nn.functional as F
import faiss

class BaseModelDot(nn.Module):
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def masked_mean(self, t, mask):
        s = torch.sum(t * mask.unsqueeze(-1).float(), axis=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s / d

    def _text_encode(self, input_ids, attention_mask):
        raise NotImplementedError

    def query_emb(self, input_ids, attention_mask):
        outputs = self._text_encode(input_ids=input_ids,
                                attention_mask=attention_mask)
        full_emb = self.masked_mean(outputs, attention_mask)

        if self.embeddingHead is not None:
            query = self.norm(self.embeddingHead(full_emb))
        else:
            query = full_emb.contiguous()
        return query

    def body_emb(self, input_ids, attention_mask):
        return self.query_emb(input_ids, attention_mask)

    def forward(self, input_ids, attention_mask, is_query, multi_chunk=False, *args):
        assert len(args) == 0
        if is_query:
            return self.query_emb(input_ids, attention_mask)
        else:
            return self.body_emb(input_ids, attention_mask)


class ContrasPQ(nn.Module):
    def init_pq(self, embedding_size=768, partition=96, centroids=256, init_index_path=None, train_rotate=False):
        self.rotate = None
        if init_index_path is not None:
            if 'OPQ' in init_index_path:
                print(f'loading codebook from OPQ index: {init_index_path}')
                opq_index = faiss.read_index(init_index_path)
                vt = faiss.downcast_VectorTransform(opq_index.chain.at(0))
                assert isinstance(vt, faiss.LinearTransform)
                opq_transform = faiss.vector_to_array(vt.A).reshape(vt.d_out, vt.d_in)
                self.rotate = nn.Parameter(torch.FloatTensor(opq_transform), requires_grad=train_rotate)

                ivf_index = faiss.downcast_index(opq_index.index)
                centroid_embeds = faiss.vector_to_array(ivf_index.pq.centroids)
                centroid_embeds = centroid_embeds.reshape(ivf_index.pq.M, ivf_index.pq.ksub, ivf_index.pq.dsub)
                self.partition = ivf_index.pq.M
                self.codebook = nn.Parameter(torch.FloatTensor(centroid_embeds), requires_grad=True)

            elif 'PQ' in init_index_path:
                self.rotate = None
                print(f'loading codebook from PQ index: {init_index_path}')
                pq_index = faiss.read_index(init_index_path)
                centroid_embeds = faiss.vector_to_array(pq_index.pq.centroids)
                centroid_embeds = centroid_embeds.reshape(pq_index.pq.M, pq_index.pq.ksub, pq_index.pq.dsub)
                self.partition = pq_index.pq.M
                self.codebook = nn.Parameter(torch.FloatTensor(centroid_embeds), requires_grad=True)
        else:
            print(f'init the codebook {partition}-{centroids}')
            self.partition = partition
            self.codebook = nn.Parameter(
                torch.empty(partition, centroids,
                            embedding_size // partition).uniform_(-1, 1)).type(
                torch.FloatTensor)
            self.rotate = None

    def rotate_vec(self, vecs):
        if self.rotate is None:
            return vecs
        return torch.matmul(vecs, self.rotate.T)

    def code_selection(self, vecs):
        vecs = vecs.view(vecs.size(0), self.partition, -1)  # B P D
        codebook = self.codebook.unsqueeze(0).expand(vecs.size(0), -1, -1, -1)  # B P K D
        proba = - torch.sum((vecs.unsqueeze(-2) - codebook) ** 2, -1)  # B P K
        proba = F.softmax(proba, -1)
        return proba

    def STEstimator(self, proba):
        index = proba.max(dim=-1, keepdim=True)[1]
        proba_hard = torch.zeros_like(proba, device=proba.device, dtype=proba.dtype).scatter_(-1, index, 1.0)
        return proba_hard.detach() - proba.detach() + proba

    def hard_vecs(self, proba):
        proba = self.STEstimator(proba)  # B P K
        proba = proba.unsqueeze(2)  # B P 1 K
        codebook = self.codebook.unsqueeze(0).expand(proba.size(0), -1, -1, -1)  # B P K D
        hard_vecs = torch.matmul(proba, codebook).squeeze(2)  # B P D
        hard_vecs = hard_vecs.view(proba.size(0), -1)  # B L
        return hard_vecs

    def quant(self, vecs):
        vecs = self.rotate_vec(vecs)
        prob = self.code_selection(vecs)
        hard_vecs = self.hard_vecs(prob)
        return hard_vecs

    def quantization_loss(self, vec, quant_vec):
        return torch.mean(torch.sum((vec - quant_vec) ** 2, dim=-1))


class RobertaDot(BaseModelDot, RobertaPreTrainedModel, ContrasPQ):
    def __init__(self, config):
        BaseModelDot.__init__(self)
        RobertaPreTrainedModel.__init__(self, config)
        ContrasPQ.__init__(self)

        if int(transformers.__version__[0]) == 4 :
            config.return_dict = False

        if hasattr(config, "output_embedding_size"):
            self.output_embedding_size = config.output_embedding_size
        else:
            self.output_embedding_size = config.hidden_size
        print("output_embedding_size", self.output_embedding_size)

        if config.use_linear:
            self.embeddingHead = nn.Linear(config.hidden_size, self.output_embedding_size)
            self.norm = nn.LayerNorm(self.output_embedding_size)
        else:
            self.embeddingHead = None
        print("use_linear", config.use_linear)

        self.apply(self._init_weights)

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        if config.use_pq:
            self.init_pq(self.output_embedding_size, config.partition, config.centroids, config.init_index_path)

    def _text_encode(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids,
                                attention_mask=attention_mask)
        return outputs

    def forward(self, all_q, all_k, all_n,
                rel_pair_mask=None, hard_pair_mask=None,
                loss_method='multi_ce',
                temperature=1,
                hard_k=None, hard_n=None, quant_weight=1e-6):

        if loss_method == 'nce':
            score = torch.matmul(all_q, all_k.T)
            positive_score = torch.diagonal(score, 0).view(-1,1)
            neg_score = score * (1-torch.eye(score.size(0), dtype=score.dtype, device=score.device))
            max_neg = torch.max(neg_score, dim=-1, keepdim=True)[0]
            logit_matrix = torch.cat([positive_score, max_neg], dim=-1)  # [B, 2]
            lsm = F.log_softmax(logit_matrix, dim=1)
            loss = -1.0 * lsm[:, 0]
            return loss.mean(), None, None, None

        elif loss_method == 'inbatch':
            score = torch.matmul(all_q, all_k.T)
            score = score / temperature
            labels = torch.arange(start=0, end=score.shape[0],
                                  dtype=torch.long, device=score.device)
            loss = F.cross_entropy(score, labels)
            return loss, None, None, None

        elif loss_method == 'multi_ce':
            score = torch.matmul(all_q, all_k.T) #B B
            n_score = torch.matmul(all_q, all_n.T) #B BN
            score = torch.cat([score, n_score], dim=-1) #B B+BN

            if hard_k is not None:
                rotate_q = self.rotate_vec(all_q, is_query=True)
                quant_score = torch.matmul(rotate_q, hard_k.T)
                quant_nscore = torch.matmul(rotate_q, hard_n.T)
                quant_score = torch.cat([quant_score, quant_nscore], dim=-1)

            if rel_pair_mask is not None and hard_pair_mask is not None:
                rel_pair_mask = rel_pair_mask + torch.eye(score.size(0), dtype=score.dtype, device=score.device)
                mask = torch.cat([rel_pair_mask, hard_pair_mask], dim=-1)
                score = score.masked_fill(mask==0, -10000)
                if hard_k is not None:
                    quant_score = quant_score.masked_fill(mask==0, -10000)

            score = score/temperature
            labels = torch.arange(start=0, end=score.shape[0],
                                  dtype=torch.long, device=score.device)
            loss = F.cross_entropy(score, labels)

            if hard_k is not None:
                quant_score = quant_score / temperature
                qloss = F.cross_entropy(quant_score, labels)
                quant_loss = quant_weight * (self.quantization_loss(all_k, hard_k) + self.quantization_loss(all_n, hard_n))
                all_loss = loss + qloss + quant_loss
                return all_loss, loss, qloss, quant_loss
            else:
                return loss, loss, 0., 0.






