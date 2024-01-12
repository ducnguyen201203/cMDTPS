from __future__ import absolute_import

import torch
from torch import nn
import torch.nn.functional as F


def euclidean_dist(x, y):
	m, n = x.size(0), y.size(0)
	xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
	yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
	dist = xx + yy
	dist.addmm_(1, -2, x, y.t())
	dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
	return dist

def cosine_dist(x, y):
	bs1, bs2 = x.size(0), y.size(0)
	frac_up = torch.matmul(x, y.transpose(0, 1))
	frac_down = (torch.sqrt(torch.sum(torch.pow(x, 2), 1))).view(bs1, 1).repeat(1, bs2) * \
	            (torch.sqrt(torch.sum(torch.pow(y, 2), 1))).view(1, bs2).repeat(bs1, 1)
	cosine = frac_up / frac_down
	return 1-cosine

def _batch_hard(mat_distance, mat_similarity, indice=False):
	sorted_mat_distance, positive_indices = torch.sort(mat_distance + (-9999999.) * (1 - mat_similarity), dim=1, descending=True)
	hard_p = sorted_mat_distance[:, 0]
	hard_p_indice = positive_indices[:, 0]
	sorted_mat_distance, negative_indices = torch.sort(mat_distance + (9999999.) * (mat_similarity), dim=1, descending=False)
	hard_n = sorted_mat_distance[:, 0]
	hard_n_indice = negative_indices[:, 0]
	if(indice):
		return hard_p, hard_n, hard_p_indice, hard_n_indice
	return hard_p, hard_n

def _batch_hard_2(mat_distance, mat_similarity, indice=False, topK=0):
	#topK should be < than  the number instances /id in a batch
	sorted_mat_distance, positive_indices = torch.sort(mat_distance + (-9999999.) * (1 - mat_similarity), dim=1, descending=True)
	hard_p = sorted_mat_distance[:, :topK]
	hard_p_indice = positive_indices[:, :topK]
	sorted_mat_distance, negative_indices = torch.sort(mat_distance + (9999999.) * (mat_similarity), dim=1, descending=False)
	hard_n = sorted_mat_distance[:, :topK]
	hard_n_indice = negative_indices[:, :topK]
	if(indice):
		return hard_p, hard_n, hard_p_indice, hard_n_indice
	return hard_p, hard_n

class TripletLoss(nn.Module):
	'''
	Compute Triplet loss augmented with Batch Hard
	Details can be seen in 'In defense of the Triplet Loss for Person Re-Identification'
	'''

	def __init__(self, margin, normalize_feature=False):
		super(TripletLoss, self).__init__()
		self.margin = margin
		self.normalize_feature = normalize_feature
		self.margin_loss = nn.MarginRankingLoss(margin=margin).cuda()

	def forward(self, emb, label):
		if self.normalize_feature:
			# equal to cosine similarity
			emb = F.normalize(emb)
		mat_dist = euclidean_dist(emb, emb)
		# mat_dist = cosine_dist(emb, emb)
		assert mat_dist.size(0) == mat_dist.size(1)
		N = mat_dist.size(0)
		mat_sim = label.expand(N, N).eq(label.expand(N, N).t()).float()

		dist_ap, dist_an = _batch_hard(mat_dist, mat_sim)
		assert dist_an.size(0)==dist_ap.size(0)
		y = torch.ones_like(dist_ap)
		loss = self.margin_loss(dist_an, dist_ap, y)
		# prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
		return loss

class SoftTripletLoss(nn.Module):

	def __init__(self, margin=None, normalize_feature=False, skip_mean=False):
		super(SoftTripletLoss, self).__init__()
		self.margin = margin
		self.normalize_feature = normalize_feature
		self.skip_mean = skip_mean

	def forward(self, emb1, emb2, label):
		if self.normalize_feature:
			# equal to cosine similarity
			emb1 = F.normalize(emb1)
			emb2 = F.normalize(emb2)

		mat_dist = euclidean_dist(emb1, emb1)
		assert mat_dist.size(0) == mat_dist.size(1)
		N = mat_dist.size(0)
		mat_sim = label.expand(N, N).eq(label.expand(N, N).t()).float()

		dist_ap, dist_an, ap_idx, an_idx = _batch_hard(mat_dist, mat_sim, indice=True)
		assert dist_an.size(0)==dist_ap.size(0)
		triple_dist = torch.stack((dist_ap, dist_an), dim=1)
		triple_dist = F.log_softmax(triple_dist, dim=1)
		if (self.margin is not None):
			loss = (- self.margin * triple_dist[:,0] - (1 - self.margin) * triple_dist[:,1])
			if self.skip_mean:
				return loss
			else:
				return loss.mean()

		mat_dist_ref = euclidean_dist(emb2, emb2)
		dist_ap_ref = torch.gather(mat_dist_ref, 1, ap_idx.view(N,1).expand(N,N))[:,0]
		dist_an_ref = torch.gather(mat_dist_ref, 1, an_idx.view(N,1).expand(N,N))[:,0]
		triple_dist_ref = torch.stack((dist_ap_ref, dist_an_ref), dim=1)
		triple_dist_ref = F.softmax(triple_dist_ref, dim=1).detach()
		loss = (- triple_dist_ref * triple_dist)
		if self.skip_mean:
			return loss 
		else:
			return loss.mean(0).sum()


class TopKTripletLoss(nn.Module):

	def __init__(self, margin=0, normalize_feature=False, skip_mean=False, topK=1):
		super(TopKTripletLoss, self).__init__()
		self.margin = margin
		self.normalize_feature = normalize_feature
		self.skip_mean = skip_mean
		self.topk = topK
	def forward(self, emb1, emb2, label):
		if self.normalize_feature:
			# equal to cosine similarity
			emb1 = F.normalize(emb1)
			emb2 = F.normalize(emb2)

		mat_dist = euclidean_dist(emb1, emb2)
		assert mat_dist.size(0) == mat_dist.size(1)
		N = mat_dist.size(0)
		mat_sim = label.expand(N, N).eq(label.expand(N, N).t()).float()

		dist_ap, dist_an, ap_idx, an_idx = _batch_hard_2(mat_dist, mat_sim, indice=True, topK=self.topk)

		assert an_idx.size(0)==ap_idx.size(0)
		dist_group_ap = torch.sum((emb1 - torch.mean(emb2[ap_idx], dim=1)) ** 2, dim=1).sqrt()
		dist_group_an = torch.sum((emb1 - torch.mean(emb2[an_idx], dim=1)) ** 2, dim=1).sqrt()

		# triple_dist = torch.stack((dist_ap, dist_an), dim=1)
		triple_dist = torch.stack((dist_group_ap, dist_group_an), dim=1)
		triple_dist = F.log_softmax(triple_dist, dim=1)

		loss = (- self.margin * triple_dist[:,0] - (1 - self.margin) * triple_dist[:,1])
		if self.skip_mean:
			return loss
		else:
			return loss.mean()

class EnhancedTopKTripletLoss(nn.Module):
	"""using Circle loss to improve TopKTripetLoss"""

	def __init__(self, margin=0, normalize_feature=False, skip_mean=False, topK=1, m = 0.25, gamma=128):
		super(EnhancedTopKTripletLoss, self).__init__()
		self.margin = margin
		self.normalize_feature = normalize_feature
		self.skip_mean = skip_mean
		self.topk = topK
		self.m = m         #margin
		self.gamma = gamma #scale factor
		self.soft_plus = nn.Softplus()
	def forward(self, emb1, emb2, label):
		if self.normalize_feature:
			# equal to cosine similarity
			emb1 = F.normalize(emb1)
			emb2 = F.normalize(emb2)

		mat_dist = euclidean_dist(emb1, emb2)
		assert mat_dist.size(0) == mat_dist.size(1)
		N = mat_dist.size(0)
		mat_sim = label.expand(N, N).eq(label.expand(N, N).t()).float()

		dist_ap, dist_an, ap_idx, an_idx = _batch_hard_2(mat_dist, mat_sim, indice=True, topK=self.topk)

		assert an_idx.size(0)==ap_idx.size(0)
		dist_group_ap = torch.sum((emb1 - torch.mean(emb2[ap_idx], dim=1)) ** 2, dim=1).sqrt()
		dist_group_an = torch.sum((emb1 - torch.mean(emb2[an_idx], dim=1)) ** 2, dim=1).sqrt()

		# triple_dist = torch.stack((dist_ap, dist_an), dim=1)
		triple_dist = torch.stack((dist_group_ap, dist_group_an), dim=1)
		triple_dist = F.log_softmax(triple_dist, dim=1)


		# Circle loss ↓
		ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
		an = torch.clamp_min(sn.detach() + self.m, min=0.)

		delta_p = 1 - self.m
		delta_n = self.m

		logit_p = - ap * (sp - delta_p) * self.gamma
		logit_n = an * (sn - delta_n) * self.gamma

		loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

		if self.skip_mean:
			return loss
		else:
			return loss.mean()