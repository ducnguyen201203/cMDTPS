from losses import objectives
from losses import ema_loss
from model.clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import math

class CORE_MODEL(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.image_encoder.name, args.image_encoder.img_size, args.image_encoder.stride_size, download_root=args.iocfg.datadir)
        self.embed_dim = base_cfg['embed_dim']
        self.vision_patch_size = base_cfg['vision_patch_size']
        
        self.logit_scale = torch.ones([]) * (1 / args.image_encoder.temperature) 
        self.eps = 1e-2
        if 'id' in args.losses.loss_names or args.ema.distillation :
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

        if "mlm" in args.losses.loss_names or "mim" in args.losses.loss_names:

            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,self.embed_dim // 64, batch_first=True)
            self.cross_modal_transformer = Transformer(width=self.embed_dim, layers=args.cross_modal.cmt_depth, heads=self.embed_dim // 64)
            scale = self.cross_modal_transformer.width**-0.5
            proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width)**-0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)
            
            if 'mlm' in args.losses.loss_names:
                self.mlm_head = nn.Sequential(
                    OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                                ('gelu', QuickGELU()),
                                ('ln', LayerNorm(self.embed_dim)),
                                ('fc', nn.Linear(self.embed_dim, args.text_encoder.vocab_size))]))
                # init mlm head
                nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
                nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

            if "mim" in args.losses.loss_names:
                self.mask_token = nn.Parameter(torch.zeros(1, 3, 1, 1), requires_grad=True)
                self.hogl = objectives.HOGLayerC(nbins=self.args.losses.mim.hog.bins, pool=self.args.losses.mim.hog.pool, 
                                    norm_pix_loss=self.args.losses.mim.hog.norm_pix_loss, 
                                    gaussian_window=self.args.losses.mim.hog.gaussian_window)
                # h,w = self.args.image_encoder.img_size
                # num_x  = (h - self.vision_patch_size) // self.args.image_encoder.stride_size + 1
                # num_y  = (w - self.vision_patch_size) // self.args.image_encoder.stride_size + 1
                # num_patches = num_x * num_y
                class_MIM = ( (self.vision_patch_size // self.args.losses.mim.hog.pool) ** 2 ) * 3 * self.args.losses.mim.hog.bins
                self.mim_head = nn.Sequential(
                    OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                                ('gelu', QuickGELU()),
                                ('ln', LayerNorm(self.embed_dim)),
                                ('fc', nn.Linear(self.embed_dim, class_MIM))]))
                # init mlm head
                nn.init.normal_(self.mim_head.dense.weight, std=fc_std)
                nn.init.normal_(self.mim_head.fc.weight, std=proj_std)

        # create momentum models
        if not self.args.ema.enable: self.ema_base_model = None
        else:
            self.ema_base_model, _ = build_CLIP_from_openai_pretrained(args.image_encoder.name, args.image_encoder.img_size, args.image_encoder.stride_size, download_root=args.iocfg.datadir) 
            if 'id' in args.losses.loss_names or args.ema.distillation:
                self.classifier_ema = nn.Linear(self.embed_dim, self.num_classes)
                nn.init.normal_(self.classifier.weight.data, std=0.001)
                nn.init.constant_(self.classifier.bias.data, val=0.0)

            if "mlm" in args.losses.loss_names or "mim" in args.losses.loss_names:
                self.ln_pre_t_emm = LayerNorm(self.embed_dim)
                self.ln_pre_i_ema = LayerNorm(self.embed_dim)
                self.ln_post_ema = LayerNorm(self.embed_dim)
                self.cross_attn_ema = nn.MultiheadAttention(self.embed_dim,self.embed_dim // 64, batch_first=True)
                self.cross_modal_transformer_ema = Transformer(width=self.embed_dim, layers=args.cross_modal.cmt_depth, heads=self.embed_dim // 64)
                
                attn_std = scale = self.cross_modal_transformer_ema.width**-0.5
                proj_std = scale * ((2 * self.cross_modal_transformer_ema.layers)**-0.5)
                fc_std = (2 * self.cross_modal_transformer_ema.width)**-0.5
                for block in self.cross_modal_transformer_ema.resblocks:
                    nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                    nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                    nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                    nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
                # init cross attn
                nn.init.normal_(self.cross_attn_ema.in_proj_weight, std=attn_std)
                nn.init.normal_(self.cross_attn_ema.out_proj.weight, std=proj_std)
                
                if 'mlm' in args.losses.loss_names:
                    self.mlm_head_ema = nn.Sequential(
                        OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                                    ('gelu', QuickGELU()),
                                    ('ln', LayerNorm(self.embed_dim)),
                                    ('fc', nn.Linear(self.embed_dim, args.text_encoder.vocab_size))]))
                    # init mlm head
                    nn.init.normal_(self.mlm_head_ema.dense.weight, std=fc_std)
                    nn.init.normal_(self.mlm_head_ema.fc.weight, std=proj_std)

                if "mim" in args.losses.loss_names:
                    self.mim_head_ema = nn.Sequential(
                        OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                                    ('gelu', QuickGELU()),
                                    ('ln', LayerNorm(self.embed_dim)),
                                    ('fc', nn.Linear(self.embed_dim, class_MIM))]))
                    # init mlm head
                    nn.init.normal_(self.mim_head.dense.weight, std=fc_std)
                    nn.init.normal_(self.mim_head.fc.weight, std=proj_std)
             


    #######################################    METHOD SECTION    ####################################################
    def _set_task(self):
        self.current_task = self.args.losses.loss_names
        print(f'Training Model with {self.current_task} tasks')
      
    def cross_former(self, q, k, v):
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NumxLengthxDim -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

    def encode_image(self, image):
        x = self.base_model.encode_image(image)
        return x[:, 0, :].float()
        # return x.float() # for CLIP ResNet visual model

    def encode_text(self, text):
        x = self.base_model.encode_text(text)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()  

    def encode_image_ema(self, image):
        x = self.ema_base_model.encode_image(image)
        return x[:, 0, :].float()
        # return x.float() # for CLIP ResNet visual model

    def encode_text_ema(self, text):
        x = self.ema_base_model.encode_text(text)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()  


    ##functions for MIM
    def mim_per_sample_block_masking(self, x, mask_ratio, block_size=16):
        batch, channel, height, width = x.shape
        # input_size = self.img_size        
        # assert height == width, f"Input height and width doesn't match ({height} != {width})."
        
        mask_size_w = width // block_size            #number patch/row
        mask_size_h = height // block_size            #number patch/row
        bw_ratio_h = height // mask_size_h                  #??
        bw_ratio_w = width // mask_size_w                  #??
        len_keep = int(mask_size_w * mask_size_h * (1 - mask_ratio)) #the number of patch will not be masked
        
        noise = torch.rand(batch, mask_size_w * mask_size_h, device=x.device)  # noise in [0, 1]
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        patch_mask = torch.ones([batch, mask_size_h * mask_size_w], device=x.device)
        patch_mask[:, :len_keep] = 0   #   0 0 0 0 0 0 0 0 0  1 1 1 1 1  1 1 1...
        patch_mask = torch.gather(patch_mask, dim=1, index=ids_restore)  #random mask by ids_restore
        patch_mask = patch_mask.reshape(batch, 1, mask_size_h, mask_size_w).long()    #path_mask
        
        pixel_mask = patch_mask.repeat(1, bw_ratio_h * bw_ratio_w, 1, 1)  #--> pixel mask of img => it's size = image's size
        pixel_mask = pixel_mask.reshape(batch, bw_ratio_h, bw_ratio_w, mask_size_h, mask_size_w).permute(0, 3, 1, 4, 2).reshape(batch, 1, height, width)
        
        if block_size > self.vision_patch_size:
            print("block size > path size --> repeat interleave")
            patch_mask = torch.repeat_interleave(patch_mask, block_size//self.vision_patch_size, dim=2)
            patch_mask = torch.repeat_interleave(patch_mask, block_size//self.vision_patch_size, dim=3)
        
        return pixel_mask, patch_mask


    def _mask_tokens(self, inputs, mask_token_index, vocab_size, special_token_indices=[49407, 49408, 49406], mlm_probability=0.15, replace_prob=0.1, orginal_prob=0.1, ignore_index=0, probability_matrix=None):

        device = inputs.device
        labels = inputs.clone() 

        # Get positions to apply mlm (mask/replace/not changed). (mlm_probability)
        if probability_matrix is None:
            probability_matrix = torch.full(labels.shape, mlm_probability, device=device) * (inputs  != 0).long()
        special_tokens_mask = torch.full(inputs.shape, False, dtype=torch.bool, device=device)
        for sp_id in special_token_indices:
            special_tokens_mask = special_tokens_mask | (inputs==sp_id) 
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0) #unmasked special tokens
        mlm_mask = torch.bernoulli(probability_matrix).bool()
        labels[~mlm_mask] = ignore_index  # We only compute loss on mlm applied tokens

        # mask  (mlm_probability * (1-replace_prob-orginal_prob))
        mask_prob = 1 - replace_prob - orginal_prob # 0.8
        mask_token_mask = torch.bernoulli(torch.full(labels.shape, mask_prob, device=device)).bool() & mlm_mask
        inputs[mask_token_mask] = mask_token_index 

        # replace with a random token (mlm_probability * replace_prob)
        rep_prob = replace_prob/(replace_prob + orginal_prob)
        replace_token_mask = torch.bernoulli(torch.full(labels.shape, rep_prob, device=device)).bool() & mlm_mask & ~mask_token_mask
        random_words = torch.randint(vocab_size, labels.shape, dtype=torch.long, device=device)
        inputs[replace_token_mask] = random_words[replace_token_mask]

        # do nothing (mlm_probability * orginal_prob)
        pass

        return inputs, labels, mlm_mask

    def _adjust_mask_prob(self, steps, max_steps, mask_prob, masking_mode):
        with torch.no_grad():
            if  masking_mode == 'cosine':
                mask_prob = mask_prob * (1 + math.cos(math.pi / max_steps * steps)) + 0.02
            elif masking_mode == 'linear':
                turn = max_steps * 0.4
                mask_prob = 0.2 + 0.1 * math.cos(math.pi / turn * steps) if steps < turn \
                        else -(0.5 / max_steps) ** 2 * (steps - turn) ** 2 + 0.1
            else:
                mask_prob = mask_prob
        return mask_prob

    ###MAIN Forward function
    def forward(self, batch):
        ret = dict()
        images = batch['images']
        caption_ids = batch['caption_ids']
        image_feats, text_feats = self.base_model(images, caption_ids) #torch.Size([B, tokens, 512]) torch.Size([1, tokens, 512])
        i_feats = image_feats[:, 0, :].float()  #get global feature
        # i_feats = image_feats.float() # for CLIP ResNet visual model
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float() #get global features
        image_features_norm = F.normalize(i_feats)
        text_features_norm = F.normalize(t_feats)
        logit_scale = self.logit_scale.exp()
        logit_scale.data = torch.clamp(logit_scale.data, max=100)
        #update ema model 
        if  self.args.ema.enable:
            self._momentum_update(self.base_model, self.ema_base_model, momentum_alpha=self.args.ema.momentum_alpha, global_step=self.args.cur_step)
            if self.args.ema.distillation:
                self._momentum_update(self.classifier, self.classifier_ema, momentum_alpha=self.args.ema.momentum_alpha, global_step=self.args.cur_step)

            if self.args.ema.enhance_mmm.enable:
                
                self._momentum_update(self.cross_attn, self.cross_attn_ema, momentum_alpha=self.args.ema.momentum_alpha, global_step=self.args.cur_step)
                self._momentum_update(self.cross_modal_transformer, self.cross_modal_transformer_ema, momentum_alpha=self.args.ema.momentum_alpha, global_step=self.args.cur_step)
                if 'mim' in self.args.losses.loss_names:
                    self._momentum_update(self.mim_head, self.mim_head_ema, momentum_alpha=self.args.ema.momentum_alpha, global_step=self.args.cur_step)
                if 'mlm' in self.args.losses.loss_names:
                    self._momentum_update(self.mlm_head, self.mlm_head_ema, momentum_alpha=self.args.ema.momentum_alpha, global_step=self.args.cur_step)

        logit_scale = self.logit_scale
        ret.update({'temperature': 1 / logit_scale})
        if 'itc' in self.current_task:
            ret.update({'itc_loss':objectives.compute_itc(i_feats, t_feats, logit_scale)})
        
        if 'sdm' in self.current_task:
            ret.update({'sdm_loss':objectives.compute_sdm(i_feats, t_feats, batch['pids'], logit_scale)})

        if 'cmpm' in self.current_task:
            ret.update({'cmpm_loss':objectives.compute_cmpm(i_feats, t_feats, batch['pids'])})
        

        image_logits = self.classifier(i_feats.half()).float()
        text_logits = self.classifier(t_feats.half()).float()
        if 'id' in self.current_task:
            ret.update({'id_loss':objectives.compute_id(image_logits, text_logits, batch['pids'])*self.args.losses.id_loss_weight})

            image_pred = torch.argmax(image_logits, dim=1)
            text_pred = torch.argmax(text_logits, dim=1)

            image_precision = (image_pred == batch['pids']).float().mean()
            text_precision = (text_pred == batch['pids']).float().mean()
            ret.update({'img_acc': image_precision})
            ret.update({'txt_acc': text_precision})


        if "ritc" in self.current_task:
            with torch.no_grad():
                idx = batch['pids']
                pos_idx = torch.eq(idx.view(-1, 1), idx).float()
                sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)

            logits_per_image_1 = logit_scale * image_features_norm @ text_features_norm.t()
            logits_per_text_1 = logit_scale * text_features_norm @ image_features_norm.t()
            img_log = F.log_softmax(logits_per_image_1, dim=1)
            txt_log = F.log_softmax(logits_per_text_1, dim=1)
            target_log = (sim_targets + self.eps).log()

            kl_img = F.kl_div(target_log, img_log, log_target=True, reduction='batchmean')
            kl_txt = F.kl_div(target_log, txt_log, log_target=True, reduction='batchmean')
            ritc_loss = 0.5 * (kl_img + kl_txt)
            ret.update({'ritc_loss': ritc_loss * self.args.losses.ritc_loss_weight})

        if 'citc' in self.current_task:
            """
            Cyclic Image-Text Contrastive Loss.
            from https://arxiv.org/pdf/2308.10045.pdf
            """
            logits_image_per_image = logit_scale * image_features_norm @ image_features_norm.t()
            logits_text_per_text = logit_scale * text_features_norm @ text_features_norm.t()
            inmodal_cyclic_loss = (logits_image_per_image - logits_text_per_text).square().mean() / (logit_scale * logit_scale)
            
            logits_text_per_image = logit_scale * image_features_norm @ text_features_norm.t()
            logits_image_per_text = logit_scale * text_features_norm @ image_features_norm.t()
            crossmodal_cyclic_loss = (logits_text_per_image - logits_image_per_text).square().mean() /  (logit_scale * logit_scale)
            
            citc_loss = self.args.losses.citc.lambda1 * inmodal_cyclic_loss + self.args.losses.citc.lambda2 * crossmodal_cyclic_loss
            ret.update({'citc_loss': citc_loss * self.args.losses.citc_loss_weight})

        if 'mlm' in self.current_task:
            mlm_prob = self._adjust_mask_prob(steps=self.args.cur_step, max_steps=self.args.total_step, 
                                              mask_prob=self.args.losses.mlm.mask_prob, masking_mode=self.args.losses.masking_mode)
            with torch.no_grad():  
                if self.args.ema.enhance_mmm.enable:
                    ema_mlm_ids, ema_mlm_labels = self._mask_tokens(batch['caption_ids'].clone(), mask_token_index=49405, \
                                                    vocab_size=self.args.text_encoder.vocab_size-3,
                                                    mlm_probability=mlm_prob, ignore_index=0)
                    ema_mlm_feats = self.ema_base_model.encode_text(ema_mlm_ids)
                    ema_mlm_pred = self.cross_former(ema_mlm_feats, image_feats, image_feats)
                    ema_mlm_pred = self.mlm_head(ema_mlm_pred)  # [batch_size, text_len, num_scopus]
        
                    ema_mlm_labels = F.one_hot(ema_mlm_labels, self.args.text_encoder.vocab_size)
                    ema_mlm_pred = torch.exp(ema_mlm_pred) / torch.sum(torch.exp(ema_mlm_pred), axis=2, keepdims=True)
                    loss = -torch.mean(torch.sum(ema_mlm_labels * torch.log(ema_mlm_pred), axis=2, keepdims=True), axis=2,  keepdims=True).squeeze(dim=2)

                    loss = loss * (ema_mlm_ids > 0) #ignore element that mlm_ids = 0
                    idx_tensor = torch.argsort(loss, dim=1, descending=True) #find the patchs having high loss
                    P = torch.sum(loss > 0, axis=1)   #identify exactly the real length of each sentence
                    num_masked_paths = torch.round(P * mlm_prob + 0.5).clamp_max(P).long()
                    num_hard_masked_paths = torch.round(num_masked_paths * self.args.ema.enhance_mmm.ratio +0.5).clamp_max(num_masked_paths).long()
                    mask_index = torch.arange(idx_tensor.shape[1]).expand(*idx_tensor.shape).to(loss.device) < num_hard_masked_paths.unsqueeze(1)
                    mask_index = mask_index * idx_tensor
                    hard_paths = torch.zeros(idx_tensor.shape).to(idx_tensor.device)
                    hard_paths.scatter_(1, mask_index, idx_tensor * 0 + 1 - mlm_prob )
                    hard_paths[:, 0] = 0 #the mask prob of cls value always is 0
                    #initial prob matrix
                    probability_matrix = torch.full(batch['caption_ids'].shape, mlm_prob, device=ema_mlm_ids.device) * (ema_mlm_ids  != 0).long()
                    probability_matrix += hard_paths

                    mlm_ids, mlm_labels, _ = self._mask_tokens(batch['caption_ids'].clone(), mask_token_index=49405, \
                                                        vocab_size=self.args.text_encoder.vocab_size-3,
                                                        mlm_probability=mlm_prob, probability_matrix=probability_matrix, ignore_index=0)
                else:
                    mlm_ids, mlm_labels, mlm_mask = self._mask_tokens(batch['caption_ids'].clone(), mask_token_index=49405, \
                                                    vocab_size=self.args.text_encoder.vocab_size-3,
                                                    mlm_probability=mlm_prob, ignore_index=0)

            mlm_feats = self.base_model.encode_text(mlm_ids)

            x = self.cross_former(mlm_feats, image_feats, image_feats)

            x = self.mlm_head(x)  # [batch_size, text_len, num_scopus]

            mlm_labels = mlm_labels.reshape(-1)
            scores = x.float().reshape(-1, self.args.text_encoder.vocab_size)
            ret.update({'mlm_loss': objectives.compute_mlm(scores, mlm_labels)*self.args.losses.mlm_loss_weight})

            pred = scores.max(1)[1]
            mlm_label_idx = torch.nonzero(mlm_labels)
            acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
            ret.update({'mlm_acc': acc})
        
        
        if 'mim' in self.current_task:
            mim_images = batch['mim_images']
            mim_prob = self._adjust_mask_prob(steps=self.args.cur_step, max_steps=self.args.total_step, 
                                              mask_prob=self.args.losses.mim.mask_prob, masking_mode=self.args.losses.masking_mode)
    
            with torch.no_grad():
                if self.args.ema.enhance_mmm.enable:
                    pixel_mask, patch_mask   = self.mim_per_sample_block_masking(mim_images, mask_ratio=mim_prob, block_size=self.args.losses.mim.hog.pool) 
                    print(patch_mask[0], patch_mask.shape)
                    B, C, H, W = mim_images.shape
                    ema_x = mim_images * (1-pixel_mask) + (pixel_mask) * self.mask_token.repeat(B, 1, H, W)   #mask with random values
                    ema_target      = self.hogl(mim_images)   #target = Bx(C*nbin)x(H/pool)x(W/pool)
                    ema_patch_mask  = patch_mask.permute(0, 2, 3, 1)
                    ema_mim_feats = self.encode_image_ema(ema_x)
                    ema_mim_feats = self.cross_former(ema_mim_feats, text_feats, text_feats)  #BxPatchsxD
                    ema_pred = self.mim_head(ema_mim_feats)[:,1:,:]   #BxPatchsxD
                    mim_loss = objectives.compute_mim(ema_pred, ema_target, ema_patch_mask, reduce_mean=False)
                    print(mim_loss, mim_loss.shape)
                    raise "haha"
                else:
                    pixel_mask, patch_mask   = self.mim_per_sample_block_masking(mim_images, mask_ratio=mim_prob, block_size=self.args.losses.mim.hog.pool) 
                    B, C, H, W = mim_images.shape
                    x = mim_images * (1-pixel_mask) + (pixel_mask) * self.mask_token.repeat(B, 1, H, W)   #mask with random values
                    target      = self.hogl(mim_images)   #target = Bx(C*nbin)x(H/pool)x(W/pool)
                    patch_mask  = patch_mask.permute(0, 2, 3, 1)


            
            
            mim_feats = self.base_model.encode_image(x)

            x = self.cross_former(mim_feats, text_feats, text_feats)  #BxPatchsxD
            pred = self.mim_head(x)   #BxPatchsxD
            pred = pred[:,1:,:] #skip cls token
            mim_loss = objectives.compute_mim(pred, target, patch_mask)
            mim_loss = mim_loss.mean() * self.args.losses.mim_loss_weight
            ret.update({'mim_loss':  mim_loss})

        if 'kntriplet' in self.current_task:
            """
            Inspire on :
            - Calibrating Cross-modal Features for Text-Based Person Searching
            - Relation Preserving Triplet Mining for Stabilising the Triplet Loss
            """
            
            kntriplet_loss = objectives.compute_ntriplet(i_feats, t_feats, batch['pids'], 
                                                        i2i_margin=self.args.losses.kntriplet.i2im,
                                                        t2t_margin=self.args.losses.kntriplet.t2tm,
                                                        i2t_margin=self.args.losses.kntriplet.i2tm,
                                                        t2i_margin=self.args.losses.kntriplet.t2im,
                                                        weights=self.args.losses.kntriplet.weights,
                                                        topK=self.args.losses.kntriplet.topk
                                                        )
            kntriplet_loss = self.args.losses.kntriplet_loss_weight * kntriplet_loss
            ret.update({'triplet_loss':  kntriplet_loss})
        
        #update ema model 
        if self.args.ema.enable:
            with torch.no_grad():
                images_ema = batch['images_ema']
                caption_ids_ema = batch['caption_ids']
                i_feats_ema = self.encode_image_ema(images_ema) #torch.Size([B, tokens, 512]) torch.Size([1, tokens, 512])
                t_feats_ema = self.encode_text_ema(caption_ids_ema) #torch.Size([B, tokens, 512]) torch.Size([1, tokens, 512])
                
                image_features_norm_ema = F.normalize(i_feats_ema)
                text_features_norm_ema = F.normalize(t_feats_ema)

                image_logits_ema = self.classifier_ema(i_feats_ema.half()).float()
                text_logits_ema = self.classifier_ema(t_feats_ema.half()).float()
                
                image_per_text_ema = logit_scale * image_features_norm_ema @ text_features_norm_ema.t()
            
            
            #CALCULATE LOSS
            criterion_ce_soft  =  ema_loss.KLDivLoss().cuda()
            distil_intra_loss_i = criterion_ce_soft(image_logits, image_logits_ema)
            distil_intra_loss_t = criterion_ce_soft(text_logits, text_logits_ema)
            distil_intra_loss   = 0.5 * (distil_intra_loss_i + distil_intra_loss_t) 
            
            image_per_text = logit_scale * image_features_norm @ text_features_norm.t()
            distil_inter_loss = ((image_per_text - image_per_text_ema) ** 2).mean()
            
            ret.update({'intra_distil_loss':distil_intra_loss*self.args.ema.intra_distil_loss_weight})
            ret.update({'inter_distil_loss':distil_inter_loss*self.args.ema.inter_distil_loss_weight})

        return ret



    @torch.no_grad()
    def copy_params(self, model, ema_model):
        for param, param_m in zip(model.parameters(), ema_model.parameters()):
            param_m.data.copy_(param.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self, model, ema_model, momentum_alpha=0.99, global_step=0):
        momentum_alpha = min(1 - 1 / (global_step + 1), momentum_alpha)        
        for param, param_m in zip(model.parameters(), ema_model.parameters()):
            param_m.data = param_m.data * momentum_alpha + param.data * (1. - momentum_alpha)




def build_model(args, num_classes=11003):
    model = CORE_MODEL(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    return model
