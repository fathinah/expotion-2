import torch
from torch import nn, einsum
from .musicgen_cc import MusicGen
from ..utilities.model_utils import freeze, print_trainable_parameters


def get_musicgen(sec, device):
    mg = MusicGen.get_pretrained(name='large', device=device)
    mg.set_generation_params(duration=sec, extend_stride=16, top_k=250)
    mg.lm.here()
    freeze(mg.lm)
    return mg


class CondMusicgen(nn.Module):
    def __init__(self, sec, device="cuda"):
        super().__init__()
        mg = get_musicgen(sec, device)
        self.musicgen = mg
        self.lm = mg.lm
        self.max_duration = sec
        self.frame_rate = 25 ##changed

    def set_training(self):
        self.lm.train()

    def forward(self, seq, desc, embed_fn, num_samples=1, mode="train",
                total_gen_len=None, prompt_tokens=None):

        mg = self.musicgen
        lm = self.lm

        attributes, _ = mg._prepare_tokens_and_attributes(desc, None)

        if mode == "train":
            with mg.autocast:
                out = lm.compute_predictions(codes=seq,
                                             embed_fn=embed_fn,
                                             conditions=attributes)
            return out
        elif mode == "inference":
            if total_gen_len is None:
                total_gen_len = int(mg.duration * mg.frame_rate)

            with mg.autocast:
                gen_tokens = lm.generate(embed_fn=embed_fn, num_samples=num_samples,
                                         prompt=None, conditions=attributes,
                                         callback=None, max_gen_len=total_gen_len, **mg.generation_params)
                return gen_tokens
        elif mode == "continuation":
            with mg.autocast:
                gen_tokens = lm.generate(embed_fn=embed_fn, num_samples=num_samples,
                                         prompt=prompt_tokens, conditions=attributes,
                                         callback=None, max_gen_len=total_gen_len, **mg.generation_params)
                return gen_tokens

    def generate(self, cp_fn, rgb_emb, desc, num_samples):
        mg = self.musicgen
        lm = self.lm

        attributes, _ = mg._prepare_tokens_and_attributes(desc, None)

        all_tokens = []
        stride_tokens = int(self.frame_rate * mg.extend_stride)
        current_gen_offset = 0
        prompt_length = 0
        prompt_tokens = None
        total_gen_len = drums.shape[-1] - 1
        total_sec = total_gen_len / 50.

        while current_gen_offset + prompt_length < total_gen_len:
            time_offset = current_gen_offset / self.frame_rate
            chunk_duration = min(total_sec - time_offset, self.max_duration)
            max_gen_len = int(chunk_duration * self.frame_rate)
            if prompt_length >= max_gen_len:
                break

            with mg.autocast:
                cond_mask = torch.ones([num_samples, max_gen_len + 1]).to(rgb_emb.device)              
                rgb_emb_clip = rgb_emb[:, current_gen_offset:current_gen_offset + max_gen_len + 1]

                embed_fn = cp_fn(rgb_emb=rgb_emb_clip,
                                 cond_mask=cond_mask,
                                  max_n_frames=max_gen_len,
                                 mode="inference")
                gen_tokens = lm.generate(num_samples=1,
                                         embed_fn=embed_fn, prompt=prompt_tokens,
                                         conditions=attributes,
                                         callback=None, max_gen_len=max_gen_len, **mg.generation_params)
            if prompt_tokens is None:
                all_tokens.append(gen_tokens)
            else:
                all_tokens.append(gen_tokens[:, :, prompt_tokens.shape[-1]:])
            prompt_tokens = gen_tokens[:, :, stride_tokens:]
            prompt_length = prompt_tokens.shape[-1]
            current_gen_offset += stride_tokens
            if current_gen_offset > 50 * 80:
                break

        gen_tokens = torch.cat(all_tokens, dim=-1)
        return gen_tokens

    def get_input_embeddings(self):
        return self.lm.emb


class EmbFn:
    '''
    Purpose: Manages the conditional embeddings and their integration into the transformer layers.
    '''
    def __init__(self, activates, fn, start_layer, max_len, inference=False, skip=None):
        self.interval = None
        self.index = -1
        self.adaptor = None
        self.start_layer = start_layer
        self.activates = activates
        self.max_len = max_len
        self.fn = fn
        self.inference = inference
        self.skip = skip

    def get_adaptor(self, tag):
        '''
        Retrieves the adaptor and gate values for a given layer.
        '''
        index = self.index
        if index < self.start_layer or tag == "cross":
            return None, None
        i = index - self.start_layer
        adaptor, gate = self.fn(i, self.activates)
        # if self.adaptor is not None:
        #    adaptor = self.adaptor + adaptor
        return adaptor, gate

    def set_state(self, prefix_q, prefix_k, prefix_v):
        self.cache[str(self.index)] = [prefix_q, prefix_k, prefix_v]

    def get_state(self):
        prefix_q, prefix_k, prefix_v = self.cache[str(self.index)]
        return prefix_q, prefix_k, prefix_v

    def clear_state(self):
        self.qkv = {}
        torch.cuda.empty_cache()

    def crop(self, tag, x):

        if self.interval is not None:
            st, ed = self.interval
            if st >= self.max_len:
                st = self.max_len - 1
                ed = st + 1
            return x[:, :, st:ed, :]
        return x

    def get_cross_attention_src(self, src):
        return src

    def modify(self, x, dt_x, gate):
        return dt_x * gate + x

    def set_uncond(self, uncond):
        self.uncond = uncond

    def get_uncond(self):
        return self.uncond

    def set_uncond_cross_attention(self, x):
        self.uncond_cross_att = x

    def get_cross(self):
        return True

    def get_status(self, tag):
        return tag == "self" and self.index >= self.start_layer

    def update_adaptor(self, adaptor):
        self.adaptor = adaptor

    def set_index(self, index):
        self.index = index

    def update_interval(self, st, ed):
        self.interval = [st, ed]


class CPTransformerLayer(nn.Module):
    '''
    Purpose: Custom transformer layer that incorporates conditional inputs.
    '''
    def __init__(self, norm1, norm2, layer_scale_1, dropout1, self_attn, layer_scale_2,
                 autocast, linear1, linear2, activation, dropout, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm1 = norm1
        self.norm2 = norm2
        self.layer_scale_1 = layer_scale_1
        self.dropout1 = dropout1
        self.self_attn = self_attn
        self.layer_scale_2 = layer_scale_2
        self.autocast = autocast
        self.linear1 = linear1
        self.linear2 = linear2
        self.activation = activation
        self.dropout = dropout

    def _ff_block(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

    def forward(self, x, cond):
        with self.autocast:
            nx = self.norm1(x) + cond
            q, k, v, o = self.self_attn(nx, nx, nx, emb_fn=None,
                                        attn_mask=None,
                                        key_padding_mask=None,
                                        need_weights=False, is_causal=False, return_qkv=True)
            x = x + self.layer_scale_1(self.dropout1(o))
            x = x + self.layer_scale_2(self._ff_block(self.norm2(x)))
        return q, k, v, x


class CPTransformer(nn.Module):
    '''
    Modifies the original transformer model to incorporate conditional embeddings starting from a specific layer.

    '''
    def __init__(self, model, emb_fn, start_layer, latent_dim, autocast, stride=50 * 10):
        super().__init__()

        self.emb_fn = {
            "emb": emb_fn
        }

        new_layers = nn.ModuleList()

        hidden_dim = 2048
    
        cond_dim = latent_dim 
        num_layers = len(model.layers) - start_layer
        max_n_frames = 250

        self.masked_embedding = nn.Parameter(
            torch.randn(num_layers, max_n_frames + 1, cond_dim),
            requires_grad=True)
        self.pos_emb = nn.Parameter(
            torch.randn(num_layers + 1, max_n_frames + 1, hidden_dim),
            requires_grad=True)
        self.encodec_emb = nn.Linear(hidden_dim, latent_dim, bias=False)
        self.merge_linear = nn.ModuleList()
        self.rgb_emb = nn.ModuleList()
        for i in range(start_layer, len(model.layers)):
            norm1 = model.layers[i].norm1
            norm2 = model.layers[i].norm2
            layer_scale_1 = model.layers[i].layer_scale_1
            dropout1 = model.layers[i].dropout1
            self_attn = model.layers[i].self_attn
            layer_scale_2 = model.layers[i].layer_scale_2
            linear1 = model.layers[i].linear1
            linear2 = model.layers[i].linear2
            activation = model.layers[i].activation
            dropout = model.layers[i].dropout
            new_layers.append(CPTransformerLayer(norm1=norm1,
                                                 norm2=norm2,
                                                 layer_scale_1=layer_scale_1,
                                                 dropout1=dropout1,
                                                 self_attn=self_attn,
                                                 linear1=linear1,
                                                 linear2=linear2,
                                                 activation=activation,
                                                 dropout=dropout,
                                                 layer_scale_2=layer_scale_2,
                                                 autocast=autocast))

            self.merge_linear.append(nn.Linear(cond_dim, hidden_dim, bias=False))
            self.rgb_emb.append(nn.Linear(128, latent_dim, bias=False))  ## to be changed

        self.layers = new_layers
        self.gates = nn.Parameter(torch.zeros([num_layers]))
        freeze(self.layers)

        self.max_n_frames = max_n_frames
        self.start_layer = start_layer
        self.num_layers = num_layers
        self.stride = stride

    def fn(self, i, activates):
        if i >= self.num_layers:
            return None, None
        return activates[i]

    def forward(self, rgb_emb, cond_mask, max_n_frames, mode, skip=None):
        cond_dim = 768
        max_n_frames = self.max_n_frames if max_n_frames is None else max_n_frames
        B, T, _ = rgb_emb.shape ### 
        o = self.pos_emb[0][None, :T].repeat(B, 1, 1)
        # cond_mask = torch.cat([torch.ones([B, T, 37]).to(o.device),
        #                        cond_mask[:, 0, :, None].repeat(1, 1, 12),
        #                        cond_mask[:, 1, :, None].repeat(1, 1, r)], -1)
        # mask_embedding = self.masked_embedding[None, :, :T].repeat(B, 1, 1, 1)
        cond_mask_expanded = cond_mask[:, :, None].repeat(1, 1, cond_dim)
        print('cond_mask_expanded')
        print(cond_mask_expanded.shape)
        mask_embedding = self.masked_embedding[None, :, :T].repeat(B, 1, 1, 1)
        outs = []
        for i in range(len(self.layers)):
            mask_embedding_per_layer = mask_embedding[:, i]  # Shape: [B, T, 768]
            # Apply the conditional mask
            cond_t = torch.where(cond_mask_expanded > 0, rgb_emb, mask_embedding_per_layer)  # Shape: [B, T, 768]
            # Merge embeddings and add positional encoding
            embedding = self.merge_linear[i](cond_t) + self.pos_emb[i + 1][None, :T].repeat(B, 1, 1)  # Shape: [B, T, hidden_dim]
            q, k, v, o = self.layers[i](o, embedding)
            outs.append([[q, k, v], self.gates[i]])

        emb_fn = EmbFn(activates=outs, fn=self.fn,
                       start_layer=self.start_layer,
                       max_len=max_n_frames,
                       inference=(mode == "inference"),
                       skip=skip)

        return emb_fn

    def save_weights(self, path):
        state_dict = {}
        sdict = self.state_dict()
        for n in sdict:
            if str.startswith(n, "layers"):
                continue
            state_dict[n] = sdict[n]
        torch.save(state_dict, path)

    def load_weights(self, path):
        self.load_state_dict(torch.load(path, map_location="cpu"), strict=False)


class CoCoMulla(nn.Module):
    def __init__(self, sec, num_layers, latent_dim):
        '''
        ... Initialize CondMusicgen and CPTransformer ...
        Purpose: The main model class that integrates the conditional transformer (CPTransformer) with the conditional MusicGen model (CondMusicgen).

        '''
        super().__init__()
        lm = CondMusicgen(sec)
        self.peft_model = lm
        self.musicgen = lm.musicgen
        self.cp_transformer = CPTransformer(self.musicgen.lm.transformer,
                                            emb_fn=self.musicgen.lm.emb,
                                            start_layer=48 - num_layers,
                                            latent_dim=latent_dim,
                                            autocast=self.musicgen.autocast)

    def set_training(self):
        self.peft_model.set_training()
        print_trainable_parameters(self)

    def save_weights(self, path):
        self.cp_transformer.save_weights(path)

    def load_weights(self, path):
        self.cp_transformer.load_weights(path)

    def forward(self, seq, rgb_emb,  desc, cond_mask,
                num_samples=1, mode="train", max_n_frames=None, prompt_tokens=None):
        print('masuk forward')
        '''
        Generates the embedding function using cp_transformer.
        Passes the sequence and embeddings to peft_model 
        for training or inference.
        '''

        embed_fn = self.cp_transformer(rgb_emb=rgb_emb,
                                       cond_mask=cond_mask,
                                        max_n_frames=max_n_frames,
                                       mode=mode, skip=None)
        out = self.peft_model(seq, desc=desc, embed_fn=embed_fn,
                              mode=mode, num_samples=num_samples, total_gen_len=max_n_frames,
                              prompt_tokens=prompt_tokens)
        return out

    def generate(self, rgb_emb, desc,
                 num_samples=1):
        '''
        Purpose: Generates music using the provided conditional inputs.
        '''
        out = self.peft_model.generate(cp_fn=self.cp_transformer, rgb_emb= rgb_emb, desc=desc, num_samples=num_samples)
        return out