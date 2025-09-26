import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Union, Dict, Any, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig

@dataclass
class AlignedSFTConfig(SFTConfig):
    teacher_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the teacher model for hidden state alignment"}
    )
    
    alignment_loss_type: str = field(
        default="cross_attention",
        metadata={"help": "Loss type: 'cross_attention' or 'pooled_mse' or 'pooled_cosine'"}
    )
    
    alignment_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for the alignment loss (alpha)"}
    )
    
    teacher_gpu: int = field(
        default=1,
        metadata={"help": "GPU device for teacher model"}
    )
    
    teacher_model_init_kwargs: Optional[Dict[str, Any]] = field(
        default=None,
        metadata={"help": "Keyword arguments for teacher model initialization"}
    )
    
    student_layers_to_align: Optional[List[int]] = field(
        default=None,
        metadata={"help": "Which student layers to align (0-30). None = last layer only"}
    )
    
    teacher_layers_to_align: Optional[List[int]] = field(
        default=None,
        metadata={"help": "Which teacher layers to align (0-35). None = last layer only"}
    )
    
    alignment_temperature: float = field(
        default=0.1,
        metadata={"help": "Temperature for cross-attention alignment"}
    )
    
    alignment_normalize: bool = field(
        default=False,
        metadata={"help": "Whether to L2 normalize hidden states before alignment"}
    )


class AlignedSFTTrainer:
    
    def _setup_teacher_model(self, args):
        teacher_model_init_kwargs = args.teacher_model_init_kwargs or {}
        torch_dtype = teacher_model_init_kwargs.get("torch_dtype")
        if isinstance(torch_dtype, str) and torch_dtype in ["bfloat16", "float16", "float32"]:
            teacher_model_init_kwargs["torch_dtype"] = getattr(torch, torch_dtype)
        
        teacher_device = f"cuda:{args.teacher_gpu}"
        teacher_model_init_kwargs["device_map"] = {"": teacher_device}
        
        self.teacher_model = AutoModelForCausalLM.from_pretrained(
            args.teacher_model_name_or_path,
            **teacher_model_init_kwargs
        )
        self.teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_name_or_path)
        
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        self.teacher_device = teacher_device
        
        if args.student_layers_to_align is None:
            self.student_layers = [-1]
        else:
            self.student_layers = args.student_layers_to_align
            
        if args.teacher_layers_to_align is None:
            self.teacher_layers = [-1]
        else:
            self.teacher_layers = args.teacher_layers_to_align
        
        if len(self.student_layers) != len(self.teacher_layers):
            raise ValueError(
                f"Number of student layers ({len(self.student_layers)}) must match "
                f"number of teacher layers ({len(self.teacher_layers)})"
            )
    
    def compute_cross_attention_alignment(
        self, 
        student_hidden, 
        teacher_hidden, 
        student_mask=None, 
        teacher_mask=None
    ):
        device = student_hidden.device
        
        if teacher_hidden.device != device:
            teacher_hidden = teacher_hidden.to(device)
        
        if teacher_mask is not None and teacher_mask.device != device:
            teacher_mask = teacher_mask.to(device)
        
        if student_mask is not None and student_mask.device != device:
            student_mask = student_mask.to(device)
        
        bs, s_len, dim = student_hidden.shape
        _, t_len, t_dim = teacher_hidden.shape
        
        if dim != t_dim:
            raise ValueError(f"Dimension mismatch: student {dim}, teacher {t_dim}")
        
        if self.args.alignment_normalize:
            student_norm = F.normalize(student_hidden, p=2, dim=-1)
            teacher_norm = F.normalize(teacher_hidden, p=2, dim=-1)
        else:
            student_norm = student_hidden
            teacher_norm = teacher_hidden
        
        similarity = torch.bmm(student_norm, teacher_norm.transpose(1, 2))
        similarity = similarity / self.args.alignment_temperature
        
        if teacher_mask is not None:
            # FIX: Convert to boolean mask
            teacher_mask_bool = teacher_mask.bool()
            teacher_mask_expanded = teacher_mask_bool.unsqueeze(1).expand(bs, s_len, t_len)
            similarity = similarity.masked_fill(~teacher_mask_expanded, -1e9)
        
        attention_weights = F.softmax(similarity, dim=-1)
        aligned_teacher = torch.bmm(attention_weights, teacher_hidden)

        # COSINE
        # student_norm = F.normalize(student_hidden, p=2, dim=-1)
        # aligned_teacher_norm = F.normalize(aligned_teacher, p=2, dim=-1)

        # cosine_sim = (student_norm * aligned_teacher_norm).sum(dim=-1)

        # # print(f"Cosine similarities - mean: {cosine_sim.mean().item():.4f}, "
        # #     f"std: {cosine_sim.std().item():.4f}, "
        # #     f"min: {cosine_sim.min().item():.4f}, "
        # #     f"max: {cosine_sim.max().item():.4f}")

        # if student_mask is not None:
        #     student_mask_float = student_mask.float()
        #     masked_cosine_sim = cosine_sim * student_mask_float
        #     valid_tokens = student_mask_float.sum()
            
        #     loss = 1 - (masked_cosine_sim.sum() / valid_tokens.clamp(min=1))
        # else:
        #     loss = 1 - cosine_sim.mean()

        # MSE
        if student_mask is not None:
            # FIX: Ensure student_mask is float for multiplication
            student_mask_expanded = student_mask.unsqueeze(-1).float()
            valid_tokens = student_mask.sum()
            
            loss = F.mse_loss(
                student_hidden * student_mask_expanded,
                aligned_teacher * student_mask_expanded,
                reduction='sum'
            ) / valid_tokens.clamp(min=1)
        else:
            loss = F.mse_loss(student_hidden, aligned_teacher)
        
        return loss, attention_weights
        
    def extract_hidden_states(self, hidden_states_tuple, layer_indices):
        if not layer_indices:
            return [hidden_states_tuple[-1]]
        
        hidden_list = []
        for idx in layer_indices:
            if idx < 0:
                idx = len(hidden_states_tuple) + idx
            if 0 <= idx < len(hidden_states_tuple):
                hidden_list.append(hidden_states_tuple[idx])
            else:
                raise ValueError(f"Layer index {idx} out of range for {len(hidden_states_tuple)} layers")
        
        return hidden_list
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        sft_loss, outputs = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        
        if self.teacher_model is None:
            return (sft_loss, outputs) if return_outputs else sft_loss
        
        texts = self.processing_class.batch_decode(
            inputs["input_ids"],
            skip_special_tokens=False
        )
        
        teacher_inputs = self.teacher_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.args.max_length,
            return_tensors="pt"
        ).to(self.teacher_device)
        
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                **teacher_inputs,
                output_hidden_states=True
            )
        
        if not hasattr(outputs, 'hidden_states') or outputs.hidden_states is None:
            inputs_for_hidden = {k: v for k, v in inputs.items() 
                               if k in ["input_ids", "attention_mask"]}
            inputs_for_hidden["output_hidden_states"] = True
            student_outputs_with_hidden = model(**inputs_for_hidden)
            student_hidden_all = student_outputs_with_hidden.hidden_states
        else:
            student_hidden_all = outputs.hidden_states
        
        teacher_hidden_all = teacher_outputs.hidden_states
        
        # print(f"Student has {len(student_hidden_all)} layers, Teacher has {len(teacher_hidden_all)} layers")
        
        student_hidden_list = self.extract_hidden_states(student_hidden_all, self.student_layers)
        teacher_hidden_list = self.extract_hidden_states(teacher_hidden_all, self.teacher_layers)
        
        total_alignment_loss = 0
        layer_losses = []
        
        for i, (s_hidden, t_hidden) in enumerate(zip(student_hidden_list, teacher_hidden_list)):
            s_layer_idx = self.student_layers[i] if self.student_layers else -1
            t_layer_idx = self.teacher_layers[i] if self.teacher_layers else -1
            
            # print(f"Aligning student layer {s_layer_idx} with teacher layer {t_layer_idx}")
            # print(f"  Student shape: {s_hidden.shape}, Teacher shape: {t_hidden.shape}")
            
            if self.args.alignment_loss_type == "cross_attention":
                layer_loss, attention_weights = self.compute_cross_attention_alignment(
                    s_hidden,
                    t_hidden,
                    inputs.get("attention_mask"),
                    teacher_inputs.get("attention_mask")
                )
                
                if i == 0 and self.model.training and hasattr(self, 'global_step'):
                    if self.global_step % 100 == 0:
                        avg_attention = attention_weights[0, :10, :20].cpu().detach().numpy()
                        print(f"Sample attention pattern (first 10x20):\n{avg_attention}")
                
            else:
                student_pooled = self.pool_hidden_states(s_hidden, inputs.get("attention_mask"))
                teacher_pooled = self.pool_hidden_states(t_hidden, teacher_inputs.get("attention_mask"))
                
                if teacher_pooled.device != student_pooled.device:
                    teacher_pooled = teacher_pooled.to(student_pooled.device)
                
                if self.args.alignment_loss_type == "pooled_mse":
                    layer_loss = F.mse_loss(student_pooled, teacher_pooled)
                elif self.args.alignment_loss_type == "pooled_cosine":
                    student_norm = F.normalize(student_pooled, p=2, dim=-1)
                    teacher_norm = F.normalize(teacher_pooled, p=2, dim=-1)
                    layer_loss = 1 - (student_norm * teacher_norm).sum(dim=-1).mean()
                else:
                    raise ValueError(f"Unknown alignment loss type: {self.args.alignment_loss_type}")
            
            layer_losses.append(layer_loss)
            total_alignment_loss += layer_loss
        
        alignment_loss = total_alignment_loss / len(layer_losses)
        
        total_loss = sft_loss + self.args.alignment_weight * alignment_loss
        
        #print(f"SFT Loss: {sft_loss.item():.4f}, Alignment Loss: {alignment_loss.item():.4f}, Total Loss: {total_loss.item():.4f}")
        #for i, ll in enumerate(layer_losses):
        #    print(f"  Layer pair {i} loss: {ll.item():.4f}")
        
        if self.model.training:
            mode = "train"
            self._metrics[mode]["alignment_loss"].append(alignment_loss.item())
            self._metrics[mode]["sft_loss"].append(sft_loss.item())
            for i, ll in enumerate(layer_losses):
                self._metrics[mode][f"layer_{i}_loss"].append(ll.item())
        
        return (total_loss, outputs) if return_outputs else total_loss
    
    def pool_hidden_states(self, hidden_states, attention_mask):
        if attention_mask is None:
            return hidden_states.mean(dim=1)
        
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_hidden / sum_mask