import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel


class CLIP_cls(nn.Module):

    def __init__(self, model_name='clip', freeze_text=True):
        super(CLIP_cls, self).__init__()
        self.clip = CLIPModel.from_pretrained(model_name, trust_remote_code=True)
        #self.clip = CLIPModel.from_pretrained(model_name)
        self.proj_dim = self.clip.text_projection.out_features
        
        
        self.head = nn.Sequential(
            nn.Linear(self.proj_dim * 2, 512),
            #nn.Linear(self.proj_dim*2, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 2)
        )
        

        #self.head.apply(self.init_weights_xavier)
        
        #torch.init.xavier_uniform_(self.head.weight)

        # freeze text encoder if needed
        #print(freeze_text)
        
        for p in self.clip.parameters():
            p.requires_grad = True
        """
        if freeze_text:
            for p in self.clip.text_model.parameters():
                p.requires_grad = False
        """
    def init_weights_xavier(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, pixel_values, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.clip(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
            img_f = F.normalize(outputs.image_embeds, dim=-1)
            txt_f = F.normalize(outputs.text_embeds, dim=-1)
        #prod = img_f * txt_f
        """
        prod = img_f * txt_f
        l1 = torch.abs(img_f - txt_f)
        cos = (img_f * txt_f).sum(dim=-1, keepdim=True)
        x = torch.cat([img_f, txt_f, prod, l1, cos], dim=-1)
        """
        x = torch.cat([img_f, txt_f], dim=-1)
        #x = prod
        logits = self.head(x)
        #logits = outputs.logits_per_image, outputs.logits_per_text
        return logits