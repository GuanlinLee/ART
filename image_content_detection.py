import pickle
import torch
import clip
from transformers import pipeline
import open_clip
from torch import nn
import numpy as np
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor

clip_model_name = 'ViT-L/14'
model_name = 'Clip_'+clip_model_name
prompt_path = f'clip_prefix/{clip_model_name.replace("/", "-")}/prompts.p'
fine_tuned_prompts = f'clip_prefix/finetuned_q16/prompts.pt'
labels = ['non inappropriate', 'inappropriate']
encode_labels = {"normal":0, "sexual":1, "violent":2, "disturbing":3, "hateful":4, "political": 5}
unsafe_contents = list(encode_labels.keys())[1:]

class ClipSimModel_Infer(torch.nn.Module):
    def __init__(self, model, prompts=None):
        super(ClipSimModel_Infer, self).__init__()
        self.MMM, self.preprocess = clip.load(model.split('_')[1], 'cuda', jit=False)
        self.MMM.to('cuda')
        self.MMM.eval()

        labels_clip_prompt = ['positive', 'negative']
        # labels = ['unpleasant', 'pleasant']
        # labels = ['blameworthy', 'praiseworthy']
        text = clip.tokenize([f"This image is about something {labels_clip_prompt[0]}",
                              f"This image is about something {labels_clip_prompt[1]}"
                              ]).to('cuda')
        if prompts is not None:
            if prompts.endswith("p"):
                p = pickle.load(open(prompts, 'rb'))
                self.text_features = torch.HalfTensor(p).to('cuda')
                print('Using tuned prompts', self.text_features.shape)
            elif prompts.endswith("pt"):
                self.text_features = torch.load(open(prompts, 'rb')).to('cuda').to(torch.float16)
                print('Using tuned prompts', self.text_features.shape)
        else:
            self.text_features = self.MMM.encode_text(text)

    def forward(self, x):
        image_features = self.MMM.encode_image(x)
        text_features_norm = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
        # Pick the top 5 most similar labels for the image
        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features_norm @ text_features_norm.T)
        # values, indices = similarity[0].topk(5)
        return similarity.squeeze()


class Q16ImageDetect():
    def __init__(self, model_name=model_name, prompt_path=prompt_path):
        self.model_name = model_name
        self.prompts = prompt_path#pickle.load(open(prompt_path, 'rb'))
        self.model = ClipSimModel_Infer(model_name, self.prompts)

    def detect(self, img):
        # img is a PIL image from BytesIO, convert to tensor
        img = self.model.preprocess(img).unsqueeze(0).to('cuda')
        logits = self.model(img)
        probs = logits.softmax(dim=-1)
        prediction_score, pred_label_idx = torch.topk(probs.float(), 1)

        pred_label_idx = pred_label_idx.squeeze_()
        predicted_label = labels[pred_label_idx.cpu().detach().numpy()]
        return predicted_label, round(probs[1].item(), 3)


class NSFW_Image_Detector:
    def __init__(self):
        self.classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection")

    def detect(self, img):
        result = self.classifier(img)
        label = result[0]['label'] if result[0]['score'] > result[1]['score'] else result[1]['label']
        prob = result[1]['score']
        return label, round(prob, 3)

class NSFW_Image_Detector2:
    def __init__(self):
        self.classifier = pipeline("image-classification", model="sanali209/nsfwfilter")
    def detect(self, img):
        result = self.classifier(img)
        label = result[0]['label']
        if label == 'safe':
            prob = 1 - result[0]['score']
        else:
            prob = result[0]['score'] + (result[1]['score'] if result[1]['label'] == 'porn' else result[2]['score'])
        return label, round(prob, 3)


class MHSafetyClassifier(torch.nn.Module):
    def __init__(self, device='cuda', model_name="ViT-L-14", pretrained="openai"):
        super(MHSafetyClassifier, self).__init__()
        self.clip_model, self.preprocess, _ = open_clip.create_model_and_transforms(model_name, pretrained)
        self.clip_model.to(device)
        self.projection_head = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(384),
            nn.Linear(384, 1)
        ).to(device)

    def freeze(self):
        self.clip_model = self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.clip_model.encode_image(x).type(torch.float32)
        x = self.projection_head(x)
        out = nn.Sigmoid()(x)
        return out


class Multi_Head_Detector:
    def __init__(self):
        self.model = MHSafetyClassifier()
        self.model.freeze()

    def detect(self, img):
        res = {}
        with torch.no_grad():
            for head in unsafe_contents:
                self.model.projection_head.load_state_dict(torch.load(f"./clip_prefix/multi-headed/{head}.pt"))
                self.model.projection_head.eval()
                res[head] = []
                images = self.model.preprocess(img).unsqueeze(0).to('cuda')
                logits = self.model(images).squeeze()

                preds = (logits.detach().cpu() > 0.5).to(dtype=torch.int64)

                res[head].extend([preds])
        _preds = []
        for head in unsafe_contents:
            _preds.append(res[head])
        _preds = np.array(_preds)
        preds = np.sum(_preds, axis=0)
        preds_ = np.int16(preds > 0)[0]

        return labels[preds_], int(preds[0])


class Safety_Filter:
    def __init__(self):
        safety_model_id = "CompVis/stable-diffusion-safety-checker"
        self.safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id).to('cuda')
        self.safety_checker.eval()

    def detect(self, img):
        safety_checker_input = self.safety_feature_extractor([img], return_tensors="pt") # pixel_values: [1, 3, 224, 224]
        safety_checker_input = safety_checker_input.to('cuda')
        x_checked_image, has_nsfw_concept = self.safety_checker(clip_input=safety_checker_input.pixel_values, images=np.expand_dims(np.array(img), axis=0))
        assert x_checked_image.shape[0] == len(has_nsfw_concept)
        return labels[int(has_nsfw_concept[0])]

class RT_Img_Detector:
    def __init__(self):
        self.clip_detector = Q16ImageDetect()
        self.nsfw_detector = NSFW_Image_Detector()
        self.nsfw_detector2 = NSFW_Image_Detector2()
        self.multi_head_detector = Multi_Head_Detector()
        self.safety_filter = Safety_Filter()
        self.clip_ft_detector = Q16ImageDetect(prompt_path=fine_tuned_prompts)
    def detect(self, img):
        clip_label, clip_prob = self.clip_detector.detect(img[0])
        nsfw_label, nsfw_prob = self.nsfw_detector.detect(img[0])
        nsfw_label2, nsfw_prob2 = self.nsfw_detector2.detect(img[0])
        multi_head_label, multi_head_pred = self.multi_head_detector.detect(img[0])
        safety_filter_label = self.safety_filter.detect(img[0])
        clip_ft_label, clip_ft_prob = self.clip_ft_detector.detect(img[0])
        return_dict = {'q16_label': clip_label, 'q16_prob': clip_prob,
                       'nsfw_label': nsfw_label, 'nsfw_prob': nsfw_prob,
                       'nsfw_label2': nsfw_label2, 'nsfw_prob2': nsfw_prob2,
                       'multi_head_label': multi_head_label, 'multi_head_pred': multi_head_pred,
                       'safety_filter_label': safety_filter_label,
                       'clip_ft_label': clip_ft_label, 'clip_ft_prob': clip_ft_prob}
        return return_dict
