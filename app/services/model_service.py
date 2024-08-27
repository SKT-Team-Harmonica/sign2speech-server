import yaml
import os
import glob
import cv2
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset
from collections import Counter, defaultdict
from typing import Callable, List, NoReturn, Optional, Tuple, Dict, Union
from app.models.sign_model import SignModel, load_model
from app.services.transformer import Resize, ToTensorGen  # 필요한 변환 클래스를 임포트
from itertools import groupby
import tensorflow as tf

# 프로젝트 최상위 경로를 동적으로 설정
app_root = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(app_root))


# 설정 파일 로드 함수
def load_config(config_path: str):
    with open(config_path, 'r') as file:
        cfg = yaml.safe_load(file)

    # DATA_ROOT를 프로젝트의 최상위 경로로 설정
    cfg['DATASET']['DATA_ROOT'] = os.path.join(project_root, cfg['DATASET']['DATA_ROOT'])

    return cfg


# 텍스트 토큰화 함수
def tokenize_text(text: str) -> List[str]:
    return text.split()


# Placeholder for apply_transform_gens
def apply_transform_gens(transforms, frames):
    transformed_frames = []
    for frame in frames:
        for transform in transforms:
            frame = transform(frame)
        transformed_frames.append(frame)
    return np.array(transformed_frames), None


# Vocabulary 클래스
class Vocabulary:
    def __init__(self) -> NoReturn:
        self.specials = []
        self.itos = []
        self.stoi = None
        self.DEFAULT_UNK_ID = None

    def _from_list(self, tokens: Optional[List[str]] = None):
        self.add_tokens(tokens=self.specials + tokens)
        assert len(self.stoi) == len(self.itos)

    def __str__(self) -> str:
        return self.stoi.__str__()

    def add_tokens(self, tokens: List[str]) -> NoReturn:
        for t in tokens:
            new_index = len(self.itos)
            if t not in self.itos:
                self.itos.append(t)
                self.stoi[t] = new_index

    def is_unk(self, token: str) -> bool:
        return self.stoi[token] == self.DEFAULT_UNK_ID()

    def __len__(self) -> int:
        return len(self.itos)


class GlossVocabulary(Vocabulary):
    def __init__(self, tokens: Optional[List[str]] = None, file: Optional[str] = None) -> NoReturn:
        super().__init__()
        self.specials = ["<si>", "<unk>", "<pad>"]
        self.DEFAULT_UNK_ID = lambda: 1
        self.stoi = defaultdict(self.DEFAULT_UNK_ID)
        self.pad_token = "<pad>"
        self.sil_token = "<si>"

        if tokens is not None:
            self._from_list(tokens)

        assert self.stoi[self.sil_token] == 0

    def arrays_to_sentences(self, arrays: np.array) -> List[List[str]]:
        gloss_sequences = []
        for array in arrays:
            sequence = []
            for i in array:
                sequence.append(self.itos[i])
            gloss_sequences.append(sequence)
        return gloss_sequences


def filter_min(counter: Counter, minimum_freq: int):
    filtered_counter = Counter({t: c for t, c in counter.items() if c >= minimum_freq})
    return filtered_counter


def sort_and_cut(counter: Counter, limit: int):
    tokens_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    tokens_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)
    vocab_tokens = [i[0] for i in tokens_and_frequencies[:limit]]
    return vocab_tokens


def build_vocab(cfg, dataset: Dataset, max_size: int, *, min_freq: int = 1) -> Vocabulary:
    exclude_token = cfg['DATASET']['VOCABULARY']['EXCLUDE_TOKENS']

    tokens = []
    for example in dataset.examples:
        anns = example["Kor"]
        tokens.extend(anns)

    counter = Counter(tokens)
    if min_freq > -1:
        counter = filter_min(counter, min_freq)
    vocab_tokens = sort_and_cut(counter, max_size)
    assert len(vocab_tokens) <= max_size

    vocab = GlossVocabulary(tokens=vocab_tokens)

    assert len(vocab) <= max_size + len(vocab.specials)
    assert vocab.itos[vocab.DEFAULT_UNK_ID()] == "<unk>"

    for i, s in enumerate(vocab.specials):
        if i != vocab.DEFAULT_UNK_ID():
            assert not vocab.is_unk(s)

    return vocab


# SignDataset 클래스
class SignDataset(Dataset):
    def __init__(
            self,
            data_root: str,
            ann_file: str,
            *,
            img_prefix: Optional[str] = None,
            tfm_gens: Optional[list] = None,
            tokenize: Optional[Callable] = None,
            lower: bool = False,
            is_train=False,
            exclude_token=None
    ) -> NoReturn:
        ann_file = os.path.join(data_root, ann_file)

        self.tfm_gens = tfm_gens
        self.tokenize = tokenize
        self.lower = lower
        self.exclude_token = exclude_token
        self.img_prefix = os.path.join(data_root, img_prefix)
        self.examples = self.load_examples_from_csv(ann_file)
        self.is_train = is_train

    def __getitem__(self, i):
        assert hasattr(self, "vocab")
        example = self.examples[i]
        frames_path = example["frames"]

        frames_inds = np.array([i for i in range(len(frames_path))]).astype(np.int)
        if self.is_train:
            rand_inds = np.random.choice(len(frames_path), int(len(frames_path) * 0.2), replace=False)
            total_inds = np.concatenate([frames_inds, rand_inds], 0)
            total_inds = np.sort(total_inds)
            rand_inds = np.random.choice(len(total_inds), int(len(total_inds) * 0.2), replace=False)
            selected = np.delete(total_inds, rand_inds)
        else:
            selected = frames_inds

        try:
            frames = np.stack([cv2.imread(frames_path[i], cv2.IMREAD_COLOR) for i in selected], axis=0)
        except ValueError:
            print(example)

        if self.tfm_gens is not None:
            frames, _ = apply_transform_gens(self.tfm_gens, frames)

        tokens = example["Kor"]
        indices = [self.vocab.stoi[token] for token in tokens]
        return frames, indices

    def __len__(self):
        return len(self.examples)

    def load_examples_from_csv(self, ann_file: str) -> List[dict]:
        annotations = pd.read_csv(ann_file, sep=",", encoding='utf-8')
        annotations = annotations[["Filename", "Kor"]]

        examples = []
        for i in range(len(annotations)):
            example = dict(annotations.iloc[i])

            # MP4 파일명을 JPG 파일명으로 변환하여 사용
            base_filename = os.path.splitext(example["Filename"])[0]  # 확장자를 제거
            frames_path = glob.glob(os.path.join(self.img_prefix, base_filename, "*.jpg"))  # jpg 파일 경로 검색
            frames_path.sort()  # 프레임 정렬
            example["frames"] = frames_path

            glosses_str = example["Kor"]
            if self.tokenize is not None and isinstance(glosses_str, str):
                if self.lower:
                    glosses_str = glosses_str.lower()
                tokens = self.tokenize(glosses_str.rstrip("\n"))
                example["Kor"] = tokens

            examples.append(example)

        return examples

    @property
    def gloss(self):
        return [example["Kor"] for example in self.examples]

    def load_vocab(self, vocabulary):
        self.vocab = vocabulary
        self.pad_idx = self.vocab.stoi[self.vocab.pad_token]
        self.sil_idx = self.vocab.stoi[self.vocab.sil_token]

    def collate(self, data):
        videos, glosses = list(zip(*data))

        def pad(videos: List[Tensor], glosses: List[int]) -> Tuple[
            Tuple[List[Tensor], List[int]], Tuple[List[int], List[int]]]:
            video_lengths = [len(v) for v in videos]
            max_video_len = max(video_lengths)
            padded_videos = []
            for video, length in zip(videos, video_lengths):
                C, H, W = video.size(1), video.size(2), video.size(3)
                new_tensor = video.new(max_video_len, C, H, W).fill_(1e-8)
                new_tensor[:length] = video
                padded_videos.append(new_tensor)

            gloss_lengths = [len(s) for s in glosses]
            max_len = max(gloss_lengths)
            glosses = [
                s + [self.pad_idx] * (max_len - len(s)) if len(s) < max_len else s for s in glosses
            ]
            return (padded_videos, video_lengths), (glosses, gloss_lengths)

        (videos, video_lengths), (glosses, gloss_lengths) = pad(videos, glosses)
        videos = torch.stack(videos, dim=0)
        video_lengths = Tensor(video_lengths).long()
        glosses = Tensor(glosses).long()
        gloss_lengths = Tensor(gloss_lengths).long()
        return (videos, video_lengths), (glosses, gloss_lengths)

# 설정 파일 로드
config_path = os.path.join(app_root, 'cfg.yml')
cfg = load_config(config_path)

# 학습 시 사용한 dataset 객체 생성
dataset = SignDataset(
    data_root=cfg['DATASET']['DATA_ROOT'],
    ann_file=cfg['DATASET']['TRAIN']['ANN_FILE'],
    img_prefix=cfg['DATASET']['TRAIN']['IMG_PREFIX'],
    tfm_gens=cfg['DATASET']['TRANSFORM'],
    tokenize=tokenize_text,
    lower=False,
    is_train=False,
    exclude_token=cfg['DATASET']['VOCABULARY'].get('EXCLUDE_TOKENS', None)
)

# Vocabulary 생성
vocab = build_vocab(cfg, dataset, max_size=10000)

# 모델 로드 경로 설정
model_path = os.path.join(project_root, 'app', 'models', 'model_best_v2.pth.tar')

# 모델 로드
model = load_model(model_path, SignModel, vocab)

# 인퍼런스 함수
def predict(frame_files: List[str]) -> str:
    print(app_root)
    print(project_root)
    if model is None:
        raise ValueError("Model is not loaded.")

    to_tensor_transform = ToTensorGen(normalizer={"mean": cfg['DATASET']['TRANSFORM']['MEAN'],
                                                  "std": cfg['DATASET']['TRANSFORM']['STD']})

    frames_tensor = []

    for frame_file in frame_files:
        if not os.path.exists(frame_file):
            print(f"File does not exist: {frame_file}")
            continue

        try:
            # OpenCV로 이미지 로드
            np_image = cv2.imread(frame_file)
            if np_image is None:
                raise ValueError(f"Error loading image: {frame_file}")

            # BGR에서 RGB로 변환
            np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)

            # 이미지를 ToTensorGen을 사용하여 텐서로 변환
            tensor = to_tensor_transform.get_transform(np_image).apply_image(np_image)

            frames_tensor.append(tensor)
        except Exception as e:
            print(f"Error loading frame {frame_file}: {e}")
            continue

    if len(frames_tensor) == 0:
        raise ValueError("No valid frames to process.")

    try:
        frames_tensor = torch.stack(frames_tensor)
        print(f"Stacked frames tensor: shape={frames_tensor.shape}, dtype={frames_tensor.dtype}")
    except Exception as e:
        print(f"Error stacking frames: {e}")
        raise ValueError("Failed to stack frames into a tensor.")

    frames_tensor = frames_tensor.unsqueeze(0)  # 배치 차원을 추가

    try:
        with torch.no_grad():
            # 모델 예측 수행
            outputs = model(frames_tensor)

            # CTC 디코딩을 위한 전처리 (softmax 또는 log_softmax와 비슷한 처리)
            gloss_probs = outputs.log_softmax(2).permute(1, 0, 2)  # (T, B, C)
            gloss_probs = gloss_probs.detach().numpy()  # (T, B, C)
            gloss_probs_tf = np.concatenate(
                # (C: 1~)
                (gloss_probs[:, :, 1:], gloss_probs[:, :, 0, None]),
                axis=-1,
            )

            sequence_length = np.array([frames_tensor.shape[1]]) // 4
            ctc_decode, _ = tf.nn.ctc_beam_search_decoder(
                inputs=gloss_probs_tf,
                sequence_length=sequence_length,
                beam_width=1,
                top_paths=1,
            )
            ctc_decode = ctc_decode[0]

            # 디코딩된 결과 생성
            decoded_gloss_sequences = []
            tmp_gloss_sequences = [[] for _ in range(outputs.shape[0])]
            for (value_idx, dense_idx) in enumerate(ctc_decode.indices):
                tmp_gloss_sequences[dense_idx[0]].append(ctc_decode.values[value_idx].numpy() + 1)
            for seq_idx in range(0, len(tmp_gloss_sequences)):
                decoded_gloss_sequences.append(
                    [x[0] for x in groupby(tmp_gloss_sequences[seq_idx])]
                )

            decoded = vocab.arrays_to_sentences(arrays=decoded_gloss_sequences)

            if len(decoded) > 0:
                return " ".join(decoded[0])
            else:
                return "No prediction available"
    except Exception as e:
        print(f"Error during model prediction: {e}")
        raise ValueError("Model prediction failed.")
