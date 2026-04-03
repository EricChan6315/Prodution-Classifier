import os
import torch
import json
import soundfile as sf
import random
import glob 
import torchaudio


NORM_TONAL_BALANCE = [-90, -3]
NORM_LOUDNESS = [-80, -3]

def normalize_feature(feature, min_val, max_val, to_tensor=True):
    normalized_feature = (feature - min_val) / (max_val - min_val) * 2 - 1 
    if to_tensor:
        return torch.from_numpy(normalized_feature)
    else:
        return normalized_feature    

class MOSdataset_moises(torch.utils.data.Dataset):
    def __init__(
        self, 
        wav_root,
        duration_sec=10,
        sr=44100,
        normalized_input=False,
        split="train"
    ):
        super().__init__()
        # arguments 
        self.wav_root   = wav_root
        self.duration_sec = duration_sec
        self.sr = sr

        self.mixing_dirs = []
        song_dirs = [d for d in os.listdir(self.wav_root) 
             if os.path.isdir(os.path.join(self.wav_root, d))]
        random.seed(42) # Ensures the "shuffle" is reproducible
        random.shuffle(song_dirs)
        for song_dir in song_dirs:
            song_dir = os.path.join(self.wav_root, song_dir)
            self.mixing_dirs.append(song_dir)
        if split == "train":
            self.mixing_dirs = self.mixing_dirs[:int(0.8*len(self.mixing_dirs))]
        if split == "valid":
            self.mixing_dirs = self.mixing_dirs[int(0.8*len(self.mixing_dirs)):int(0.9*len(self.mixing_dirs))]
        if split == "test":
            self.mixing_dirs = self.mixing_dirs[int(0.9*len(self.mixing_dirs)):]

        self.inputs = {}
        self.chunks = []

        # conds = []
        for idx, song_dir in enumerate(self.mixing_dirs):
            stem_path = os.path.join(song_dir, "stem.wav")
            submix_path = os.path.join(song_dir, "submix.wav")
            raw_path = os.path.join(song_dir, "raw.wav")
            self.inputs[idx] = {
                'stem': stem_path,
                'submix': submix_path,
                'raw': raw_path,
            }
            assert sf.info(stem_path).frames == sf.info(submix_path).frames and sf.info(submix_path).frames == sf.info(raw_path).frames
            min_length = sf.info(stem_path).frames
            win_len = sf.info(stem_path).samplerate*self.duration_sec
            hop_length = win_len // 2
            last_chunk_start_frame = min_length - win_len  + 1
            for offset in range(0, last_chunk_start_frame, hop_length):
                self.chunks.append((idx, offset))

        random.seed(1223)
        random.shuffle(self.chunks)

    def __getitem__(self, idx):
        wav_id, offset = self.chunks[idx]
        stem_path = self.inputs[wav_id]['stem']
        win_len = sf.info(stem_path).samplerate*self.duration_sec

        stem, sr = torchaudio.load(
            self.inputs[wav_id]['stem'],
            frame_offset=offset,
            num_frames=win_len
        )
        submix, sr = torchaudio.load(
            self.inputs[wav_id]['submix'],
            frame_offset=offset,
            num_frames=win_len
        )
        raw, sr = torchaudio.load(
            self.inputs[wav_id]['raw'],
            frame_offset=offset,
            num_frames=win_len
        )

        
        if sr != self.sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sr)
            stem = resampler(stem)
            submix = resampler(submix)
            raw= resampler(raw)

        good_mix = stem + submix
        bad_mix = raw + submix

        good_mix = good_mix.mean(dim=0)
        bad_mix = bad_mix.mean(dim=0)

        return good_mix, bad_mix
        
    def __len__(self):
        return len(self.chunks)