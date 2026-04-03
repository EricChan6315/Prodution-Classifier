import os, sys, json
import torch
import torch.nn as nn
from model import MERT_AES
import soundfile as sf
import torchaudio
import utils as utils
from utils import load_config

SUPPORTED_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".aiff", ".aif"}

def inference(args):
    # ======= enable cudnn benchmarking =======
    torch.backends.cudnn.benchmark = True
    
    # ======= model =======
    model = MERT_AES(
        proj_num_layer=args.model.proj_num_layer,
        proj_ln=args.model.proj_ln,
        proj_act_fn=args.model.proj_act_fn,
        proj_dropout=args.model.proj_dropout,
        output_dim=args.model.output_dim,
        binary_classification=args.model.binary_classification,
        freeze_encoder=args.model.freeze_encoder,
    )

    # ======= load checkpoint =======
    ckpt = torch.load(args.inference.ckpt_path, map_location=args.device)
    model.load_state_dict(ckpt)

    model.to(args.device)
    model.eval()

    # ======= collect audio files =======
    input_path = args.input_path
    if os.path.isdir(input_path):
        audio_files = [
            os.path.join(input_path, f)
            for f in sorted(os.listdir(input_path))
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
        ]
        if not audio_files:
            raise ValueError(f"No supported audio files found in directory: {input_path}")
    elif os.path.isfile(input_path):
        audio_files = [input_path]
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    # ======= run inference =======
    pred_results = {}
    for audio_path in audio_files:
        mixes = load_audio(args, audio_path)
        label = 0.0
        with torch.no_grad():
            for mix in mixes:
                mix = mix.unsqueeze(0)   # (1, T)
                pred_label = model(mix).item()
                print(pred_label)
                label += pred_label / len(mixes)

        pred_results[os.path.basename(audio_path)] = "good" if label > 0.5 else "bad"
        print(f"{os.path.basename(audio_path)}: {pred_results[os.path.basename(audio_path)]} (score={label:.4f})")

    with open(args.output_path, "w") as f:
        json.dump(pred_results, f, indent=2)

        
def load_audio(args, mix_path):
    info = sf.info(mix_path)
    min_length = info.frames
    win_len = int(info.samplerate * args.data.duration_sec)
    hop_length = win_len // 2

    last_chunk_start_frame = min_length - win_len + 1
    mixes = []

    for offset in range(0, last_chunk_start_frame, hop_length):
        mix, sr = torchaudio.load(
            mix_path,
            frame_offset=offset,
            num_frames=win_len
        )
        if sr != args.data.sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=args.data.sr)
            mix = resampler(mix)

        mix = mix.mean(dim=0)  # stereo -> mono: (T,)

        mix = mix.to(args.device)

        mixes.append(mix)

    return mixes


if __name__ == "__main__":
    assert len(sys.argv) == 3, "Usage: python3 ./inference.py <input_path> <output_path>"
    args = load_config("./model_ckpt/model/config.yaml")
    args.inference.ckpt_path = "./model_ckpt/model/best_acc_params.pth"
    args.input_path = sys.argv[1]
    args.output_path = sys.argv[2]
    inference(args)