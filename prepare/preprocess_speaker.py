import os
import sys
# insert the root dir
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
import argparse

from tqdm import tqdm
from argparse import RawTextHelpFormatter
from speaker.models.lstm import LSTMSpeakerEncoder
from speaker.config import SpeakerEncoderConfig

from speaker.utils.audio import AudioProcessor
from speaker.infer import read_json


def get_spk_wavs(dataset_path, output_path):
    wav_files = []
    os.makedirs(f"./{output_path}", exist_ok=True)
    for spks in os.listdir(dataset_path):
        if os.path.isdir(f"./{dataset_path}/{spks}"):
            if os.path.exists(f"./{output_path}/{spks}"):
                print(f">>>>>>>>>>{spks}<<<<<<<<<<")
                continue
            os.makedirs(f"./{output_path}/{spks}")
            for file in os.listdir(f"./{dataset_path}/{spks}"):
                if file.endswith(".wav"):
                    wav_files.append(f"./{dataset_path}/{spks}/{file}")
        elif spks.endswith(".wav"):
            wav_files.append(f"./{dataset_path}/{spks}")
    return wav_files


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="""Compute embedding vectors for each wav file in a dataset.""",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument("dataset_path", type=str, help="Path to dataset waves.")
    parser.add_argument(
        "output_path", type=str, help="path for output speaker/speaker_wavs.npy."
    )
    parser.add_argument("--use_cuda", type=bool, help="flag to set cuda.", default=True)
    args = parser.parse_args()
    dataset_path = args.dataset_path
    output_path = args.output_path
    # model
    args.model_path = os.path.join("speaker_pretrain", "best_model.pth.tar")
    args.config_path = os.path.join("speaker_pretrain", "config.json")
    # config
    config_dict = read_json(args.config_path)

    # model
    config = SpeakerEncoderConfig(config_dict)
    config.from_dict(config_dict)

    speaker_encoder = LSTMSpeakerEncoder(
        config.model_params["input_dim"],
        config.model_params["proj_dim"],
        config.model_params["lstm_dim"],
        config.model_params["num_lstm_layers"],
    )

    speaker_encoder.load_checkpoint(args.model_path, eval=True, use_cuda=args.use_cuda)

    # preprocess
    speaker_encoder_ap = AudioProcessor(**config.audio)
    # normalize the input audio level and trim silences
    speaker_encoder_ap.do_sound_norm = True
    speaker_encoder_ap.do_trim_silence = True

    wav_files = get_spk_wavs(dataset_path, output_path)

    # compute speaker embeddings
    for idx, wav_file in enumerate(tqdm(wav_files)):
        waveform = speaker_encoder_ap.load_wav(
            wav_file, sr=speaker_encoder_ap.sample_rate
        )
        spec = speaker_encoder_ap.melspectrogram(waveform)
        spec = torch.from_numpy(spec.T)
        if args.use_cuda:
            spec = spec.cuda()
        spec = spec.unsqueeze(0)
        embed = speaker_encoder.compute_embedding(spec).detach().cpu().numpy()
        embed = embed.squeeze()
        embed_path = wav_file.replace(dataset_path, output_path)
        embed_path = embed_path.replace(".wav", ".spk")
        np.save(embed_path, embed, allow_pickle=False)
