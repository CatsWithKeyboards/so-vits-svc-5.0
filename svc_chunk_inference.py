import os
import torch
import librosa
import argparse
import numpy as np
import torchcrepe

from omegaconf import OmegaConf
from scipy.io.wavfile import write
from vits.models import SynthesizerInfer


def load_svc_model(checkpoint_path, model):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    saved_state_dict = checkpoint_dict["model_g"]
    state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k] = saved_state_dict[k]
    model.load_state_dict(new_state_dict)
    return model


def compute_f0_nn(filename, device):
    audio, sr = librosa.load(filename, sr=16000)
    assert sr == 16000
    # Load audio
    audio = torch.tensor(np.copy(audio))[None]
    # Here we'll use a 20 millisecond hop length
    hop_length = 320
    # Provide a sensible frequency range for your domain (upper limit is 2006 Hz)
    # This would be a reasonable range for speech
    fmin = 50
    fmax = 1000
    # Select a model capacity--one of "tiny" or "full"
    model = "full"
    # Pick a batch size that doesn't cause memory errors on your gpu
    batch_size = 512
    # Compute pitch using first gpu
    pitch, periodicity = torchcrepe.predict(
        audio,
        sr,
        hop_length,
        fmin,
        fmax,
        model,
        batch_size=batch_size,
        device=device,
        return_periodicity=True,
    )
    pitch = np.repeat(pitch, 2, -1)  # 320 -> 160 * 2
    periodicity = np.repeat(periodicity, 2, -1)  # 320 -> 160 * 2
    # CREPE was not trained on silent audio. some error on silent need filter.
    periodicity = torchcrepe.filter.median(periodicity, 9)
    pitch = torchcrepe.filter.mean(pitch, 9)
    pitch[periodicity < 0.1] = 0
    pitch = pitch.squeeze(0)
    return pitch

def process_chunk(model, spk, ppg, pit, len_chunk, device):
    with torch.no_grad():
        spk = spk.unsqueeze(0).to(device)
        ppg = ppg.unsqueeze(0).to(device)
        pit = pit.unsqueeze(0).to(device)
        len_chunk = torch.LongTensor([len_chunk]).to(device)
        audio = model(ppg, pit, spk, len_chunk)
        audio = audio[0, 0].data.cpu().detach().numpy()
    return audio

def main(args):
    if (args.ppg == None):
        args.ppg = "svc_tmp.ppg.npy"
        print(
            f"Auto run : python whisper/inference.py -w {args.wave} -p {args.ppg}")
        os.system(f"python whisper/inference.py -w {args.wave} -p {args.ppg}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hp = OmegaConf.load(args.config)
    model = SynthesizerInfer(
        hp.data.filter_length // 2 + 1,
        hp.data.segment_size // hp.data.hop_length,
        hp)
    load_svc_model(args.model, model)
    model.eval()
    model.to(device)

    spk = np.load(args.spk)
    spk = torch.FloatTensor(spk)

    ppg = np.load(args.ppg)
    ppg = np.repeat(ppg, 2, 0)  # 320 PPG -> 160 * 2
    ppg = torch.FloatTensor(ppg)

    pit = compute_f0_nn(args.wave, device)

    if (args.statics == None):
        print("don't use pitch shift")
    else:
        source = pit[pit > 0]
        source_ave = source.mean()
        source_min = source.min()
        source_max = source.max()
        print(f"source pitch statics: mean={source_ave:0.1f}, \
                min={source_min:0.1f}, max={source_max:0.1f}")
        singer_ave, singer_min, singer_max = np.load(args.statics)
        print(f"singer pitch statics: mean={singer_ave:0.1f}, \
                min={singer_min:0.1f}, max={singer_max:0.1f}")

        shift = np.log2(singer_ave/source_ave) * 12
        if (singer_ave >= source_ave):
            shift = np.floor(shift)
        else:
            shift = np.ceil(shift)
        shift = 2 ** (shift / 12)
        pit = pit * shift

    pit = torch.FloatTensor(pit)

    len_pit = pit.size()[0]
    len_ppg = ppg.size()[0]
    len_min = min(len_pit, len_ppg)
    pit = pit[:len_min]
    ppg = ppg[:len_min, :]

    seconds_per_chunk = 10
    samples_per_second = hp.data.sampling_rate // hp.data.hop_length
    chunk_size = seconds_per_chunk * samples_per_second
    num_chunks = int(np.ceil(len_ppg / chunk_size))

    print(f"Chunk size: {chunk_size}")
    print(f"Number of chunks: {num_chunks}")

    audio_out = np.zeros((0,))

    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len_ppg)

        ppg_chunk = ppg[start:end, :]
        pit_chunk = pit[start:end]

        len_chunk = end - start
        print(f"Processing chunk {i + 1}/{num_chunks}, size: {len_chunk}")

        audio_chunk = process_chunk(model, spk, ppg_chunk, pit_chunk, len_chunk, device)
        audio_out = np.concatenate((audio_out, audio_chunk))

    # Ensure the output audio has the same number of samples as the input
    audio_out = audio_out[:len_min * hp.data.hop_length]

    # Convert audio data to int16 format
    audio_out = (audio_out * 32767).astype(np.int16)

    print(f"Output audio length: {len(audio_out)}")
    print(f"Expected output length: {len_min * hp.data.hop_length}")

    write(f"svc_out_{args.spk.split('/')[-1].replace('.npy', '')}_{args.wave}", hp.data.sampling_rate, audio_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for config.")
    parser.add_argument('-m', '--model', type=str, required=True,
                        help="path of model for evaluation")
    parser.add_argument('-w', '--wave', type=str, required=True,
                        help="Path of raw audio.")
    parser.add_argument('-s', '--spk', type=str, required=True,
                        help="Path of speaker.")
    parser.add_argument('-p', '--ppg', type=str,
                        help="Path of content vector.")
    parser.add_argument('-t', '--statics', type=str,
                        help="Path of pitch statics.")
    args = parser.parse_args()

    main(args)
