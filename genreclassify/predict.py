#type: ignore
import numpy as np


import click

import torch
import librosa

from train import model


def create_mfccs(file, n_mfcc=13, hop_length=512, n_fft=2040):
    signal, sr = librosa.load(file)
    if not sr == 22050:
        return "Signal rate is not 44.1 khz"
    if len(signal) < sr * 30:
        return "Wav file is less than 30 seconds"

    signal = signal[: sr * 30]  # take 30 seconds of the .wav file

    # extract mfccs
    MFCCs = librosa.feature.mfcc(
        y=signal, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)

    return MFCCs


@click.command()
@click.option("--file", "-F", help="File path", default="music.wav", required=True)
@click.option("--gpu", is_flag=True, required=False, type=bool, show_default=True)
def prediction(file, gpu):
    if gpu:
        model.load_state_dict(torch.load(
            "music-model.pt", map_location="cuda"))
    else:
        model.load_state_dict(torch.load("music-model.pt", map_location="cpu"))

    mfccs = create_mfccs(file)

    # if the function returns string then leave it as be
    if isinstance(mfccs, str):
        print(mfccs)
    else:
        genres = ["nephop", "gajal", "lok_dohori", "rock"]

        # transpose it
        mfccs_t = mfccs.T

        # reshape the mfccs to [1, seq len, dimension]
        mfccs = mfccs_t[np.newaxis, :, :]

        # change to torch tensor
        mfccs_tensor = torch.from_numpy(mfccs).float()

        # predict
        predict = model(mfccs_tensor).squeeze(0)

        # find maximum prediction
        predict_max = predict.argmax(dim=0).item()

        print(genres[predict_max])


prediction()
