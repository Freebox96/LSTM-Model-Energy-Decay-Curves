import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import pyroomacoustics as pra
from scipy import stats
import acoustics as ac
import os
import pandas as pd
import csv
from scipy.io import wavfile

class RoomAcousticParameters:
    def __init__(self, filename):
        self.filename = filename
        self.fs, self.rir, self.idx_start = self._readwav(filename)
        #self.rir = np.expand_dims(self.rir, axis=0)

    # Reading the room impulse response
    def _readwav(self, Filename):
        '''
        Diese Funktion liest eine wav-Datei ein. Falls es sich um eine Stereodatei handelt, wird diese in eine Monodatei
        umgewandelt.
        :param Filename: Dateiname (string)
        :return: Fs -> Abtastrate der wav-Datei
        :return: Data -> Daten aus der wav-Datei (Array)
        :return: IdxStart -> Index (X-Wert Mal Samplerate) des Beginns der Impulsantwort (hier der erste Wert > 0.02)
        '''
        Fs, Data = wavfile.read(Filename)
        try:
            if Data.shape[1] == 2:                              # if Stereo
                Data = Data[:, 0]                               # Convert in Mono
        except:
            pass

        Data = np.float64(Data)
        Data = Data / max(np.abs(Data))                         # Normalization to amplitude = 1

        Y = np.abs(Data)

        IdxStart = 0
        for i in range(len(Y)):                                 # Determine the first value Y > 0.02 -> start of the impulse response
            if Y[i] > 0.02:
                IdxStart = i
                break

        return Fs, Data, IdxStart

    def EDC(self, normalize: bool = True) -> np.ndarray:
        """
        Energy Decay Curve - backwards integration, normalized to [0..1]

        Parameters
        ----------
        audio : np.ndarray
            audio array with shape (channels, samples)
        normalize : bool, default: `True`
            normalize to [0..1]

        Returns
        -------
        np.ndarray
            cumulated energy (backwards integration)
        """
        energy = np.atleast_2d(self.rir**2)
        reverted = np.cumsum(energy[..., ::-1], axis=-1)
        result = reverted[:, ::-1]
        if normalize and np.all(result[:, 0] != 0.0):
            result = (result.T / result[:, 0]).T.flatten()
        return result

    def EDT(self) -> np.ndarray:
        """
        Early Decay Time - when is the energy dropping by 10dB (returning seconds)

        Parameters
        ----------
        normalized_EDC : np.ndarray
            array with result from EDC function
        fs : int, default: `TCSettings.fs`
            samplerate

        Returns
        -------
        np.ndarray
            times for each channel when a drop by 10dB is reached
        """

        edc_db = 10*np.log10(self.EDC() + 1e-30) # power to db
        times = np.argmin(edc_db > -10, axis=-1)
        return times / self.fs

    def RT60_direct(self) -> np.ndarray:
        """
        Reverberation Time RT60 - when the energy of an impulse response dropped
        by 60dB. The drop is examined after the first drop of 5dB. The method uses
        the values of an energy decay curve directly without truncation at the end.

        ---------
        Parameters
        Normalized_EDC : np.ndarray array with result from EDC function with shape (channels/bands, samples)
        fs : int, default : 44100

        Returns
        -------
        np.ndarray
            RT60 times for each channel/band
        """
        thr_begin = 10 ** (-5 / 10) # db to power
        thr_end = 10 ** (-65 / 10) # db to power
        # thr_begin = conversions.db_to_power(-5)
        # thr_end = conversions.db_to_power(-65)

        normalized_EDC = self.EDC()

        begin = np.argmax(normalized_EDC < thr_begin, axis=-1)
        #print(begin)
        end = np.argmin(normalized_EDC > thr_end, axis=-1)
        #print(end)
            # Ensure 'end' is an array and adjust invalid values
        end = np.array(end)  # Convert to array if scalar
        if np.isscalar(end):
            if end == 0:
                end = normalized_EDC.shape[-1] - 1
        else:
            end[end == 0] = normalized_EDC.shape[-1] - 1

        # Ensure begin < end
        valid = begin < end
        times = np.where(valid, end - begin, 0)

        times = end - begin
        return times / self.fs
    
    def plot_edc(self, edc):
        # edc = pra.experimental.parameters.energy_decay_curve(self.rir)
        t = np.arange(edc.size) / self.fs
        plt.plot(t, edc)
        plt.xlabel('Time [s]')
        plt.ylabel('Energy')
        plt.title('Energy Decay Curve')
        plt.show()

        # Schröder Plot and Reverberation Time
    def schroederrt(self, LevelHigh=-5, LevelLow=-25, plot=True):
        '''
        Diese Funktion berechnet den Schröder-Plot aus der gegebenen Impulsantwort. Anschließend wird eine Gerade auf dem
        Plot gesucht (manuell). Diese Gerade wird auf 60dB extrapoliert, sodass daraus die Nachhallzeit TN bestimmt werden
        kann.
        :param Fs: Abtastrate (Integer)
        :param Input: Audiosignal (Array oder Liste)
        :param Filename: Dateiname (String)
        :return: None
        '''

        Energy = []

        InputRev = list(reversed(self.rir))

        for i in range(len(InputRev)):                  # Calculation of backward integrated energy values
            e = np.square(InputRev[i]) / self.fs
            if i > 0:                                   # Calculation of backward integrated energy values
                e = e + Energy[i - 1]                   # added, all others yes
            Energy.append(e)

        Energy = list(reversed(Energy))

        Energy = 10 * np.log10(Energy / Energy[0]+1e-30)

        LevelDiff = np.abs(LevelLow - LevelHigh)

        Levels = [LevelHigh, LevelLow]

        for i in range(len(Levels)):
            for e in range(len(Energy)):                # Iteration over all energy values ​​until first occurrence of
                if Energy[e] < Levels[i]:               # LevelHigh/LevelLow is reached
                    Levels[i] = e / self.fs                  # save the time value at this point
                    break

        TimeHigh = Levels[0]                            # Storing the determined values ​​in individual variables
        TimeLow = Levels[1]

        Factor = np.abs(60 / (LevelLow - LevelHigh))    # Calculation of the factor by which the time value is multiplied
                                                        # must be extrapolated to 60 dB
        TN = (TimeLow - TimeHigh) * Factor              # Calculating the reverberation time

        print(
            "Die frequenzunabhägnige Nachhallzeit RT{} beträgt {:.2f} Sekunden (extrapoliert auf 60 dB).".format(LevelDiff,
                                                                                                                TN))
        # Correlation factor:
        YPlot = Energy[int(TimeHigh * self.fs):int(TimeLow * self.fs)]
        YSecant = np.linspace(LevelLow, LevelHigh, len(YPlot))
        YSecant = np.flip(YSecant)                 # Turn over so that both lists run from small to large time value

        EPlot = np.sum(YPlot ** 2)
        ESecant = np.sum(YSecant ** 2)

        KorrFactor = np.sum(YPlot * YSecant) / np.sqrt(EPlot * ESecant)
        print("Der Korrelationsfaktor beträgt r = {:.3f}".format(KorrFactor))
        if KorrFactor < np.abs(0.995):
            print("Der Korrelationsfaktor sollte mindestens r = 0.995 betragen.")

        if plot == True:
            # Plotausgabe:
            fig, ax = plt.subplots()
            x = np.linspace(0, len(Energy) / self.fs, len(Energy))
            ax.set_title("Schröder Plot " + str(self.filename))

            ax.plot([TN, TN], [-65, 5], label="Nachhallzei TN", color="royalblue",
                    linestyle="dotted")                                                     # Vertical line at TN
            ax.annotate("TN={:.2f}s".format(TN), xy=(TN + 0.1, -60), color="royalblue")     # Label TN

            # Schröder plot:
            ax.plot(x, Energy, label="Schröder Plot", linewidth=2.5, color="salmon")

            # Secant between selected level limits
            ax.plot([TimeHigh, TimeLow], [LevelHigh, LevelLow],
                    label="Sekante {}dB bis {}dB, k={:.3f}".format(LevelHigh, LevelLow, KorrFactor),
                    color="dodgerblue", marker="o")

            # Extrapolation auf 60dB:
            ax.plot([0, TN], [0, -60],
                    label="Extrapolation der Sekante auf 0dB bis -60dB", color="lightskyblue", marker="o")

            ax.set_ylim(-63, 3)
            ax.grid()
            ax.legend()
            ax.set_xlabel("Zeit t [s]")
            ax.set_ylabel("Pegel L [dB]")
            #plt.savefig("Schröder Plot")
            plt.show()
        
        return TN, KorrFactor


    def calrityindex(self, CXX):
        '''
        Diese Funktion berechnet das Deutlichkeitsmaß C50 oder das Klarheitsmaß C80.
        :param Fs: Samplerate (Integer)
        :param Input: Audiosignal (Array oder Liste)
        :param IdxStart: Index des Beginns der Impulsantwort (Integer)
        :param CXX: C50 oder C80 (String)
        :return: None
        '''
        Limit = None                                # Empty variable -> for calculating C50 / 80
        WhichIndex = None                           # Empty variable -> Name of the quantity to be calculated

        if CXX == "C50":
            Limit = 0.05
            WhichIndex = "Deutlichkeitsmaß"

        elif CXX == "C80":
            Limit = 0.08
            WhichIndex = "Klarheitsmaß"

        SumToLimit = np.sum(np.square(self.rir[self.idx_start:(self.idx_start + (int(self.fs*Limit)))]))  # Integral (sum) up to 50 or 80 ms
        SumFromLimit = np.sum(np.square(self.rir[(self.idx_start + (int(self.fs*Limit))):]))        # Integral (sum) from 50 or 80 ms

        c = 10 * np.log10(SumToLimit / SumFromLimit)

        print("Das frequenzunabhängige {} C{} beträgt {:.2f} dB".format(WhichIndex, int(Limit * 1000), c))

        return c
    

if __name__ == "__main__":

    input_folder_path = "dataset/RIRs"
    output_folder_path = "dataset/room_acoustic_parameters"

    filenames = os.listdir(input_folder_path)
    rirs = [sf.read(f"{input_folder_path}/{rir_file}")[0] for rir_file in filenames]
    os.makedirs(output_folder_path, exist_ok=True)

    output = pd.DataFrame(columns=["filename", "RT60_RT20", "Corr_RT60_RT20", "RT60_RT30", "Corr_RT60_RT30"])
    
    for i, filename in enumerate(filenames):

        # Create an instance of RoomAcousticParameters
        rap = RoomAcousticParameters(f'{input_folder_path}/{filename}')
        print(rap.rir.shape)

        edc = rap.EDC()
        print(edc.size)
        np.save(f"{output_folder_path}/EDC/{filename.split('.')[0]}_edc.npy", edc)
        #rap.plot_edc(10*np.log10(edc))
        rap.plot_edc(edc)

        #edt = rap.EDT()
        #print(f"EDT: {edt} seconds")

        rt60_direct = rap.RT60_direct()
        print(f"RT60 direct: {rt60_direct} seconds")

        rt60, Corr_rt60 = rap.schroederrt(LevelHigh=-5, LevelLow=-65, plot=False)

        rt60_rt20, Corr_rt60rt20 = rap.schroederrt(plot=False)

        rt60_rt30, Corr_rt60rt30 = rap.schroederrt(LevelHigh=-5, LevelLow=-35, plot=False)

        #c50 = rap.calrityindex("C50")
        #c80 = rap.calrityindex("C80")

        output.loc[i] = [filename, rt60_rt20, Corr_rt60rt20, rt60_rt30, Corr_rt60rt30]


    output = output.round(3)
    csv_file = "EDC/dataset/room_acoustic_parameters/room_acoustic_parameters.csv"
    output.to_csv(csv_file, index=False)


