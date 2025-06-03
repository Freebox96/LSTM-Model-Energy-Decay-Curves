import pyroomacoustics as pra
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import json
import os
import re
from scipy.signal import spectrogram
import csv
import pandas as pd


class Room():
    def __init__(self, id, width, length, height, material_case_index, material_cases, source, receiver):
        """
        Initialize the room with multiple material cases.

        :param name: Name of the room (str)
        :param size: Tuple of dimensions (width, height, depth)
        :param sources: List of source positions [(x, y, z), ...]
        :param receivers: List of receiver positions [(x, y, z), ...]
        :param material_cases: List of cases, where each case is a dict with wall, ceiling, and floor materials
        """
        #read the input CSV file to dataframe
        # Extract the room dimensions and positions from the DataFrame
        self.corners = [[0, 0], [0, width], [length, width], [length, 0]]
        self.height = height
        self.fs = 44100
        #self.material = str(input_row['absorption_coeffs'])
        self.walls = [material_cases.iloc[0], material_cases.iloc[1], material_cases.iloc[2], material_cases.iloc[3]]
        self.floor = material_cases.iloc[4]
        self.ceiling = material_cases.iloc[5]
        self.source = source
        self.receiver = receiver
        self.material_case_index = material_case_index
        self.id = str(id).zfill(3)
        self.filename = f"RIR_{self.id}_case{self.material_case_index}"

    def _create_room(self):
        rooms = []
        room = pra.Room.from_corners(
            corners=np.array(self.corners).T,
            fs=self.fs,
            max_order=3, # at 5 script hangs after 106 combinations
            materials=[pra.Material(material) for material in self.walls],
            ray_tracing=True,
            air_absorption=True
        )

        room.extrude(self.height, materials={'ceiling': pra.Material(self.ceiling), 'floor': pra.Material(self.floor)})

        # Set ray tracing parameters
        room.set_ray_tracing(receiver_radius=0.5, n_rays=5000, energy_thres=1e-3) # at 20000 script hangs after 6 combinations 1e-5 after 106 combinations 10000 after 106

        room.add_source(self.source)
        
        mic_positions = np.array([self.receiver]).T
        room.add_microphone_array(
            pra.MicrophoneArray(mic_positions, room.fs)
            )

        return room

    def calculate_rir(self, save=0):
        """
        Calculate the Room Impulse Response for a specific source, receiver, and material case.

        :param source_index: Index of the source in the source list
        :param receiver_index: Index of the receiver in the receiver list
        :param material_case: Material configuration (dict)
        :return: Simulated RIR (mocked as None)
        """

        room = self._create_room()

        # Compute the RIR
        room.compute_rir()
        rir = room.rir[0][0]

        if save == 1:
            self._save_rir(rir)

        return rir
    
    def _save_rir(self, rir):
        rir_normalized = rir / np.max(np.abs(rir)) 

        rir_normalized = rir_normalized.flatten()
        # Save the RIR to a WAV file using soundfile
        sf.write(f"RIRs_simulated_v2/{self.filename}.wav", rir, self.fs)


class Visualize():
    def __init__(self, fs, input_folder_path, output_folder_path):
        self.fs = fs
        self.filenames = os.listdir(input_folder_path)
        self.rirs = [sf.read(f"{input_folder_path}/{rir_file}")[0] for rir_file in self.filenames]
        self.output_folder_path = output_folder_path
        os.makedirs(output_folder_path, exist_ok=True)

    def _extract_information_from_filename(self, filename):

        # Extract numbers for each component using regex
        match = re.match(r"room(\d+)_case(\d+)_source(\d+)_receiver(\d+)\.wav", filename)
        if match:
            room = int(match.group(1))      # Extract and convert "room1" -> 1
            case = int(match.group(2))      # Extract and convert "case1" -> 1
            source = int(match.group(3))    # Extract and convert "source1" -> 1
            receiver = int(match.group(4))  # Extract and convert "receiver1" -> 1
            title_string = f"Room {room}, Case {case}, Source {source}, Receiver {receiver}"
            return title_string
        else:
            raise ValueError("Filename does not match the expected pattern.")
        

    def plot_rir(self, time=1, freq=1, spec=1):
        #self.rirs = self.rirs[0:2]
        #self.filenames = self.filenames[0:2]
        self.title_strings = [self._extract_information_from_filename(filename) for filename in self.filenames]

        if time == 1:
            for i, rir in enumerate(self.rirs):
                fig, ax = plt.subplots(figsize=(10, 6))
                x = np.arange(len(rir)) / self.fs
                ax.plot(x, rir, color='lightsalmon')
                ax.set_title(f'RIR in Time Domain: {self.title_strings[i]}')
                ax.set_xlabel('Time [s]')
                ax.set_ylabel('Amplitude')
                ax.set_ylim(-1, 1)
                ax.grid(True)
                plt.savefig(f'{self.output_folder_path}/timedomain_{self.filenames[i]}.png')
                plt.show()

        if freq == 1:
            # plot the fourier transform of the RIR (not in time)
            for i, rir in enumerate(self.rirs):

                bin_width = 50  # Frequency bin width in Hz
                # Compute the FFT of the RIR
                rir_fft = np.fft.fft(rir)
                freq_bins = np.fft.fftfreq(len(rir), d=1/self.fs)
                magnitude = np.abs(rir_fft)
                
                # Take only the positive frequencies
                positive_freqs = freq_bins[:len(rir)//2]
                positive_magnitude = magnitude[:len(rir)//2]
                
                # Normalize the magnitude
                positive_magnitude /= np.max(positive_magnitude)
                
                # Aggregate magnitudes into bins of width `bin_width`
                bin_edges = np.logspace(np.log10(positive_freqs[1]), np.log10(positive_freqs[-1]), num=150)
                binned_magnitude, _ = np.histogram(positive_freqs, bins=bin_edges, weights=positive_magnitude)
                
                # Normalize the binned magnitudes
                binned_magnitude /= np.max(binned_magnitude)
                
                # Plot the bar plot
                ax = plt.subplots(figsize=(10, 6))[1]
                ax.bar(bin_edges[:-1], binned_magnitude, width=np.diff(bin_edges), align='edge', color='lightsalmon')
                ax.set_title(f"RIR in Frequency Domain: {self.title_strings[i]}")
                ax.set_xlabel("Frequency [Hz]")
                ax.set_ylabel("Magnitude")
                ax.set_xscale('log')
                ax.set_xlim(50, 8000)
                ax.set_xticks([100, 500, 200, 1000, 2000, 5000, 8000])
                ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
                ax.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=0)
                ax.set_axisbelow(True)
                plt.savefig(f'{self.output_folder_path}/freqdomain_{self.filenames[i]}.png')
                plt.show()


        if spec == 1:
            for i, rir in enumerate(self.rirs):
                # Compute the spectrogram
                frequencies, times, Sxx = spectrogram(rir, fs=self.fs, window='hann', nperseg=256, noverlap=128)
                
                # Convert to decibels for better visualization
                Sxx_dB = 10 * np.log10(Sxx + 1e-10)  # Add small constant to avoid log(0)
                
                # Plot the spectrogram with logarithmic frequency axis
                fig, ax = plt.subplots(figsize=(10, 6))
                cax = ax.pcolormesh(times, frequencies, Sxx_dB, shading='gouraud', cmap='viridis')
                fig.colorbar(cax, ax=ax, label='Magnitude (dB)')
                ax.set_yscale('log')
                ax.set_title(f"Spectrogram: {self.title_strings[i]}")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Frequency (Hz)")
                ax.set_yticks([100, 500, 200, 1000, 2000, 5000, 8000])
                ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
                ax.set_ylim(frequencies[1], self.fs / 2)  # Show up to Nyquist frequency, avoid zero frequency
                plt.savefig(f'{self.output_folder_path}/spectrogram_{self.filenames[i]}.png')
                plt.show()

    



if __name__ == "__main__":

    plt.show = lambda: None # to disable plotting
    #plt.savefig = lambda *args, **kwargs: None # to disable saving

    df = pd.read_csv("random-rooms/full_source_receiver_data.csv", dtype={'ID': str})
    material_cases = pd.read_csv("random-rooms/room_material_combinations.csv", dtype={'ID': str})

    combos = []
    for row in df.iloc[:].iterrows():
        id = row[1]['ID']
        # if int(id) < 107:
        #     continue

        width = row[1]['Width']
        length = row[1]['Length']
        height = row[1]['Height']
        src = (row[1]['Source_X'], row[1]['Source_Y'], 1.5)
        rec = (row[1]['Receiver_X'], row[1]['Receiver_Y'], 1.5) 

        if src == rec:
            print(f"Skipping ID {id}: source and receiver are the same")
            continue
        if min(width, length, height) < 0.5:
            print(f"Skipping ID {id}: room too small")
            continue


        print(f"Processing id {id}")



        for material_case in material_cases.iloc[:].iterrows():

            print(f"Room ID {id}, Case {material_case[0]}: src={src}, rec={rec}, dims=({width},{length},{height})")
            print(f"Materials: {material_case[1].to_dict()}")
       
            try:
                room = Room(
                    id = id,
                    width = width,
                    length = length,
                    height = height,
                    material_case_index = material_case[0],
                    material_cases = material_case[1],
                    source = src,
                    receiver = rec
                )

                room.calculate_rir(save=1)
        

            except Exception as e:
                # missing_key = row['absorption_coeffs']
                # missing_keys.append(missing_key)
                #print(f"Skipping row {index} due to missing key: {missing_key}")
                print(f"Error: {e}")
                continue
                
        
    
        
    
    #print(missing_keys)




    #Visualize(fs=16000, input_folder_path=input_folder, output_folder_path=output_folder).plot_rir(time=1, freq=1, spec=1)

    #room.calculate_rir(source_index=1, receiver_index=0, material_case_index=0)


