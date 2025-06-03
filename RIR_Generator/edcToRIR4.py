import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

# Function to calculate decay rate from RT60 (Reverberation Time)
def calculate_decay_rate(rt60):
    return 3 / rt60  # Decay rate is inversely proportional to RT60

# Generate RIR with exponential decay influenced by RT60, keeping sign intact (only for reflections)
def generate_rir(rt60, duration=2, sample_rate=44100):
    time_vector = np.linspace(0, duration, int(sample_rate * duration))
    rir_noise = np.random.randn(len(time_vector))  # White noise for RIR (with both positive and negative values)
    decay_rate = calculate_decay_rate(rt60)
    
    # Apply exponential decay with sign retention (no absolute value)
    rir_exponential_decay = rir_noise * np.exp(-decay_rate * time_vector)
    
    return rir_exponential_decay, time_vector

# Compute the Energy Decay Curve (EDC) from RIR
def compute_edc(rir):
    #rir, samplerate = sf.read('rir01.wav')
    #plt.plot(rir)
    squared_rir = rir**2
    edc = np.cumsum(squared_rir) / np.sum(squared_rir)
    
    #edc1 = 10*np.log10(np.cumsum(squared_rir[..., ::-1], axis=-1)[..., ::-1] / np.sum(squared_rir))
    #edc = np.cumsum(squared_rir[..., ::-1], axis=-1)[..., ::-1] / np.sum(squared_rir)
    plt.plot(edc)
    plt.xlabel('Time [s]')
    plt.ylabel('EDC')    
    plt.show()
    return edc

# Reconstruct the RIR from the EDC (Ensure dual-sign RIR)
def reconstruct_rir_from_edc(edc, duration=2, sample_rate=44100):
    #differential_edc = np.diff(edc[...,::-1], prepend=0)
    differential_edc = np.diff(edc, prepend=0)
    reconstructed_rir = np.sqrt(differential_edc)
    
    # Ensure dual-sign by randomizing the sign for each element
    sign_pattern = np.random.choice([-1, 1], size=reconstructed_rir.shape)
    reconstructed_rir = reconstructed_rir * sign_pattern
    
    reconstructed_rir = reconstructed_rir / np.max(np.abs(reconstructed_rir))  # Normalize
    return reconstructed_rir

# Function to generate a dual polarity pulse (representing direct sound)
def generate_dual_polarity_pulse(sample_rate, duration=10):
    pulse = np.zeros(duration)
    pulse[::2] = 1  # Assign alternating polarity (+1, -1)
    pulse[1::2] = -1
    return pulse

# Function to add direct sound (dual polarity pulse) before the rest of the RIR
def add_direct_sound_with_reflections(rir, sample_rate=44100, distance=2, speed_of_sound=343, gap_time=0.005, pulse_duration=2):
    # Calculate direct sound delay (time taken for sound to travel from source to receiver)
    delay_samples = int((distance / speed_of_sound) * sample_rate)
    
    # Generate a dual-polarity pulse for the direct sound (10 samples)
    direct_sound_pulse = generate_dual_polarity_pulse(sample_rate, duration=pulse_duration)
    
    # Create the final RIR with direct sound pulse and the rest of the RIR after the gap
    total_length = len(rir)
    gap_samples = int(gap_time * sample_rate)
    
    # Ensure the final RIR can fit both direct sound and reflections
    rir_with_direct_sound_and_reflections = np.zeros(total_length + delay_samples + gap_samples + len(direct_sound_pulse))
    
    # Place the direct sound pulse at the beginning of the signal
    rir_with_direct_sound_and_reflections[delay_samples:delay_samples + len(direct_sound_pulse)] = direct_sound_pulse
    
    # Place the reflection part after the gap, starting from the gap position
    rir_with_direct_sound_and_reflections[delay_samples + len(direct_sound_pulse) + gap_samples:] = rir
    
    return rir_with_direct_sound_and_reflections

# Compute Mean Squared Error (MSE)
def compute_mse(rir_orig, rir_reconstructed):
    # Trim the reconstructed RIR to match the length of the original RIR
    min_length = min(len(rir_orig), len(rir_reconstructed))
    rir_orig_trimmed = rir_orig[:min_length]
    rir_reconstructed_trimmed = rir_reconstructed[:min_length]
    
    return np.mean((rir_orig_trimmed - rir_reconstructed_trimmed)**2)

# Compute Peak Signal-to-Noise Ratio (PSNR)
def compute_psnr(rir_orig, rir_reconstructed):
    mse = compute_mse(rir_orig, rir_reconstructed)
    psnr = 10 * np.log10(np.max(rir_orig**2) / mse)  # MAX is 1 for normalized values
    return psnr

# Main Function to Simulate RIR, EDC, and Reconstruct RIR with Direct Sound and Reflection
def main(rt60):
    rir, time_vector = generate_rir(rt60)
    edc = compute_edc(rir)
    reconstructed_rir = reconstruct_rir_from_edc(edc)
    
    # Add direct sound and reflections with the assumed distance (2 meters) and 5ms gap
    rir_with_direct_sound_and_reflections = add_direct_sound_with_reflections(reconstructed_rir, distance=2, gap_time=0.005)
    
    # Calculate MSE and PSNR
    mse = compute_mse(rir, rir_with_direct_sound_and_reflections)
    psnr = compute_psnr(rir, rir_with_direct_sound_and_reflections)
    
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Peak Signal-to-Noise Ratio (PSNR): {psnr} dB")
    
    # Plotting
    plt.figure(figsize=(18, 6))

    # Plot Original RIR with Direct Sound and Reflections
    plt.subplot(1, 3, 1)
    plt.plot(time_vector, rir_with_direct_sound_and_reflections[:len(time_vector)], label=f"Original RIR with Direct Sound (RT60 = {rt60}s)", color='blue')
    plt.title("Original RIR with Direct Sound and Reflections")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid(True)

    # Plot Energy Decay Curve (EDC) in dB
    plt.subplot(1, 3, 2)
    edc_db = 20 * np.log10(1 - edc)  # Reversed EDC in dB scale
    plt.plot(time_vector, edc_db, label="Reversed EDC in dB", color='green')
    plt.title("Energy Decay Curve (EDC) in dB")
    plt.xlabel("Time [s]")
    plt.ylabel("EDC [dB]")
    plt.ylim(-90, 0)  # Limit the y-axis to -60 dB for better visualization
    plt.grid(True)

    # Plot Reconstructed RIR from EDC
    plt.subplot(1, 3, 3)
    plt.plot(time_vector, rir_with_direct_sound_and_reflections[:len(time_vector)], label="Reconstructed RIR from EDC", color='red')
    plt.title("Reconstructed RIR from EDC")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Example: Generate RIR for RT60 = 0.3 seconds
main(rt60=1)
print("Done")
