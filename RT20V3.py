import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def compute_t20_from_edcs(data, fs=44100, start_dB=-5, end_dB=-25):
    num_signals, N = data.shape
    time = np.arange(N) / fs
    T20_values = []

    for i in range(num_signals):
        edc = data[i]  # Skip first column if it's time or index
        edc = edc / np.max(edc)
        #edc_db = 10 * np.log10(edc + np.finfo(float).eps)

        if np.min(edc) > end_dB or np.max(edc) < start_dB:
            T20_values.append(np.nan)
            continue

        try:
            start_idx = np.where(edc <= start_dB)[0][0]
            end_idx = np.where(edc <= end_dB)[0][0]
        except IndexError:
            T20_values.append(np.nan)
            continue

        if end_idx <= start_idx or (end_idx - start_idx < 2):
            T20_values.append(np.nan)
            continue

        t_fit = time[start_idx:end_idx]
        edc_fit = edc[start_idx:end_idx]

        if len(t_fit) < 2:
            T20_values.append(np.nan)
            continue

        slope, intercept, *_ = stats.linregress(t_fit, edc_fit)
        T20 = -60 / slope
        T20_values.append(T20)

    return np.array(T20_values)


# Function to calculate EDT, T20, T30, C50, C80, D50 and D80 from EDCs
def calculate_acoustics_from_edc(edc, fs, plot=False):
    edc = np.asarray(edc)

    if edc.max() == 0 or np.isnan(edc).any():
        return {k: np.nan for k in ['EDT', 'T20', 'T30', 'C50', 'C80', 'D50', 'D80']}

    edc = edc #/ np.max(edc)  # Normalize EDC
    #edc_db = 10 * np.log10(edc + 1e-12)
    time = np.arange(len(edc)) / fs

    def decay_time(start_db, end_db):
        try:
            if edc[0] < start_db:
                return np.nan, [0, 0]  # Not enough dynamic range

            idx_start = np.where(edc <= start_db)[0][0]
            idx_end = np.where(edc <= end_db)[0][0]

            if idx_end <= idx_start:
                return np.nan, [0, 0]

            p = np.polyfit(time[idx_start:idx_end], edc[idx_start:idx_end], 1)
            slope = p[0]
            rt = -60 / slope if slope != 0 else np.nan
            return rt, p
        except IndexError:
            return np.nan, [0, 0]

    def clarity(t_ms):
        idx = int((t_ms / 1000.0) * fs)
        if idx >= len(edc):
            return np.nan
        early_energy = np.sum(edc[:idx])
        late_energy = np.sum(edc[idx:])
        if late_energy == 0:
            return np.nan
        #return 10 * np.log10(early_energy / late_energy)
        return early_energy / late_energy

    def definition(t_ms):
        idx = int((t_ms / 1000.0) * fs)
        total_energy = np.sum(edc)
        if total_energy == 0:
            return np.nan
        return 100 * np.sum(edc[:idx]) / total_energy

    # Calculate decay times
    edt, p_edt = decay_time(0, -10)
    t20, p_t20 = decay_time(-5, -25)
    t30, p_t30 = decay_time(-5, -35)

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(time, edc, label="Schroeder Curve", color='black')

        def plot_fit(p, label, color, start_db, end_db):
            try:
                idx_start = np.where(edc <= start_db)[0][0]
                idx_end = np.where(edc <= end_db)[0][0]
                t_fit = time[idx_start:idx_end]
                fit_vals = np.polyval(p, t_fit)
                plt.plot(t_fit, fit_vals, '--', label=label, color=color)
            except IndexError:
                pass

        plot_fit(p_edt, "EDT Fit", "blue", 0, -10)
        plot_fit(p_t20, "T20 Fit", "green", -5, -25)
        plot_fit(p_t30, "T30 Fit", "red", -5, -35)

        plt.xlabel("Time (s)")
        plt.ylabel("EDC (dB)")
        plt.title("Energy Decay Curve and Reverberation Fits")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return {
        "EDT": edt,
        "T20": t20,
        "T30": t30,
        "C50": clarity(50),
        "C80": clarity(80),
        "D50": definition(50),
        "D80": definition(80)
    }

# Main script
if __name__ == "__main__":
    fs = 44100
    target_length = 44100
    #rooms_to_process = "dense_model100e_scalingX"
    output_folder = "model_results"
    model_name = "mhe_model_100e_scalingXseperate"

    actual_edcs = pd.read_csv(f"true_curves{model_name}.csv").values
    predicted_edcs = pd.read_csv(f"predicted_curves{model_name}.csv").values
    output_csv = f"{output_folder}/Comparison_results_{model_name}.csv"

    print("csv files loaded")

    results = []

    for idx in range(len(actual_edcs)):
        actual = actual_edcs[idx]
        predicted = predicted_edcs[idx]

        params_actual = calculate_acoustics_from_edc(actual, fs, plot=False)
        params_pred = calculate_acoustics_from_edc(predicted, fs, plot=False)

        row = {'Sample': idx}
        for key in params_actual:
            row[f'{key}_actual'] = params_actual[key]
            row[f'{key}_pred'] = params_pred[key]
            row[f'{key}_error'] = params_pred[key] - params_actual[key]
        results.append(row)

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

    # Error analysis & plotting
    metrics = {}
    params = ['EDT', 'T20', 'T30', 'C50', 'C80', 'D50', 'D80']

    for param in params:
        actual_vals = results_df[f"{param}_actual"]
        pred_vals = results_df[f"{param}_pred"]

        mask = ~np.isnan(actual_vals) & ~np.isnan(pred_vals)
        if mask.sum() == 0:
            continue

        actual_vals = actual_vals[mask]
        pred_vals = pred_vals[mask]

        mae = mean_absolute_error(actual_vals, pred_vals)
        rmse = mean_squared_error(actual_vals, pred_vals, squared=False)
        r2 = r2_score(actual_vals, pred_vals)
        metrics[param] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}

        print(f"{param}: MAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.3f}")
        
        # Plotting of Actual vs Predicted values    
        plt.figure(figsize=(6, 5))
        plt.scatter(actual_vals, pred_vals, alpha=0.7, edgecolors='k')
        min_val = min(actual_vals.min(), pred_vals.min())
        max_val = max(actual_vals.max(), pred_vals.max())
        plt.plot([min_val, max_val], [min_val, max_val], '--', color='gray', label='Ideal Match')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'{param} Comparison')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_folder}/{param}_comparison_{model_name}.png")
        #plt.show()
        print(f"Plot saved as {param}_comparison_{model_name}.png")


    """
    # ===================
    # EDCs Preprocessing from RT20V2
    # ===================
    """
    # Compute T20 for both
    actual_T20 = compute_t20_from_edcs(actual_edcs)
    predicted_T20 = compute_t20_from_edcs(predicted_edcs)

    # Convert to T60 (T20 * 3)
    actual_T60 = actual_T20 * 3
    predicted_T60 = predicted_T20 * 3

    # Combine and Export
    results20_df = pd.DataFrame({
        'T20_actual': actual_T20,
        'T20_predicted': predicted_T20
    })

    results60_df = pd.DataFrame({
            'T60_actual': actual_T60,
            'T60_predicted': predicted_T60
    })
    
    
    # Save the results to CSV files
    results20_df.to_csv(f"{output_folder}/T20_comparison_RT20V2_{model_name}.csv", index=False, header=True)
    print(f"T20_comparison_RT20V2_{model_name}.csv")
    results60_df.to_csv(f"{output_folder}/T60_comparison_RT20V2_{model_name}.csv", index=False, header=True)
    print(f"T60_comparison_RT20V2_{model_name}.csv")

    
    # Optional: Plot example of single EDC for debugging
    # Uncomment the following lines to visualize a single EDC and its T20 estimation
    exampleEDCPlot = False  # Set to True to plot an example EDC
    if exampleEDCPlot == True:
        i = 100
        fs = 44100
        N = actual_edcs.shape[1]
        time = np.arange(N) / fs
        edc = actual_edcs[i] / np.max(actual_edcs[i])
        #edc_db = 10 * np.log10(edc + np.finfo(float).eps)
        start_idx = np.argmax(edc <= -5)
        end_idx = np.argmax(edc <= -25)
        t_fit = time[start_idx:end_idx]
        slope, intercept, *_ = stats.linregress(t_fit, edc[start_idx:end_idx])
        print(f"Estimated T20 for Actual EDC #{i}: {actual_T20[i]:.3f} seconds")
        print(f"Estimated T60 for Actual EDC #{i}: {actual_T60[i]:.3f} seconds")
        plt.figure(figsize=(10, 5))
        plt.plot(time, edc, label=f'Actual EDC #{i}', color='blue')
        plt.plot(t_fit, slope * t_fit + intercept, 'r--', label='Linear Fit (-5 to -25 dB)', linewidth=2)
        plt.title(f'T20 Estimation for Actual EDC #{i}')
        plt.xlabel('Time (s)')
        plt.ylabel('Energy Decay (dB)')
        plt.legend()
        plt.grid(True)
        #plt.show()
        print("Plotting example EDC completed.")    

    # Load data for comparison
    # Load the CSV file containing the actual and predicted T20 values
    df = pd.read_csv(f"{output_folder}/T20_comparison_RT20V2_{model_name}.csv")
    actual = df["T20_actual"].values
    predicted = df["T20_predicted"].values

    # Remove NaNs for fair comparison
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
    actual = actual[mask]
    predicted = predicted[mask]

    # Compute Error Metrics
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    print(f"Mean Absolute Error (MAE): {mae:.3f} s")
    print(f"Root Mean Squared Error (RMSE): {rmse:.3f} s")
    print(f"R² Score: {r2:.3f}")

    # --- Plot 1: Actual vs Predicted T60 ---
    plt.figure(figsize=(10, 5))
    plt.plot(actual, label='T20 Actual', marker='o', color='blue')
    plt.plot(predicted, label='T20 Predicted', marker='x', color='red', alpha=0.8)
    plt.xlabel("EDC Index")
    plt.ylabel("T20 (seconds)")
    plt.title("Actual vs Predicted T20 per EDC")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_folder}/Actual vs Predicted T20 per EDC_{model_name}.png")
    #plt.show()
    
    # Bar Plot of T20 values
    indices = np.arange(len(actual))
    bar_width = 0.4
    plt.figure(figsize=(14, 6))
    plt.bar(indices - bar_width/2, actual, width=bar_width, label='T20 Actual', color='skyblue')
    plt.bar(indices + bar_width/2, predicted, width=bar_width, label='T20 Predicted', color='salmon')
    plt.xlabel("EDC Index")
    plt.ylabel("T20 (seconds)")
    plt.title("Comparison of Actual and Predicted T20 Values")
    #plt.xticks(indices, rotation=90 if len(actual) > 20 else 0)
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{output_folder}/Actual vs Predicted T20 per EDC Bar_Plot {model_name}.png")
    #plt.show()

    # Box Plot of T20 values
    data_to_plot = [actual, predicted]
    plt.figure(figsize=(6, 5))
    plt.boxplot(data_to_plot, labels=["T20 Actual", "T20 Predicted"], patch_artist=True,
                boxprops=dict(facecolor='lightgreen'),
                medianprops=dict(color='red', linewidth=2))

    plt.title("Box Plot of Actual vs Predicted T20")
    plt.ylabel("T20 (seconds)")
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{output_folder}/Actual vs Predicted T20 per EDC Box_Plot {model_name}.png")
    #plt.show()
    
    # Scatter Plot of T20 values
    plt.figure(figsize=(8, 6))
    plt.scatter(actual, predicted, alpha=0.7, edgecolor='k', label='Predicted vs Actual')
    plt.plot([min(actual), max(actual)], [min(actual), max(actual)], 'r--', label='Ideal (y = x)')
    plt.xlabel("T20 Actual (s)")
    plt.ylabel("T20 Predicted (s)")
    plt.title("T20 Prediction Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_folder}/Actual vs Predicted T20 per EDC Scattered Plot {model_name}.png")
    #plt.show()

    # Histogram of Errors for T20 values
    errors = actual - predicted
    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=30, color='skyblue', edgecolor='black')
    plt.axvline(0, color='red', linestyle='--', label='Zero Error')
    plt.xlabel("T20 Error (Actual - Predicted) [s]")
    plt.ylabel("Count")
    plt.title("Histogram of T20 Prediction Errors")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_folder}/Actual vs Predicted T20 per EDC Histogram {model_name}.png")
    #plt.show()
    
    # Box + Strip Plot of T20 values
    boxStripPlot = False  # Set to True to generate Box + Strip Plot    
    if boxStripPlot == True:
        df_plot = pd.DataFrame({
            'T20': np.concatenate([actual, predicted]),
            'Type': ['Actual'] * len(actual) + ['Predicted'] * len(predicted)
        })
        plt.figure(figsize=(6, 5))
        sns.boxplot(x='Type', y='T20', data=df_plot, palette='pastel', showfliers=True)
        sns.stripplot(x='Type', y='T20', data=df_plot, color='black', alpha=0.4, jitter=True)
        plt.title("Box + Strip Plot of T20")
        plt.ylabel("T20 (seconds)")
        plt.grid(True, axis='y', linestyle='--', alpha=0.4)
        plt.tight_layout()
        #plt.show()
        print("Plots generated successfully.")
    
    print("All tasks completed successfully.") 