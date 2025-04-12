import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.cm as cm

# Directory containing the CSVs

# def rl_graphs():
#     data_dir = "./Data/RL/obstacle_level_4"
#     pattern = r"RL_swarm_(\d+)_run_(\d)\.csv"

#     files_by_robot_count = {}

#     for filepath in glob(os.path.join(data_dir, "RL_swarm_*.csv")):
#         filename = os.path.basename(filepath)
#         match = re.match(pattern, filename)
#         if match:
#             num_robots = int(match.group(1))
#             run_id = int(match.group(2))
#             files_by_robot_count.setdefault(num_robots, []).append(filepath)

#     output_dir = os.path.join(data_dir, "plots_combined")
#     os.makedirs(output_dir, exist_ok=True)

#     processed_data = {}

#     for num_robots, file_list in files_by_robot_count.items():
#         print(f"Processing {len(file_list)} files for {num_robots} robots")

#         dataframes = [pd.read_csv(file) for file in file_list]
#         base_columns = dataframes[0].columns
#         dataframes = [df[base_columns] for df in dataframes]

#         max_len = max(len(df) for df in dataframes)
#         padded_dfs = []
#         for df in dataframes:
#             if len(df) < max_len:
#                 pad_size = max_len - len(df)
#                 pad_df = pd.DataFrame(np.nan, index=range(pad_size), columns=df.columns)
#                 df = pd.concat([df, pad_df], ignore_index=True)
#             padded_dfs.append(df)

#         data_stack = np.stack([df.values for df in padded_dfs])
#         mean_data = np.nanmean(data_stack, axis=0)
#         std_data = np.nanstd(data_stack, axis=0)

#         if 'Timestep' in base_columns:
#             timesteps = padded_dfs[0]['Timestep'].values
#         else:
#             timesteps = np.arange(max_len)

#         for col_idx, col_name in enumerate(base_columns):
#             if col_name == 'Timestep':
#                 continue
#             if col_name not in processed_data:
#                 processed_data[col_name] = {}
#             processed_data[col_name][num_robots] = (timesteps, mean_data[:, col_idx], std_data[:, col_idx])

#     # === Plot each metric over Timestep ===
#     for col_name, robot_data in processed_data.items():
#         plt.figure(figsize=(10, 5))
#         for num_robots, (timesteps, mean_vals, std_vals) in sorted(robot_data.items()):
#             plt.plot(timesteps, mean_vals, label=f"{num_robots} robots")
#             plt.fill_between(timesteps, mean_vals - std_vals, mean_vals + std_vals, alpha=0.2)
#         plt.xlabel("Timestep")
#         plt.ylabel(col_name)
#         plt.title(f"{col_name} vs Timestep for Different Robot Counts")
#         plt.legend(title="Robot Count")
#         plt.grid(True)
#         plt.tight_layout()
#         plt.savefig(os.path.join(output_dir, f"{col_name}_comparison.png"))
#         plt.close()

#     # === Rolling R² over Timestep ===
#     window_size = 50  # Rolling window size in timesteps
#     r2_over_time = {}  # {robot_count: (timesteps, mean_r2, std_r2)}

#     for num_robots, file_list in files_by_robot_count.items():
#         dataframes = [pd.read_csv(file) for file in file_list]
#         r2_matrix = []

#         for df in dataframes:
#             if 'TotalCost' in df.columns and 'Timestep' in df.columns:
#                 df = df.dropna(subset=['TotalCost', 'Timestep'])
#                 timesteps = df['Timestep'].values
#                 cost_vals = df['TotalCost'].values

#                 r2_list = []
#                 for i in range(len(df)):
#                     if i < window_size:
#                         r2_list.append(np.nan)
#                         continue
#                     X = timesteps[i - window_size:i].reshape(-1, 1)
#                     y = cost_vals[i - window_size:i]
#                     if len(np.unique(X)) > 1:
#                         model = LinearRegression().fit(X, y)
#                         y_pred = model.predict(X)
#                         r2 = r2_score(y, y_pred)
#                         r2_list.append(r2)
#                     else:
#                         r2_list.append(np.nan)
#                 r2_matrix.append(r2_list)

#         r2_array = np.array(r2_matrix)
#         mean_r2 = np.nanmean(r2_array, axis=0)
#         std_r2 = np.nanstd(r2_array, axis=0)
#         r2_over_time[num_robots] = (timesteps, mean_r2, std_r2)

#     # === Plot R² vs Timestep ===
#     plt.figure(figsize=(10, 5))
#     for num_robots, (timesteps, mean_r2, std_r2) in sorted(r2_over_time.items()):
#         plt.plot(timesteps, mean_r2, label=f"{num_robots} robots")
#         plt.fill_between(timesteps, mean_r2 - std_r2, mean_r2 + std_r2, alpha=0.2)
#     plt.axhline(1.0, linestyle='--', color='gray', label='Perfect Fit (R² = 1)')
#     plt.xlabel("Timestep")
#     plt.ylabel("Rolling R² (TotalCost)")
#     plt.title(f"Smoothness of Control Over Time (Window = {window_size})")
#     plt.legend(title="Robot Count")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, "R2_Timestep.png"))
#     plt.close()

#     print("✅ All RL plots (metrics + R² over time) saved to:", output_dir)

def plot_episode_rewards(log_dir, output_dir):
    reward_files = glob(os.path.join(log_dir, "episode_rewards_log_*.txt"))
    reward_data = {}

    for filepath in reward_files:
        match = re.search(r"episode_rewards_log_(\d+)", os.path.basename(filepath))
        if not match:
            continue
        num_robots = int(match.group(1))

        episodes, rewards = [], []
        with open(filepath, 'r') as f:
            for line in f:
                ep_match = re.search(r'Episode\s+(\d+)', line)
                reward_match = re.search(r'Reward:\s+([-\d\.eE]+)', line)
                if ep_match and reward_match:
                    episodes.append(int(ep_match.group(1)))
                    rewards.append(float(reward_match.group(1)))

        if episodes and rewards:
            df = pd.DataFrame({'Episode': episodes, 'Reward': rewards})
            reward_data[num_robots] = df

    # === Smoothing ===
    smoothed_data = {}
    window = 10

    for num_robots, df in reward_data.items():
        df = df.sort_values('Episode').reset_index(drop=True)
        df['Smoothed'] = df['Reward'].rolling(window=window, min_periods=1).mean()
        smoothed_data[num_robots] = df

    # === Split-view Plot ===
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))  # No shared Y-axis
    colormap = plt.cm.plasma
    color_list = colormap(np.linspace(0, 1, len(smoothed_data)))

    for i, (num_robots, df) in enumerate(sorted(smoothed_data.items())):
        color = color_list[i]
        label = f"{num_robots} robots"

        # Early training (Episodes 0–30)
        early = df[df['Episode'] <= 30]
        axs[0].plot(early['Episode'], early['Smoothed'], label=label, color=color, linewidth=2)

        # Late training (Episodes >30)
        late = df[df['Episode'] > 30]
        axs[1].plot(late['Episode'], late['Smoothed'], label=label, color=color, linewidth=2)

    # Axis labels and styling
    axs[0].set_title("Early Training (Episodes 0–30)")
    axs[1].set_title("Late Training (Episodes 31–End)")
    for ax in axs:
        ax.set_xlabel("Episode")
        ax.grid(True, linestyle="--", alpha=0.5)
    axs[0].set_ylabel("Reward")

    axs[1].legend(title="Swarm Size", fontsize=9, title_fontsize=10, loc='lower right')

    plt.suptitle("Episode Rewards During Training", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save plot
    output_path = os.path.join(output_dir, "episode_rewards_split_nostd.png")
    plt.savefig(output_path)
    plt.close()

    print(f"✅ Smoothed reward plot (no std) saved to: {output_path}")

def rl_graphs():
    data_dir = "./Data/RL/obstacle_level_4"
    pattern = r"RL_swarm_(\d+)_run_(\d)\.csv"

    files_by_robot_count = {}

    for filepath in glob(os.path.join(data_dir, "RL_swarm_*.csv")):
        filename = os.path.basename(filepath)
        match = re.match(pattern, filename)
        if match:
            num_robots = int(match.group(1))
            run_id = int(match.group(2))
            files_by_robot_count.setdefault(num_robots, []).append(filepath)

    output_dir = os.path.join(data_dir, "plots_combined")
    os.makedirs(output_dir, exist_ok=True)
    
    plot_episode_rewards(data_dir, output_dir)

    processed_data = {}
    r2_scores_by_robot = {}

    for num_robots, file_list in files_by_robot_count.items():
        print(f"Processing {len(file_list)} files for {num_robots} robots")

        dataframes = [pd.read_csv(file) for file in file_list]
        base_columns = dataframes[0].columns
        dataframes = [df[base_columns] for df in dataframes]

        max_len = max(len(df) for df in dataframes)
        padded_dfs = []
        for df in dataframes:
            if len(df) < max_len:
                pad_size = max_len - len(df)
                pad_df = pd.DataFrame(np.nan, index=range(pad_size), columns=df.columns)
                df = pd.concat([df, pad_df], ignore_index=True)
            padded_dfs.append(df)

        data_stack = np.stack([df.values for df in padded_dfs])
        mean_data = np.nanmean(data_stack, axis=0)
        std_data = np.nanstd(data_stack, axis=0)

        if 'Timestep' in base_columns:
            timesteps = padded_dfs[0]['Timestep'].values
        else:
            timesteps = np.arange(max_len)

        for col_idx, col_name in enumerate(base_columns):
            if col_name == 'Timestep':
                continue
            if col_name not in processed_data:
                processed_data[col_name] = {}
            processed_data[col_name][num_robots] = (timesteps, mean_data[:, col_idx], std_data[:, col_idx])

        # R² calculation for total_cost vs Timestep
        r2_list = []
        for df in dataframes:
            if 'TotalCost' in df.columns and 'Timestep' in df.columns:
                clean_df = df.dropna(subset=['TotalCost', 'Timestep'])
                if len(clean_df) > 1:
                    X = clean_df['Timestep'].values.reshape(-1, 1)
                    y = clean_df['TotalCost'].values
                    model = LinearRegression().fit(X, y)
                    y_pred = model.predict(X)
                    r2 = r2_score(y, y_pred)
                    r2_list.append(r2)
        if r2_list:
            r2_scores_by_robot[num_robots] = r2_list

    # === Plot metric over Timestep ===
    for col_name, robot_data in processed_data.items():
        plt.figure(figsize=(12, 6))
        colors = cm.viridis(np.linspace(0, 1, len(robot_data)))

        for idx, (num_robots, (timesteps, mean_vals, std_vals)) in enumerate(sorted(robot_data.items())):
            # Smooth
            mean_vals = pd.Series(mean_vals).rolling(50, min_periods=1).mean().values
            std_vals = pd.Series(std_vals).rolling(50, min_periods=1).mean().values

            plt.plot(timesteps, mean_vals, label=f"{num_robots} robots", color=colors[idx], linewidth=2)
            plt.fill_between(timesteps, mean_vals - std_vals, mean_vals + std_vals, color=colors[idx], alpha=0.2)

        plt.xlabel("Timestep [s]")
        plt.ylabel(col_name)
        plt.title(f"{col_name} vs Timestep for Different Robot Counts")
        plt.legend(title="Robot Count", loc="upper right", fontsize=9)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{col_name}_comparison_smooth.png"))
        plt.close()

    # === Plot R² values ===
    plt.figure(figsize=(8, 5))
    for num_robots, r2_list in sorted(r2_scores_by_robot.items()):
        mean_r2 = np.mean(r2_list)
        std_r2 = np.std(r2_list)
        plt.errorbar(num_robots, mean_r2, yerr=std_r2, fmt='o', capsize=5, label=f"{num_robots} robots")
    plt.axhline(1.0, linestyle='--', color='gray', label='Perfect Fit (R² = 1)')
    plt.xlabel("Number of Robots")
    plt.ylabel("R² Value (Control Cost)")
    plt.title("Smoothness of Control (R²)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "R2_ControlCost.png"))
    plt.close()

    print("✅ All RL combined and performance plots saved to:", output_dir)


def pso_graphs():
    data_dir = "./Data/PSO/obstacle_level_4"
    pattern = r"optimal_run_swarm_(\d+)_run_(\d)\.csv"

    files_by_robot_count = {}

    for filepath in glob(os.path.join(data_dir, "optimal_run_swarm_*.csv")):
        filename = os.path.basename(filepath)
        match = re.match(pattern, filename)
        if match:
            num_robots = int(match.group(1))
            run_id = int(match.group(2))
            files_by_robot_count.setdefault(num_robots, []).append(filepath)

    output_dir = os.path.join(data_dir, "plots_combined")
    os.makedirs(output_dir, exist_ok=True)

    processed_data = {}
    r2_scores_by_robot = {}

    for num_robots, file_list in files_by_robot_count.items():
        print(f"Processing {len(file_list)} files for {num_robots} robots")

        dataframes = [pd.read_csv(file) for file in file_list]
        base_columns = dataframes[0].columns
        dataframes = [df[base_columns] for df in dataframes]

        max_len = max(len(df) for df in dataframes)
        padded_dfs = []
        for df in dataframes:
            if len(df) < max_len:
                pad_size = max_len - len(df)
                pad_df = pd.DataFrame(np.nan, index=range(pad_size), columns=df.columns)
                df = pd.concat([df, pad_df], ignore_index=True)
            padded_dfs.append(df)

        data_stack = np.stack([df.values for df in padded_dfs])
        mean_data = np.nanmean(data_stack, axis=0)
        std_data = np.nanstd(data_stack, axis=0)

        if 'timestep' in base_columns:
            timesteps = padded_dfs[0]['timestep'].values
        else:
            timesteps = np.arange(max_len)

        for col_idx, col_name in enumerate(base_columns):
            if col_name == 'timestep':
                continue
            if col_name not in processed_data:
                processed_data[col_name] = {}
            processed_data[col_name][num_robots] = (
                timesteps,
                mean_data[:, col_idx],
                std_data[:, col_idx]
            )

        # === R² calculation for total_cost vs timestep ===
        r2_list = []
        for df in dataframes:
            if 'total_cost' in df.columns and 'timestep' in df.columns:
                clean_df = df.dropna(subset=['total_cost', 'timestep'])
                if len(clean_df) > 1:
                    X = clean_df['timestep'].values.reshape(-1, 1)
                    y = clean_df['total_cost'].values
                    model = LinearRegression().fit(X, y)
                    y_pred = model.predict(X)
                    r2 = r2_score(y, y_pred)
                    r2_list.append(r2)
        if r2_list:
            r2_scores_by_robot[num_robots] = r2_list

    # === Plot metrics over timestep (with smoothing & styling) ===
    for col_name, robot_data in processed_data.items():
        plt.figure(figsize=(12, 6))
        colors = cm.viridis(np.linspace(0, 1, len(robot_data)))

        for idx, (num_robots, (timesteps, mean_vals, std_vals)) in enumerate(sorted(robot_data.items())):
            mean_vals = pd.Series(mean_vals).rolling(window=50, min_periods=1).mean().values
            std_vals = pd.Series(std_vals).rolling(window=50, min_periods=1).mean().values

            plt.plot(timesteps, mean_vals, label=f"{num_robots} robots", color=colors[idx], linewidth=2)
            plt.fill_between(timesteps, mean_vals - std_vals, mean_vals + std_vals,
                             color=colors[idx], alpha=0.2)

        plt.xlabel("Timestep [s]")
        plt.ylabel(col_name)
        plt.title(f"{col_name} vs Timestep for Different Robot Counts")
        plt.legend(title="Robot Count", loc="upper right", fontsize=9)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{col_name}_comparison.png"))
        plt.close()

    # === R² plot ===
    plt.figure(figsize=(8, 5))
    for num_robots, r2_list in sorted(r2_scores_by_robot.items()):
        mean_r2 = np.mean(r2_list)
        std_r2 = np.std(r2_list)
        plt.errorbar(num_robots, mean_r2, yerr=std_r2, fmt='o', capsize=5, label=f"{num_robots} robots")
    plt.axhline(1.0, linestyle='--', color='gray', label='Perfect Fit (R² = 1)')
    plt.xlabel("Number of Robots")
    plt.ylabel("R² Value (Total Cost)")
    plt.title("Smoothness of Control (R²)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "R2_ControlCost.png"))
    plt.close()

    print("✅ All PSO combined and performance plots saved to:", output_dir)
    
    
def get_avg_pso_data():
    data_dir = "./Data/PSO/obstacle_level_4"
    pattern = r"optimal_run_swarm_20_run_(\d)\.csv"
    file_paths = [f for f in glob(os.path.join(data_dir, "optimal_run_swarm_20_run_*.csv")) if re.match(pattern, os.path.basename(f))]

    if not file_paths:
        print("No data found.")
        return pd.DataFrame(), pd.DataFrame()

    dataframes = [pd.read_csv(file) for file in file_paths]
    base_columns = dataframes[0].columns
    max_len = max(len(df) for df in dataframes)

    padded_dfs = []
    for df in dataframes:
        if len(df) < max_len:
            pad = pd.DataFrame(np.nan, index=range(max_len - len(df)), columns=df.columns)
            df = pd.concat([df, pad], ignore_index=True)
        padded_dfs.append(df[base_columns])

    # === Compute mean DataFrame ===
    data_stack = np.stack([df.values for df in padded_dfs])
    mean_data = np.nanmean(data_stack, axis=0)
    avg_df = pd.DataFrame(mean_data, columns=base_columns)
    if 'timestep' not in avg_df.columns:
        avg_df.insert(0, 'timestep', np.arange(len(avg_df)))

    # === Compute R² per metric, averaged across runs ===
    r2_results = []

    for col in base_columns:
        if col == "timestep":
            continue
        r2_list = []
        for df in dataframes:
            clean = df.dropna(subset=["timestep", col])
            if len(clean) > 1:
                X = clean["timestep"].values.reshape(-1, 1)
                y = clean[col].values
                model = LinearRegression().fit(X, y)
                r2 = r2_score(y, model.predict(X))
                r2_list.append(r2)
        if r2_list:
            r2_mean = np.mean(r2_list)
            r2_std = np.std(r2_list)
            r2_results.append((col, round(r2_mean, 4), round(r2_std, 4)))

    r2_df = pd.DataFrame(r2_results, columns=["Metric", "R² (mean)", "R² (std)"])
    return avg_df, r2_df

def get_avg_rl_data():
    data_dir = "./Data/RL/obstacle_level_4"
    pattern = r"RL_swarm_15_run_(\d)\.csv"
    file_paths = [f for f in glob(os.path.join(data_dir, "RL_swarm_15_run_*.csv")) if re.match(pattern, os.path.basename(f))]

    if not file_paths:
        print("No data found.")
        return pd.DataFrame(), pd.DataFrame()

    dataframes = [pd.read_csv(file) for file in file_paths]
    base_columns = dataframes[0].columns
    max_len = max(len(df) for df in dataframes)

    padded_dfs = []
    for df in dataframes:
        if len(df) < max_len:
            pad = pd.DataFrame(np.nan, index=range(max_len - len(df)), columns=df.columns)
            df = pd.concat([df, pad], ignore_index=True)
        padded_dfs.append(df[base_columns])

    # === Compute mean DataFrame ===
    data_stack = np.stack([df.values for df in padded_dfs])
    mean_data = np.nanmean(data_stack, axis=0)
    avg_df = pd.DataFrame(mean_data, columns=base_columns)
    if 'Timestep' not in avg_df.columns:
        avg_df.insert(0, 'Timestep', np.arange(len(avg_df)))

    # === Compute R² per metric, averaged across runs ===
    r2_results = []

    for col in base_columns:
        if col == "Timestep":
            continue
        r2_list = []
        for df in dataframes:
            clean = df.dropna(subset=["Timestep", col])
            if len(clean) > 1:
                X = clean["Timestep"].values.reshape(-1, 1)
                y = clean[col].values
                model = LinearRegression().fit(X, y)
                r2 = r2_score(y, model.predict(X))
                r2_list.append(r2)
        if r2_list:
            r2_mean = np.mean(r2_list)
            r2_std = np.std(r2_list)
            r2_results.append((col, round(r2_mean, 4), round(r2_std, 4)))

    r2_df = pd.DataFrame(r2_results, columns=["Metric", "R² (mean)", "R² (std)"])
    return avg_df, r2_df
  
if __name__ == "__main__":
    # rl_graphs()
    # pso_graphs()
    
    print("PPO Data for swarm size 20")
    print("========================================")
    avg_df, r2_df = get_avg_pso_data()
    if not avg_df.empty:
        for column in avg_df.columns:
            if column != "Timestep":
                print(f"Column: {column}")
                print(f"Mean: {avg_df[column].mean()}")
                print(f"Max: {avg_df[column].max()}")
                print(f"Min: {avg_df[column].min()}")
                print("-" * 30)
                
    print("R² DataFrame:")
    print(r2_df)
    
    print("PPO Data for swarm size 15")
    print("========================================")
    avg_df_rl, r2_df_rl = get_avg_rl_data()
    if not avg_df_rl.empty:
        for column in avg_df_rl.columns:
            if column != "Timestep":
                print(f"Column: {column}")
                print(f"Mean: {avg_df_rl[column].mean()}")
                print(f"Max: {avg_df_rl[column].max()}")
                print(f"Min: {avg_df_rl[column].min()}")
                print("-" * 30)
                
    print("R² DataFrame:")
    print(r2_df_rl)
