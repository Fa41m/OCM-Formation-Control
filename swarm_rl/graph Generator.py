import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.cm as cm
import seaborn as sns

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

    # Define a consistent colormap for all graphs
    unique_robot_counts = sorted(reward_data.keys())
    colors = {num_robots: cm.tab10(idx % 10) for idx, num_robots in enumerate(unique_robot_counts)}

    for num_robots, df in reward_data.items():
        df = df.sort_values('Episode').reset_index(drop=True)
        df['Smoothed'] = df['Reward'].rolling(window=window, min_periods=1).mean()
        smoothed_data[num_robots] = df

    # === Split-view Plot ===
    fig, axs = plt.subplots(1, 2, figsize=(14, 8))  # No shared Y-axis
    
    fig.suptitle("PPO Episode Rewards", fontsize=20, weight='bold')

    for num_robots, df in sorted(smoothed_data.items()):
        color = colors[num_robots]
        label = f"{num_robots} robots"

        # Early training (Episodes 0–30)
        early = df[df['Episode'] <= 60]
        axs[0].plot(early['Episode'], early['Smoothed'], label=label, color=color, linewidth=2)

        # Late training (Episodes >30)
        late = df[df['Episode'] > 60]
        axs[1].plot(late['Episode'], late['Smoothed'], label=label, color=color, linewidth=2)

    # Axis labels and styling
    axs[0].set_title("Early Training (Episodes 0–60)", fontsize=18, weight='bold')
    axs[1].set_title("Late Training (Episodes 61–End)", fontsize=18, weight='bold') 
    for ax in axs:
        ax.grid(True, linestyle="--", alpha=0.5)

    axs[1].legend(fontsize=18, loc='best')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save plot
    output_path = os.path.join(output_dir, "episode_rewards_split_nostd.png")
    plt.savefig(output_path)
    plt.close()

    # Save legend as a separate image
    fig_legend = plt.figure(figsize=(4, 3))  # Increased size
    handles, labels = axs[1].get_legend_handles_labels()
    fig_legend.legend(handles, labels, title="Swarm Size", title_fontsize=20, fontsize=18, loc='center')
    legend_path = os.path.join(output_dir, "legend_swarm_sizes.png")
    fig_legend.savefig(legend_path)
    plt.close(fig_legend)

    print(f"✅ Smoothed reward plot (no std) saved to: {output_path}")
    print(f"✅ Legend saved to: {legend_path}")

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
    
    summary_records = []

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
            
        # Dataframe for Anova and Tukey 
        
        for run_idx, df in enumerate(dataframes):
            clean_df = df.dropna()

            total_cost = clean_df['TotalCost'].sum() if 'TotalCost' in clean_df.columns else np.nan
            collisions = clean_df['Collisions'].sum() if 'Collisions' in clean_df.columns else np.nan
            control_cost = clean_df['ControlCost'].sum() if 'ControlCost' in clean_df.columns else np.nan
            alignment_cost = clean_df['AlignmentCost'].sum() if 'AlignmentCost' in clean_df.columns else np.nan

            # R² Calculation
            if 'TotalCost' in clean_df.columns and 'Timestep' in clean_df.columns:
                X = clean_df['Timestep'].values.reshape(-1, 1)
                y = clean_df['TotalCost'].values

                if len(np.unique(X)) > 1:
                    model = LinearRegression().fit(X, y)
                    y_pred = model.predict(X)
                    r2 = r2_score(y, y_pred)
                else:
                    r2 = np.nan
            else:
                r2 = np.nan
                
            # AUC calculations (using trapezoidal rule)
            if 'ControlCost' in clean_df.columns:
                auc_control = np.trapz(clean_df['ControlCost'].values, clean_df['Timestep'].values)
            else:
                auc_control = np.nan

            if 'AlignmentCost' in clean_df.columns:
                auc_alignment = np.trapz(clean_df['AlignmentCost'].values, clean_df['Timestep'].values)
            else:
                auc_alignment = np.nan
                
            collisions_per_robot = collisions / num_robots if pd.notna(collisions) else np.nan

            summary_records.append({
                'swarm_size': num_robots,
                'run_id': run_idx + 1,
                'total_cost': total_cost,
                'collisions': collisions,
                'control_cost': control_cost,
                'alignment_cost': alignment_cost,
                'r2_total_cost': r2,
                'auc_control_cost': auc_control,
                'auc_alignment_cost': auc_alignment,
                'collisions_per_robot': collisions_per_robot
            })

    # Define a consistent colormap for all graphs
    unique_robot_counts = sorted(processed_data[next(iter(processed_data))].keys())
    colors = {num_robots: cm.tab10(idx % 10) for idx, num_robots in enumerate(unique_robot_counts)}
    # === Plot metric over Timestep ===
    for col_name, robot_data in processed_data.items():
        plt.figure(figsize=(12, 6))

        for num_robots, (timesteps, mean_vals, std_vals) in sorted(robot_data.items()):
            # Smooth
            mean_vals = pd.Series(mean_vals).rolling(50, min_periods=1).mean().values
            std_vals = pd.Series(std_vals).rolling(50, min_periods=1).mean().values

            plt.plot(timesteps, mean_vals, label=f"{num_robots} robots", color=colors[num_robots], linewidth=2)
            plt.fill_between(timesteps, mean_vals - std_vals, mean_vals + std_vals, color=colors[num_robots], alpha=0.2)

        plt.legend(loc="best", fontsize=18)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.95])  # Adjust rect to prevent cutoff
        plt.xlabel("Timestep [s]", fontsize=18, weight='bold')
        plt.ylabel(col_name, fontsize=18, weight='bold')
        plt.title(f"{col_name} vs Timestep for Different Robot Counts", weight='bold', fontsize=20)
        plt.savefig(os.path.join(output_dir, f"{col_name}.png"))
        plt.close()

    # === Plot R² values ===
    # plt.figure(figsize=(8, 5))
    # for num_robots, r2_list in sorted(r2_scores_by_robot.items()):
    #     mean_r2 = np.mean(r2_list)
    #     std_r2 = np.std(r2_list)
    #     plt.errorbar(num_robots, mean_r2, yerr=std_r2, fmt='o', capsize=5, color=colors[num_robots], label=f"{num_robots} robots")
    # plt.axhline(1.0, linestyle='--', color='gray', label='Perfect Fit (R² = 1)')
    # plt.grid(True, linestyle="--", alpha=0.5)
    # # plt.legend(title="Robot Count", fontsize=18)
    # plt.tight_layout()
    # plt.savefig(os.path.join(output_dir, "R2_ControlCost.png"))
    # plt.close()

    # print("✅ All RL combined and performance plots saved to:", output_dir)
    
    summary_df = pd.DataFrame(summary_records)
    
    unique_robot_counts = sorted(summary_df['swarm_size'].unique())
    colors = {num_robots: cm.tab10(idx % 10) for idx, num_robots in enumerate(unique_robot_counts)}

    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11
    })

    for metric in ['auc_control_cost', 'auc_alignment_cost', 'collisions_per_robot', 'r2_total_cost']:
        grouped = summary_df.groupby('swarm_size')[metric]
        means = grouped.mean()
        stds = grouped.std()

        swarm_sizes = means.index.values
        x = np.arange(len(swarm_sizes))
        colors_list = [colors[int(num_robots)] for num_robots in swarm_sizes]

        if metric == 'collisions_per_robot':
            plt.figure(figsize=(10, 6))

            summary_df['swarm_size'] = summary_df['swarm_size'].astype(str)  # Ensure matching types
            palette = {str(size): colors[int(size)] for size in summary_df['swarm_size'].unique()}



            sns.stripplot(
                x='swarm_size',
                y='collisions_per_robot',
                data=summary_df,
                palette=palette,
                size=8,
                jitter=True,
                edgecolor='black',
                linewidth=0.8
            )

            plt.xlabel("Swarm Size (Number of Robots)", weight='bold', fontsize=18)
            plt.ylabel("Collisions Per Robot", weight='bold', fontsize=18)
            plt.title("Collisions Per Robot per Swarm Size", weight='bold', fontsize=20)
            plt.grid(axis='y', linestyle='--', alpha=0.6)

            plot_filename = os.path.join(output_dir, f"{metric}_stripplot.png")
            plt.tight_layout()
            plt.savefig(plot_filename, dpi=300)
            plt.close()


        else:
            # Use normal plotting for all other metrics
            plt.figure(figsize=(10, 6))
            for i, (xi, mean, std, color, size) in enumerate(zip(x, means.values, stds.values, colors_list, swarm_sizes)):
                plt.errorbar(
                    xi,
                    mean,
                    yerr=std,
                    fmt='o',
                    color=color,
                    ecolor=color,
                    elinewidth=2,
                    capsize=6,
                    capthick=1.5,
                    markersize=8,
                    linestyle='none',
                    zorder=3,
                    label=f'{size} robots'
                )

            plt.xticks(x, swarm_sizes)
            plt.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
            plt.xlabel("Swarm Size (Number of Robots)", weight='bold', fontsize=18)
            plt.ylabel(metric.replace('_', ' ').title(), weight='bold', fontsize=18)
            plt.title(f"{metric.replace('_', ' ').title()} per Swarm Size", weight='bold', fontsize=20)
            plt.legend(fontsize=18, loc='best')

            plot_filename = os.path.join(output_dir, f"{metric}_errorbar_plot.png")
            plt.tight_layout()
            plt.savefig(plot_filename, dpi=300)
            plt.close()
        
    metrics_to_test = ['total_cost', 'collisions', 'auc_control_cost', 'auc_alignment_cost', 'r2_total_cost']
    
    # save the summary dataframe to a CSV file
    summary_csv_path = os.path.join(output_dir, "summary_statistics.csv")
    summary_df.to_csv(summary_csv_path, index=False)

    # Create a CSV file with the average of each metric for each swarm size
    avg_metrics = summary_df.groupby('swarm_size').mean().reset_index()
    avg_csv_path = os.path.join(output_dir, "average_metrics_per_swarm_size.csv")
    avg_metrics.to_csv(avg_csv_path, index=False)
    print(f"📄 Average metrics per swarm size saved to {avg_csv_path}")
    print(f"📄 Summary statistics saved to {summary_csv_path}")


    for metric in metrics_to_test:
        print(f"\n===== Analyzing {metric} =====")
        run_anova_and_tukey(
            df_long=summary_df.rename(columns={'swarm_size': 'group'}),  # rename for formula compatibility
            metric_col=metric,
            group_col='group',
            output_dir=output_dir
        )


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
    summary_records = []

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
           
        # Dataframe for Anova and Tukey 
        
        for run_idx, df in enumerate(dataframes):
            clean_df = df.dropna()

            total_cost = clean_df['total_cost'].sum() if 'total_cost' in clean_df.columns else np.nan
            collisions = clean_df['collisions'].sum() if 'collisions' in clean_df.columns else np.nan
            control_cost = clean_df['control_cost'].sum() if 'control_cost' in clean_df.columns else np.nan
            alignment_cost = clean_df['alignment_cost'].sum() if 'alignment_cost' in clean_df.columns else np.nan

            # R² Calculation
            if 'total_cost' in clean_df.columns and 'timestep' in clean_df.columns:
                X = clean_df['timestep'].values.reshape(-1, 1)
                y = clean_df['total_cost'].values
                if len(np.unique(X)) > 1:
                    model = LinearRegression().fit(X, y)
                    y_pred = model.predict(X)
                    r2 = r2_score(y, y_pred)
                else:
                    r2 = np.nan
            else:
                r2 = np.nan
                
            # AUC calculations (using trapezoidal rule)
            if 'control_cost' in clean_df.columns:
                auc_control = np.trapz(clean_df['control_cost'].values, clean_df['timestep'].values)
            else:
                auc_control = np.nan

            if 'alignment_cost' in clean_df.columns:
                auc_alignment = np.trapz(clean_df['alignment_cost'].values, clean_df['timestep'].values)
            else:
                auc_alignment = np.nan

            collisions_per_robot = collisions / num_robots if pd.notna(collisions) else np.nan

            summary_records.append({
                'swarm_size': num_robots,
                'run_id': run_idx + 1,
                'total_cost': total_cost,
                'collisions': collisions,
                'control_cost': control_cost,
                'alignment_cost': alignment_cost,
                'r2_total_cost': r2,
                'auc_control_cost': auc_control,
                'auc_alignment_cost': auc_alignment,
                'collisions_per_robot': collisions_per_robot
            })

    # Define a consistent colormap for all graphs
    unique_robot_counts = sorted(processed_data[next(iter(processed_data))].keys())
    colors = {num_robots: cm.tab10(idx % 10) for idx, num_robots in enumerate(unique_robot_counts)}

    # === Plot metrics over timestep (with smoothing & styling) ===
    for col_name, robot_data in processed_data.items():
        plt.figure(figsize=(12, 6))

        for num_robots, (timesteps, mean_vals, std_vals) in sorted(robot_data.items()):
            mean_vals = pd.Series(mean_vals).rolling(window=50, min_periods=1).mean().values
            std_vals = pd.Series(std_vals).rolling(window=50, min_periods=1).mean().values

            plt.plot(timesteps, mean_vals, label=f"{num_robots} robots", color=colors[num_robots], linewidth=2)
            plt.fill_between(timesteps, mean_vals - std_vals, mean_vals + std_vals,
                             color=colors[num_robots], alpha=0.2)

        plt.legend(loc="best", fontsize=18)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.95])  # Adjust rect to prevent cutoff
        plt.xlabel("Timestep [s]", fontsize=18, weight='bold')
        plt.ylabel(col_name, fontsize=18, weight='bold')
        plt.title(f"{col_name} vs Timestep for Different Robot Counts", weight='bold', fontsize=20)
        plt.savefig(os.path.join(output_dir, f"{col_name}.png"))
        plt.close()

    # === R² plot ===
    plt.figure(figsize=(8, 5))
    for num_robots, r2_list in sorted(r2_scores_by_robot.items()):
        mean_r2 = np.mean(r2_list)
        std_r2 = np.std(r2_list)
        plt.errorbar(num_robots, mean_r2, yerr=std_r2, fmt='o', capsize=5, color=colors[num_robots], label=f"{num_robots} robots")
    plt.axhline(1.0, linestyle='--', color='gray', label='Perfect Fit (R² = 1)')
    plt.grid(True, linestyle="--", alpha=0.5)
    # plt.legend(fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "R2_ControlCost.png"))
    plt.close()

    print("✅ All PSO combined and performance plots saved to:", output_dir)
    
    summary_df = pd.DataFrame(summary_records)

    unique_robot_counts = sorted(summary_df['swarm_size'].unique())
    colors = {num_robots: cm.tab10(idx % 10) for idx, num_robots in enumerate(unique_robot_counts)}

    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11
    })

    for metric in ['auc_control_cost', 'auc_alignment_cost', 'collisions_per_robot', 'r2_total_cost']:
        grouped = summary_df.groupby('swarm_size')[metric]
        means = grouped.mean()
        stds = grouped.std()

        swarm_sizes = means.index.values
        x = np.arange(len(swarm_sizes))
        colors_list = [colors[int(num_robots)] for num_robots in swarm_sizes]

        if metric == 'collisions_per_robot':
            plt.figure(figsize=(10, 6))

            summary_df['swarm_size'] = summary_df['swarm_size'].astype(str)  # Ensure matching types
            palette = {str(size): colors[int(size)] for size in summary_df['swarm_size'].unique()}



            sns.stripplot(
                x='swarm_size',
                y='collisions_per_robot',
                data=summary_df,
                palette=palette,
                size=8,
                jitter=True,
                edgecolor='black',
                linewidth=0.8
            )

            plt.xlabel("Swarm Size (Number of Robots)", weight='bold', fontsize=18)
            plt.ylabel("Collisions Per Robot", weight='bold', fontsize=18)
            plt.title("Collisions Per Robot per Swarm Size", weight='bold', fontsize=20)
            plt.grid(axis='y', linestyle='--', alpha=0.6)

            plot_filename = os.path.join(output_dir, f"{metric}_stripplot.png")
            plt.tight_layout()
            plt.savefig(plot_filename, dpi=300)
            plt.close()


        else:
            # Use normal plotting for all other metrics
            plt.figure(figsize=(10, 6))
            for i, (xi, mean, std, color, size) in enumerate(zip(x, means.values, stds.values, colors_list, swarm_sizes)):
                plt.errorbar(
                    xi,
                    mean,
                    yerr=std,
                    fmt='o',
                    color=color,
                    ecolor=color,
                    elinewidth=2,
                    capsize=6,
                    capthick=1.5,
                    markersize=8,
                    linestyle='none',
                    zorder=3,
                    label=f'{size} robots'
                )

            plt.xticks(x, swarm_sizes)
            plt.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
            plt.xlabel("Swarm Size (Number of Robots)", weight='bold', fontsize=18)
            plt.ylabel(metric.replace('_', ' ').title(), weight='bold', fontsize=18)
            plt.title(f"{metric.replace('_', ' ').title()} per Swarm Size", weight='bold', fontsize=20)
            plt.legend(fontsize=18, loc='best')

            plot_filename = os.path.join(output_dir, f"{metric}_errorbar_plot.png")
            plt.tight_layout()
            plt.savefig(plot_filename, dpi=300)
            plt.close()
    
    metrics_to_test = ['total_cost', 'collisions', 'auc_control_cost', 'auc_alignment_cost', 'r2_total_cost']
    
    # save the summary dataframe to a CSV file
    summary_csv_path = os.path.join(output_dir, "summary_statistics.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    avg_metrics = summary_df.groupby('swarm_size').mean().reset_index()
    avg_csv_path = os.path.join(output_dir, "average_metrics_per_swarm_size.csv")
    avg_metrics.to_csv(avg_csv_path, index=False)
    print(f"📄 Summary statistics saved to {summary_csv_path}")

    for metric in metrics_to_test:
        print(f"\n===== Analyzing {metric} =====")
        run_anova_and_tukey(
            df_long=summary_df.rename(columns={'swarm_size': 'group'}),  # rename for formula compatibility
            metric_col=metric,
            group_col='group',
            output_dir=output_dir
        )
    
    
    
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

import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
from statsmodels.formula.api import ols
def run_anova_and_tukey(df_long, metric_col="value", group_col="group", alpha=0.05, output_dir="./"):
    model = ols(f"{metric_col} ~ C({group_col})", data=df_long).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    # Save ANOVA results
    # anova_csv_path = os.path.join(output_dir, f"anova_{metric_col}.csv")
    safe_name = metric_col.replace(" ", "_").replace("/", "_")

    anova_csv_path = os.path.join(output_dir, f"anova_{safe_name}.csv")
    anova_table.to_csv(anova_csv_path)
    print(f"📄 ANOVA result saved to {anova_csv_path}")

    p_value = anova_table["PR(>F)"][0]
    if p_value < alpha:
        tukey = pairwise_tukeyhsd(endog=df_long[metric_col], groups=df_long[group_col], alpha=alpha)
        tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])

        # Save Tukey results
        tukey_csv_path = os.path.join(output_dir, f"tukey_{safe_name}.csv")
        tukey_df.to_csv(tukey_csv_path, index=False)
        print(f"✅ Significant differences! Tukey HSD saved to {tukey_csv_path}")
    else:
        print(f"❌ No significant difference in '{metric_col}' (p = {p_value:.4f})")
        
def build_tables_pso():
    import pandas as pd
    import dataframe_image as dfi
    input_dir = "./Data/PSO/obstacle_level_4/plots_combined"
    # Define file paths
    anova_files = {
        'AUC Alignment Cost': os.path.join(input_dir, 'anova_auc_alignment_cost.csv'),
        'AUC Control Cost': os.path.join(input_dir, 'anova_auc_control_cost.csv'),
        'Collisions': os.path.join(input_dir, 'anova_collisions.csv'),
        'R2 Total Cost': os.path.join(input_dir, 'anova_r2_total_cost.csv'),
        'Total Cost': os.path.join(input_dir, 'anova_total_cost.csv')
    }

    tukey_files = {
        'AUC Alignment Cost': os.path.join(input_dir, 'tukey_auc_alignment_cost.csv'),
        'AUC Control Cost': os.path.join(input_dir, 'tukey_auc_control_cost.csv'),
        'Collisions': os.path.join(input_dir, 'tukey_collisions.csv'),
        'Total Cost': os.path.join(input_dir, 'tukey_total_cost.csv'),
    }

    # Build ANOVA summary table
    anova_summary = []
    for metric, path in anova_files.items():
        df = pd.read_csv(path)
        row = df[df['Unnamed: 0'] == 'C(group)'].iloc[0]
        F = row['F']
        p = row['PR(>F)']
        anova_summary.append({
            'Metric': metric,
            'F value': round(F, 3),
            'p value': f"{p:.4f}" if p >= 0.0001 else "<0.0001",
            'Null Hypothesis': f"Accepts" if p >= 0.05 else "Rejects"
        })
    anova_df = pd.DataFrame(anova_summary, index=None)

    # Build Tukey HSD summary table (significant comparisons only)
    tukey_summary = []
    for metric, path in tukey_files.items():
        df = pd.read_csv(path)
        sigs = df[df['reject'] == True]
        # Check for "all pairwise"
        total_pairs = df.shape[0]
        # if sigs.shape[0] == total_pairs:
        #     tukey_summary.append({
        #         'Metric': metric,
        #         'Comparison': 'All pairwise',
        #         'Mean Difference': '',
        #         'Adj. p value': '',
        #         'Direction': ''
        #     })
        # else:
        for _, r in sigs.iterrows():
            comp = f"{int(r['group1'])} vs {int(r['group2'])}"
            md = round(r['meandiff'], 4)
            padj = f"{r['p-adj']:.4f}" if r['p-adj'] >= 0.0001 else "<0.0001"
            direction = 'group {} > {}'.format(int(r['group1']), int(r['group2'])) if r['meandiff'] > 0 else 'group {} < {}'.format(int(r['group1']), int(r['group2']))
            tukey_summary.append({
                'Metric': metric,
                'Comparison': comp,
                'Mean Difference': md,
                'Adj. p value': padj,
                'Direction': direction
            })
    tukey_df = pd.DataFrame(tukey_summary, index=None)

    # Save the tables as images

    anova_image_path = os.path.join(input_dir, "anova_summary.png")
    tukey_image_path = os.path.join(input_dir, "tukey_summary.png")

    dfi.export(anova_df.style.hide(axis="index"), anova_image_path)
    dfi.export(tukey_df.style.hide(axis="index"), tukey_image_path)


    print(f"📊 ANOVA summary saved as image: {anova_image_path}")
    print(f"📊 Tukey HSD summary saved as image: {tukey_image_path}")

def build_tables_ppo():
    import pandas as pd
    import dataframe_image as dfi
    input_dir = "./Data/RL/obstacle_level_4/plots_combined"
    # Define file paths
    anova_files = {
        'AUC Alignment Cost': os.path.join(input_dir, 'anova_auc_alignment_cost.csv'),
        'AUC Control Cost': os.path.join(input_dir, 'anova_auc_control_cost.csv'),
        'Collisions': os.path.join(input_dir, 'anova_collisions.csv'),
        'R2 Total Cost': os.path.join(input_dir, 'anova_r2_total_cost.csv'),
        'Total Cost': os.path.join(input_dir, 'anova_total_cost.csv')
    }

    tukey_files = {
        'AUC Alignment Cost': os.path.join(input_dir, 'tukey_auc_alignment_cost.csv'),
        'AUC Control Cost': os.path.join(input_dir, 'tukey_auc_control_cost.csv'),
        'Collisions': os.path.join(input_dir, 'tukey_collisions.csv'),
        'Total Cost': os.path.join(input_dir, 'tukey_total_cost.csv'),
    }

    # Build ANOVA summary table
    anova_summary = []
    for metric, path in anova_files.items():
        df = pd.read_csv(path)
        row = df[df['Unnamed: 0'] == 'C(group)'].iloc[0]
        F = row['F']
        p = row['PR(>F)']
        anova_summary.append({
            'Metric': metric,
            'F value': round(F, 3),
            'p value': f"{p:.4f}" if p >= 0.0001 else "<0.0001",
            'Null Hypothesis': f"Accepts" if p >= 0.05 else "Rejects"
        })
    anova_df = pd.DataFrame(anova_summary, index=None)

    # Build Tukey HSD summary table (significant comparisons only)
    tukey_summary = []
    for metric, path in tukey_files.items():
        df = pd.read_csv(path)
        sigs = df[df['reject'] == True]
        # Check for "all pairwise"
        # total_pairs = df.shape[0]
        # if sigs.shape[0] == total_pairs:
        #     tukey_summary.append({
        #         'Metric': metric,
        #         'Comparison': 'All pairwise',
        #         'Mean Difference': '',
        #         'Adj. p value': '',
        #         'Direction': ''
        #     })
        # else:
        for _, r in sigs.iterrows():
            comp = f"{int(r['group1'])} vs {int(r['group2'])}"
            md = round(r['meandiff'], 4)
            padj = f"{r['p-adj']:.4f}" if r['p-adj'] >= 0.0001 else "<0.0001"
            direction = 'group {} > {}'.format(int(r['group1']), int(r['group2'])) if r['meandiff'] > 0 else 'group {} < {}'.format(int(r['group1']), int(r['group2']))
            tukey_summary.append({
                'Metric': metric,
                'Comparison': comp,
                'Mean Difference': md,
                'Adj. p value': padj,
                'Direction': direction
            })
    tukey_df = pd.DataFrame(tukey_summary, index=None)

    # Save the tables as images

    anova_image_path = os.path.join(input_dir, "anova_summary.png")
    tukey_image_path = os.path.join(input_dir, "tukey_summary.png")

    dfi.export(anova_df.style.hide(axis="index"), anova_image_path)
    dfi.export(tukey_df.style.hide(axis="index"), tukey_image_path)


    print(f"📊 ANOVA summary saved as image: {anova_image_path}")
    print(f"📊 Tukey HSD summary saved as image: {tukey_image_path}")


if __name__ == "__main__":
    # rl_graphs()
    pso_graphs()
    
    # build_tables_pso()
    # build_tables_ppo()
    
    # print("PPO Data for swarm size 20")
    # print("========================================")
    # avg_df, r2_df = get_avg_pso_data()
    # if not avg_df.empty:
    #     for column in avg_df.columns:
    #         if column != "Timestep":
    #             print(f"Column: {column}")
    #             print(f"Mean: {avg_df[column].mean()}")
    #             print(f"Max: {avg_df[column].max()}")
    #             print(f"Min: {avg_df[column].min()}")
    #             print("-" * 30)
                
    # print("R² DataFrame:")
    # print(r2_df)
    
    # print("PPO Data for swarm size 15")
    # print("========================================")
    # avg_df_rl, r2_df_rl = get_avg_rl_data()
    # if not avg_df_rl.empty:
    #     for column in avg_df_rl.columns:
    #         if column != "Timestep":
    #             print(f"Column: {column}")
    #             print(f"Mean: {avg_df_rl[column].mean()}")
    #             print(f"Max: {avg_df_rl[column].max()}")
    #             print(f"Min: {avg_df_rl[column].min()}")
    #             print("-" * 30)
                
    # print("R² DataFrame:")
    # print(r2_df_rl)
