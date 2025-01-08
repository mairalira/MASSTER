import pandas as pd
import os
import sys
from pathlib import Path
from autorank import autorank, latex_report, plot_stats

# Absolute path using Path
project_root = Path(__file__).resolve().parent.parent
# Adding path to sys.path   
sys.path.append(str(project_root))

from config import *

class ActiveLearningEvaluator:
    def __init__(self, dataset_name, metric_name, n_epochs):
        self.dataset_name = dataset_name
        self.metric_name = metric_name
        self.n_epochs = n_epochs
        self.methods = ['greedy', 'instance', 'qbcrf', 'random', 'rtal', 'upperbound']
        self.iterations = iterations

    def assemble_auc(self):
        auc_data = {method: [] for method in self.methods}
        
        for method in self.methods:
            file_path = Path(f'reports/active_learning/{self.dataset_name}/{self.metric_name}/{method}_{self.metric_name}.csv')
            
            if file_path.exists():
                df = pd.read_csv(file_path, index_col=0)
                if 'AUC' in df.columns:
                    auc_values = df['AUC'].values[:self.iterations]
                    auc_data[method] = auc_values
                else:
                    auc_data[method] = [None] * self.iterations
            else:
                auc_data[method] = [None] * self.iterations
        
        index = [f'Iteration {i+1}' for i in range(self.iterations)]
        self.auc_df = pd.DataFrame(auc_data, index=index)
        
        return self.auc_df
    
    def assemble_reports(self):
        data = {method: [] for method in self.methods}
        
        for method in self.methods:
            file_path = Path(f'reports/active_learning/{self.dataset_name}/{self.metric_name}/{method}_{self.metric_name}.csv')
            
            if file_path.exists():
                df = pd.read_csv(file_path, index_col=0)
                averages = df.loc['Average']
                data[method] = averages.values[:self.n_epochs]
            else:
                data[method] = [None] * self.n_epochs
        
        index = [f'Epoch {i+1}' for i in range(self.n_epochs)]
        self.average_iterations_df = pd.DataFrame(data, index=index)
        
        if 'AUC' in self.average_iterations_df.index:
            self.average_iterations_df = self.average_iterations_df.drop('AUC')
        
        return self.average_iterations_df
    
    def summarize_iterations(self):
        summary_data = {method: [] for method in self.methods}
        
        for method in self.methods:
            file_path = Path(f'reports/active_learning/{self.dataset_name}/{self.metric_name}/{method}_{self.metric_name}.csv')
            if file_path.exists():
                df = pd.read_csv(file_path, index_col=0)
                if 'AUC' in df.columns:
                    df = df.drop(columns=['AUC'])
                for i in range(self.iterations):
                    iteration_column = f'Iteration {i+1}'
                    if iteration_column in df.index:
                        summary_data[method].append(df.loc[iteration_column].mean())
                    else:
                        summary_data[method].append(None)
            else:
                summary_data[method] = [None] * self.iterations

        index = [f'Iteration {i+1}' for i in range(self.iterations)]
        self.average_epochs_df = pd.DataFrame(summary_data, index=index)
        return self.average_epochs_df
    
    def save_reports(self):
        output_path_auc = Path(f'reports/active_learning/{self.dataset_name}/{self.metric_name}/resume_auc.csv')
        self.auc_df.to_csv(output_path_auc)

        output_path_iterations = Path(f'reports/active_learning/{self.dataset_name}/{self.metric_name}/resume_iterations.csv')
        self.average_iterations_df.to_csv(output_path_iterations)

        output_path_epochs = Path(f'reports/active_learning/{self.dataset_name}/{self.metric_name}/resume_epochs.csv')
        self.average_epochs_df.to_csv(output_path_epochs)

        description_auc_df = self.auc_df.describe()
        description_auc_path = Path(f'reports/active_learning/{self.dataset_name}/{self.metric_name}/resume_auc_description.csv')
        description_auc_df.to_csv(description_auc_path)
        
        description_df = self.average_iterations_df.describe()
        description_path = Path(f'reports/active_learning/{self.dataset_name}/{self.metric_name}/resume_iterations_description.csv')
        description_df.to_csv(description_path)

        description_epochs_df = self.average_epochs_df.describe()
        description_epochs_path = Path(f'reports/active_learning/{self.dataset_name}/{self.metric_name}/resume_epochs_description.csv')
        description_epochs_df.to_csv(description_epochs_path)

    def save_summary_metrics(self, typing):
        if typing == 'auc':
            df_type = self.auc_df
        if typing == 'iterations':
            df_type = self.average_iterations_df
        if typing == 'epochs':
            df_type = self.average_epochs_df

        summary_data = {method: [] for method in self.methods}
        
        for method in self.methods:
            mean_values = []
            std_values = []
            for dataset in dataset_names:
                description_path = Path(f'reports/active_learning/{dataset}/{self.metric_name}/resume_{typing}_description.csv')
                if description_path.exists():
                    description_df = pd.read_csv(description_path, index_col=0)
                    mean_value = description_df.at['mean', method]
                    std_value = description_df.at['std', method]
                    mean_values.append(f"{mean_value:.3f} +/- {std_value:.3f}")
                else:
                    print(f"File not found: {description_path}")  
                    mean_values.append(None)
            summary_data[method] = mean_values

        summary_df = pd.DataFrame(summary_data, index=dataset_names)
        output_path = Path(f'reports/active_learning/summary_{typing}_{self.metric_name}.csv')
        summary_df.to_csv(output_path)

    def run_autorank(self, typing):
        if typing == 'auc':
            df_type = self.auc_df
        if typing == 'iterations':
            df_type = self.average_iterations_df
        if typing == 'epochs':
            df_type = self.average_epochs_df
        result = autorank(df_type, alpha=0.05, verbose=False)
        report_path = Path(f'reports/active_learning/{self.dataset_name}/{self.metric_name}/autorank_report_{typing}.txt')
        with open(report_path, 'w') as report_file:
            old_stdout = sys.stdout
            sys.stdout = report_file
            latex_report(result, decimal_places=3, complete_document=False)
            sys.stdout = old_stdout
        ax = plot_stats(result, allow_insignificant=True)  # Allow insignificant results
        fig = ax.get_figure()
        plot_path = Path(f'reports/active_learning/{self.dataset_name}/{self.metric_name}/autorank_plot_{typing}.png')
        fig.savefig(plot_path)

def run_reports(dataset_names, metric_names):
    for dataset in dataset_names:
        for metric in metric_names:
            evaluator = ActiveLearningEvaluator(dataset, metric, n_epochs)
            auc_df = evaluator.assemble_auc()
            average_iterations_df = evaluator.assemble_reports()
            average_epochs_df = evaluator.summarize_iterations()
            evaluator.save_reports()
            evaluator.run_autorank('auc')
            evaluator.save_summary_metrics('auc')
            evaluator.run_autorank('iterations')
            evaluator.save_summary_metrics('iterations')
            evaluator.run_autorank('epochs')
            evaluator.save_summary_metrics('epochs')
            

n_epochs = N_EPOCHS
iterations = ITERATIONS
dataset_names = ['atp7d', 'friedman', 'mp5spec', 'musicOrigin2', 'oes97']
metric_names = ['arrmse', 'ca', 'mae', 'mse', 'r2'] 
run_reports(dataset_names, metric_names)
