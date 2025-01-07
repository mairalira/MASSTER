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
        self.average_df = pd.DataFrame(data, index=index)
        
        if 'AUC' in self.average_df.index:
            self.average_df = self.average_df.drop('AUC')
        
        return self.average_df
    
    def save_reports(self):
        output_path = Path(f'reports/active_learning/{self.dataset_name}/{self.metric_name}/resume.csv')
        self.average_df.to_csv(output_path)
        
        description_df = self.average_df.describe()
        description_path = Path(f'reports/active_learning/{self.dataset_name}/{self.metric_name}/resume_description.csv')
        description_df.to_csv(description_path)

    def run_autorank(self):
        result = autorank(self.average_df, alpha=0.05, verbose=False)
        report_path = Path(f'reports/active_learning/{self.dataset_name}/{self.metric_name}/autorank_report.txt')
        with open(report_path, 'w') as report_file:
            old_stdout = sys.stdout
            sys.stdout = report_file
            latex_report(result, decimal_places=3, complete_document=False)
            sys.stdout = old_stdout
        ax = plot_stats(result)
        fig = ax.get_figure()
        plot_path = Path(f'reports/active_learning/{self.dataset_name}/{self.metric_name}/autorank_plot.png')
        fig.savefig(plot_path)

# Example usage
#dataset_name = 'atp7d'
#metric_name = 'arrmse'
n_epochs = N_EPOCHS

#evaluator = ActiveLearningEvaluator(dataset_name, metric_name, n_epochs)
#average_df = evaluator.assemble_reports()
#evaluator.save_reports()
#evaluator.run_autorank()

def run_reports(dataset_names, metric_names):
    for dataset in dataset_names:
        for metric in metric_names:
            evaluator = ActiveLearningEvaluator(dataset, metric, n_epochs)
            average_df = evaluator.assemble_reports()
            evaluator.save_reports()
            evaluator.run_autorank()

dataset_names = ['atp7d', 'friedman', 'mp5spec', 'musicOrigin2', 'oes97']
metric_names = ['arrmse', 'ca', 'mae', 'mse', 'r2'] 
run_reports(dataset_names, metric_names)
