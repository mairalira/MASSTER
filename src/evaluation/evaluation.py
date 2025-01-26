import pandas as pd
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from autorank import autorank, latex_report, plot_stats

# Absolute path using Path
project_root = Path(__file__).resolve().parent.parent
# Adding path to sys.path   
sys.path.append(str(project_root))

from config import *

class Evaluator:
    def __init__(self, dataset_name, metric_name, iterations, folds):
        self.dataset_name = dataset_name
        self.metric_name = metric_name
        self.iterations = iterations
        self.folds = folds
        self.all_methods = ['target_qbc', 'masster_cotraining', 'masster_self_learning', 'pct']

    # method is in self.all_methods
    def generate_path(self, fold_number, method):
        if method == 'target_qbc':
            type = 'active_learning'
            file_path = Path(f'reports\{type}\{self.dataset_name}\{method}_results_fold_{fold_number}.csv')
        if method == 'masster_cotraining' or method == 'masster_self_learning':
            type = 'proposed_method'
            file_path = Path(f'reports\{type}\{self.dataset_name}\{method}_fold_{fold_number}.csv')
        if method == 'pct':
            type = 'semi_supervised_learning'
            file_path = Path(f'reports\{type}\{self.dataset_name}\{method}_results_fold_{fold_number}.csv')
        return file_path
    
    def read_clean_dataframe(self, file_path):
        if file_path.exists():
            df = pd.read_csv(file_path, index_col=0)
            iteration_list = list(range(self.iterations)) 
            df = df[df['Iterations'].isin(iteration_list)]
        else:
            print(file_path)
        return df
    
    def concate_dataframes(self, method):
        self.full_df = pd.DataFrame()
        for fold_number in list(range(self.folds)):
            file_path = self.generate_path(fold_number, method)
            df = self.read_clean_dataframe(file_path)

            self.full_df = pd.concat([self.full_df, df], axis=0)
        
        self.full_df['Method'] = method

        return self.full_df #by method
    
    def compute_auc(self, method):
        full_df = self.full_df # by method
        auc_df = pd.DataFrame(index=full_df.index.unique(), columns=['R2', 'MSE', 'MAE', 'CA', 'ARRMSE'])

        for fold_index in full_df.index.unique():
            fold_df = full_df.loc[fold_index]
            for metric in auc_df.columns:
                x = fold_df['Iterations']
                y = fold_df[metric]
                auc_value = auc(x, y)
                auc_df.loc[fold_index, metric] = auc_value

        auc_df['Method'] = method
        self.auc_df = auc_df
        return self.auc_df
    
    def compile_methods(self):
        # for each metric
        resume_auc = pd.DataFrame()
        for unique_method in self.all_methods:
            self.method = unique_method
            full_df = self.concate_dataframes(unique_method)
            auc_df = self.compute_auc(unique_method)
            resume_auc[unique_method] = auc_df[self.metric_name]
            self.resume_auc = resume_auc

        return self.resume_auc
    
    def save_reports(self):
        output_path_auc = Path(f'reports/paper_evaluation/{self.dataset_name}/resume_auc_{self.metric_name}.csv')
        output_path_auc.parent.mkdir(parents=True, exist_ok=True) 
        self.resume_auc.to_csv(output_path_auc)

        description_auc_df = self.resume_auc.describe()
        description_auc_path = Path(f'reports/paper_evaluation/{self.dataset_name}/resume_auc_{self.metric_name}_description.csv')
        description_auc_df.to_csv(description_auc_path)

    def run_autorank(self):
        self.resume_auc = self.resume_auc.apply(pd.to_numeric, errors='coerce')
        self.resume_auc.dropna(inplace=True)
        self.resume_auc.reset_index(drop=True, inplace=True)
        
        result = autorank(self.resume_auc, alpha=0.05, verbose=False)
        report_path = Path(f'reports/paper_evaluation/{self.dataset_name}/{self.metric_name}_autorank_report_auc.txt')
        with open(report_path, 'w') as report_file:
            old_stdout = sys.stdout
            sys.stdout = report_file
            latex_report(result, decimal_places=3, complete_document=False)
            sys.stdout = old_stdout
        ax = plot_stats(result, allow_insignificant=True) 
        fig = ax.get_figure()
        plot_path = Path(f'reports/paper_evaluation/{self.dataset_name}/{self.metric_name}_autorank_plot_auc.png')
        fig.savefig(plot_path)
        plt.close(fig) 
        plt.close('all')

    def generate_subplot_image(self, dataset_names, metric_names):
        legend_labels = {
            'target_qbc': 'Active Learning',
            'masster_cotraining': 'MASSTER - CT',
            'masster_self_learning': 'MASSTER - SL',
            'pct': 'SSL - PCT',
        }
        fig, axes = plt.subplots(len(metric_names), len(dataset_names), figsize=(5*len(dataset_names), 4*len(metric_names)), sharex=True)
        fig.subplots_adjust(bottom=0.15, top = 0.95)
    
        for i, dataset in enumerate(dataset_names):
            for j, metric in enumerate(metric_names):
                ax = axes[j, i]
                for method in self.methods:
                    file_path = Path(f'reports/active_learning/{dataset}/{metric}/{method}_{metric}.csv')
                    if file_path.exists():
                        df = pd.read_csv(file_path, index_col=0)
                        if 'AUC' in df.columns:
                            df = df.drop(columns=['AUC'])
                        if 'Average' in df.index:
                            mean_values = df.loc['Average']
                            ax.plot(range(1, len(mean_values) + 1), mean_values, label=legend_labels[method])
                    
                if i == 0:
                    ax.set_ylabel(metric, fontsize = 18)
                if j == 0:
                    ax.set_title(dataset, fontsize = 18)
                if j == 1:
                    ax.set_xlabel('Epoch', fontsize = 14)
        
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=len(self.methods), fontsize = 18)
        plt.savefig('reports/active_learning/summary_subplot_image.png')
        plt.close(fig)



def run_reports(dataset_names, metric_names, iterations, folds):
    for dataset in dataset_names:
        for metric in metric_names:
            evaluator = Evaluator(dataset, metric, iterations, folds)
            resume_auc = evaluator.compile_methods()
            evaluator.save_reports()
            evaluator.run_autorank()


iterations = ITERATIONS
folds = K_FOLDS
dataset_names = ['atp7d']
metric_names = ['ARRMSE', 'CA', 'MAE', 'MSE', 'R2'] 
run_reports(dataset_names, metric_names, iterations, folds)