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
        self.folds = 5 #folds
        self.all_methods = ['target_qbc', 'masster_cotraining', 'masster_self_learning', 'self_learning', 'cotraining']

    # method is in self.all_methods
    def generate_path(self, fold_number, method):
        if method == 'target_qbc':
            type = 'active_learning'
            file_path = Path(f'reports\{type}\{self.dataset_name}\{method}_results_fold_{fold_number}.csv')
        if method == 'masster_cotraining' or method == 'masster_self_learning':
            type = 'proposed_method'
            file_path = Path(f'reports\{type}\{self.dataset_name}\{method}_fold_{fold_number}.csv')
        if method == 'self_learning':
            type = 'semi_supervised_learning'
            file_path = Path(f'reports\{type}\{self.dataset_name}\\target_{method}_results_fold_{fold_number}.csv')
        if method == 'cotraining':
            type = 'semi_supervised_learning'
            file_path = Path(f'reports\{type}\{self.dataset_name}\\target_{method}_results_fold_{fold_number}.csv')
        if method == 'pct':
            type = 'pct'
            file_path = Path(f'reports\{type}\{self.dataset_name}\{method}_fold_{fold_number}.csv')
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
    
    def calculate_auc(self,fold_df, metrics):
        auc_values = {}
        for metric in metrics:
            x = fold_df['Iterations']
            y = fold_df[metric]
            auc_value = auc(x, y)  # Calcula a AUC
            auc_values[metric] = auc_value
        return auc_values
    
    def normalize_auc(self,auc_df):
        for metric in auc_df.columns:
            mean_value = auc_df[metric].mean()
            std_dev = auc_df[metric].std()
            if std_dev > 0:
                auc_df[metric] = (auc_df[metric] - mean_value) / std_dev
            else:
                auc_df[metric] = 0
        return auc_df
    
    def process_and_save_auc(self, method):
        full_df = self.full_df 
        metrics = ['R2', 'MSE', 'MAE', 'CA', 'ARRMSE']
        auc_df = pd.DataFrame(index=full_df.index.unique(), columns=metrics)

        for fold_index in full_df.index.unique():
            fold_df = full_df.loc[fold_index]
            auc_values = self.calculate_auc(fold_df, metrics)
            for metric, value in auc_values.items():
                auc_df.loc[fold_index, metric] = value

        auc_df = self.normalize_auc(auc_df)
        auc_df['Method'] = method

        # Concatenate the new auc_df with the existing auc_complete_df
        if hasattr(self, 'auc_complete_df'):
            self.auc_complete_df = pd.concat([self.auc_complete_df, auc_df], axis=0)
        else:
            self.auc_complete_df = auc_df

        # Save the complete DataFrame to a CSV file
        output_dir = Path(f'reports/paper_evaluation/{self.dataset_name}')
        output_dir.mkdir(parents=True, exist_ok=True)
        complete_output_path = Path(f'reports/paper_evaluation/{self.dataset_name}/full_auc_{self.metric_name}.csv')
        self.auc_complete_df.to_csv(complete_output_path, index=True)

        # Generate separate DataFrames for each metric and save them
        for metric in metrics:
            metric_df = self.auc_complete_df.pivot(index=self.auc_complete_df.index, columns='Method', values=metric)
            metric_output_path = Path(f'reports/paper_evaluation/{self.dataset_name}/full_auc_{metric}.csv')
            metric_df.to_csv(metric_output_path, index=True)

            # Generate and save the describe DataFrame for each metric
            description_metric_df = metric_df.describe()
            description_output_path = Path(f'reports/paper_evaluation/{self.dataset_name}/describe_auc_{metric}.csv')
            description_metric_df.to_csv(description_output_path, index=True)

        return self.auc_complete_df

    def compute_auc(self, method):
        self.full_df = self.concate_dataframes(method)
        return self.process_and_save_auc(method)

    
    def compute_average(self, method):
        full_df = self.full_df  # by method
        avg_df = pd.DataFrame(index=range(self.iterations), columns=['R2', 'MSE', 'MAE', 'CA', 'ARRMSE'])

        for iteration in range(self.iterations):
            iteration_df = full_df[full_df['Iterations'] == iteration]
            for metric in avg_df.columns:
                avg_value = iteration_df[metric].mean()
                avg_df.loc[iteration, metric] = avg_value

        avg_df['Method'] = method
        self.avg_df = avg_df
        return self.avg_df
    
    def compile_methods(self):
        # for each metric
        resume_auc = pd.DataFrame()
        resume_avg = pd.DataFrame()
        for unique_method in self.all_methods:
            self.method = unique_method
            full_df = self.concate_dataframes(unique_method)
            complete_auc_df= self.compute_auc(unique_method)
            resume_auc[unique_method] = complete_auc_df[self.metric_name]
            self.resume_auc = resume_auc

            avg_df = self.compute_average(unique_method)
            resume_avg[unique_method] = avg_df[self.metric_name]
            self.resume_avg = resume_avg

            output_path_avg = Path(f'reports/paper_evaluation/{self.dataset_name}/resume_avg_{self.metric_name}.csv')
            self.resume_avg.to_csv(output_path_avg)

        return 
    
        
    def save_summary_metrics(self):
        df_type = self.auc_df
        
        summary_data = {method: [] for method in self.all_methods}
        
        for method in self.all_methods:
            mean_values = []
            std_values = []
            for dataset in dataset_names:
                description_path = Path(f'reports/paper_evaluation/{self.dataset_name}/resume_auc_{self.metric_name}_description.csv')
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
        output_path = Path(f'reports/paper_evaluation/summary_auc_{self.metric_name}.csv')
        summary_df.to_csv(output_path)

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
            'self_learning': 'SSL - SL',
            'cotraining': 'SSL - CT'
        }
        #fig, axes = plt.subplots(len(metric_names), len(dataset_names), figsize=(5*len(dataset_names), 4*len(metric_names)), sharex=True)
        #fig.subplots_adjust(bottom=0.15, top = 0.95)
    
        if len(dataset_names) == 1 and len(metric_names) == 1:
            fig, ax = plt.subplots(figsize=(5, 4))
            axes = [[ax]]
        elif len(dataset_names) == 1:
            fig, axes = plt.subplots(len(metric_names), 1, figsize=(5*len(metric_names), 4*len(metric_names)), sharex=True)
            axes = [[ax] for ax in axes]
        elif len(metric_names) == 1:
            fig, axes = plt.subplots(1, len(dataset_names), figsize=(5*len(dataset_names), 4), sharex=True)
            axes = [axes]
        else:
            fig, axes = plt.subplots(len(metric_names), len(dataset_names), figsize=(5*len(dataset_names), 4*len(metric_names)), sharex=True)
        
        fig.subplots_adjust(bottom=0.15, top=0.95)
        
        for i, dataset in enumerate(dataset_names):
            for j, metric in enumerate(metric_names):
                ax = axes[j][i]
                for method in self.all_methods:
                    file_path = Path(f'reports/paper_evaluation/{dataset}/resume_avg_{metric}.csv')
                    if file_path.exists():
                        df = pd.read_csv(file_path, index_col=0)
                        mean_values = df[method].values
                        ax.plot(range(1, len(mean_values) + 1), mean_values, label=legend_labels[method])
                        
                if i == 0:
                    ax.set_ylabel(metric, fontsize=18)
                if j == 0:
                    ax.set_title(dataset, fontsize=18)
                if j == len(metric_names) - 1:
                    ax.set_xlabel('Iteration', fontsize=14)
        
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=len(self.all_methods), fontsize=18)
        plt.savefig('reports/paper_evaluation/summary_subplot_image.png')
        plt.close(fig)

def compile_summary_reports(metric_name):
    all_summaries = []
    summary_path = Path(f'reports/paper_evaluation/summary_auc_{metric_name}.csv', index_col = 0)
    if summary_path.exists():
        summary_df = pd.read_csv(summary_path)
        if 'Unnamed: 0' in summary_df.columns:
            summary_df.rename(columns={'Unnamed: 0': 'dataset'}, inplace=True)  # Rename the column to 'dataset'
        all_summaries.append(summary_df)
    
    final_summary_df = pd.concat(all_summaries, ignore_index=True)

    cols = ['dataset'] + [col for col in final_summary_df.columns if col not in ['dataset']]
    final_summary_df = final_summary_df[cols]

    final_summary_df = final_summary_df.sort_values(by='dataset')

    final_summary_path = Path(f'reports/paper_evaluation/summary_auc_{metric_name}.csv')
    final_summary_df.to_csv(final_summary_path)




iterations = ITERATIONS
folds = 5
dataset_names = ['enb', 'friedman', 'jura', 'mp5spec', 'musicOrigin2', 'oes97', 'osales']
#dataset_names = ['atp7d', 'jura', 'enb', 'mp5spec',]
metric_names = ['ARRMSE', 'CA', 'MAE', 'MSE', 'R2'] 
#metric_names = ['ARRMSE', 'R2', 'MSE']


def run_reports(dataset_names, metric_names, iterations, folds):
    for dataset in dataset_names:
        for metric in metric_names:
            evaluator = Evaluator(dataset, metric, iterations, folds)
            evaluator.compile_methods()
            evaluator.run_autorank()
            evaluator.save_summary_metrics()

    evaluator.generate_subplot_image(dataset_names, metric_names)
    for metric in metric_names:
        compile_summary_reports(metric)

run_reports(dataset_names, metric_names, iterations, folds)

class MultiEvaluator:
    def __init__(self, dataset_names, metric_names, all_methods):
        self.dataset_names = dataset_names
        self.metric_names = metric_names
        self.all_methods = all_methods

    def fetch_auc(self):
        for metric in self.metric_names:
            general_auc = pd.DataFrame(index=self.dataset_names, columns=self.all_methods)
            for dataset in self.dataset_names:
                output_path_auc = Path(f'reports/paper_evaluation/{dataset}/resume_auc_{metric}.csv')
                df_auc = pd.read_csv(output_path_auc, index_col=0)
                mean_auc = df_auc.mean(axis=0)
                general_auc.loc[dataset] = mean_auc[self.all_methods].values
            output_path_mean_auc = Path(f'reports/paper_evaluation/resume_auc_{metric}.csv')
            general_auc.to_csv(output_path_mean_auc)

    def multi_autorank(self):
        for metric in self.metric_names:
            output_path_mean_auc = Path(f'reports/paper_evaluation/resume_auc_{metric}.csv')
            general_auc = pd.read_csv(output_path_mean_auc, index_col=0)
            result = autorank(general_auc, alpha=0.05, verbose=False)
            report_path = Path(f'reports/paper_evaluation/{metric}_autorank_report_auc.txt')
            with open(report_path, 'w') as report_file:
                old_stdout = sys.stdout
                sys.stdout = report_file
                latex_report(result, decimal_places=3, complete_document=False)
                sys.stdout = old_stdout
            ax = plot_stats(result, allow_insignificant=True) 
            fig = ax.get_figure()
            fig.set_size_inches(12, 4)
            plot_path = Path(f'reports/paper_evaluation/{metric}_autorank_plot_auc.png')
            fig.savefig(plot_path)
            plt.close(fig) 
            plt.close('all')

def run_multi_evaluation(dataset_names, metric_names, all_methods):
    multi_eval_module = MultiEvaluator(dataset_names, metric_names, all_methods)
    multi_eval_module.fetch_auc()
    multi_eval_module.multi_autorank()

dataset_names = ['enb', 'friedman', 'jura', 'mp5spec', 'musicOrigin2', 'oes97', 'osales']
metric_names = ['ARRMSE', 'CA', 'MAE', 'MSE', 'R2'] 
all_methods = ['target_qbc', 'masster_cotraining', 'masster_self_learning', 'self_learning']
run_multi_evaluation(dataset_names, metric_names, all_methods)