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

class ActiveLearningEvaluator:
    def __init__(self, dataset_name, metric_name, n_epochs, considered_epoch):
        self.dataset_name = dataset_name
        self.metric_name = metric_name
        self.n_epochs = n_epochs
        self.methods_autorank = ['greedy', 'instance', 'qbcrf', 'random', 'rtal']
        self.methods = ['greedy', 'instance', 'qbcrf', 'random', 'rtal', 'upperbound']
        self.iterations = iterations
        self.considered_epoch = considered_epoch

    def assemble_auc(self):
        auc_data = {method: [] for method in self.methods_autorank}
        
        for method in self.methods_autorank:
            file_path = Path(f'reports/active_learning_only/{self.dataset_name}/{self.metric_name}/{method}_{self.metric_name}.csv')
            if file_path.exists():
                df = pd.read_csv(file_path, index_col=0)
                if 'AUC' in df.columns:
                    df = df.drop(columns=['AUC'])

                epoch_columns = [f'Epoch {i+1}' for i in range(self.considered_epoch)]
                df = df[epoch_columns]

                for i in range(self.iterations):
                    iteration_data = df.iloc[i, :self.considered_epoch]
                    x = range(1, self.considered_epoch + 1)
                    auc_value = auc(x, iteration_data)
                    auc_data[method].append(auc_value)
        
        index = [f'Iteration {i+1}' for i in range(self.iterations)]
        self.auc_df = pd.DataFrame(auc_data, index=index)

        # Normalizar os valores de AUC usando z-score
        for method in self.methods_autorank:
            auc_values = self.auc_df[method]
            mean_value = auc_values.mean()
            std_dev = auc_values.std()
            
            # Evitar divisão por zero ao normalizar
            if std_dev > 0:
                self.auc_df[method] = (auc_values - mean_value) / std_dev
            else:
                self.auc_df[method] = 0  # Ajustar para 0 se todos os valores forem iguais
        
        
        return self.auc_df
    
    def save_reports(self):
        output_path_auc = Path(f'reports/active_learning_only/{self.dataset_name}/{self.metric_name}/resume_auc_{self.considered_epoch}.csv')
        self.auc_df.to_csv(output_path_auc)

        description_auc_df = self.auc_df.describe()
        description_auc_path = Path(f'reports/active_learning_only/{self.dataset_name}/{self.metric_name}/resume_auc_{self.considered_epoch}_description.csv')
        description_auc_df.to_csv(description_auc_path)
        
    def save_summary_metrics(self):
        df_type = self.auc_df
        
        summary_data = {method: [] for method in self.methods_autorank}
        
        for method in self.methods_autorank:
            mean_values = []
            std_values = []
            for dataset in dataset_names:
                description_path = Path(f'reports/active_learning_only/{dataset}/{self.metric_name}/resume_auc_{self.considered_epoch}_description.csv')
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
        output_path = Path(f'reports/active_learning_only/summary_auc_{self.considered_epoch}_{self.metric_name}.csv')
        summary_df.to_csv(output_path)

    def run_autorank(self):
        df_type = self.auc_df
        result = autorank(df_type, alpha=0.05, verbose=False)
        report_path = Path(f'reports/active_learning_only/{self.dataset_name}/{self.metric_name}/autorank_report_auc_{self.considered_epoch}.txt')
        with open(report_path, 'w') as report_file:
            old_stdout = sys.stdout
            sys.stdout = report_file
            latex_report(result, decimal_places=3, complete_document=False)
            sys.stdout = old_stdout
        ax = plot_stats(result, allow_insignificant=True) 
        fig = ax.get_figure()
        plot_path = Path(f'reports/active_learning_only/{self.dataset_name}/{self.metric_name}/autorank_plot_auc_{self.considered_epoch}.png')
        fig.savefig(plot_path)
        plt.close(fig) 
        plt.close('all')

    def generate_subplot_image(self, dataset_names, metric_names):
        legend_labels = {
            'greedy': 'Greedy',
            'instance': 'Instance-based QBC',
            'qbcrf': 'Target-based QBC',
            'random': 'Random',
            'rtal': 'RT-AL',
            'upperbound': 'Upper-bound'
        }
        fig, axes = plt.subplots(len(metric_names), len(dataset_names), figsize=(5*len(dataset_names), 4*len(metric_names)), sharex=True)
        fig.subplots_adjust(bottom=0.15, top = 0.95)
    
        for i, dataset in enumerate(dataset_names):
            for j, metric in enumerate(metric_names):
                ax = axes[j, i]
                for method in self.methods:
                    file_path = Path(f'reports/active_learning_only/{dataset}/{metric}/{method}_{metric}.csv')
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
        plt.savefig('reports/active_learning_only/summary_subplot_image.png')
        plt.close(fig)

def compile_summary_reports(considered_epochs, metric_name):
    all_summaries = []
    for considered_epoch in considered_epochs:
        summary_path = Path(f'reports/active_learning_only/summary_auc_{considered_epoch}_{metric_name}.csv', index_col = 0)
        if summary_path.exists():
            summary_df = pd.read_csv(summary_path)
            if 'Unnamed: 0' in summary_df.columns:
                summary_df.rename(columns={'Unnamed: 0': 'dataset'}, inplace=True)  # Rename the column to 'dataset'
            summary_df['considered_epoch'] = considered_epoch
            all_summaries.append(summary_df)
    
    final_summary_df = pd.concat(all_summaries, ignore_index=True)

    cols = ['dataset','considered_epoch'] + [col for col in final_summary_df.columns if col not in ['dataset','considered_epoch']]
    final_summary_df = final_summary_df[cols]

    final_summary_df = final_summary_df.sort_values(by='dataset')

    final_summary_path = Path(f'reports/active_learning_only/summary_auc_{metric_name}.csv')
    final_summary_df.to_csv(final_summary_path)

def run_reports(dataset_names, metric_names, considered_epochs, method_names):
    for considered_epoch in considered_epochs:
        for dataset in dataset_names:
            for metric in metric_names:
                evaluator = ActiveLearningEvaluator(dataset, metric, n_epochs, considered_epoch)
                auc_df = evaluator.assemble_auc()
                evaluator.save_reports()
                #evaluator.run_autorank()
                evaluator.save_summary_metrics()
                
    #evaluator.generate_subplot_image(dataset_names, metric_names)
 
#n_epochs = 15
iterations = ITERATIONS

#considered_epochs = [int(n_epochs/3), int(n_epochs*(2/3)), n_epochs]
#dataset_names = ['atp7d', 'friedman', 'mp5spec', 'musicOrigin2', 'rf2', 'oes97', 'enb', 'osales', 'wq', 'scm1d', 'jura']
#dataset_names = ['atp7d', 'friedman', 'jura', 'mp5spec', 'oes97', 'rf2', 'scm1d', 'wq']
#metric_names = ['arrmse', 'r2']
#metric_names = ['arrmse', 'ca', 'mae', 'mse', 'r2'] 
#method_names = ['greedy', 'instance', 'qbcrf', 'random', 'rtal', 'upperbound']
#run_reports(dataset_names, metric_names, considered_epochs, method_names)

class MultiEvaluatorActive:
    def __init__(self, dataset_names, metric_names, all_methods, considered_epoch):
        self.dataset_names = dataset_names
        self.metric_names = metric_names
        self.all_methods = all_methods
        self.considered_epoch = considered_epoch

    def fetch_auc(self):
        for metric in self.metric_names:
            general_auc = pd.DataFrame(index=self.dataset_names, columns=self.all_methods)
            for dataset in self.dataset_names:
                output_path_auc = Path(f'reports/active_learning_only/{dataset}/{metric}/resume_auc_{self.considered_epoch}.csv')
                df_auc = pd.read_csv(output_path_auc, index_col=0)
                print(df_auc)
                mean_auc = df_auc.mean(axis=0)
                general_auc.loc[dataset] = mean_auc[self.all_methods].values
            output_path_mean_auc = Path(f'reports/active_learning_only/resume_auc_{metric}.csv')
            general_auc.to_csv(output_path_mean_auc)

    def multi_autorank(self):
        for metric in self.metric_names:
            output_path_mean_auc = Path(f'reports/active_learning_only/resume_auc_{metric}.csv')
            general_auc = pd.read_csv(output_path_mean_auc, index_col=0)
            #print(general_auc)
            result = autorank(general_auc, alpha=0.05, verbose=False)
            report_path = Path(f'reports/active_learning_only/{metric}_autorank_report_auc.txt')
            with open(report_path, 'w') as report_file:
                old_stdout = sys.stdout
                sys.stdout = report_file
                latex_report(result, decimal_places=3, complete_document=False)
                sys.stdout = old_stdout
            ax = plot_stats(result, allow_insignificant=True) 
            fig = ax.get_figure()
            plot_path = Path(f'reports/active_learning_only/{metric}_autorank_plot_auc.png')
            fig.savefig(plot_path)
            plt.close(fig) 
            plt.close('all')

def run_multi_evaluation(dataset_names, metric_names, all_methods, considered_epoch):
    multi_eval_module = MultiEvaluatorActive(dataset_names, metric_names, all_methods, considered_epoch)
    multi_eval_module.fetch_auc()
    multi_eval_module.multi_autorank()

n_epochs = 15
considered_epoch = int(n_epochs/3)#, int(n_epochs*(2/3)), n_epochs]
dataset_names = ['atp7d', 'friedman', 'mp5spec', 'musicOrigin2', 'rf2', 'oes97', 'enb', 'osales', 'wq', 'scm1d', 'jura']
#dataset_names = ['atp7d', 'friedman', 'jura', 'mp5spec', 'oes97', 'rf2', 'scm1d', 'wq']
#metric_names = ['arrmse', 'r2']
metric_names = ['arrmse', 'ca', 'mae', 'mse', 'r2'] 
all_methods = ['greedy', 'instance', 'qbcrf', 'random', 'rtal']#, 'upperbound']
run_multi_evaluation(dataset_names, metric_names, all_methods, considered_epoch)




