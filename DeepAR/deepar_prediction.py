import argparse
import logging
import os
import json

import numpy as np
import torch
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm

import utils
import model.net as net
from dataloader import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = logging.getLogger('DeepAR.Eval')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='goog_stock_processed', help='Name of the dataset')
parser.add_argument('--data-folder', default='data', help='Parent dir of the dataset')
parser.add_argument('--model-name', default='goog_base_model', help='Directory containing params.json')
parser.add_argument('--relative-metrics', action='store_true', help='Whether to normalize the metrics by label scales')
parser.add_argument('--sampling', action='store_true', help='Whether to sample during evaluation')
parser.add_argument('--restore-file', default='best',
                   help='Optional, name of the file in --model_dir containing weights to reload before training')
# Add parameters for saving prediction results
parser.add_argument('--save-predictions', action='store_true', help='Whether to save predictions to CSV')
parser.add_argument('--predictions-file', default='predictions.csv', 
                   help='Path to save predictions CSV file (relative to model directory)')
parser.add_argument('--test-start', default='2024-09-18', 
                   help='Test data start date (YYYY-MM-DD), used when saving predictions')
parser.add_argument('--test-end', default='2025-03-15', 
                   help='Test data end date (YYYY-MM-DD), limit predictions up to this date')


def evaluate(model, loss_fn, test_loader, params, plot_num, sample=True):
    '''Evaluate the model on the test set.
    Args:
        model: (torch.nn.Module) the Deep AR model
        loss_fn: a function that takes outputs and labels per timestep, and then computes the loss for the batch
        test_loader: load test data and labels
        params: (Params) hyperparameters
        plot_num: (-1): evaluation from evaluate.py; else (epoch): evaluation on epoch
        sample: (boolean) do ancestral sampling or directly use output mu from last time step
    '''
    model.eval()
    with torch.no_grad():
        n_batches = len(test_loader)
        if n_batches <= 1:
            plot_batch = 0
        else:
            plot_batch = np.random.randint(n_batches-1)

        summary_metric = {}
        raw_metrics = utils.init_metrics(sample=sample)
        
        # Collect all predictions and actual values
        all_samples = []
        all_sample_mu = []
        all_labels = []
        
        for i, (test_batch, id_batch, v, labels) in enumerate(tqdm(test_loader)):
            test_batch = test_batch.permute(1, 0, 2).to(torch.float32).to(params.device)
            id_batch = id_batch.unsqueeze(0).to(params.device)
            v_batch = v.to(torch.float32).to(params.device)
            labels = labels.to(torch.float32).to(params.device)
            batch_size = test_batch.shape[1]
            input_mu = torch.zeros(batch_size, params.test_predict_start, device=params.device) # scaled
            input_sigma = torch.zeros(batch_size, params.test_predict_start, device=params.device) # scaled
            hidden = model.init_hidden(batch_size)
            cell = model.init_cell(batch_size)

            for t in range(params.test_predict_start):
                # if z_t is missing, replace it by output mu from the last time step
                zero_index = (test_batch[t,:,0] == 0)
                if t > 0 and torch.sum(zero_index) > 0:
                    test_batch[t,zero_index,0] = mu[zero_index]

                mu, sigma, hidden, cell = model(test_batch[t].unsqueeze(0), id_batch, hidden, cell)
                input_mu[:,t] = v_batch[:, 0] * mu + v_batch[:, 1]
                input_sigma[:,t] = v_batch[:, 0] * sigma

            if sample:
                samples, sample_mu, sample_sigma = model.test(test_batch, v_batch, id_batch, hidden, cell, sampling=True)
                # Collect predictions and actual values
                all_samples.append(samples.cpu().numpy())
                all_sample_mu.append(sample_mu.cpu().numpy())
                all_labels.append(labels[:, params.test_predict_start:].cpu().numpy())
                
                raw_metrics = utils.update_metrics(raw_metrics, input_mu, input_sigma, sample_mu, labels, params.test_predict_start, samples, relative = params.relative_metrics)
            else:
                sample_mu, sample_sigma = model.test(test_batch, v_batch, id_batch, hidden, cell)
                # Collect predictions and actual values
                all_sample_mu.append(sample_mu.cpu().numpy())
                all_labels.append(labels[:, params.test_predict_start:].cpu().numpy())
                
                raw_metrics = utils.update_metrics(raw_metrics, input_mu, input_sigma, sample_mu, labels, params.test_predict_start, relative = params.relative_metrics)

            if i == plot_batch:
                if sample:
                    sample_metrics = utils.get_metrics(sample_mu, labels, params.test_predict_start, samples, relative = params.relative_metrics)
                else:
                    sample_metrics = utils.get_metrics(sample_mu, labels, params.test_predict_start, relative = params.relative_metrics)                
                # select 10 from samples with highest error and 10 from the rest
                top_10_nd_sample = (-sample_metrics['ND']).argsort()[:batch_size // 10]  # hard coded to be 10
                chosen = set(top_10_nd_sample.tolist())
                all_samples_set = set(range(batch_size))
                not_chosen = np.asarray(list(all_samples_set - chosen))
                if batch_size < 100: # make sure there are enough unique samples to choose top 10 from
                    random_sample_10 = np.random.choice(top_10_nd_sample, size=10, replace=True)
                else:
                    random_sample_10 = np.random.choice(top_10_nd_sample, size=10, replace=False)
                if batch_size < 12: # make sure there are enough unique samples to choose bottom 90 from
                    random_sample_90 = np.random.choice(not_chosen, size=10, replace=True)
                else:
                    random_sample_90 = np.random.choice(not_chosen, size=10, replace=False)
                combined_sample = np.concatenate((random_sample_10, random_sample_90))

                label_plot = labels[combined_sample].data.cpu().numpy()
                predict_mu = sample_mu[combined_sample].data.cpu().numpy()
                predict_sigma = sample_sigma[combined_sample].data.cpu().numpy()
                plot_mu = np.concatenate((input_mu[combined_sample].data.cpu().numpy(), predict_mu), axis=1)
                plot_sigma = np.concatenate((input_sigma[combined_sample].data.cpu().numpy(), predict_sigma), axis=1)
                plot_metrics = {_k: _v[combined_sample] for _k, _v in sample_metrics.items()}
                plot_eight_windows(params.plot_dir, plot_mu, plot_sigma, label_plot, params.test_window, params.test_predict_start, plot_num, plot_metrics, sample)

        # Merge predictions and actual values from all batches
        if all_sample_mu:
            predictions = np.concatenate(all_sample_mu, axis=0)
            targets = np.concatenate(all_labels, axis=0)
        else:
            predictions = np.array([])
            targets = np.array([])
        
        summary_metric = utils.final_metrics(raw_metrics, sampling=sample)
        metrics_string = '; '.join('{}: {:05.3f}'.format(k, v) for k, v in summary_metric.items())
        logger.info('- Full test metrics: ' + metrics_string)
        
        # Add predictions and actual values to summary_metric
        summary_metric['predictions'] = predictions.tolist() if predictions.size > 0 else []
        summary_metric['targets'] = targets.tolist() if targets.size > 0 else []
    
    return summary_metric


def plot_eight_windows(plot_dir,
                       predict_values,
                       predict_sigma,
                       labels,
                       window_size,
                       predict_start,
                       plot_num,
                       plot_metrics,
                       sampling=False):

    x = np.arange(window_size)
    f = plt.figure(figsize=(8, 42), constrained_layout=True)
    nrows = 21
    ncols = 1
    ax = f.subplots(nrows, ncols)

    for k in range(nrows):
        if k == 10:
            ax[k].plot(x, x, color='g')
            ax[k].plot(x, x[::-1], color='g')
            ax[k].set_title('This separates top 10 and bottom 90', fontsize=10)
            continue
        m = k if k < 10 else k - 1
        ax[k].plot(x, predict_values[m], color='b')
        ax[k].fill_between(x[predict_start:], predict_values[m, predict_start:] - 2 * predict_sigma[m, predict_start:],
                         predict_values[m, predict_start:] + 2 * predict_sigma[m, predict_start:], color='blue',
                         alpha=0.2)
        ax[k].plot(x, labels[m, :], color='r')
        ax[k].axvline(predict_start, color='g', linestyle='dashed')

        plot_metrics_str = f'ND: {plot_metrics["ND"][m]: .3f} ' \
            f'RMSE: {plot_metrics["RMSE"][m]: .3f}'
        if sampling:
            plot_metrics_str += f' rou90: {plot_metrics["rou90"][m]: .3f} ' \
                                f'rou50: {plot_metrics["rou50"][m]: .3f}'

        ax[k].set_title(plot_metrics_str, fontsize=10)

    f.savefig(os.path.join(plot_dir, str(plot_num) + '.png'))
    plt.close()

def save_predictions_to_csv(predictions, targets, test_start, test_end, stride_size, output_file):
    """
    Save prediction results to a CSV file, using original stock data as actual values
    """
    import pandas as pd
    from datetime import datetime, timedelta
    
    # Create date series
    start_date = datetime.strptime(test_start, '%Y-%m-%d')
    end_date = datetime.strptime(test_end, '%Y-%m-%d') if test_end else None
    
    # Make sure predictions is a 2D array
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    
    # Reshape predictions into daily format
    daily_predictions = []
    daily_dates = []
    
    for i in range(predictions.shape[0]):
        base_date = start_date + timedelta(days=i*stride_size)
        
        # For each prediction step
        for j in range(predictions.shape[1]):
            # Current date = base date + j days
            curr_date = base_date + timedelta(days=j)
            
            # Skip if end date is set and current date exceeds it
            if end_date and curr_date > end_date:
                continue
                
            # Get prediction value
            pred_value = predictions[i, j]
            
            daily_predictions.append(pred_value)
            daily_dates.append(curr_date)
    
    # Create prediction DataFrame
    df = pd.DataFrame({
        'Date': daily_dates,
        'prediction': daily_predictions
    })
    
    # Remove duplicate dates (keep last prediction)
    df = df.drop_duplicates(subset='Date', keep='last')
    df = df.sort_values('Date')
    
    # Load original stock data
    original_data_path = '../data/stock/goog_stock_wsenti.csv'  # Adjust to your file path
    logger.info(f"Loading original stock data: {original_data_path}")
    try:
        original_data = pd.read_csv('../data/stock/goog_stock_wsenti.csv')
        original_data['Date'] = pd.to_datetime(original_data['Date']).dt.tz_localize(None)
        df['Date'] = df['Date'].dt.tz_localize(None)
        df = pd.merge(df, original_data[['Date', 'Daily Return']], 
                     on='Date', how='inner')
        
        # Rename columns
        df.rename(columns={'Daily Return': 'actual'}, inplace=True)
        
        # Add prediction error
        df['error'] = df['prediction'] - df['actual']
        df['abs_error'] = abs(df['error'])
        
        # Add direction correctness (only for non-NaN values)
        df['correct_direction'] = np.nan
        valid_indices = ~df['prediction'].isna() & ~df['actual'].isna()
        df.loc[valid_indices, 'correct_direction'] = (df.loc[valid_indices, 'prediction'] * df.loc[valid_indices, 'actual']) > 0
        
        # Calculate evaluation metrics
        valid_df = df.dropna(subset=['actual'])
        if len(valid_df) > 0:
            mae = valid_df['abs_error'].mean()
            rmse = np.sqrt((valid_df['error'] ** 2).mean())
            dir_acc = valid_df['correct_direction'].mean()
            
            logger.info(f"Evaluation with original data - MAE: {mae:.6f}, RMSE: {rmse:.6f}, Direction Accuracy: {dir_acc:.4f}")
    except Exception as e:
        logger.error(f"Error loading original data: {e}")
        raise ValueError("Unable to access original stock data. Please ensure the data file path is correct and contains the required columns.")
    
    # Date range information
    min_date = df['Date'].min().strftime('%Y-%m-%d')
    max_date = df['Date'].max().strftime('%Y-%m-%d')
    logger.info(f"Prediction date range: {min_date} to {max_date}")
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    logger.info(f"Prediction results saved to: {output_file}")
    
    # Visualize predictions
    plt.figure(figsize=(12, 6))
    
    # Plot actual values (if available)
    if not df['actual'].isna().all():
        plt.plot(df['Date'], df['actual'], 'o-', label='Actual', color='blue', alpha=0.7, markersize=3)
    
    # Plot predicted values
    plt.plot(df['Date'], df['prediction'], 'x--', label='Prediction', color='red', alpha=0.7, markersize=3)
    
    plt.title(f'DeepAR Predictions vs Actual Values ({min_date} to {max_date})')
    plt.xlabel('Date')
    plt.ylabel('Daily Return')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save chart
    plot_file = output_file.replace('.csv', '_plot.png')
    plt.savefig(plot_file)
    logger.info(f"Prediction visualization chart saved to: {plot_file}")
    
    return df

if __name__ == '__main__':
    # Load the parameters
    args = parser.parse_args()
    model_dir = os.path.join('experiments', args.model_name) 
    json_path = os.path.join(model_dir, 'params.json')
    data_dir = os.path.join(args.data_folder, args.dataset)
    assert os.path.isfile(json_path), 'No json configuration file found at {}'.format(json_path)
    params = utils.Params(json_path)

    utils.set_logger(os.path.join(model_dir, 'deepar-prediction.log'))

    params.relative_metrics = args.relative_metrics
    params.sampling = args.sampling
    params.model_dir = model_dir
    params.plot_dir = os.path.join(model_dir, 'figures')
    
    # Ensure plot directory exists
    os.makedirs(params.plot_dir, exist_ok=True)
    
    cuda_exist = torch.cuda.is_available()  # use GPU is available

    # Set random seeds for reproducible experiments if necessary
    if cuda_exist:
        params.device = torch.device('cuda')
        # torch.cuda.manual_seed(240)
        logger.info('Using Cuda...')
        model = net.Net(params).cuda()
    else:
        params.device = torch.device('cpu')
        # torch.manual_seed(230)
        logger.info('Not using cuda...')
        model = net.Net(params)

    # Create the input data pipeline
    logger.info('Loading the datasets...')

    test_set = TestDataset(data_dir, args.dataset, params.num_class)
    test_loader = DataLoader(test_set, batch_size=params.predict_batch, sampler=RandomSampler(test_set), num_workers=4)
    logger.info('- done.')

    print('model: ', model)
    loss_fn = net.loss_fn

    logger.info('Starting evaluation')

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(model_dir, args.restore_file + '.pth.tar'), model)

    # Run evaluation
    test_metrics = evaluate(model, loss_fn, test_loader, params, -1, params.sampling)
    
    # Extract predictions and target data from test_metrics
    predictions = np.array(test_metrics.pop('predictions', []))
    targets = np.array(test_metrics.pop('targets', []))
    
    # Save evaluation metrics (without predictions and target data)
    save_path = os.path.join(model_dir, 'metrics_test_{}.json'.format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)
    
    # If prediction results need to be saved
    if args.save_predictions and predictions.size > 0 and targets.size > 0:
        # Get stride_size
        stride_size = getattr(params, 'stride_size', 5)
        
        # Save prediction results to CSV
        predictions_file = os.path.join(model_dir, args.predictions_file)
        save_predictions_to_csv(predictions, targets, args.test_start, args.test_end, stride_size, predictions_file)
        
        # Output statistics
        logger.info("Prediction generation complete. Summary statistics:")
        
        # Simple statistics - using the first column of predictions and targets
        pred_col = predictions[:, 0] if predictions.ndim > 1 else predictions
        target_col = targets[:, 0] if targets.ndim > 1 else targets
        
        # Calculate metrics
        mae = np.mean(np.abs(pred_col - target_col))
        rmse = np.sqrt(np.mean((pred_col - target_col)**2))
        direction_acc = np.mean((pred_col * target_col) > 0)
        
        logger.info(f"Number of samples: {len(pred_col)}")
        logger.info(f"Mean Absolute Error (MAE): {mae:.6f}")
        logger.info(f"Root Mean Square Error (RMSE): {rmse:.6f}")
        logger.info(f"Direction Accuracy: {direction_acc:.4f}")
        
        # Separately save prediction and target data (if needed)
        predictions_data = {
            'predictions': predictions.tolist(),
            'targets': targets.tolist()
        }
        
        predictions_json_path = os.path.join(model_dir, 'predictions_data_{}.json'.format(args.restore_file))
        with open(predictions_json_path, 'w') as f:
            json.dump(predictions_data, f)
        
        logger.info(f"Prediction data saved to: {predictions_json_path}")