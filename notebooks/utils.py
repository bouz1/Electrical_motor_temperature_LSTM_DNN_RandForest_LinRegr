import matplotlib.pyplot as plt
import numpy as np
def plot_prediction_real(y, yh, label='train set', fig_size=(5, 5), \
                         show_plot=True,offset=None,return_fig=False):
    
    """
    This function plots the real values against the predicted values. It's useful for visualizing the performance of a regression model.
    
    Parameters:
    y (array-like): The real values.
    yh (array-like): The predicted values.
    label (str, optional): The title of the plot. Defaults to 'train set'.
    fig_size (tuple, optional): The size of the figure. Defaults to (5, 5).
    show_plot (bool, optional): Whether to display the plot. Defaults to True.
    offset (float, optional): The offset to apply to the limits of the plot. If None, it's calculated as 20% of the range of the data. Defaults to None.
    return_fig (bool, optional): Whether to return the figure object. Defaults to False.
    """
    
    # Calculate the maximum and minimum values in the data
    max_ = max(y.max(), yh.max())
    min_ = min(y.min(), yh.min())
    
    # If no offset is provided, calculate it as 20% of the range of the data
    if offset is None: 
        offset = (max_ - min_) * 0.2
    
    # Adjust the maximum and minimum values by the offset
    max_ = max_ - offset
    min_ = min_ + offset   
    
    # Color plot 
    r='#b7190f'
    b='#26619c'
    
    # Create a new figure
    fig = plt.figure(figsize=fig_size)
    
    # Plot the predicted values against the real values
    plt.scatter(y, yh, label='prediction', c=b, s=0.5)  
    
    # Plot the line y=x, which represents perfect prediction
    plt.plot([min_, max_], [min_, max_], label='no error', c=r)
    
    # plot the cuvre no error +10°C 
    plt.plot([min_, max_], [min_+10, max_+10],'--', label='no error +10°C',\
             c=r)
    
    # plot the cuvre no error -10°C 
    plt.plot([min_, max_], [min_-10, max_-10],':', label='no error -10°C',\
             c=r)
    
    # Set the labels of the axes
    plt.xlabel('real values')
    plt.ylabel('predicted values')
    
    # Add a legend
    plt.legend()
    
    # Set the title of the plot
    plt.title(label)
    
    # Add a grid
    plt.grid()
    
    # If show_plot is True, display the plot
    if show_plot:
        plt.show()
    
    # If return_fig is True, return the figure object
    if return_fig: 
        return fig




def plot_y_yh_time(y, yh, Ts=0.5, title='', return_fig=False, fig_size=(12,5),\
                   show_plot=True):
    """
    This function plots the real and predicted values over time. It's useful for visualizing the performance of a time series prediction model.
    
    Parameters:
    y (array-like): The real values.
    yh (array-like): The predicted values.
    Ts (float, optional): The sampling time in seconds. Defaults to 0.5.
    title (str, optional): The title of the plot. Defaults to ''.
    return_fig (bool, optional): Whether to return the figure object. Defaults to False.
    fig_size (tuple, optional): The size of the figure. Defaults to (12,5).
    show_plot (bool, optional): Whether to display the plot. Defaults to True.
    """
    
    # Color plot 
    r='#b7190f'
    b='#26619c'
    
    # Calculate the end time based on the length of the data and the sampling time
    end = Ts * len(y)
    
    # Generate the time values
    time = np.linspace(0, end)
    
    # Create a new figure
    fig = plt.figure(figsize=fig_size)
    
    # Plot the predicted values
    plt.plot(yh, label='Predicted', c=b)
    
    # Plot the real values
    plt.plot(y, label='Real', c=r)
    
    # Add a legend
    plt.legend()
    
    # Add a grid
    plt.grid()
    
    # Set the labels of the axes
    plt.xlabel('time s')
    plt.ylabel('Rotor temperature °C')
    
    # Set the title of the plot
    plt.title(title)
    
    # If show_plot is True, display the plot
    if show_plot: 
        plt.show()
    
    # If return_fig is True, return the figure object
    if return_fig: 
        return fig
    
    
    
def plot_error(y_train, yh_train, y_test, yh_test, bins_in=40, show_plot=True, \
               fig_size=(5, 5),return_fig=False):  
    """
    This function plots the histogram of prediction errors for both training and testing data. It also plots the normal probability density function (pdf) for the errors.
    
    Parameters:
    y_train (array-like): The real values for the training set.
    yh_train (array-like): The predicted values for the training set.
    y_test (array-like): The real values for the test set.
    yh_test (array-like): The predicted values for the test set.
    bins_in (int, optional): The number of bins for the histogram. Defaults to 40.
    show_plot (bool, optional): Whether to display the plot. Defaults to True.
    fig_size (tuple, optional): The size of the figure. Defaults to (5, 5).
    return_fig (bool, optional): Whether to return the figure object. Defaults to False.
    """
    
    # Create a new figure with two y-axes
    fig, ax1 = plt.subplots(figsize=fig_size)
    ax2 = ax1.twinx()

    # Calculate the prediction error for the test set
    error_test = y_test - yh_test
    
    # Color plot 
    r='#b7190f'
    b='#26619c'
    
    # Dictionnary to store mean STD of train/test error
    dic={}
    
    # Plot the histogram of the test error
    counts, bins = np.histogram(error_test, bins=bins_in, density=True)
    bins = 0.5 * (bins[:-1] + bins[1:])
    ax2.plot(bins, counts, label='test', c=r, marker='.')
    
    # Plot the normal pdf for the test error
    mean, std = error_test.mean(), error_test.std()
    x = np.linspace(error_test.min(), error_test.max(), bins_in)
    ax2.plot(x, norm_pdf(x, mean, std), label='test Norm pdf', c=r, alpha=0.4,\
             linestyle='--')
    dic['test']={'mean':mean,'std':std}

    # Calculate the prediction error for the training set
    error_train = y_train - yh_train
    
    # Plot the histogram of the training error
    counts, bins = np.histogram(error_train, bins=bins_in, density=True)
    bins = 0.5 * (bins[:-1] + bins[1:])
    ax1.plot(bins, counts, label='train', c=b, marker='.')
    
    # Plot the normal pdf for the training error
    mean, std = error_train.mean(), error_train.std()
    x = np.linspace(error_train.min(), error_train.max(), bins_in)
    ax1.plot(x, norm_pdf(x, mean, std), label='train Norm pdf', c=b, alpha=0.4,\
             linestyle='--')
    dic['train']={'mean':mean,'std':std}

    # Set the labels of the axes
    ax1.set_xlabel('prediction error Bins')
    ax1.set_ylabel('Density (train)', color=b)
    ax2.set_ylabel('Density (test)', color=r)

    # Set the title of the plot
    plt.suptitle("Error Histogram")

    # Add a grid
    ax1.grid(which='major',  color=b, linewidth=1, alpha=0.2)
    ax2.grid(which='major',  color=r, linewidth=1, alpha=0.2)

    # Set the colors of the y-axis ticks
    ax2.tick_params(axis='y', colors=r)
    ax1.tick_params(axis='y', colors=b)

    # Add a legend
    fig.legend()
    
    # If show_plot is True, display the plot
    if show_plot: 
        plt.show()

    # Print the Dictionnary of mean/STD of train/test error
    print(dic)
    
    # If return_fig is True, return the figure object
    if return_fig: 
        return fig

    
def norm_pdf(x, mean, std):
    """
    This function calculates the probability density function (pdf) of a normal distribution.
    
    Parameters:
    x (float or array-like): The point(s) at which to evaluate the pdf.
    mean (float): The mean of the normal distribution.
    std (float): The standard deviation of the normal distribution.
    
    Returns:
    float or array-like: The pdf of the normal distribution at x.
    """
    
    # Calculate the pdf of the normal distribution at x
    # The formula for the pdf of a normal distribution is:
    # (1 / (std * sqrt(2*pi))) * exp(-0.5 * ((x - mean) / std)^2)
    return np.exp(-0.5 * ((x - mean) / std)**2) / (std * np.sqrt(2 * np.pi))


from sklearn.metrics import mean_squared_error as MSE
import numpy as np

# https://math.stackexchange.com/questions/488964/the-definition-of-nmse-normalized-mean-square-error
def RMSE(y_actual, y_predicted):
    """
    This function calculates the Root Mean Square Error (RMSE) between the actual and predicted values.
    
    Parameters:
    y_actual (array-like): The actual values.
    y_predicted (array-like): The predicted values.
    
    Returns:
    float: The RMSE between the actual and predicted values.
    """
    
    # Calculate the RMSE
    # The RMSE is the square root of the mean square error (MSE)
    return np.sqrt(MSE(y_actual, y_predicted))

def NMSE(y_actual, y_predicted):
    """
    This function calculates the Normalized Mean Square Error (NMSE) between the actual and predicted values.
    
    Parameters:
    y_actual (array-like): The actual values.
    y_predicted (array-like): The predicted values.
    
    Returns:
    float: The NMSE between the actual and predicted values.
    """
    
    # Calculate the NMSE
    # The NMSE is the MSE between the actual and predicted values, normalized by the MSE between the actual values and zero
    return MSE(y_actual, y_predicted) / MSE(y_actual, np.zeros(len(y_actual)))




from scipy.stats import loguniform,uniform
class loguniform_int:
    """Integer valued version of the log-uniform distribution"""

    def __init__(self, a, b):
        self._distribution = loguniform(a, b)

    def rvs(self, *args, **kwargs):
        """Random variable sample"""
        return self._distribution.rvs(*args, **kwargs).astype(int)
class uniform_int:
    """Integer valued version of the uniform distribution"""

    def __init__(self, a, b):
        self._distribution = uniform(a, b)

    def rvs(self, *args, **kwargs):
        """Random variable sample"""
        return self._distribution.rvs(*args, **kwargs).astype(int)
    
from sklearn.metrics import r2_score as R2
def local_metrics(y,yh):
    dic={}
    dic['r2_score']=R2(y,yh)
    dic['MSE']=MSE(y,yh)
    dic['RMSE']=RMSE(y,yh)
    dic['NMSE']=NMSE(y,yh)
    return dic







def split_with_groupes(df, groups_col, test_size=0.2, seed=0, n_ether=10): 
    """
    This function splits a DataFrame into training and testing sets based on unique groups. 
    The split is done in such a way that there is no overlap between the groups in the training and testing sets.
    
    Parameters:
    df (DataFrame): The DataFrame to be split.
    groups_col (str): The column name in df that contains the group identifiers.
    test_size (float): The proportion of the dataset to include in the test split.
    seed (int): The seed for the random number generator.
    n_ether (int): The number of iterations for shuffling and splitting the groups.
    
    Returns:
    df_train (DataFrame): The training set.
    df_test (DataFrame): The test set.
    """
    
    # Get a list of unique groups
    groups = df[groups_col].unique().tolist()
    
    # Set the seed for the random number generator
    np.random.seed(seed) 
    
    # Initialize an empty list to store the training and testing DataFrames
    arr = []
    
    # Shuffle the groups and split them into training and testing sets for n_ether iterations
    for i in range(n_ether):
        np.random.shuffle(groups)
        i_test = int(len(groups) * (1 - test_size))
        groups_train = groups[:i_test]  
        groups_test = groups[i_test:]
        
        # Create the training and testing DataFrames based on the groups
        df_train = df[df[groups_col].isin(groups_train)]
        df_test = df[df[groups_col].isin(groups_test)]
        
        # Append the training and testing DataFrames to the list
        arr.append([df_train, df_test])
    
    # Initialize an empty list to store the proportions of the test set sizes
    arr_results = []
    
    # Calculate the proportion of the test set size for each iteration
    for elem in arr: 
        df_train, df_test = elem
        arr_results.append(len(df_test) / len(df))
    
    # Calculate the absolute difference between the actual and desired test set sizes
    arr_results = np.abs(np.array(arr_results) - test_size)
    
    # Find the iteration with the test set size closest to the desired size
    i_min = np.argmin(arr_results)
    
    # Get the training and testing DataFrames for the iteration with the closest test set size
    df_train, df_test = arr[i_min]
    
    # Return the training and testing DataFrames
    return df_train, df_test
