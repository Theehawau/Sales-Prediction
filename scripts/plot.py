import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

def scatter_plot(x:str,y:str,df:pd.DataFrame,title:str,split=None,col=None):
    """
    This function makes a scatter plot.
    x = Column name for X-axis
    y = Column name for y-axis
    df = DataFrame name
    split = Categorical variable for hue
    col = Categorical variable for multi plots
    title = plot title
    """
    if (col == None):
        sns.relplot(x=x, y=y, hue=split,data=df, height=7, aspect=2).set(title=title)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel(xlabel=x,fontsize=18)
        plt.ylabel(ylabel=y,fontsize=18)
        plt.show()
    else:
        rel = sns.relplot(x=x, y=y,hue=split,data=df,col=col,col_wrap=2)
        rel.fig.suptitle(t=title,fontsize=30)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel(xlabel=x,fontsize=18)
        plt.ylabel(ylabel=y,fontsize=18)
        plt.show()