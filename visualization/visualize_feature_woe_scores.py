import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
sns.set()


def woe_plot(plot_df, bar_width=0.32, figsize=(10, 6), rotation=45, 
             save_img=False):
  ''' Line graph of Weight of Evidence scores for bins

  Parameters
  ----------
  df: Dataframe of feature metrics including feature bins, WoE and 
  Information Value (if available)

  figsize: Numerical tuple of figure size
  rotation: Rotation of xtick labels

  Returns
#   -------
  Plot showing the distribution of feature bins and the relationship between 
  bins and WoE scores
  '''

  #Creating two axis of different scales with one placed on the right side of 
  #the plot
  fig, ax_woe = plt.subplots(figsize=figsize)
  ax_distr = ax_woe.twinx()

  #Plotting the WoE scores for the bins
  sns.pointplot(data=plot_df, x=plot_df.columns[0], y=plot_df.columns[8], 
                linestyle='-', ax=ax_woe)
  ax_woe.set_xlabel(f'{plot_df.columns[0][:-4]} bin')
  ax_woe.set_xticklabels(labels=plot_df.iloc[:, 0], rotation=rotation)
  ax_woe.set_ylabel('WoE')
  ax_woe.legend(['WoE'], bbox_to_anchor=(1.3, 1), facecolor='inherit')

  #Plotting the Distribution of the bins
  sns.barplot(data=plot_df, x=plot_df.columns[0], y=plot_df.columns[1], 
              ax=ax_distr, color='teal', alpha=0.7)
  ax_distr.set_xlabel(f'{plot_df.columns[0]}')
  ax_distr.set_ylabel('Count')
  ax_distr.legend(['Count'], bbox_to_anchor=(1.3, 0.9), facecolor='teal')

  #Change bar width
  def change_width(ax, new_width):
    for patch in ax.patches:
      current_width = patch.get_width()
      diff = current_width - new_width

      patch.set_width(new_width)

      patch.set_x(patch.get_x() + diff * 0.5)
  
  change_width(ax_distr, bar_width)



  #Adding a title and information value information to the plot
  plt.suptitle(f'WoE scores and distribution of {plot_df.columns[0][:-4]} bins')
  plt.text(1, 0.7, f"Information Value= {plot_df['IV'][0]:.3f}", 
           transform=plt.gcf().transFigure)

  if save_img:
    IMAGE_PATH = '../../reports/figures/'
    image_file = f'{IMAGE_PATH}{plot_df.columns[0][:-4]}_woe_plot.png'
    fig.savefig(image_file, bbox_inches='tight')

  plt.show()

def bad_prob_plot(plot_df, bar_width=0.32, figsize=(10, 6), rotation=45, 
             save_img=False):
  ''' Line graph of Weight of Evidence scores for bins

  Parameters
  ----------
  df: Dataframe of feature metrics including feature bins, WoE and 
  Information Value (if available)

  figsize: Numerical tuple of figure size
  rotation: Rotation of xtick labels

  Returns
#   -------
  Plot showing the distribution of feature bins and the relationship between 
  bins and WoE scores
  '''

  #Creating two axis of different scales with one placed on the right side of 
  #the plot
  fig, ax_woe = plt.subplots(figsize=figsize)
  ax_distr = ax_woe.twinx()

  #Plotting the Bad probability scores for the bins
  sns.pointplot(data=plot_df, x=plot_df.columns[0], y=plot_df.columns[2], 
                linestyles='-', ax=ax_woe)
  ax_woe.set_xlabel(f'{plot_df.columns[0][:-4]} bin')
  ax_woe.set_xticklabels(labels=plot_df.iloc[:, 0], rotation=rotation)
  ax_woe.set_ylabel('bad_probability')
  ax_woe.legend(['bad_probability'], bbox_to_anchor=(1.3, 1), facecolor='inherit')

  #Plotting the Distribution of the bins
  sns.barplot(data=plot_df, x=plot_df.columns[0], y=plot_df.columns[1], 
              ax=ax_distr, color='teal', alpha=0.7)
  ax_distr.set_xlabel(f'{plot_df.columns[0]}')
  ax_distr.set_ylabel('Count')
  ax_distr.legend(['Count'], bbox_to_anchor=(1.3, 0.9), facecolor='teal')

  #Change bar width
  def change_width(ax, new_width):
    for patch in ax.patches:
      current_width = patch.get_width()
      diff = current_width - new_width

      patch.set_width(new_width)

      patch.set_x(patch.get_x() + diff * 0.5)
  
  change_width(ax_distr, bar_width)



  #Adding a title and information value information to the plot
  plt.suptitle(f'Bad probability scores and distribution of {plot_df.columns[0][:-4]} bins')
  plt.text(1, 0.7, f"Information Value= {plot_df['IV'][0]:.3f}", 
           transform=plt.gcf().transFigure)

  if save_img:
    IMAGE_PATH = '../../reports/figures/'
    image_file = f'{IMAGE_PATH}{plot_df.columns[0][:-4]}_bad_prob_plot.png'
    fig.savefig(image_file, bbox_inches='tight')

  plt.show()