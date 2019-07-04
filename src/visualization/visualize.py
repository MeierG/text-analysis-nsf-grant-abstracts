import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path

from bokeh.io import output_notebook, output_file, show, export_png
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Range1d, DataRange1d, Plot, CategoricalColorMapper, LabelSet, BoxAnnotation
from bokeh.models.tools import HoverTool
from bokeh.models.glyphs import Circle, Text
from bokeh.palettes import Spectral5
from bokeh.transform import factor_cmap
from bokeh.models.annotations import Label
from bokeh.io import export_svgs
from bokeh.layouts import gridplot
from bokeh.models import LinearColorMapper, ColorBar
from bokeh.transform import transform

output_notebook()

data_path = Path("../data/")
img_path = Path("../reports/figures/")

def plot_hist(df, prefix=None, colname = None, title = 'Award amounts\n', xlabel='Award amount (log10)'):
    """return log transgromed histogram of desired column with fixed paramenters
    and layout"""

    img_path = Path("../reports/figures/")

    _=plt.rcParams['figure.figsize'] = [14,6]
    _=plt.style.use(['fivethirtyeight'])
    _=plt.rcParams['xtick.labelsize']=15
    _=plt.rcParams['ytick.labelsize']=15

    # plot histogram using log
    _=plt.hist(np.log10(df[colname]), bins=200, color= '#3182bd')
    _=plt.title(title, fontsize = 30)

    #plot logtransformed col
    _=plt.xlabel(xlabel, fontsize=25)
    _=plt.ylabel('number of awards', fontsize=25)

    # add padding for title and labels
    _=plt.tight_layout()
    # save figure in the figures folder
    #_=plt.savefig(img_path /'hist.png', dpi = 250, pad_inches=0.5)
    image_name = str(prefix) +'hist.png'
    _=plt.savefig(img_path / image_name, dpi = 250, pad_inches=0.5)

    # return the plot
    plt.show()

def plot_boxplot(df, colname = None,
                 title = 'Out of the ordinary billion dollar funded proposal\n',
                vertical_line=9):

    """return boxplot of desidered column"""

    # set parameters for the box plot
    _=plt.rcParams['figure.figsize'] = [14,6]
    _=plt.style.use(['fivethirtyeight'])

    # use log transform to plot the award amountt
    ax = sns.boxplot(x=np.log10(df[colname]), linewidth=1, color = '#3182bd')
    _=plt.title(title, fontsize = 30)
    _=plt.xlabel('Award amount (log10)', fontsize=25)

    # show location of billion dollar grant
    _=plt.axvline(vertical_line, c = '#31a354', animated = True, linewidth = 4, linestyle = '--')

    # add padding for title and labels
    _=plt.tight_layout()

    # save plot
    _=plt.savefig(img_path / 'boxplot.png', dpi = 220, pad_inches=0.5)

    plt.show()


def plot_fundingOvertime(df, col1, col2, col_transform = 1000000000, left=2015, right=2016.5):
    """return interactive line plot using bokeh"""

    print('\n*** INTERACTIVE MODE: HOVER OVER THE GRAPH TO SEE AWARD TOTALS FOR EACH YEAR***')
    grouped = pd.DataFrame(df.groupby([col1])[col2].sum())
    grouped.reset_index(inplace=True)

# set amounts by billion dollars
    grouped[col2]=grouped[col2]/col_transform
    source = ColumnDataSource(grouped)

# initialize the figure
    p = figure(plot_width = 1000,
               plot_height = 450,
               title = 'Award funding has increased over time with 2011 seeing the largest funding amounts')

    # create the plot
    p.line(x=col1,
           y=col2,
           line_width=6,
           source=source, color = 'green')

    # set formating parameters
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.background_fill_color = "AliceBlue"
    p.title.text_font_size = "16pt"
    p.title.text_color = 'MidnightBlue'
    p.xaxis.axis_label_text_font_size = '15pt'
    p.yaxis.axis_label_text_font_size = '15pt'
    p.yaxis.axis_label = 'Amount awarded in US Billion'
    p.xaxis.major_label_text_font_size = '12pt'

    # add shaded box to highlight year with greatest funding
    box = BoxAnnotation(left=left, right=right,
                    line_width=1,
                    line_color='black',
                    line_dash='dashed',
                    fill_alpha=0.2,
                    fill_color='green')
    # add box to plot
    p.add_layout(box)

    # create label for the box
    label = Label(x=2016,
                  y=6.220,
                  x_offset=12,
                  text="$6.22 b.awarded in 2016",
                  text_baseline="middle")

    # add to plot
    p.add_layout(label)

    # add interactive hover tool that shows the amount awarded
    hover = HoverTool()
    hover.tooltips = [("Total amount awarded ", "@AwardAmount")]

    hover.mode = 'vline'
    p.add_tools(hover)

    # export plots
    _=export_png(p, filename = img_path / 'fundingovertime.png')
    output_file(img_path/'fundingovertime.html')

    p.output_backend = "svg"
    export_svgs(p, filename=img_path/"fundingovertime.svg")

    #display plot
    show(p)

def plot_projectsOverTime(df, col, column_line = None, operation = 'count'):
    """plot count of projects over time"""

    print('\n*** INTERACTIVE MODE: HOVER OVER THE GRAPH TO SEE COUNTS FOR EACH YEAR***')

    # create a subsett of year and number of projects
    counts = df.groupby([col]).agg(operation)
    counts.reset_index(inplace = True)

    # create a column data source to plot in bokeh
    source = ColumnDataSource(counts)

    # initialize the plot
    p = figure(plot_width = 1000,
           plot_height = 450,
           title = 'The largest number of approved proposals were funded for 2016')

    # plot the trend line
    p.line(x=col, y=column_line,
       line_width=6, source=source)

    # set parameters
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.background_fill_color = "AliceBlue"
    p.title.text_font_size = "16pt"
    p.title.text_color = 'MidnightBlue'
    p.xaxis.axis_label_text_font_size = '15pt'
    p.yaxis.axis_label_text_font_size = '15pt'

    p.yaxis.axis_label = 'Total number of approved projects'
    p.xaxis.major_label_text_font_size = '12pt'

    # create annotation
    box = BoxAnnotation(left=2014.5, right=2016.5,
                    line_width=1, line_color='black', line_dash='dashed',
                    fill_alpha=0.2, fill_color='orange')

    # add annotation to plot
    p.add_layout(box)

    # create and set label
    label = Label(x=2016, y=12872, x_offset=12, text="12,872 projects in 2016", text_baseline="middle")
    p.add_layout(label)

    # add interactive hover tool
    hover = HoverTool()
    hover.tooltips = [("Total number of projects ", "@AbstractNarration"), ('year', '@year')]

    hover.mode = 'vline'
    p.add_tools(hover)

    # export plots
    _=export_png(p, filename = img_path / 'projectsovertime.png')
    output_file(img_path/'projectsovertime.html')

    p.output_backend = "svg"
    export_svgs(p, filename=img_path/"projectsovertime.svg")

    #display plot
    show(p)


def plot_awardsPerDivision(df, title, NSF_org, aggregate = list(),
                           sortby=tuple(),
                           n_directs = 5):

    # set a subset to get statistics on awards by directorate
    directorates = df.groupby([NSF_org]).\
    agg((aggregate)).\
    sort_values(sortby,
    ascending = False).head(n_directs)

    directorates.columns = directorates.columns.get_level_values(1)

    directorates.reset_index(inplace = True)

    # set datasource to visualize in Bokeh
    source = ColumnDataSource(directorates)
    direct = source.data[NSF_org].tolist()

    # initialize the plot
    p = figure(x_range=direct, plot_width = 1000,
           plot_height = 550,
           tools = 'box_select, wheel_zoom, reset, save')

    # set color for each categoory
    color_mapper = CategoricalColorMapper(factors = direct,
                                     palette=['MidnightBlue',
                                              'DodgerBlue',
                                              'CornflowerBlue',
                                              'DeepSkyBlue', 'grey'])

    # create plot
    p.circle(x = NSF_org,
             y = 'count',
             source = source,
             color = dict(transform = color_mapper, field=NSF_org),
             nonselection_alpha = 0.2,
             size = 35,
             legend = NSF_org)

    # set plot formating and displaying options
    p.title.text = title
    p.title.text_font_size = "15pt"
    p.title.text_color = 'MidnightBlue'
    p.xaxis.axis_label_text_font_size = '14pt'
    p.yaxis.axis_label_text_font_size = '14pt'
    p.xaxis.axis_label = 'Directorate'
    p.yaxis.axis_label = 'Number of projects funded'
    p.xaxis.major_tick_line_color = None
    p.xaxis.minor_tick_line_color = None
    p.xaxis.major_label_text_font_size = '0pt'  # turn off x-axis tick labels

    # create hover tool
    hover = HoverTool()
    hover.tooltips = [("Directorate ", " @NSF_org"),
                  ("Counts ", "@count"),
                  ("Average funding", "@mean"),
                  ("Max funding awarded ", "@max")]

    hover.mode = 'vline'
    p.add_tools(hover)

    # export plots
    _=export_png(p, filename = img_path / 'awardsbydir.png')
    output_file(img_path/'awardsbydir.html')

    p.output_backend = "svg"
    export_svgs(p, filename=img_path/"awardsbydir.svg")

    #display plot
    show(p)
    print(directorates)


def plot_wordsOverTime(df, col, column_line = None, operation = 'count', title = 'Words over time'):
    """plot count of projects over time"""

    print('\n*** INTERACTIVE MODE: HOVER OVER THE GRAPH TO SEE COUNTS FOR EACH YEAR***')

    # create a subsett of year and number of projects
    counts = df.groupby([col]).agg(operation)
    counts.reset_index(inplace = True)

    # create a column data source to plot in bokeh
    source = ColumnDataSource(counts)

    # initialize the plot
    p = figure(plot_width = 1000,
           plot_height = 450,
           title = title)

    # plot the trend line
    p.line(x=col, y=column_line,
       line_width=6, source=source)

    # set parameters
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.background_fill_color = "AliceBlue"
    p.title.text_font_size = "16pt"
    p.title.text_color = 'MidnightBlue'
    p.xaxis.axis_label_text_font_size = '15pt'
    p.yaxis.axis_label_text_font_size = '15pt'

    p.yaxis.axis_label = 'Total number of words'
    p.xaxis.major_label_text_font_size = '12pt'

    # create annotation
    box = BoxAnnotation(left=2014.5, right=2016.5,
                    line_width=1, line_color='black', line_dash='dashed',
                    fill_alpha=0.2, fill_color='orange')

    # add annotation to plot
    p.add_layout(box)

    # add interactive hover tool
    hover = HoverTool()
    hover.tooltips = [("Total number of words ", "@word_count"), ('year', '@year')]

    hover.mode = 'vline'
    p.add_tools(hover)

    # export plots
    _=export_png(p, filename = img_path / 'wordsovertime.png')
    output_file(img_path/'wordsovertime.html')

    p.output_backend = "svg"
    export_svgs(p, filename=img_path/"wordsovertime.svg")

    #display plot
    show(p)


def plot_fleschScore(df, bins=100):

    _=sns.scatterplot(x="flesch_score", y="amount_awarded_per_day", data=df)
    _=plt.rcParams['figure.figsize'] = [14,6]
    _=plt.style.use(['fivethirtyeight'])
    _=plt.rcParams['xtick.labelsize']=15
    _=plt.rcParams['ytick.labelsize']=15

    #plot logtransformed col
    _=plt.xlabel('flesch score', fontsize=25)
    _=plt.ylabel('amount_awarded_per_day', fontsize=25)
    _=plt.title('Flesch reading ease vs. Amount awarded', fontsize = 30)

    # add padding for title and labels
    _=plt.tight_layout()
    # save figure in the figures folder
    _=plt.savefig(img_path /'fleschscore.png', dpi = 250, pad_inches=0.5)

    _=plt.show()

    _=plt.hist(df['flesch_score'], bins)
    _=plt.title('Flesch reading ease', fontsize = 30)
    _=plt.rcParams['figure.figsize'] = [14,6]
    _=plt.style.use(['fivethirtyeight'])
    _=plt.rcParams['xtick.labelsize']=15
    _=plt.rcParams['ytick.labelsize']=15
    # return the plot
    plt.show()

def plot_wordsAwards(df,amount_awarded_per_day,
                     word_count,
                     y_axis_type='log',
                     title= 'Funding vs Number of words used in the abstract'):

    wm = df[[amount_awarded_per_day, word_count]]

    source = ColumnDataSource(wm)
    color_mapper = LinearColorMapper(palette="Viridis256",
                                 low=wm['amount_awarded_per_day'].min(),
                                 high=wm['amount_awarded_per_day'].max())

    p = figure(plot_width = 1000,
           plot_height = 450,
           tools = 'box_select, wheel_zoom, reset, save',
           toolbar_location='above',
           x_axis_label='word count',
           y_axis_label='amount awarded per day (log)',
           y_axis_type=y_axis_type)

    p.circle(x=word_count,
         y=amount_awarded_per_day,
         color=transform(amount_awarded_per_day,
                         color_mapper),
         size=10, alpha=0.6, source=wm)
    p.title.text =title

    color_bar = ColorBar(color_mapper=color_mapper,
                     label_standoff=15,
                     location=(0,0), title='Amount awarded per day')

    p.add_layout(color_bar, 'right')
    p.title.text_font_size = "16pt"
    p.title.text_color = 'MidnightBlue'
    p.xaxis.axis_label_text_font_size = '15pt'
    p.yaxis.axis_label_text_font_size = '15pt'

    # export plots
    _=export_png(p, filename = img_path / 'wordawards.png')
    #output_file(img_path/'file.html')

    #p.output_backend = "svg"
    #export_svgs(p, filename=img_path/"file.svg")

    #display plot
    show(p)

def plot_scatter(x, y, data, title, xlabel, ylabel, prefix="first", scale = None, color='#3182bd'):
    try:
        _=sns.set_style("dark")
        _=sns.scatterplot(x=x, y=y, data=data, color='#3182bd')
        _=plt.rcParams['figure.figsize'] = [14,6]
        _=plt.xscale(scale)
        _=plt.grid = None
        _=plt.style.use(['fivethirtyeight'])
        _=plt.rcParams['xtick.labelsize']=15
        _=plt.rcParams['ytick.labelsize']=15


        _=plt.xlabel(xlabel, fontsize=25)
        _=plt.ylabel(ylabel, fontsize=25)
        _=plt.title(title, fontsize = 30)

        # add padding for title and labels
        _=plt.tight_layout()

        image_name = prefix +'scatterplot.png'
        _=plt.savefig(img_path / image_name, dpi = 250, pad_inches=0.5)

        plt.show()
    except AttributeError:
        plt.show()

def uniqueFunding(df):

    uniquewords_funding = df[['amount_awarded_per_day', 'unique_words']]

    source = ColumnDataSource(uniquewords_funding)
    color_mapper = LinearColorMapper(palette="Viridis256",
                                 low=uniquewords_funding['amount_awarded_per_day'].min(),
                                 high=uniquewords_funding['amount_awarded_per_day'].max())

    p = figure(plot_width = 1000, plot_height = 450,
           tools = 'box_select, wheel_zoom, reset, save',
           x_axis_label='unique word count',
           y_axis_label='amount awarded per day (log)',
           toolbar_location='above',
           y_axis_type="log")

    p.circle(x='unique_words',
         y='amount_awarded_per_day',
         color=transform('amount_awarded_per_day', color_mapper),
         size=10, alpha=0.5,
         source=uniquewords_funding)

    color_bar = ColorBar(color_mapper=color_mapper,
                     label_standoff=15, location=(0,0),
                     title='Amount per day')

    p.add_layout(color_bar, 'right')
    p.title.text ='Funding vs Number of unique words in the abstract'
    p.title.text_font_size = "16pt"
    p.title.text_color = 'MidnightBlue'
    p.xaxis.axis_label_text_font_size = '15pt'
    p.yaxis.axis_label_text_font_size = '15pt'

    p.add_layout(color_bar, 'right')
    p.title.text ='Funding vs Number of unique words in the abstract'
    p.title.text_font_size = "16pt"
    p.title.text_color = 'MidnightBlue'
    p.xaxis.axis_label_text_font_size = '15pt'
    p.yaxis.axis_label_text_font_size = '15pt'

    show(p)
def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""

    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, 1+n) / n

    return x, y

def ecdf_words(df):
    wordcount_x, wordcount_y = ecdf(df['word_count'])
    uniqueword_x, uniqueword_y = ecdf(df['unique_words'])

    _=plt.rcParams['figure.figsize'] = [14,6]
    _=plt.style.use(['fivethirtyeight'])
    #_=plt.grid(b=None)
    _=plt.rcParams['xtick.labelsize']=15
    _=plt.rcParams['ytick.labelsize']=15
    _=sns.set_style(style = 'darkgrid')

    # Generate plot
    _=plt.plot(wordcount_x, wordcount_y, marker='*', linestyle = 'none', markersize = 8)
    _=plt.plot(uniqueword_x, uniqueword_y, marker='*', linestyle = 'none', markersize = 8)

    # Make the margins
    _=plt.margins(0.02)


    # Annotate the plot
    _=plt.title('Proportion of total words and unique words per abstract\n', fontsize = 25)

    _=plt.legend(('word count', 'unique words'),
             loc='lower right',
             fontsize = 25)


    _= plt.xlabel('Number of words per abstract', fontsize=25)
    _= plt.ylabel('ECDF', fontsize=25)

    # add padding for title and labels
    _=plt.tight_layout()

    # Display the plot
    _=plt.savefig(img_path / 'ecdf',dpi = 220, pad_inches=0.5)

    plt.show()

def plot_genderDist(df):
    """plot distribution of program officer gender by year"""
    palette = 'muted'

    gender = df.groupby(['year','NSF_org','predGender_po']).size().reset_index().sort_values(0, ascending = False)

    _=plt.rcParams["figure.figsize"] = [18,8]

    _=sns.catplot(x="year", y=0, hue="predGender_po",  kind="boxen", data=gender,height=7, aspect=2, legend_out = False, palette='muted')

    _=plt.ylabel('Number of signed proposals', fontsize = 20)
    _=plt.title('Number of signed proposals by gender', fontsize = 20)

    # add padding for title and labels
    _=plt.tight_layout()

    # Display the plot
    _=plt.savefig(img_path / 'gender.png',dpi = 220, pad_inches=0.5)

    plt.show()
