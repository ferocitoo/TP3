import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib.collections import PathCollection

from matplotlib.patches import Rectangle
from matplotlib.patches import ConnectionPatch


dark_blue = (0.0, 0.0, 0.4)
light_blue = (0.6, 0.8, 1.0)

dark_orange = (0.9, 0.5, 0.0)
light_orange = (1.0, 0.8, 0.2)

dark_green = (0, 0.3, 0.6)
light_green = (0.2, 0.7, 0.3)

dark_pink= (0.3,0.2,0.8)
light_pink = (1,0.5,0.9)

def convert_RGB_01(r,g,b) :
    r = r/255
    g = g/255
    b = b/255
    
    return (r,g,b)

blue = convert_RGB_01(59.0,117.0,175.0)
red = convert_RGB_01(197.0,58.0,50)
orange = convert_RGB_01(238.0,138.0,54.0)
purple = convert_RGB_01(118.0,45.0,121.0)
green = convert_RGB_01(145.0,188.0,134.0)
dark_yellow = convert_RGB_01(184.0,156.0,61.0)


def remove_ticks(axis,indexes_to_remove, ax = plt) :
    if axis == 'x' : 
        current_ticks = ax.get_xticks()
        current_ticks = list(current_ticks)
        for index in sorted(indexes_to_remove, reverse=True):
            del current_ticks[index]
        ax.set_xticks(current_ticks)
    if axis == 'y' : 
        current_ticks = ax.get_yticks()
        current_ticks = list(current_ticks)
        for index in sorted(indexes_to_remove, reverse=True):
            del current_ticks[index]
        ax.set_yticks(current_ticks)
        
def add_ticks(axis,new_ticks_values, new_ticks_texts,text_size = 25, ax = plt, divide = 1) :
    if axis == 'x' : 
        new_ticks_positions = list(ax.get_xticks()) + new_ticks_values
        new_ticks_labels = [tick/(10**divide) if tick not in new_ticks_values else new_ticks_texts[new_ticks_values.index(tick)] for tick in new_ticks_positions]

        ax.set_xticks(new_ticks_positions)
        ax.set_xticklabels(new_ticks_labels)
        
        for i in range(len(new_ticks_values)) : 
            ticks_labels = ax.get_xticks()
            ticks = ax.get_xticklabels()
            index = np.where(ticks_labels == new_ticks_values[i])[0]
            ticks[index[0]].set_fontsize(text_size)
            
    if axis == 'y' :
        new_ticks_positions = list(ax.get_yticks()) + new_ticks_values
        new_ticks_labels = [str(tick) if tick not in new_ticks_values else new_ticks_texts[new_ticks_values.index(tick)] for tick in new_ticks_positions]

        ax.set_yticks(new_ticks_positions)
        ax.set_yticklabels(new_ticks_labels)
        
        for i in range(len(new_ticks_values)) : 
            ticks_labels = ax.get_yticks()
            ticks = ax.get_yticklabels()
            index = np.where(ticks_labels == new_ticks_values[i])[0]
            ticks[index[0]].set_fontsize(text_size)
            
    first_x = ax.get_xlim()[0]
    last_x = ax.get_xlim()[1]
    
    range_x = abs(last_x - first_x)
    
    last_y = ax.get_ylim()[1]
    first_y = ax.get_ylim()[0]
    
    range_y = abs(last_y - first_y)
    
    if axis == 'x' :
        ax.text(last_x-range_x*0.05 ,first_y-range_y*0.12, f'1e{divide}', fontsize = 12)
    
    if axis == 'y' :
        ax.text(first_x-range_x*0.12 ,last_y-range_y*0.05, f'1e{divide}', fontsize = 12)
    



def zoom_in_plot(ax,X,Y,xlim,ylim,x_zoom,y_zoom, colors,markers,linesytles,corner1 = ["top","top"],corner2 = ["top","top"],multiply = 2, linewidth = 2, title = "", ticks = False) :
    N = len(X)
    x_range = abs(ax.get_xlim()[1] - ax.get_xlim()[0])
    y_range = abs(ax.get_ylim()[1] - ax.get_ylim()[0])
    
    width_zoom = multiply*abs(xlim[1] - xlim[0])/x_range
    height_zoom = multiply*abs(ylim[1] - ylim[0])/y_range

    x_zoom_top_left_corner = ax.get_xlim()[0] + x_zoom*x_range 
    y_zoom_top_left_corner = ax.get_ylim()[0] + y_zoom*y_range  + y_range * height_zoom
    
    x_zoom_low_right_corner = ax.get_xlim()[0] + x_zoom*x_range  + x_range * width_zoom
    y_zoom_low_right_corner = ax.get_ylim()[0] + y_zoom*y_range 
    
    x_zoom_top_right_corner = ax.get_xlim()[0] + x_zoom*x_range  + x_range * width_zoom
    y_zoom_top_right_corner = ax.get_ylim()[0] + y_zoom*y_range  + y_range * height_zoom
    
    x_zoom_low_left_corner = ax.get_xlim()[0] + x_zoom*x_range
    y_zoom_low_left_corner = ax.get_ylim()[0] + y_zoom*y_range



    axin = ax.inset_axes([x_zoom, y_zoom, width_zoom, height_zoom])  # [x, y, width, height]
    for i in range(N) :
        axin.plot(X[i], Y[i], color = colors[i], linestyle = linesytles[i], marker = markers[i])
    axin.grid()
    axin.set_xlim(xlim[0], xlim[1])
    axin.set_ylim(ylim[0], ylim[1])
    
    axin.set_title(title)
    
    if not ticks :
        axin.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

    rect = Rectangle((xlim[0], ylim[0]), xlim[1] - xlim[0], ylim[1] - ylim[0], 
                 linewidth=linewidth, edgecolor='black', facecolor='none')
    ax.add_patch(rect)
    
    # Add an arrow annotation from the center of the rectangle to the zoomed plot
    arrow_props = dict(arrowstyle="-", color='black',zorder=10) 
    
    if corner1[0] == "top" :
        x1 = x_zoom_top_left_corner
        y1 = y_zoom_top_left_corner
    else :
        x1 = x_zoom_low_left_corner
        y1 = y_zoom_low_left_corner

    if corner1[1] == "top" :
        x2 = x_zoom_top_right_corner
        y2 = y_zoom_top_right_corner
    else :
        x2 = x_zoom_low_right_corner
        y2 = y_zoom_low_right_corner

        
        
    if corner2[0] == "top" :
        xx1 = xlim[0]
        yy1 = ylim[1]
    else :
        xx1 = xlim[0]
        yy1 = ylim[0]
        
    if corner2[1] == "top" :
        xx2 = xlim[1]
        yy2 = ylim[1]
    else :
        xx2 = xlim[1]
        yy2 = ylim[0]
        
        
        
        
    arrow = ConnectionPatch((xx1, yy1), (x1, y1), "data", "data", **arrow_props)
    ax.add_artist(arrow)

    arrow = ConnectionPatch((xx2, yy2), (x2, y2), "data", "data", **arrow_props)
    ax.add_artist(arrow)
    
    


def return_number_with_precision(number, n, abs = False):
    if abs : 
        number = np.abs(number)
    if np.abs(number) < 10 ** (-n):
        formatted_number = 10**(-n)
    else:
        formatted_number = "{:.{}f}".format(number, n)
    return formatted_number

def linear_fit(x,y,color,
               
               #digits after coma
               precisions,
               
               #multiply a by 10^multiply
               multiply = 0,
               
               ax = plt, bounds = [0,0],
               
               a_err= 0, b_err = 0, linewidth = 3, plot = True) :
    x = remove_nan_values(x)
    y = remove_nan_values(y)
    n = len(x)
    
    abool = False
    bbool = False
    
    if a_err != 0 :
        abool = True
        
    if b_err != 0 :
        bbool = True
    
    a,b = np.polyfit(x,y,deg=1)

    if bounds != [0,0] : 
        x_fit = np.linspace(bounds[0],bounds[1],10)
    else : 
        x_fit = np.linspace(x.min(),x.max(),10)
        
    a_error = np.sqrt(np.sum((y-a*x-b)**2)/((n-2)*np.sum((x-np.mean(x))**2)))
    b_error = np.sqrt(np.sum(x**2)/n) * a_error
    
    y_fit = a*x_fit + b
    
    to_return = [a,a_error,b,b_error]
    
    
    if multiply != 0 : 
        a = a * pow(10,multiply)
        a_error = a_error * pow(10,multiply)
        b = b * pow(10,multiply)
        b_error = b_error * pow(10,multiply)
        
    if abool :
        a_error = a_err
        
    if bbool :
        b_error = b_err
        
    b_sign = np.sign(b)

    a_error = return_number_with_precision(a_error,precisions[0])
    b_error = return_number_with_precision(b_error,precisions[1])
    a = return_number_with_precision(a,precisions[0])
    b = return_number_with_precision(b,precisions[1],abs = True)
    print(b)
    
    
    if multiply != 0 : 
        label = rf"$y = [({a} \pm {a_error})x" + f"{'+' if b_sign > 0 else '-'}"+ rf"({b} \pm {b_error})] \cdot 10^{-multiply}$"
    else : 
        label = rf"$y = ({a} \pm {a_error})x" + f"{'+' if b_sign > 0 else '-'}"+ rf"({b} \pm {b_error}) $"
    
    if plot :
        ax.plot(x_fit,y_fit,color=color,label=label,linewidth = linewidth,linestyle="--")
    
    return to_return


def remove_nan_values(input_list):
    return np.array([value for value in input_list if not np.isnan(value)])

def scatter_multiple (colors, labels, X,Y,markers, hide = [], markersize = 10, ax = plt, alpha = 1) : 
    n = len(X)
    
    for i in range(n): 
        if not i in hide : 
            ax.scatter(X[i],Y[i],label = labels[i], marker = markers[i], color = colors[i], s=markersize, alpha=alpha) 

def errorbars_multiple (colors, labels, X,Y,markers, Y_error, X_error = [], ecolors = [], markersize = 20, capsize = 10, capthick = 1,ax = plt) : 
    n = len(X)
    
    if ecolors == [] : 
        ecolors = colors
        
    if X_error == [] : 
        for i in range(n): 
            ax.errorbar(X[i],Y[i],label = labels[i], yerr=Y_error[i], linestyle = '', marker=markers[i], capsize=capsize,capthick=capthick,ecolor=ecolors[i],color=colors[i],markersize=markersize)
    else : 
        for i in range(n):  
            ax.errorbar(X[i],Y[i],label = labels[i], xerr=X_error[i], yerr=Y_error[i], linestyle = '', marker=markers[i], capsize=capsize,capthick=capthick,ecolor=ecolors[i],color=colors[i],markersize=markersize)

def create_figure_and_apply_format(figsize,
                                   #label and axis settings
                                   xlabel, ylabel, xy_fontsize=22, tick_fontsize=18, 
                                   
                                   #grid or not
                                   grid_bool = True) : 
    
    fig = create_fig(figsize)
    
    ax = fig.gca()
    
    set_axis_and_tick_properties(ax,xlabel, ylabel, xy_fontsize, tick_fontsize)
    
    #set_legend_properties(ax, ncol, loc, fontsize, fontweight, fontstyle, text_color, border_color, border_linewidth)
    
    if grid_bool : 
        ax.grid()
    
    plt.tight_layout()
    
    return ax,fig
    

def create_fig(figsize) : 
    return plt.figure(figsize = figsize) 


def import_csv(filename = "CSV.csv") : 
    file_path = filename

    with open(file_path, 'r') as file:
        lines = file.readlines()

    modified_lines = [line.replace(',', '.') for line in lines]

    with open(file_path, 'w') as file:
        file.writelines(modified_lines)



    data = np.genfromtxt(file_path, delimiter=';', skip_header=1,  dtype=float)

    return data


def set_legend_properties(ax,markersize=15,ncol = 1, loc = "best", fontsize=20, fontweight='normal', fontstyle='italic', text_color='black', border_color='black', border_linewidth=2):
        
    # i = 0
    # j= 0
    
    # proxy_artists = [0 for i in range(len(handles))]
    
    # for h in handles :
    #     if type(h) == Line2D :
    #         proxy_artists[i] = h
    #     if type(h) ==  PathCollection:
    #         proxy_artists[i] = Line2D([0], [0], linestyle='', marker=markers[j], markersize=markersize, color=colors[j])
    #         j+=1
    #     i+=1
    
    
    legend = ax.legend(ncol = ncol, loc = loc)
    

    for label in legend.get_texts():
        label.set_fontsize(fontsize)
        label.set_fontweight(fontweight)
        label.set_fontstyle(fontstyle)
        label.set_color(text_color)

    legend.set_frame_on(True)
    legend.get_frame().set_linewidth(border_linewidth)
    legend.get_frame().set_edgecolor(border_color)
    
    plt.tight_layout()
 
def set_axis_and_tick_properties(ax,x_label, y_label, xy_fontsize=22, tick_fontsize=18):
    ax.set_xlabel(x_label, fontsize=xy_fontsize)
    ax.set_ylabel(y_label, fontsize=xy_fontsize)
    ax.tick_params('both', labelsize = tick_fontsize)

def savefig(fig, filename, ext = "png") :
    if ext == "pdf" : 
        fig.savefig("pdf/" + filename + "." + ext)
    else : 
        fig.savefig("png/" + filename + "." + ext)

def save_png(fig, png_name = "test.png") : 
    fig.savefig("png/" + png_name)
    
def save_pdf(fig,pdf_name) : 
    fig.savefig("pdf/" + pdf_name)
    
def x_axis_divide(ax,divide = 1000) : 
    
    def divide_by(x, pos):
        return f'{x/divide:.2f}'
    formatter = FuncFormatter(divide_by)

    ax.xaxis.set_major_formatter(formatter)
        
def y_axis_divide(ax,divide = 1000) : 
    def divide_by(y,pos):
        return f'{y/divide:.2f}'
    formatter = FuncFormatter(divide_by)

    ax.yaxis.set_major_formatter(formatter)
    