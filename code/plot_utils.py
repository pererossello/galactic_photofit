import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import subprocess
import shutil
import ffmpeg
import utils as ut


def simp_figure(
    fig_size=100,
    fig_w=1080, fig_h=1080,
    text_size=1, grid=True, theme="dark",
    color='#000000',
    dpi=300,
    layout='constrained'
):
    ratio = fig_w / fig_h
    fig_width = fig_w / dpi
    fig_height = fig_h / dpi
    fig_size = fig_width * fig_height
    fs = np.sqrt(fig_size)
    fig = plt.figure(
        figsize=(fig_width, fig_height),
        dpi=dpi,  # Default dpi, will adjust later for saving
        layout=layout,
    )



    ax = fig.subplots()

    if theme == 'dark':
        fig.patch.set_facecolor(color)
        plt.rcParams.update({"text.color": "white"})
        ax.set_facecolor(color)
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("white")
        ax.spines["top"].set_color("white")
        ax.spines["left"].set_color("white")
        ax.spines["right"].set_color("white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")

    ax.xaxis.set_tick_params(which="minor", bottom=False)
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=1.5 * text_size * fs,
        size=fs * 0.5,
        width=fs * 0.15,
    )
    if grid:
        ax.grid(
            which="major",
            linewidth=fs * 0.015,
            color="white" if theme == "dark" else "black",
        )
    for spine in ax.spines.values():
        spine.set_linewidth(fs * 0.15)

    # axes equal
    #ax.set_aspect("equal")
            
    return fig, ax, fs

def figure_skeleton(
    fig_size=20, ratio=1,
    fig_w=512, fig_h=512,
    text_size=1, 
    dpi=300,
    layout=None,
    tick_direction='in',
    minor=True,
    top_bool=False
):
    if ratio is not None:
        fs = np.sqrt(fig_size)
        fig = plt.figure(
            figsize=(np.sqrt(ratio * fig_size), np.sqrt(fig_size / ratio)),
            dpi=dpi,
            layout=layout,
        )
    else:
        dpi = dpi
        ratio = fig_w / fig_h
        fig_width = fig_w / dpi
        fig_height = fig_h / dpi
        fig_size = fig_width * fig_height
        fs = np.sqrt(fig_size)
        fig = plt.figure(
            figsize=(fig_width, fig_height),
            dpi=dpi,  # Default dpi, will adjust later for saving
            layout=layout,
        )

    # The parent GridSpec
    gs0 = gridspec.GridSpec(2, 3, figure=fig,
    wspace=0.125*fs, height_ratios=[1, 1])

    # Placeholder for storing the various axes
    axes = []

    # Loop through columns
    for col in range(3):
        # Create the top row subplot for this column
        ax_top = fig.add_subplot(gs0[0, col])
        axes.append([ax_top])

        # Create a child GridSpec for the bottom row subplot in this column
        gs_child = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs0[1, col], hspace=0, height_ratios=[4, 1])
        
        # Create the two subplots for this child GridSpec
        for row in range(2):
            ax = fig.add_subplot(gs_child[row, 0])
            axes[col].append(ax)

    def flatten(l):
        return [item for sublist in l for item in sublist]

    inner_pos=[0.235, 0.29, 0.07, 0.15] 
    ax_in = fig.add_axes(inner_pos)
    
    for i, ax in enumerate(flatten(axes)+[ax_in]):

        ax.grid(
            which="major",
            linewidth=fs * 0.015,
            color="black")
        
        for spine in ax.spines.values():
            spine.set_linewidth(fs * 0.15)

        wdth = 0.125
        lbs = 1 if i != 9 else 0.75
        top_bool = True if i != 9 else False
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=lbs * text_size * fs,
            size=fs * 0.5,
            width=fs * wdth,
            pad=0.45*fs,
            top=top_bool,
            direction=tick_direction
        )

        if minor:
            ax.minorticks_on()
            ax.tick_params(axis='both', which="minor", 
            direction=tick_direction,
            top=top_bool,
            size=fs * 0.25, width=fs * wdth,)

    for i in range(3):
        axes[i][1].xaxis.tick_top()

    return fig, axes, fs, ax_in

def initialize_figure(
    fig_size=20, ratio=1,
    fig_w=512, fig_h=512,
    text_size=1, subplots=(1, 1), grid=True, theme="dark",
    color='#222222',
    dpi=300,
    wr=None, hr=None, hmerge=None, wmerge=None,
    layout='constrained',
    hspace=None, wspace=None,
    tick_direction='out',
    minor=False,
    top_bool=False
):
    """
    Initialize a Matplotlib figure with a specified size, aspect ratio, text size, and theme.

    Parameters:
    fig_size (float): The size of the figure.
    ratio (float): The aspect ratio of the figure.
    text_size (float): The base text size for the figure.
    subplots (tuple): The number of subplots, specified as a tuple (rows, cols).
    grid (bool): Whether to display a grid on the figure.
    theme (str): The theme for the figure ("dark" or any other string for a light theme).

    Returns:
    fig (matplotlib.figure.Figure): The initialized Matplotlib figure.
    ax (list): A 2D list of axes for the subplots.
    fs (float): The scaling factor for the figure size.
    """
    if ratio is not None:
        fs = np.sqrt(fig_size)
        fig = plt.figure(
            figsize=(np.sqrt(ratio * fig_size), np.sqrt(fig_size / ratio)),
            dpi=dpi,
            layout=layout,
        )
    else:
        dpi = dpi
        ratio = fig_w / fig_h
        fig_width = fig_w / dpi
        fig_height = fig_h / dpi
        fig_size = fig_width * fig_height
        fs = np.sqrt(fig_size)
        fig = plt.figure(
            figsize=(fig_width, fig_height),
            dpi=dpi,  # Default dpi, will adjust later for saving
            layout=layout,
        )

    if wr is None:
        wr_ = [1] * subplots[1]
    else:
        wr_ = wr
    if hr is None:
        hr_ = [1] * subplots[0]
    else:
        hr_ = hr
    

    gs = mpl.gridspec.GridSpec(subplots[0], subplots[1], figure=fig, width_ratios=wr_, height_ratios=hr_, hspace=hspace, wspace=wspace)


    ax = [[None] * subplots[1] for _ in range(subplots[0])]

    if theme == "dark":
        fig.patch.set_facecolor(color)
        plt.rcParams.update({"text.color": "white"})

    for i in range(subplots[0]):
        for j in range(subplots[1]):
            
            if hmerge is not None:
                if i in hmerge:
                    ax[i][j] = fig.add_subplot(gs[i, :])
                else:
                    ax[i][j] = fig.add_subplot(gs[i, j])
            elif wmerge is not None:
                if j in wmerge:
                    ax[i][j] = fig.add_subplot(gs[:, j])
                else:
                    ax[i][j] = fig.add_subplot(gs[i, j])
            else:
                ax[i][j] = fig.add_subplot(gs[i, j])

            if theme == "dark":
                ax[i][j].set_facecolor(color)
                ax[i][j].tick_params(colors="white")
                ax[i][j].spines["bottom"].set_color("white")
                ax[i][j].spines["top"].set_color("white")
                ax[i][j].spines["left"].set_color("white")
                ax[i][j].spines["right"].set_color("white")
                ax[i][j].xaxis.label.set_color("white")
                ax[i][j].yaxis.label.set_color("white")

            #ax[i][j].xaxis.set_tick_params(which="minor", bottom=False)

            if grid:
                ax[i][j].grid(
                    which="major",
                    linewidth=fs * 0.015,
                    color="white" if theme == "dark" else "black",
                )
            for spine in ax[i][j].spines.values():
                spine.set_linewidth(fs * 0.15)

            ax[i][j].tick_params(
                axis="both",
                which="major",
                labelsize=1.5 * text_size * fs,
                size=fs * 0.5,
                width=fs * 0.15,
                top=top_bool,
                direction=tick_direction
            )

            if minor:
                ax[i][j].minorticks_on()
                ax[i][j].tick_params(axis='both', which="minor", 
                direction=tick_direction,
                top=top_bool,
                size=fs * 0.25, width=fs * 0.15,)

    if hmerge is not None:
        for k in hmerge:
            for l in range(1, subplots[1]):
                fig.delaxes(ax[k][l])

    if wmerge is not None:
        for k in wmerge:
            for l in range(1, subplots[0]):
                fig.delaxes(ax[l][k])
            
    
    return fig, ax, fs, gs



def plot_val(axs, xs, dic, lab, fs, ylim2=None, xlim=None, ylim1=None, ylabel1='', ylabel2='', quot=True, tls = 1):

    s = 0.1

    vals = dic[lab][0]
    vals_err = dic[lab][1]

    ax0, ax1 = axs[0], axs[1]

    #ax0.scatter(xs, vals, s=s*fs, color='k', marker='s')
    ax0.plot(xs, vals, lw=s*fs, color='k')
    lower_bound = vals - vals_err
    upper_bound = vals + vals_err
    ax0.fill_between(xs, lower_bound, upper_bound, 
    color='grey', alpha=0.5, edgecolor=None)

    if quot == True:
        err = vals_err/vals
    else:
        err = vals_err
    #ax1.scatter(xs, err, s=s*fs, color='k', marker='s')
    ax1.plot(xs, err, lw=s*fs, color='k')
    #ax1.set_ylim(*ylim2)

    #get current xtick values (ie xtick labels)
    # defines new xtick labels which are the old ones multiplied by a constnat R

    # for ax in ut.flatten(axs):
    #     ax.set_xlim(1, (xs*R)[-1])
    #     ax.tick_params(axis='both', labelsize=1.25*fs, pad=0.45*fs)

    R, d, fact = 0.396, 115.3e6, 206265
    current_xticks = ax1.get_xticks()[1:]
    new_xtick_labels = np.around(current_xticks * d / fact * 1e-3, decimals=1)
    ax0.xaxis.tick_top()
    ax0.set_xticklabels(new_xtick_labels)

    #set xlabel
    ax0.set_xlabel('Distance (kpc)', fontsize=fs*tls)
    ax1.set_xlabel('Distance (arcsec)', fontsize=fs*tls)
    ax0.xaxis.set_label_position('top')

    ax0.set_ylabel(lab, fontsize=fs*tls)
    ax1.set_ylabel(ylabel2, fontsize=fs*tls)

    ax0.xaxis.labelpad = 0.5 * fs
    ax0.yaxis.labelpad = 0.75 * fs

    ax1.xaxis.labelpad = 0.5 * fs
    ax1.yaxis.labelpad = 0.75* fs

    if xlim is not None:
        ax0.set_xlim(*xlim)
        ax1.set_xlim(*xlim)
    if ylim2 is not None:
        ax1.set_ylim(*ylim2)
    if ylim1 is not None:
        ax0.set_ylim(*ylim1)

    return 


def plot_val2(axs, xs, dic, lab, fs, fit, ylim2=None, xlim=None, ylim1=None, ylabel1='', ylabel2='', quot=True, tls = 1):

    s = 0.1

    vals = dic[lab][0]
    vals_err = dic[lab][1]
    vals_err_fit = vals - fit

    ax0, ax1 = axs[0], axs[1]

    #ax0.scatter(xs, vals, s=s*fs, color='k', marker='s')
    ax0.plot(xs, vals, lw=s*fs, color='k')
    lower_bound = vals - vals_err
    upper_bound = vals + vals_err
    ax0.fill_between(xs, lower_bound, upper_bound, 
    color='grey', alpha=0.5, edgecolor=None)

    if quot == True:
        err = vals_err_fit/vals
    else:
        err = vals_err_fit
    #ax1.scatter(xs, err, s=s*fs, color='k', marker='s')
    ax1.plot(xs, err, lw=s*fs, color='k')
    #ax1.set_ylim(*ylim2)

    #get current xtick values (ie xtick labels)
    # defines new xtick labels which are the old ones multiplied by a constnat R

    # for ax in ut.flatten(axs):
    #     ax.set_xlim(1, (xs*R)[-1])
    #     ax.tick_params(axis='both', labelsize=1.25*fs, pad=0.45*fs)

    R, d, fact = 0.396, 115.3e6, 206265
    current_xticks = ax1.get_xticks()[1:]
    new_xtick_labels = np.around(current_xticks * d / fact * 1e-3, decimals=1)
    ax0.xaxis.tick_top()
    ax0.set_xticklabels(new_xtick_labels)

    #set xlabel
    ax0.set_xlabel('Distance (kpc)', fontsize=fs*tls)
    ax1.set_xlabel('Distance (arcsec)', fontsize=fs*tls)
    ax0.xaxis.set_label_position('top')

    ax0.set_ylabel(lab, fontsize=fs*tls)
    ax1.set_ylabel(ylabel2, fontsize=fs*tls)

    ax0.xaxis.labelpad = 0.5 * fs
    ax0.yaxis.labelpad = 0.75 * fs

    ax1.xaxis.labelpad = 0.5 * fs
    ax1.yaxis.labelpad = 0.75* fs

    if xlim is not None:
        ax0.set_xlim(*xlim)
        ax1.set_xlim(*xlim)
    if ylim2 is not None:
        ax1.set_ylim(*ylim2)
    if ylim1 is not None:
        ax0.set_ylim(*ylim1)

    return 

def magplot(data, ax, fs, cmap='gist_ncar_r', vms=None, 
            cbar_label='$\mu_r$ (mag/arcsec$^2$)',
            cbar=True,
            wdth=0.15, lgth=0.35, tls=1, ls=1.1, pad=-0.035, shrink=1,
            title='', mask=None, mean=False, chi=None, xtop=True, yright=True):

    if vms is None:
        vms = [np.min(data), np.max(data)]
    else:
        if vms[0] is None:
            vms[0] = np.min(data)
        if vms[1] is None:
            vms[1] = np.max(data)

    R = 0.396
    lenx, leny = data.shape[1], data.shape[0]
    x = np.linspace(0, lenx * R, lenx)
    y = np.linspace(0, leny * R, leny)
    

    im = ax.imshow(data, cmap=cmap, origin='lower',
                   vmin=vms[0], vmax=vms[1],
                   extent=(x[0], x[-1], y[0], y[-1]))

    


    # im = ax.pcolormesh(x, y, data,
    # vmin=vms[0], vmax=vms[1])
    if cbar:
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', 
                            location='left', shrink=shrink, pad=pad*fs)
        cbar.set_label(cbar_label, size=ls * fs, labelpad=0.1*fs)  # Label for the color bar
        cbar.ax.tick_params(labelsize=tls * fs, width=wdth * fs, length=lgth * fs)
        ticks = cbar.get_ticks()
        for spine in cbar.ax.spines.values():
            spine.set_linewidth(0.15 * fs) 

    ax.set_title(title, fontsize=1.25*fs, color='k')
    ax.tick_params(labelsize=tls * fs, width=wdth * fs, length=lgth * fs)

 # Adjust the factor to make it as thin as you like

    # if mask is not None:
    #     plt.rcParams['hatch.linewidth'] = 0.125
    #     hatches = ax.contourf(1 - mask, levels=[-0.5, 0.5], colors='none', hatches=['xxxxxxxx'], alpha=0, zorder=2, 
    #     extent=(0, lenx * R, 0, leny * R))  
    if mask is not None:
        gray_mask = np.where(mask==1, 0, np.nan)  # Create a mask where the value is 0.5 where mask==0, and NaN otherwise
        ax.imshow(gray_mask, cmap='gray_r', origin='lower', extent=(0, lenx * R, 0, leny * R), alpha=1, zorder=2)

    if mean==True:
        mean = ut.get_mean(data)
        box = dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3', linewidth=0.05*fs)
        ax.text(0.95, 0.975, f'Mean = {mean:.2e}', transform=ax.transAxes, fontsize=0.75*fs, ha='right', va='top',
        bbox=box)

    if chi is not None:
        box = dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3', linewidth=0.05*fs)
        ax.text(0.95, 0.975, f'$\chi^2 = {chi:.2f}$', transform=ax.transAxes, fontsize=0.75*fs, ha='right', va='top',
        bbox=box)

    ax.set_xlabel('X (arcsec)', fontsize=tls * fs)
    ax.set_ylabel('Y (arcsec)', fontsize=tls * fs)
    #place label on the  right

    if xtop:
        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()
    if yright:
        ax.yaxis.set_label_position('right')
        ax.yaxis.tick_right()
    # ax.tick_params(labelsize=tls * fs, width=wdth * fs, length=lgth * fs)
    #label pad
    ax.xaxis.labelpad = 0.5 * fs
    ax.yaxis.labelpad = 0.08* fs

    if cbar is not False:

        return im, cbar
    
    else:
        return im






def png_to_gif(fold, title='video', outfold=None, fps=36, 
               digit_format='04d', quality=1000):

    # Get a list of all .mp4 files in the directory
    files = [f for f in os.listdir(fold) if f.endswith('.png')]
    files.sort()

    name = os.path.splitext(files[0])[0]
    basename = name.split('_')[0]

    ffmpeg_path = 'ffmpeg'  
    framerate = fps

    if outfold==None:
        abs_path = os.path.abspath(fold)
        parent_folder = os.path.dirname(abs_path)+'\\'
    else:
        parent_folder = outfold
        if not os.path.exists(parent_folder):
            os.makedirs(parent_folder)
            
    output_file = parent_folder + "{}.gif".format(title,framerate)

    # Create a palette for better GIF quality
    palette_file = parent_folder + "palette.png"
    palette_command = f'{ffmpeg_path} -i {fold}{basename}_%04d.png -vf "fps={framerate},scale={quality}:-1:flags=lanczos,palettegen" -y {palette_file}'
    subprocess.run(palette_command, shell=True)
    print(palette_file)

    # Use the palette to create the GIF
    gif_command = f'{ffmpeg_path} -r {framerate} -i {fold}{basename}_%04d.png -i {palette_file} -lavfi "fps={framerate},scale={quality}:-1:flags=lanczos [x]; [x][1:v] paletteuse" -y {output_file}'
    subprocess.run(gif_command, shell=True)



def fit_plot(data, data_fit, data_mask, title='', savefold='../figures/fit/', filename='plot'):

    data_res = data - data_fit

    res, rat = 1080, 3
    subplots = (1, 3)
    fig, axs, fs, gs = initialize_figure(ratio=None, fig_w=res*rat, fig_h=res, subplots=subplots,
        theme=None)

    cmap = mpl.colormaps['jet']

    vmin = np.min([np.min(data), np.min(data_fit)])
    vmax = np.max([np.max(data), np.max(data_fit)])

    im = axs[0][0].imshow(data, cmap=cmap, origin='lower',
            vmin=vmin, vmax=vmax)
    im_fit = axs[0][1].imshow(data_fit, cmap=cmap, origin='lower',
            vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, ax=axs[0][0], orientation='vertical', shrink=1, location='left')
    cbar.set_label('Intensity', size=1.5 * fs)  # Label for the color bar
    cbar.ax.tick_params(labelsize=1.2 * fs, width=0.2 * fs, length=0.5 * fs)  # Tick 


    cmap_res = mpl.colormaps['RdBu_r']
    vmin_res, vmax_res = np.min(data_fit), np.max(data_fit)
    vres = 100
    im_res = axs[0][2].imshow(data_res, cmap=cmap_res, origin='lower',
                            vmin=-vres, vmax=vres)
    cbar_res = plt.colorbar(im_res, ax=axs[0][2], orientation='vertical', shrink=1)
    cbar_res.set_label('Intensity', size=1.5 * fs)  # Label for the color bar
    cbar_res.ax.tick_params(labelsize=1.2 * fs, width=0.2 * fs, length=0.5 * fs)  # Tick 

    plt.rcParams['hatch.linewidth'] = 0.5
    hatches = axs[0][0].contourf(1 - data_mask, levels=[-0.5, 0.5], colors='none', hatches=['////'], alpha=0, zorder=2)

    for ax in ut.flatten(axs):
        ax.set_xticks([])
        ax.set_yticks([])


    fig.suptitle(title, size=2.5 * fs)
    if not os.path.exists(savefold):
        os.makedirs(savefold)
    filename = f'{filename}.png'
    savepath = os.path.join(savefold, filename)

    # # save figure
    fig.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.show()

    return 



def plot_lines_(xs, vals, vals_err, ylim2=None, ylabel1='', ylabel2=''):

    res, rat, subplots = 720, 1, (2,1)
    fig, axs, fs, gs = initialize_figure(ratio=None, fig_w=rat*res, fig_h=res, 
                                            subplots=subplots, theme=None, layout=None,
                                            hr=[3,1], hspace=0.0, tick_direction='in', minor=True,
                                            top_bool=True)

    R = 0.396
    s = 0.2
    d = 115.3e6
    fact = 206265

    axs[0][0].scatter(xs*R, vals, s=s*fs, color='k', marker='s')
    lower_bound = vals - vals_err
    upper_bound = vals + vals_err
    axs[0][0].fill_between(xs*R, lower_bound, upper_bound, 
    color='grey', alpha=0.5, edgecolor=None)


    #axs[0][0].tick_params(axis='x', labelbottom=False)

    axs[1][0].scatter(xs*R, vals_err/vals, s=s*fs, color='k', marker='s')
    axs[1][0].set_ylim(*ylim2)

    #get current xtick values (ie xtick labels)
    # defines new xtick labels which are the old ones multiplied by a constnat R

    for ax in ut.flatten(axs):
        ax.set_xlim(1, (xs*R)[-1])
        ax.tick_params(axis='both', labelsize=1.25*fs, pad=0.45*fs)

    current_xticks = axs[1][0].get_xticks()
    print(current_xticks)
    new_xtick_labels = np.around(current_xticks * d / fact * 1e-3, decimals=1)
    axs[0][0].xaxis.tick_top()
    axs[0][0].set_xticklabels(new_xtick_labels)

    ts = 1.5
    #set xlabel
    axs[0][0].set_xlabel('Distance (kpc)', fontsize=fs*ts)
    axs[1][0].set_xlabel('Distance (arcsec)', fontsize=fs*ts)
    axs[0][0].xaxis.set_label_position('top')

    axs[0][0].set_ylabel(ylabel1, fontsize=fs*ts)
    axs[1][0].set_ylabel(ylabel2, fontsize=fs*ts)

    return fig


