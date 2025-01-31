def matplotlib_setup():
    import seaborn as sns
    import matplotlib as mpl
    
    sns.set_theme(style="white")
    mpl.rc('xtick.minor', visible=True) 
    mpl.rc('ytick.minor', visible=True) 
    mpl.rc('xtick', direction='in', top=True, bottom=True, labelsize=16) 
    mpl.rc('ytick', direction='in', right=True, left=True, labelsize=16)
    mpl.rc('axes', labelsize=18)
    mpl.rc('legend', fontsize=14)
