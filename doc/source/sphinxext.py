"""Functions for documentation configuration only, importable by sphinx"""
# To reset resolution setting for each sphinx-gallery example
def reset_mpl(gallery_conf, fname):
    # To get a good resolution for displayed figures
    from matplotlib import pyplot
    pyplot.rcParams['figure.dpi'] = 600
    pyplot.rcParams['savefig.dpi'] = 600