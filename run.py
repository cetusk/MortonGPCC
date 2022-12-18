# standard modules
import numpy as np
import matplotlib.pyplot as plt

# Morton G-PCC module
import morton_gpcc as mgpcc

# generate random sample points
def generateSamplePoints (
    num_points:int, spatial_dim:int=2,
    min_bound:tuple=(0.0,100.0), max_bound:tuple=(0.0,100.0)
) -> np.ndarray:
    _x0, _y0 = min_bound
    _x1, _y1 = max_bound
    _pts = np.random.rand ( num_points, spatial_dim )
    _pts [ :, 0 ] = ( _x1-_x0 ) * _pts [ :, 0 ] + _x0
    _pts [ :, 1 ] = ( _y1-_y0 ) * _pts [ :, 1 ] + _y0
    return _pts

# plot sampled points
def plotPoints (
    points:np.ndarray,
    out_path:str=None, plot_color:str="blue",
    new_plot:bool=True, continue_plot:bool=False,
    figure=None, axis=None
):
    if new_plot:
        assert figure is None and axis is None
        figure = plt.figure ()
        axis = figure.add_subplot ( 1, 1, 1 )
    else:
        assert figure is not None and axis is not None
    axis.scatter (
        points [ :, 0 ], points [ :, 1 ],
        s=100, linewidths=2,
        c=plot_color, alpha=0.5, edgecolors=plot_color
    )
    figure.tight_layout ()
    if out_path is not None:
        plt.savefig ( out_path )
    if not continue_plot:
        plt.clf ()
    else:
        return figure, axis

if __name__ == "__main__":

    # level of detail and apatial dimension
    depth = 8
    dim = 2

    # bounds
    x0 = 0.0; x1 = 100.0
    y0 = 0.0; y1 = 100.0

    # number of sampling points
    numPoints = 100

    # create Morton G-PCC instance
    mt = mgpcc.MortonGPCC (
        min_bound=( x0, y0 ), max_bound=( x1, y1 ),
        spatial_dim=dim, tree_depth=depth
    )

    # generate points
    points = generateSamplePoints ( numPoints, dim, min_bound=( x0, y0 ), max_bound=( x1, y1 ) )
    # save input points as binary of float32
    with open ( "input.bin", "wb" ) as fh:
        fh.write ( points.astype ( np.float32 ).tobytes () )

    # compression and dump file as binary of uint16 ( and get Morton indices )
    mortonIndices = mt.compress ( points, dump_path="encoded.bin" )
    # decompress by the dumped file and get vertices
    decodedPoints = mt.decompress ( dump_path="encoded.bin" )

    # plot in/out points
    fig, ax = plotPoints (
        points, plot_color="black",
        continue_plot=True
    )
    plotPoints (
        decodedPoints, plot_color="green",
        out_path = "points_fromfile.png",
        new_plot=False, figure=fig, axis=ax
    )

    # decompress by the Morton indices and get vertices
    decodedPoints = mt.decompress ( data=mortonIndices )

    # plot in/out points
    fig, ax = plotPoints (
        points, plot_color="black",
        new_plot=True, continue_plot=True
    )
    plotPoints (
        decodedPoints, plot_color="red",
        out_path = "points_fromstream.png",
        new_plot=False, figure=fig, axis=ax
    )
