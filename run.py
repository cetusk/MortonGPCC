# standard modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Morton G-PCC module
import morton_gpcc as mgpcc

# generate random sample points
def generateSamplePoints (
    num_points:int, spatial_dimension:int=2,
    min_bound:np.ndarray=np.array([0.0,0.0]),
    max_bound:np.ndarray=np.array([100.0,100.0])
) -> np.ndarray:
    assert min_bound.shape [ 0 ] == spatial_dimension
    assert max_bound.shape [ 0 ] == spatial_dimension
    _pts = np.random.rand ( num_points, spatial_dimension )
    return ( max_bound-min_bound ) * _pts + min_bound

# plot sampled points
def plotPoints (
    points:np.ndarray,
    out_path:str=None, plot_color:str="blue", color_alpha:float=0.5,
    project_3d:bool=False,
    new_plot:bool=True, continue_plot:bool=False,
    figure=None, axis=None
):
    if new_plot:
        assert figure is None and axis is None
        figure = plt.figure ()
        axis = figure.add_subplot ( 1, 1, 1 ) \
            if not project_3d \
            else figure.add_subplot ( 1, 1, 1, projection="3d" )
    else:
        assert figure is not None and axis is not None
    if not project_3d:
        axis.scatter (
            points [ :, 0 ], points [ :, 1 ],
            s=100, linewidths=2,
            c=plot_color, alpha=color_alpha, edgecolors=plot_color
        )
    else:
        axis.scatter (
            points [ :, 0 ], points [ :, 1 ], points [ :, 2 ],
            s=100, linewidths=2,
            c=plot_color, alpha=color_alpha, edgecolors=plot_color
        )
    figure.tight_layout ()
    if out_path is not None:
        plt.savefig ( out_path )
    if not continue_plot:
        plt.clf ()
    else:
        return figure, axis

def test_2d ():

    # level of detail and spatial dimension
    depth = 8
    dim = 2

    # bounds
    x0 = 0.0; x1 = 100.0
    y0 = 0.0; y1 = 100.0

    # number of sampling points
    numPoints = 100

    #---------------------

    minBound = np.array ( [ x0, y0 ] )
    maxBound = np.array ( [ x1, y1 ] )

    # create Morton G-PCC instance
    pcc = mgpcc.MortonGPCC (
        min_bound=minBound, max_bound=maxBound,
        spatial_dimension=dim, level_of_detail=depth
    )

    # generate points
    points = generateSamplePoints ( numPoints, dim, min_bound=minBound, max_bound=maxBound )
    # save input points as binary of float32
    with open ( "input.bin", "wb" ) as fh:
        fh.write ( points.astype ( np.float32 ).tobytes () )

    # compression and dump file as binary of uint16 ( and get Morton indices )
    mortonIndices = pcc.compress ( points, dump_path="encoded.bin" )
    # decompress by the dumped file and get vertices
    decodedPoints = pcc.decompress ( file_path="encoded.bin" )

    # plot in/out points
    fig, ax = plotPoints (
        points, plot_color="blue", color_alpha=0.7,
        continue_plot=True
    )
    plotPoints (
        decodedPoints, plot_color="green", color_alpha=0.3,
        out_path="points_fromfile.png",
        new_plot=False, figure=fig, axis=ax
    )

    # decompress by the Morton indices and get vertices
    decodedPoints = pcc.decompress ( data=mortonIndices )

    # plot in/out points
    fig, ax = plotPoints (
        points, plot_color="blue", color_alpha=0.7,
        new_plot=True, continue_plot=True
    )
    plotPoints (
        decodedPoints, plot_color="red", color_alpha=0.3,
        out_path="points_fromstream.png",
        new_plot=False, figure=fig, axis=ax
    )

def test_3d ():

    # level of detail and spatial dimension
    depth = 8
    dim = 3

    # bounds
    x0 = 0.0; x1 = 100.0
    y0 = 0.0; y1 = 100.0
    z0 = 0.0; z1 = 100.0

    # number of sampling points
    numPoints = 100

    #---------------------

    minBound = np.array ( [ x0, y0, z0 ] )
    maxBound = np.array ( [ x1, y1, z1 ] )

    # create Morton G-PCC instance
    pcc = mgpcc.MortonGPCC (
        min_bound=minBound, max_bound=maxBound,
        spatial_dimension=dim, level_of_detail=depth
    )

    # generate points
    points = generateSamplePoints ( numPoints, dim, min_bound=minBound, max_bound=maxBound )
    # save input points as binary of float32
    with open ( "input.bin", "wb" ) as fh:
        fh.write ( points.astype ( np.float32 ).tobytes () )

    # compression and dump file as binary of uint32 ( and get Morton indices )
    mortonIndices = pcc.compress ( points, dump_path="encoded.bin" )
    # decompress by the dumped file and get vertices
    decodedPoints = pcc.decompress ( file_path="encoded.bin" )

    # plot in/out points
    fig, ax = plotPoints (
        points, plot_color="blue", color_alpha=0.5,
        project_3d=True,
        continue_plot=True
    )
    plotPoints (
        decodedPoints, plot_color="green", color_alpha=0.3,
        out_path="points_fromfile.png",
        project_3d=True,
        new_plot=False, figure=fig, axis=ax
    )

    # decompress by the Morton indices and get vertices
    decodedPoints = pcc.decompress ( data=mortonIndices )

    # plot in/out points
    fig, ax = plotPoints (
        points, plot_color="blue", color_alpha=0.5,
        project_3d=True,
        new_plot=True, continue_plot=True
    )
    plotPoints (
        decodedPoints, plot_color="red", color_alpha=0.3,
        out_path="points_fromstream.png",
        project_3d=True,
        new_plot=False, figure=fig, axis=ax
    )

def errorAnalysis (
    spatial_dimension:int,
    level_of_detail:int,
    bounding_size:float,
    number_of_points:int
):

    # level of detail and spatial dimension
    depth = level_of_detail
    dim = spatial_dimension

    # bound size
    size = bounding_size

    # number of sampling points
    numPoints = number_of_points

    #---------------------

    minBound = np.array ( [  0.0 for _ in range ( dim ) ] )
    maxBound = np.array ( [ size for _ in range ( dim ) ] )

    # create Morton G-PCC instance
    pcc = mgpcc.MortonGPCC (
        min_bound=minBound, max_bound=maxBound,
        spatial_dimension=dim, level_of_detail=depth
    )

    # generate points
    points = generateSamplePoints ( numPoints, dim, min_bound=minBound, max_bound=maxBound )

    # compression and get Morton indices
    mortonIndices = pcc.compress ( points )
    # decompress by the Morton indices and get vertices
    decodedPoints = pcc.decompress ( data=mortonIndices )

    # distance error between ground truth and decoded points
    validMask = pcc.innerMask ( points )
    npts = points [ validMask, : ].shape [ 0 ]
    return np.sqrt (
        (
            np.linalg.norm (
                decodedPoints [ validMask, : ] - points [ validMask, : ],
                axis=1
            ) ** 2
        ).sum () / npts
    )

def errorPlot () -> None:
    # tag = "2dim10000bnd"
    tag = "3dim1000pts"
    fname = "out.%s.log" % tag

    lod = []
    error = []

    fig = plt.figure ()
    ax = fig.add_subplot ( 1, 1, 1 )
    ax.set_xlabel ( "Level of detail" )
    # ax.set_ylabel ( "Distance error per point ( RMSE )" )
    ax.set_ylabel ( "Bounding size" )
    ax.set_yscale ( "log" )

    plotCount = 0
    with open ( fname, "r" ) as fh:
        lines = fh.readlines ()
        for line in lines:
            seg = line.split ( "," )
            if len ( seg ) == 1:
                color = cm.Paired ( plotCount % 12 )
                plotCount += 1
                ax.plot (
                    lod, error,
                    # label="%d points" % npts,
                    label="%d of bound size" % bnd,
                    color=color, linewidth=2,
                    marker="o", markerfacecolor=color, markersize=8
                )
                ax.legend ()
                lod = []
                error = []
                continue
            lod.append ( int ( seg [ 0 ] ) )
            # npts = int ( seg [ 1 ] )
            bnd = float ( seg [ 1 ] )
            error.append ( float ( seg [ 2 ] ) )

    fig.tight_layout ()
    plt.savefig ( "lod-vs-error.%s.png" % tag )
    plt.clf ()

    exit ()

if __name__ == "__main__":
    # test_2d ()
    test_3d ()

    exit ()

    #--- Error analysis ---#

    # errorPlot ()

    dim = 3
    npts = 1000
    # size = 10000.0
    # with open ( "out.%ddim%dbnd.log" % ( dim, int ( size ) ), "w" ) as fh:
    with open ( "out.%ddim%dpts.log" % ( dim, npts ), "w" ) as fh:
        # for npts in [ 10, 100, 1000, 10000, 100000 ]:
        for bnd in [ 10.0, 100.0, 1000.0, 10000.0, 100000.0 ]:
            for lod in range ( 1, 8+1 ):
                print (
                    "[Main] %d-dim, %.3f bnd, %d pts, %d lod" % (
                        dim, bnd, npts, lod
                    )
                )
                error = errorAnalysis (
                    spatial_dimension=dim,
                    level_of_detail=lod,
                    bounding_size=bnd,
                    number_of_points=npts
                )
                fh.write (
                    "%d,%.3f,%.3f\n" % (
                        lod, bnd, error
                    )
                )
            fh.write ( "\n" )
