# standard modules
import numpy as np
import matplotlib.pyplot as plt

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
    numPoints = 300

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
    return np.sqrt (
        (
            np.linalg.norm (
                decodedPoints [ validMask, : ] - points [ validMask, : ],
                axis=1
            ) ** 2
        ).sum ()
    )


if __name__ == "__main__":
    # test_2d ()
    # test_3d ()

    print ( "LoD,Dimension,NumPoints,Size,Error" )
    for lod in range ( 3, 8+1 ):
        error = errorAnalysis (
            spatial_dimension=3,
            level_of_detail=lod,
            bounding_size=100.0,
            number_of_points=1000
        )
        print ( lod, 3, 1000, 100.0, error )
