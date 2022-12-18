import numpy as np

class MortonGPCC:
    # depth limitation
    _MAX_DEPTH = 8

    # constructor
    def __init__ (
        self,
        min_bound:tuple, max_bound:tuple,
        spatial_dim:int=2, tree_depth:int=3
    ) -> None:
        # check dimension and depth
        assert len ( min_bound ) == spatial_dim and len ( max_bound ) == spatial_dim
        assert tree_depth > 0 and tree_depth <= MortonGPCC._MAX_DEPTH
        # boundary
        self._x0, self._y0 = min_bound
        self._x1, self._y1 = max_bound
        assert self._x1 - self._x0 > 0
        assert self._y1 - self._y0 > 0
        # tree depth
        self._dim   = spatial_dim
        self._depth = tree_depth
        # leaf grid size: ( x1-x0 ) / 2**tree_depth
        self._dx = ( self._x1-self._x0 ) / ( 1 << self._depth )
        self._dy = ( self._y1-self._y0 ) / ( 1 << self._depth )
        # number of grids: 2^d ( 1-dim ), 2^md=(2^d)^m ( m-dim )
        self._numGrids1d = 2 ** self._depth
        self._numGrids = self._numGrids1d ** self._dim
        # array of Morton index
        self._data = []

    def pushVertex ( self, point:tuple ) -> None:
        # check dimension
        assert len ( point ) == self._dim
        # skip when the point in external
        if not self.isInside ( point ): return
        # Morton index
        _idx = self.getVertexIndex ( point )
        # append to memory
        self._data.append ( _idx )

    def pushVertices ( self, points:np.ndarray ) -> None:
        # check dimension
        assert points.shape [ 1 ] == self._dim

    def compress ( self, points:np.ndarray, dump_path:str=None ) -> np.ndarray:
        # check dimension
        _, _dim = points.shape [ : 2 ]
        assert _dim == self._dim
        # inner indices
        _validIdx = self.innerIndices ( points )
        # convert to Morton indices
        _data = self.getVertexIndices ( points [ _validIdx, : ] ).astype ( np.uint16 )
        # write out as binary file
        if dump_path is not None:
            # number of points
            _npts = len ( _data )
            with open ( dump_path, "wb" ) as fh:
                fh.write (
                    # header: dim, depth, npts, x0, y0, x1, y0
                    np.array ( [ self._dim, self._depth ], dtype=np.uint8 ).tobytes ()
                    + np.array ( [ _npts ], dtype=np.uint32 ).tobytes ()
                    + np.array ( [ self._x0, self._y0, self._x1, self._y1 ], dtype=np.float32 ).tobytes ()
                    # body
                    + _data.tobytes ()
                )
        # return
        return _data

    def decompress ( self, data:np.ndarray=None, dump_path:str=None ) -> np.ndarray:
        # check arg
        if dump_path is not None:
            # read file as binary
            with open ( dump_path, "rb" ) as fh:
                # get binary stream
                stream = fh.read ()
                # header: dim, depth, npts, x0, y0, x1, y0
                buf = np.frombuffer ( stream, dtype=np.uint8, count=2, offset=0 )
                _dim, _depth = buf
                buf = np.frombuffer ( stream, dtype=np.uint32, count=1, offset=1*2 )
                _npts = buf [ 0 ]
                buf = np.frombuffer ( stream, dtype=np.float32, count=4, offset=1*2+4*1 )
                _x0, _y0, _x1, _y1 = buf
                _dx = ( _x1-_x0 ) / ( 1 << _depth )
                _dy = ( _y1-_y0 ) / ( 1 << _depth )
                # body
                _data = np.frombuffer ( stream, dtype=np.uint16, count=_npts, offset=1*2+4*1+4*4 )
                # initialize points
                decodedPoints = np.zeros ( ( _npts, 2 ) )
                # binary format
                rep = "{:" + str ( self._dim*self._depth ).zfill ( 2 ) + "b}"
                # convert Morton index to coordinate values
                for j, morton_index in enumerate ( _data.tolist () ):
                    bin_index = rep.format ( morton_index )
                    _ix = int ( bin_index [ 1 : _dim*_depth : _dim ], 2 )
                    _iy = int ( bin_index [ 0 : _dim*_depth : _dim ], 2 )
                    _x = _x0 + _dx*_ix
                    _y = _y0 + _dy*_iy
                    decodedPoints [ j, 0 ] = _x
                    decodedPoints [ j, 1 ] = _y
                return decodedPoints
        else:
            # check arg
            assert data is not None
            # initialize points
            _npts = data.shape [ 0 ]
            decodedPoints = np.zeros ( ( _npts, 2 ) )
            # convert Morton index to coordinate values
            for j, morton_index in enumerate ( data.tolist () ):
                _x, _y = self.getVertexFromIndex ( morton_index )
                decodedPoints [ j, 0 ] = _x
                decodedPoints [ j, 1 ] = _y
            return decodedPoints

    def isInside ( self, point:tuple ) -> bool:
        # check dimension
        assert len ( point ) == self._dim
        _x, _y = point
        return (
            _x > self._x0 and _x < self._x1 and
            _y > self._y0 and _y < self._y1
        )

    def innerIndices ( self, points:np.ndarray ) -> np.ndarray:
        # check dimension: assuming the shape is num points x dim
        assert points.shape [ 1 ] == self._dim
        # calc grid indices
        _ixs = ( ( points [ :, 0 ] - self._x0 ) / self._dx ).astype ( np.int32 )
        _iys = ( ( points [ :, 1 ] - self._y0 ) / self._dy ).astype ( np.int32 )
        return np.where ( ( _ixs >= 0 ) & ( _ixs < self._numGrids1d ) & ( _iys >= 0 ) & ( _iys < self._numGrids1d ) ) [ 0 ]

    def getVertexIndex ( self, point:tuple ) -> int:
        # check dimension
        assert len ( point ) == self._dim
        _ix = int ( ( point [ 0 ] - self._x0 ) / self._dx )
        _iy = int ( ( point [ 1 ] - self._y0 ) / self._dy )
        return self._convertToSeparatedBits ( _ix ) | self._convertToSeparatedBits ( _iy ) << 1

    def getVertexIndices ( self, points:np.ndarray ) -> np.ndarray:
        # check dimension: assuming the shape is num points x dim
        assert points.shape [ 1 ] == self._dim
        _ix = ( ( points [ :, 0 ] - self._x0 ) / self._dx ).astype ( np.int32 )
        _iy = ( ( points [ :, 1 ] - self._y0 ) / self._dy ).astype ( np.int32 )
        return self._convertToSeparatedBits ( _ix ) | self._convertToSeparatedBits ( _iy ) << 1

    def getGridFromIndex ( self, morton_index:int ) -> list:
        # binary format
        rep = "{:" + str ( self._dim*self._depth ).zfill ( 2 ) + "b}"
        bin_index = rep.format ( morton_index )
        # convert to grid index for each depth level
        return [
            int ( bin_index [ d*self._dim : ( d+1 )*self._dim ], 2 )
            for d in range ( self._depth )
        ]

    def getVertexFromIndex ( self, morton_index:int ) -> tuple:
        # binary format
        rep = "{:" + str ( self._dim*self._depth ).zfill ( 2 ) + "b}"
        bin_index = rep.format ( morton_index )
        # extract each components
        _ix = int ( bin_index [ 1 : self._dim*self._depth : self._dim ], 2 )
        _iy = int ( bin_index [ 0 : self._dim*self._depth : self._dim ], 2 )
        # convert to coordinate values
        _x = self._x0 + self._dx*_ix
        _y = self._y0 + self._dy*_iy
        return ( _x, _y )

    ### private functions

    @staticmethod
    def _convertToSeparatedBits ( grid_index:int ) -> int:
        _idx = grid_index
        _idx = ( _idx | ( _idx<<8 ) ) & 0x00ff00ff
        _idx = ( _idx | ( _idx<<4 ) ) & 0x0f0f0f0f
        _idx = ( _idx | ( _idx<<2 ) ) & 0x33333333
        return ( _idx | ( _idx<<1 ) ) & 0x55555555

    ### getter

    @property
    def data ( self ) -> np.ndarray:
        return np.array ( self._data, dtype=np.uint16 )
