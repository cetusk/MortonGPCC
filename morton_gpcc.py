import numpy as np

class MortonGPCC:
    # depth limitation
    _MAX_DEPTH = 8

    # constructor
    def __init__ (
        self,
        min_bound:np.ndarray, max_bound:np.ndarray,
        spatial_dim:int=2, tree_depth:int=3
    ) -> None:
        # check dimension and depth
        assert min_bound.shape [ 0 ] == spatial_dim and max_bound.shape [ 0 ] == spatial_dim
        assert tree_depth > 0 and tree_depth <= MortonGPCC._MAX_DEPTH
        # boundaries
        self._minBound = min_bound.copy ()  # x0, y0, z0, ...
        self._maxBound = max_bound.copy ()  # x1, y1, z1, ...
        self._boundSize = self._maxBound - self._minBound  # x1-x0, ...
        assert np.all ( self._boundSize > 0 )
        # tree depth
        self._dim   = spatial_dim
        self._depth = tree_depth
        # leaf grid size: ( x1-x0 ) / 2**tree_depth, ...
        self._gridSize = self._boundSize / ( 1 << self._depth )
        # number of grids: 2^d ( 1-dim ), 2^md=(2^d)^m ( m-dim )
        self._numGrids1d = 2 ** self._depth
        self._numGrids = self._numGrids1d ** self._dim
        # array of Morton index
        self._data = []
        # data type
        self._numBits = 16  # dim*depth < 16
        self._dataType = np.uint16
        # ToDo
        # if self._dim*self._depth > 16 and self._dim*self._depth <= 32:
        #     self._numBits = 32
        #     self._dataType = np.uint32

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
        # check dimension: assuming the shape is num points x dim
        assert points.shape [ 1 ] == self._dim
        _validMask = self.innerMask ( points )
        self._data = self.getVertexIndices ( points [ _validMask, : ] ).astype ( self._dataType )

    def compress ( self, points:np.ndarray, dump_path:str=None ) -> np.ndarray:
        # check dimension: assuming the shape is num points x dim
        _, _dim = points.shape [ : 2 ]
        assert _dim == self._dim
        # inner indices
        _validMask = self.innerMask ( points )
        # convert to Morton indices
        _data = self.getVertexIndices ( points [ _validMask, : ] ).astype ( self._dataType )
        # write out as binary file
        if dump_path is not None:
            # number of points
            _npts = len ( _data )
            with open ( dump_path, "wb" ) as fh:
                fh.write (
                    # header: dim, depth, npts, minbounds(x0,y0,..), maxbounds(x1,y1,...)
                    np.array ( [ self._dim, self._depth ], dtype=np.uint8 ).tobytes ()
                    + np.array ( [ _npts ], dtype=np.uint32 ).tobytes ()
                    + self._minBound.astype ( np.float32 ).tobytes ()
                    + self._maxBound.astype ( np.float32 ).tobytes ()
                    # body
                    + _data.tobytes ()
                )
        # return
        return _data

    def decompress ( self, data:np.ndarray=None, file_path:str=None ) -> np.ndarray:
        # check arg
        if file_path is not None:
            # read file as binary
            with open ( file_path, "rb" ) as fh:
                # get binary stream
                stream = fh.read ()
                # header: dim, depth, npts, minbounds(x0,y0,..), maxbounds(x1,y1,...)
                buf = np.frombuffer ( stream, dtype=np.uint8, count=2, offset=0 )
                _dim, _depth = buf; offset = 1*2
                buf = np.frombuffer ( stream, dtype=np.uint32, count=1, offset=offset )
                _npts = buf [ 0 ]; offset += 4*1
                _minBound = np.frombuffer ( stream, dtype=np.float32, count=_dim, offset=offset )
                offset += 4*_dim
                _maxBound = np.frombuffer ( stream, dtype=np.float32, count=_dim, offset=offset )
                offset += 4*_dim
                _boundSize = _maxBound - _minBound
                _gridSize = _boundSize / ( 1 << _depth )
                # body
                _data = np.frombuffer ( stream, dtype=self._dataType, count=_npts, offset=offset )
                # convert Morton indices to coordinate values
                return self.getVerticesFromIndices ( _data, _dim, _minBound, _gridSize )
        else:
            # check arg
            assert data is not None
            # convert Morton indices to coordinate values
            return self.getVerticesFromIndices ( data )

    def isInside ( self, point:tuple ) -> bool:
        # check dimension
        assert len ( point ) == self._dim
        _point = np.array ( [ [ pt for pt in point ] ] )  # 1 x dim
        # must be returned [ True, True, True ]
        return self.innerMask ( _point ).shape [ 0 ] == self._dim

    def innerMask ( self, points:np.ndarray ) -> np.ndarray:
        # check dimension: assuming the shape is num points x dim
        assert points.shape [ 1 ] == self._dim
        # calc grid indices
        _indices = ( ( points - self._minBound ) / self._gridSize ).astype ( np.int32 )
        return np.all ( ( _indices >= 0 ) & ( _indices < self._numGrids1d ), axis=1 )

    def getVertexIndex ( self, point:tuple ) -> int:
        # check dimension
        assert len ( point ) == self._dim
        _mortonIndex = 0
        for d in range ( self._dim ):
            _gridIndex = int ( ( point [ d ] - self._minBound [ d ] ) / self._gridSize [ d ] )
            _mortonIndex |= self._convertToSeparatedBits ( _gridIndex ) << d
        return _mortonIndex

    def getVertexIndices ( self, points:np.ndarray ) -> np.ndarray:
        # check dimension: assuming the shape is num points x dim
        assert points.shape [ 1 ] == self._dim
        _gridIndices = ( ( points - self._minBound ) / self._gridSize ).astype ( np.int32 )
        _mortonIndices = np.zeros ( points.shape [ 0 ], dtype=np.int )
        for d in range ( self._dim ):
            _mortonIndices |= self._convertToSeparatedBits ( _gridIndices [ :, d ] ) << d
        return _mortonIndices

    def getVertexFromIndex (
        self, morton_index:int,
        dimension:int=None, min_bound:np.ndarray=None, grid_size:np.ndarray=None
    ) -> np.ndarray:
        # set force fixed value if specified
        _dim = self._dim if dimension is None else dimension
        _minBound = self._minBound if min_bound is None else min_bound
        _gridSize = self._gridSize if grid_size is None else grid_size
        # extract each components
        _indices = np.zeros ( _dim, dtype=self._dataType )
        for d in range ( _dim ):
            _indices [ d ] = self._convertFromSeparatedBits ( morton_index ) >> d
        # convert to coordinate values
        return _minBound + _gridSize*_indices

    def getVerticesFromIndices (
        self, morton_index:np.ndarray,
        dimension:int=None, min_bound:np.ndarray=None, grid_size:np.ndarray=None
    ) -> np.ndarray:
        # set force fixed value if specified
        _dim = self._dim if dimension is None else dimension
        _minBound = self._minBound if min_bound is None else min_bound
        _gridSize = self._gridSize if grid_size is None else grid_size
        # extract each components
        _npts = morton_index.shape [ 0 ]
        _indices = np.zeros ( ( _npts, _dim ), dtype=self._dataType )
        for d in range ( _dim ):
            _indices [ :, d ] = self._convertFromSeparatedBits ( morton_index >> d )
        # convert to coordinate values
        return _minBound + _gridSize*_indices

    ### private functions

    def _convertToSeparatedBits ( self, grid_index:int ) -> int:
        assert self._dim == 2 or self._dim == 3, "2 or 3 dimension are only supported"
        _idx = grid_index
        if self._dim == 2:
            _idx = ( _idx | ( _idx<<8 ) ) & 0x00ff00ff
            _idx = ( _idx | ( _idx<<4 ) ) & 0x0f0f0f0f
            _idx = ( _idx | ( _idx<<2 ) ) & 0x33333333
            return ( _idx | ( _idx<<1 ) ) & 0x55555555
        elif self._dim == 3:
            _idx = ( _idx | ( _idx<<8 ) ) & 0x0f00f00f
            _idx = ( _idx | ( _idx<<4 ) ) & 0xc30c30c3
            return ( _idx | ( _idx<<2 ) ) & 0x49249249
        else:
            return None

    def _convertFromSeparatedBits ( self, morton_index:int ) -> int:
        assert self._dim == 2 or self._dim == 3, "2 or 3 dimension are only supported"
        _idx = morton_index
        if self._dim == 2:
            _idx = _idx & 0x5555
            _idx = ( _idx | _idx>>1 ) & 0x3333
            _idx = ( _idx | _idx>>2 ) & 0x0f0f
            return ( _idx | _idx>>4 ) & 0x00ff
        elif self._dim == 3:
            _idx = _idx & 0x249249
            _idx = ( _idx | _idx>>2 ) & 0x6db6db
            _idx = ( _idx | _idx>>2 ) & 0x0c30c3
            _idx = ( _idx | _idx>>4 ) & 0x00f00f
            return ( _idx | _idx>>8 ) & 0x0000ff
        else:
            return None

    ### getter

    @property
    def data ( self ) -> np.ndarray:
        return np.array ( self._data, dtype=self._dataType )


# 16 bits
# 0000 0000 0000 0000 xxxx xxxx xxxx xxxx
# 0000 0000 xxxx xxxx xxxx xxxx 0000 0000
# --------------------------------------- ( or )
# 0000 0000 xxxx xxxx xxxx xxxx xxxx xxxx
# 0000 1111 0000 0000 1111 0000 0000 1111 ( 0x0f00f00f )
# --------------------------------------- ( and )
# 0000 0000 0000 0000 xxxx 0000 0000 xxxx
# 0000 0000 0000 xxxx 0000 0000 xxxx 0000
# --------------------------------------- ( or )
# 0000 0000 0000 xxxx xxxx 0000 xxxx xxxx
# 1100 0011 0000 1100 0011 0000 1100 0011 ( 0xc30c30c3 )
# --------------------------------------- ( and )
# 0000 0000 0000 xx00 00xx 0000 xx00 00xx
# 0000 0000 00xx 0000 xx00 00xx 0000 xx00
# --------------------------------------- ( or )
# 0000 0000 00xx xx00 xxxx 00xx xx00 xxxx
# 0100 1001 0010 0100 1001 0010 0100 1001 ( 0x49249249 )
# --------------------------------------- ( and )
# 0000 0000 00x0 0x00 x00x 00x0 0x00 x00x

# yayb ycyd yeyf ygyh
# 0101 0101 0101 0101 ( 0x5555 )
# ------------------- ( and )
# 0a0b 0c0d 0e0f 0g0h
# 00a0 b0c0 d0e0 f0g0 ( >> 1 )
# ------------------- ( or )
# 0aab bccd deef fggh
# 0011 0011 0011 0011 ( 0x3333 )
# ------------------- ( and )
# 00ab 00cd 00ef 00gh
# 0000 ab00 cd00 ef00 ( >> 2 )
# ------------------- ( or )
# 00ab abcd cdef efgh
# 0000 1111 0000 1111 ( 0x0f0f )
# ------------------- ( and )
# 0000 abcd 0000 efgh
# 0000 0000 abcd 0000 ( >> 4 )
# ------------------- ( or )
# 0000 abcd abcd efgh
# 0000 0000 1111 1111 ( 0x00ff )
# ------------------- ( and )
# 0000 0000 abcd efgh

# zyaz ybzy czyd zyez yfzy gzyh
# 0010 0100 1001 0010 0100 1001 ( 0x249249 )
# ----------------------------- ( and )
# 00a0 0b00 c00d 00e0 0f00 g00h
# 0000 a00b 00c0 0d00 e00f 00g0 ( >> 2 )
# ----------------------------- ( or )
# 00a0 ab0b c0cd 0de0 ef0f g0gh
# 0110 1101 1011 0110 1101 1011 ( 0x6db6db )
# ----------------------------- ( and )
# 00a0 ab0b c0cd 0de0 ef0f g0gh
# 0000 00ab 0bc0 cd0d e0ef 0fg0 ( >> 2 )
# ----------------------------- ( or )
# 00a0 abab cbcd cded efef gfgh
# 0000 1100 0011 0000 1100 0011 ( 0x0c30c3 )
# ----------------------------- ( and )
# 0000 ab00 00cd 0000 ef00 00gh
# 0000 0000 ab00 00cd 0000 ef00 ( >> 4 )
# ----------------------------- ( or )
# 0000 ab00 abcd 00cd ef00 efgh
# 0000 0000 1111 0000 0000 1111 ( 0x00f00f )
# ----------------------------- ( and )
# 0000 0000 abcd 0000 0000 efgh
# 0000 0000 0000 0000 abcd 0000 ( >> 8 )
# ----------------------------- ( or )
# 0000 0000 abcd 0000 abcd efgh
# 0000 0000 0000 0000 1111 1111 ( 0x0000ff )
# ----------------------------- ( and )
# 0000 0000 0000 0000 abcd efgh

# 24 bits
# 0000 0000 0000 0000 0000 0000 xxxx xxxx xxxx xxxx xxxx xxxx ( 48 bits )
# 0000 0000 0000 0000 xxxx xxxx xxxx xxxx xxxx xxxx 0000 0000
# ----------------------------------------------------------- ( or )
# 0000 0000 0000 0000 xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx
# 0000 0000 1111 0000 0000 1111 0000 0000 1111 0000 0000 1111 ( 0x00f00f00f00f )
# ----------------------------------------------------------- ( and )
# 0000 0000 0000 0000 0000 xxxx 0000 0000 xxxx 0000 0000 xxxx
# 0000 0000 0000 0000 xxxx 0000 0000 xxxx 0000 0000 xxxx 0000
# ----------------------------------------------------------- ( or )
# 0000 0000 0000 0000 xxxx xxxx 0000 xxxx xxxx 0000 xxxx xxxx
# 0000 1100 0011 0000 1100 0011 0000 1100 0011 0000 1100 0011 ( 0x0c30c30c30c3 )
# ----------------------------------------------------------- ( and )
# 0000 0000 0000 0000 xx00 00xx 0000 xx00 00xx 0000 xx00 00xx
# 0000 0000 0000 00xx 0000 xx00 00xx 0000 xx00 00xx 0000 xx00
# ----------------------------------------------------------- ( or )
# 0000 0000 0000 00xx xx00 xxxx 00xx xx00 xxxx 00xx xx00 xxxx
# 0010 0100 1001 0010 0100 1001 0010 0100 1001 0010 0100 1001 ( 0x249249249249 )
# ----------------------------------------------------------- ( and )
# 0000 0000 0000 00x0 0x00 x00x 00x0 0x00 x00x 00x0 0x00 x00x