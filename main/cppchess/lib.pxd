from libcpp.string cimport string
from libcpp.optional cimport optional
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map

cdef extern from "./src/chess.cpp":
    pass

cdef extern from "./src/chess.h" namespace "chess":

    string __author__ "chess::__author___"
    string __email__ "chess::__email__"
    string __version__ "chess::__version__"

    ctypedef int _Square "chess::Square"
    ctypedef int _PieceType "chess::PieceType"
    ctypedef bint _Color "chess::Color"

    _Color WHITE = True
    _Color BLACK = False

    cdef string STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    cdef enum class _Termination "chess::Termination":
        CHECKMATE = 0
        STALEMATE = 1
        INSUFFICIENT_MATERIAL = 2
        SEVENTYFIVE_MOVES = 3
        FIVEFOLD_REPETITION = 4
        FIFTY_MOVES = 5
        THREEFOLD_REPETITION = 6
        VARIANT_WIN = 7
        VARIANT_LOSS = 8
        VARIANT_DRAW = 9

    cdef cppclass _Outcome "chess::Outcome":
        _Termination termination
        optional[_Color] winner
        _Outcome(_Termination, optional[_Color])
        string result()

    cdef cppclass _Move "chess::Move":
        bint operator==(_Move)
        bint operator!=(_Move)
        _Square from_square
        _Square to_square
        optional[_PieceType] promotion
        optional[_PieceType] drop
        _Move(_Square, _Square, optional[_PieceType], optional[_PieceType]) except +
        string uci() const

    # Static method for "chess::Move"
    cdef _Move move_from_uci "chess::Move::from_uci" (string) 

    cdef cppclass _Piece "chess::Piece":
        _PieceType piece_type
        _Color color
        char symbol()
        _Piece(_PieceType, _Color)

    ctypedef unsigned long _Bitboard "chess::Bitboard"
    cdef _Bitboard BB_EMPTY = 0
    cdef _Bitboard BB_ALL = 18446744073709551615

    cdef cppclass _LegalMoveGenerator "chess::LegalMoveGenerator":
        _LegalMoveGenerator(_Board)
        int count()
        vector[_Move].iterator begin()
        vector[_Move].iterator end()


    cdef cppclass _Board "chess::Board":
        # properties
        _Color turn
        int fullmove_number
        int halfmove_clock
        vector[_Move] move_stack
        
        # Constructors
        _Board(optional[string], bint)
        _Board copy() # Not an actual constructor

        # Methods
        int ply()
        string fen(bool, string, bint)
        _Move push_san(const string)
        _Move push_uci(const string)
        _Move pop()
        bint is_game_over(bint)
        optional[_Outcome] outcome(bint)
        bint is_repetition(int)
        bint is_checkmate()
        bint is_stalemate()
        bint is_insufficient_material()
        bint is_fifty_moves()
        bint has_kingside_castling_rights(_Color)
        bint has_queenside_castling_rights(_Color)
        unordered_map[_Square, _Piece] piece_map(_Bitboard)
        optional[_Piece] piece_at(_Square)
        _LegalMoveGenerator legal_moves()
        vector[_Move] generate_legal_moves(_Bitboard, _Bitboard)
        bint is_legal(_Move)