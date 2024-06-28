from .lib cimport _Color, _PieceType, _Square, _Move, _Piece, _Board, _Outcome, STARTING_FEN, BB_ALL, move_from_uci
from cython.operator cimport dereference as deref, postincrement as postinc, preincrement as inc
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.optional cimport optional, make_optional
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map

WHITE = True
BLACK = False

# Cpp classes are prefixed with an underscore to avoid name clashes with Python classes
cdef class Outcome:
    cdef _Outcome *outcome_ptr

    def __cinit__(self, termination, winner=None):
        cdef optional[_Color] _winner
        if winner is not None:
            _winner = make_optional[_Color](<_Color>winner)
        self.outcome_ptr = new _Outcome(termination, _winner)

    def __dealloc__(self):
        del self.outcome_ptr
    
    def result(self):
        return self.outcome_ptr.result()

    @property
    def termination(self):
        return self.outcome_ptr.termination

    @property
    def winner(self):
        cdef optional[_Color] winner = self.outcome_ptr.winner
        return winner.value() if winner.has_value() else None

cdef class Move:
    cdef _Move *move_ptr

    def __cinit__(self, from_square, to_square, promotion=None, drop=None):
        cdef optional[_PieceType] _promotion
        cdef optional[_PieceType] _drop
        if promotion is not None:
            _promotion = make_optional[_PieceType](<_PieceType>promotion)
        if drop is not None:
            _drop = make_optional[_PieceType](<_PieceType>drop)
        self.move_ptr = new _Move(from_square, to_square, _promotion, _drop)

    def __dealloc__(self):
        del self.move_ptr

    def __repr__(self):
        return self.uci()

    def uci(self):
        return self.move_ptr.uci().decode('utf-8')

    @property
    def from_square(self):
        return self.move_ptr.from_square

    @property
    def to_square(self):
        return self.move_ptr.to_square

    @property
    def promotion(self):
        cdef optional[_PieceType] promotion = self.move_ptr.promotion
        return promotion.value() if promotion.has_value() else None

    @property
    def drop(self):
        cdef optional[_PieceType] drop = self.move_ptr.drop
        return drop.value() if drop.has_value() else None

    @staticmethod
    def from_uci(uci):
        cdef unique_ptr[_Move] move = make_unique[_Move](move_from_uci(uci.encode('utf-8')))
        return Move(deref(move).from_square, deref(move).to_square)

cdef class Piece:
    cdef _Piece *piece_ptr

    def __cinit__(self, piece_type, color):
        self.piece_ptr = new _Piece(piece_type, color)

    def __dealloc__(self):
        del self.piece_ptr

    def symbol(self):
        cdef char symbol = self.piece_ptr.symbol()
        return chr(symbol)

cdef class Board:
    cdef _Board *board_ptr

    def __cinit__(self, fen=None, chess960=False):
        cdef optional[string] _fen
        if fen is not None:
            bytes = fen.encode('utf-8')
            _fen = make_optional[string](<string>bytes)
        else:
            _fen = make_optional[string](<string>STARTING_FEN)
        self.board_ptr = new _Board(_fen, chess960)

    def __dealloc__(self):
        del self.board_ptr

    def __repr__(self):
        return self.fen()
    
    def copy(self):
        cdef unique_ptr[_Board] board = make_unique[_Board](<_Board>self.board_ptr.copy())
        new_board = Board()
        new_board.board_ptr = board.release()
        return new_board

    def ply(self):
        return self.board_ptr.ply()

    def fen(self):
        return self.board_ptr.fen(False, "legal".encode('utf-8'), False).decode('utf-8')

    def push_uci(self, uci):
        self.board_ptr.push_uci(uci.encode('utf-8'))

    def push_san(self, san):
        self.board_ptr.push_san(san.encode('utf-8'))

    def piece_map(self):
        cdef unordered_map[_Square, _Piece] piece_map = self.board_ptr.piece_map(BB_ALL)
        cdef unordered_map[_Square, _Piece].iterator it = piece_map.begin()
        while it != piece_map.end():
            piece = Piece(deref(it).second.piece_type, deref(it).second.color)
            yield (<int>deref(it).first, piece)
            postinc(it)

    def piece_at(self, square):
        cdef optional[_Piece] piece = self.board_ptr.piece_at(square)
        if not piece.has_value():
            return None
        return Piece(piece.value().piece_type, piece.value().color) 

    def pop(self):
        cdef unique_ptr[_Move] move = make_unique[_Move](self.board_ptr.pop())
        return Move(
            deref(move).from_square,
            deref(move).to_square,
            deref(move).promotion.value() if deref(move).promotion.has_value() else None,
            deref(move).drop.value() if deref(move).drop.has_value() else None
            )

    def is_repetition(self, count):
        return self.board_ptr.is_repetition(count)

    def has_kingside_castling_rights(self, color):
        return self.board_ptr.has_kingside_castling_rights(color)

    def has_queenside_castling_rights(self, color):
        return self.board_ptr.has_queenside_castling_rights(color)

    def is_game_over(self, claim_draw=False):
        return self.board_ptr.is_game_over(claim_draw)

    # Custom outcome method, excluding variant win/loss, seventyfive moves and fivefold repetition
    def outcome(self, claim_draw=True):
        cdef optional[_Outcome] outcome_opt = self.board_ptr.outcome(claim_draw)
        if not outcome_opt.has_value():
            return None
        return Outcome(
            outcome_opt.value().termination, 
            outcome_opt.value().winner.value() if outcome_opt.value().winner.has_value() else None
        )       

    @property
    def turn(self):
        return self.board_ptr.turn

    @property
    def move_stack(self):
        cdef vector[_Move] move_stack = self.board_ptr.move_stack
        cdef vector[_Move].iterator it = move_stack.begin()
        cdef list moves = []
        while it != move_stack.end():
            moves.append(
                Move(
                    deref(it).from_square,
                    deref(it).to_square, 
                    deref(it).promotion.value() if deref(it).promotion.has_value() else None, 
                    deref(it).drop.value() if deref(it).drop.has_value() else None
                ) 
            )
            postinc(it)
        return moves

    def generate_legal_moves(self):
        cdef vector[_Move] move_stack = self.board_ptr.generate_legal_moves(BB_ALL, BB_ALL)
        cdef vector[_Move].iterator it = move_stack.begin()
        cdef list moves = []
        while it != move_stack.end():
            moves.append(
                Move(
                    deref(it).from_square,
                    deref(it).to_square, 
                    deref(it).promotion.value() if deref(it).promotion.has_value() else None, 
                    deref(it).drop.value() if deref(it).drop.has_value() else None
                ) 
            )
            postinc(it)
        return moves

    @property
    def legal_moves(self):
        cdef vector[_Move] move_stack = self.board_ptr.generate_legal_moves(BB_ALL, BB_ALL)
        cdef vector[_Move].iterator it = move_stack.begin()
        while it != move_stack.end():
            yield(
                Move(
                    deref(it).from_square,
                    deref(it).to_square, 
                    deref(it).promotion.value() if deref(it).promotion.has_value() else None, 
                    deref(it).drop.value() if deref(it).drop.has_value() else None
                )
            )               
            postinc(it)        

    @property
    def fullmove_number(self):
        return self.board_ptr.fullmove_number

    @property
    def halfmove_clock(self):
        return self.board_ptr.halfmove_clock
