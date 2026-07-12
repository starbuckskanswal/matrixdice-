import json
import logging
import shutil
import sys
import random
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Final, ClassVar, Literal
from functools import cached_property

import numpy as np
from numpy.linalg import LinAlgError

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

VERSION: Final[str] = "1.2.0"
AUTHOR: Final[str] = "MatrixDice Team"
LICENSE: Final[str] = "MIT"

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.WARNING) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.setLevel(level)


import os
setup_logging(logging.DEBUG if os.environ.get("MATRIXDICE_VERBOSE") else logging.WARNING)


class MatrixDiceError(Exception):
    pass


class ValidationError(MatrixDiceError):
    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__("\n".join(errors))


class RotationError(MatrixDiceError):
    pass


class GenerationError(MatrixDiceError):
    pass


class SerializationError(MatrixDiceError):
    pass


class Position(Enum):
    FRONT = 0
    BACK = 1
    LEFT = 2
    RIGHT = 3
    TOP = 4
    BOTTOM = 5


class Rotation(Enum):
    RIGHT = auto()
    LEFT = auto()
    UP = auto()
    DOWN = auto()
    CLOCKWISE = auto()
    COUNTERCLOCKWISE = auto()


class GenerationStrategy(Enum):
    DIAGONAL = auto()
    RANDOM_CONSTRUCTIVE = auto()
    UPPER_TRIANGULAR = auto()
    SPARSE = auto()


@dataclass
class GenerationStats:
    successful_row_ops: int = 0
    successful_col_ops: int = 0
    rejected_ops: int = 0
    restarts: int = 0
    generation_time: float = 0.0

    def merge(self, other: "GenerationStats") -> None:
        self.successful_row_ops += other.successful_row_ops
        self.successful_col_ops += other.successful_col_ops
        self.rejected_ops += other.rejected_ops
        self.restarts += other.restarts
        self.generation_time += other.generation_time


class MatrixGenerator:
    _MIN_ENTRY: ClassVar[int] = 0
    _MAX_ENTRY: ClassVar[int] = 9
    _SHAPE: ClassVar[Tuple[int, int]] = (3, 3)
    _MIN_SUCCESSFUL_OPS: ClassVar[int] = 500
    _MAX_SUCCESSFUL_OPS: ClassVar[int] = 5000
    _ATTEMPTS_PER_OP: ClassVar[int] = 20

    @classmethod
    def generate(cls, face_number: int, strategy: GenerationStrategy) -> Tuple[np.ndarray, GenerationStats]:
        if face_number < 1 or face_number > 6:
            raise GenerationError(f"Face number must be between 1 and 6, got {face_number}")

        if strategy == GenerationStrategy.DIAGONAL:
            return cls._diagonal(face_number), GenerationStats()
        elif strategy == GenerationStrategy.RANDOM_CONSTRUCTIVE:
            return cls._random_constructive(face_number)
        elif strategy == GenerationStrategy.UPPER_TRIANGULAR:
            return cls._upper_triangular(face_number), GenerationStats()
        elif strategy == GenerationStrategy.SPARSE:
            return cls._sparse(face_number), GenerationStats()
        else:
            raise GenerationError(f"Unknown generation strategy: {strategy}")

    @staticmethod
    def _diagonal(face_number: int) -> np.ndarray:
        return np.diag([face_number, 1, 1]).astype(np.int64)

    @classmethod
    def _upper_triangular(cls, face_number: int) -> np.ndarray:
        mat = np.diag([face_number, 1, 1]).astype(np.int64)
        for i in range(3):
            for j in range(i + 1, 3):
                mat[i, j] = random.randint(0, 3)
        return mat

    @classmethod
    def _sparse(cls, face_number: int) -> np.ndarray:
        return cls._diagonal(face_number)

    @classmethod
    def _apply_row_operation(cls, mat: np.ndarray, i: int, j: int, mult: int) -> np.ndarray:
        new_mat = mat.copy()
        new_mat[j, :] += mult * new_mat[i, :]
        return new_mat

    @classmethod
    def _apply_column_operation(cls, mat: np.ndarray, i: int, j: int, mult: int) -> np.ndarray:
        new_mat = mat.copy()
        new_mat[:, j] += mult * new_mat[:, i]
        return new_mat

    @classmethod
    def _apply_row_swap(cls, mat: np.ndarray, i: int, j: int) -> np.ndarray:
        new_mat = mat.copy()
        new_mat[[i, j], :] = new_mat[[j, i], :]
        return new_mat

    @classmethod
    def _apply_column_swap(cls, mat: np.ndarray, i: int, j: int) -> np.ndarray:
        new_mat = mat.copy()
        new_mat[:, [i, j]] = new_mat[:, [j, i]]
        return new_mat

    @classmethod
    def _check_bounds(cls, mat: np.ndarray) -> bool:
        return bool(np.all((mat >= cls._MIN_ENTRY) & (mat <= cls._MAX_ENTRY)))

    @classmethod
    def _nonzero_mult(cls, low: int = -3, high: int = 3) -> int:
        return random.choice([m for m in range(low, high + 1) if m != 0])

    @classmethod
    def _try_operation(cls, mat: np.ndarray, op_type: str) -> Optional[np.ndarray]:
        if op_type == "row_add":
            i, j = random.sample(range(3), 2)
            new_mat = cls._apply_row_operation(mat, i, j, cls._nonzero_mult())
            return new_mat if cls._check_bounds(new_mat) else None

        if op_type == "col_add":
            i, j = random.sample(range(3), 2)
            new_mat = cls._apply_column_operation(mat, i, j, cls._nonzero_mult())
            return new_mat if cls._check_bounds(new_mat) else None

        if op_type == "multi_row_add":
            temp = mat.copy()
            for _ in range(random.choice([2, 3])):
                i, j = random.sample(range(3), 2)
                temp = cls._apply_row_operation(temp, i, j, cls._nonzero_mult(-2, 2))
                if not cls._check_bounds(temp):
                    return None
            return temp

        if op_type == "multi_col_add":
            temp = mat.copy()
            for _ in range(random.choice([2, 3])):
                i, j = random.sample(range(3), 2)
                temp = cls._apply_column_operation(temp, i, j, cls._nonzero_mult(-2, 2))
                if not cls._check_bounds(temp):
                    return None
            return temp

        if op_type == "row_swap_pair":
            temp = mat.copy()
            for _ in range(2):
                i, j = random.sample(range(3), 2)
                temp = cls._apply_row_swap(temp, i, j)
            return temp if cls._check_bounds(temp) else None

        if op_type == "col_swap_pair":
            temp = mat.copy()
            for _ in range(2):
                i, j = random.sample(range(3), 2)
                temp = cls._apply_column_swap(temp, i, j)
            return temp if cls._check_bounds(temp) else None

        raise GenerationError(f"Unknown operation type: {op_type}")

    @classmethod
    def _random_constructive(cls, face_number: int) -> Tuple[np.ndarray, GenerationStats]:
        start = time.perf_counter()
        stats = GenerationStats()
        op_types = [
            "row_add", "col_add",
            "multi_row_add", "multi_col_add",
            "row_swap_pair", "col_swap_pair",
        ]

        while True:
            mat = cls._diagonal(face_number).copy()
            target_ops = random.randint(cls._MIN_SUCCESSFUL_OPS, cls._MAX_SUCCESSFUL_OPS)
            max_attempts = target_ops * cls._ATTEMPTS_PER_OP

            successful = 0
            attempts = 0
            row_ops = 0
            col_ops = 0
            rejected = 0

            while successful < target_ops and attempts < max_attempts:
                attempts += 1
                op_type = random.choice(op_types)
                candidate = cls._try_operation(mat, op_type)
                if candidate is not None:
                    mat = candidate
                    successful += 1
                    if "row" in op_type:
                        row_ops += 1
                    else:
                        col_ops += 1
                else:
                    rejected += 1

            det = int(round(np.linalg.det(mat)))
            if det == face_number:
                stats.successful_row_ops = row_ops
                stats.successful_col_ops = col_ops
                stats.rejected_ops = rejected
                stats.generation_time = time.perf_counter() - start
                logger.debug(
                    f"Face {face_number}: generated after {successful} successful ops "
                    f"({row_ops} row, {col_ops} col), {rejected} rejected, "
                    f"{stats.restarts} restart(s)."
                )
                return mat, stats

            stats.restarts += 1
            logger.debug(
                f"Face {face_number}: determinant drifted to {det}, restarting generation "
                f"(restart #{stats.restarts})."
            )


@dataclass
class Face:
    face_number: int
    matrix: np.ndarray

    def __post_init__(self) -> None:
        if self.matrix.shape != (3, 3):
            raise ValueError("Matrix must be 3x3.")
        if not np.issubdtype(self.matrix.dtype, np.integer):
            self.matrix = self.matrix.astype(np.int64)

    @cached_property
    def determinant(self) -> int:
        det = np.linalg.det(self.matrix)
        return int(round(det))

    @cached_property
    def exact_determinant(self) -> float:
        return float(np.linalg.det(self.matrix))

    @cached_property
    def rank(self) -> int:
        return int(np.linalg.matrix_rank(self.matrix))

    @cached_property
    def trace(self) -> int:
        return int(np.trace(self.matrix))

    @cached_property
    def eigenvalues(self) -> np.ndarray:
        return np.linalg.eigvals(self.matrix)

    @cached_property
    def invertible(self) -> bool:
        return not np.isclose(self.determinant, 0.0)

    @cached_property
    def inverse(self) -> Optional[np.ndarray]:
        if self.invertible:
            try:
                return np.linalg.inv(self.matrix)
            except LinAlgError:
                return None
        return None

    @cached_property
    def condition_number(self) -> Optional[float]:
        if self.invertible:
            try:
                return float(np.linalg.cond(self.matrix))
            except LinAlgError:
                return None
        return None

    @cached_property
    def density(self) -> float:
        return float(np.count_nonzero(self.matrix)) / self.matrix.size

    @cached_property
    def sparsity(self) -> float:
        return 1.0 - self.density

    def verify_inverse(self, tolerance: float = 1e-6) -> bool:
        if self.inverse is None:
            return False
        identity_check = self.matrix @ self.inverse
        return bool(np.allclose(identity_check, np.eye(3), atol=tolerance))

    def has_duplicate_eigenvalues(self, tolerance: float = 1e-6) -> bool:
        eigs = self.eigenvalues
        for a in range(len(eigs)):
            for b in range(a + 1, len(eigs)):
                if np.isclose(eigs[a], eigs[b], atol=tolerance):
                    return True
        return False

    def has_integer_overflow_risk(self) -> bool:
        max_safe = np.iinfo(np.int64).max
        return bool(np.any(np.abs(self.matrix.astype(np.int64)) > max_safe // 1000))

    def validate(self) -> List[str]:
        errors = []

        if self.matrix.shape != (3, 3):
            errors.append(f"Face {self.face_number}: matrix must be 3x3, got {self.matrix.shape}")

        if not np.issubdtype(self.matrix.dtype, np.integer):
            errors.append(
                f"Face {self.face_number}: entries must be integers, got dtype {self.matrix.dtype}"
            )

        if not np.all((self.matrix >= 0) & (self.matrix <= 9)):
            errors.append(f"Face {self.face_number}: entries must be between 0 and 9")

        if abs(self.exact_determinant - self.face_number) > 1e-6:
            errors.append(
                f"Face {self.face_number}: determinant {self.determinant} "
                f"does not equal face number {self.face_number}"
            )

        if not self.invertible:
            errors.append(f"Face {self.face_number}: matrix is not invertible (determinant zero)")

        if self.rank != 3:
            errors.append(f"Face {self.face_number}: rank {self.rank} is not 3 (not full rank)")

        if self.invertible and not self.verify_inverse():
            errors.append(f"Face {self.face_number}: A @ A^-1 does not match the identity within tolerance")

        if self.has_integer_overflow_risk():
            errors.append(f"Face {self.face_number}: entries at risk of integer overflow")

        if self.has_duplicate_eigenvalues():
            logger.debug(f"Face {self.face_number}: has (near-)duplicate eigenvalues.")

        if self.condition_number is not None and self.condition_number > 200:
            logger.debug(
                f"Face {self.face_number}: high condition number ({self.condition_number:.2f}), "
                "matrix is close to singular relative to its scale."
            )

        if not errors:
            logger.debug(f"Face {self.face_number} validated successfully.")

        return errors

    def _format_eigenvalues(self, tolerance: float = 1e-6) -> str:
        """Format eigenvalues compactly for console display.

        ``np.linalg.eigvals`` always returns a complex-dtype array for a
        real matrix whose eigenvalues *could* be complex, even when every
        imaginary part actually computed out to (numerically) zero. That
        forces numpy to print every entry as e.g. ``12.625+0.j``, which
        is needlessly wide and is what caused combined display rows to
        exceed typical terminal widths and get soft-wrapped mid-field.
        Here, real-valued eigenvalues are printed as plain floats, and
        only genuinely complex ones keep their ``a+bj`` form.
        """
        parts: List[str] = []
        for value in self.eigenvalues:
            if abs(value.imag) < tolerance:
                parts.append(f"{value.real:.2f}")
            else:
                sign = "+" if value.imag >= 0 else "-"
                parts.append(f"{value.real:.2f}{sign}{abs(value.imag):.2f}j")
        return "[" + ", ".join(parts) + "]"

    def display_lines(self, verbose: bool = False) -> List[str]:
        lines = []
        lines.append(f"Face {self.face_number}")
        mat_str = np.array2string(self.matrix, separator=' ')
        lines.extend(mat_str.split('\n'))
        lines.append(f"Determinant : {self.determinant}")
        lines.append(f"Rank        : {self.rank}")
        lines.append(f"Trace       : {self.trace}")
        lines.append(f"Invertible  : {'Yes' if self.invertible else 'No'}")
        if verbose:
            lines.append(f"Eigenvalues : {self._format_eigenvalues()}")
            if self.condition_number is not None:
                lines.append(f"Condition # : {self.condition_number:.3f}")
            else:
                lines.append("Condition # : N/A")
        return lines

    def to_dict(self) -> Dict[str, Any]:
        return {
            "face_number": self.face_number,
            "matrix": self.matrix.tolist(),
            "determinant": self.determinant,
            "rank": self.rank,
            "trace": self.trace,
            "invertible": self.invertible,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Face":
        return cls(face_number=data["face_number"], matrix=np.array(data["matrix"], dtype=np.int64))


class Validator:
    @staticmethod
    def validate_cube(cube: "MatrixCube") -> None:
        start = time.perf_counter()
        logger.debug("Validating cube...")
        all_errors = []

        for pos, face in cube.faces.items():
            errors = face.validate()
            if errors:
                all_errors.extend([f"Position {pos.name}: {e}" for e in errors])

        if len(cube.faces) != 6:
            all_errors.append(f"Cube has {len(cube.faces)} faces, expected 6.")

        expected_positions = set(Position)
        actual_positions = set(cube.faces.keys())
        if actual_positions != expected_positions:
            missing = expected_positions - actual_positions
            extra = actual_positions - expected_positions
            if missing:
                all_errors.append(f"Missing positions: {[p.name for p in missing]}")
            if extra:
                all_errors.append(f"Extra positions: {[p.name for p in extra]}")

        face_numbers = {face.face_number for face in cube.faces.values()}
        if face_numbers != {1, 2, 3, 4, 5, 6}:
            all_errors.append(f"Face numbers incorrect: {face_numbers}")

        dets = [face.determinant for face in cube.faces.values()]
        if sorted(dets) != [1, 2, 3, 4, 5, 6]:
            all_errors.append(f"Determinants not exactly 1..6: {dets}")

        for face in cube.faces.values():
            if face.face_number != face.determinant:
                all_errors.append(
                    f"Face {face.face_number} has determinant {face.determinant}, mismatch."
                )

        positions = list(cube.faces.keys())
        for a in range(len(positions)):
            for b in range(a + 1, len(positions)):
                mat_a = cube.faces[positions[a]].matrix
                mat_b = cube.faces[positions[b]].matrix
                if np.array_equal(mat_a, mat_b):
                    all_errors.append(
                        f"Duplicate matrix detected between {positions[a].name} and {positions[b].name}."
                    )

        cube.last_validation_time = time.perf_counter() - start

        if all_errors:
            logger.debug("Cube validation failed with %d errors.", len(all_errors))
            raise ValidationError(all_errors)

        logger.debug("Cube validation successful.")


class MatrixCube:
    _ROTATION_MAPS = {
        Rotation.RIGHT: {
            Position.FRONT: Position.LEFT,
            Position.LEFT: Position.BACK,
            Position.BACK: Position.RIGHT,
            Position.RIGHT: Position.FRONT,
            Position.TOP: Position.TOP,
            Position.BOTTOM: Position.BOTTOM,
        },
        Rotation.LEFT: {
            Position.FRONT: Position.RIGHT,
            Position.RIGHT: Position.BACK,
            Position.BACK: Position.LEFT,
            Position.LEFT: Position.FRONT,
            Position.TOP: Position.TOP,
            Position.BOTTOM: Position.BOTTOM,
        },
        Rotation.UP: {
            Position.FRONT: Position.TOP,
            Position.TOP: Position.BACK,
            Position.BACK: Position.BOTTOM,
            Position.BOTTOM: Position.FRONT,
            Position.LEFT: Position.LEFT,
            Position.RIGHT: Position.RIGHT,
        },
        Rotation.DOWN: {
            Position.FRONT: Position.BOTTOM,
            Position.BOTTOM: Position.BACK,
            Position.BACK: Position.TOP,
            Position.TOP: Position.FRONT,
            Position.LEFT: Position.LEFT,
            Position.RIGHT: Position.RIGHT,
        },
    }
    _ROTATION_MAPS[Rotation.CLOCKWISE] = _ROTATION_MAPS[Rotation.RIGHT]
    _ROTATION_MAPS[Rotation.COUNTERCLOCKWISE] = _ROTATION_MAPS[Rotation.LEFT]

    def __init__(self, strategy: GenerationStrategy = GenerationStrategy.DIAGONAL):
        self.faces: Dict[Position, Face] = {}
        self.move_count: int = 0
        self.rotation_history: List[Rotation] = []
        self.generation_strategy = strategy
        self.generation_stats = GenerationStats()
        self.last_validation_time: float = 0.0
        logger.debug("MatrixCube instance created with strategy %s.", strategy.name)

    def generate_faces(self, strategy: Optional[GenerationStrategy] = None) -> None:
        if strategy is None:
            strategy = self.generation_strategy

        logger.debug("Generating cube faces using strategy %s...", strategy.name)
        self.faces.clear()
        self.generation_stats = GenerationStats()
        pos_map = {
            1: Position.FRONT,
            2: Position.BACK,
            3: Position.LEFT,
            4: Position.RIGHT,
            5: Position.TOP,
            6: Position.BOTTOM,
        }
        for i in range(1, 7):
            matrix, stats = MatrixGenerator.generate(i, strategy)
            face = Face(face_number=i, matrix=matrix)
            self.faces[pos_map[i]] = face
            self.generation_stats.merge(stats)
            logger.debug("Face %d generated and assigned to %s.", i, pos_map[i].name)

        logger.debug("All faces generated.")

    def generate_random_cube(self) -> None:
        self.generate_faces(GenerationStrategy.RANDOM_CONSTRUCTIVE)
        logger.debug("Random cube generated.")

    def validate(self) -> None:
        Validator.validate_cube(self)

    def _apply_rotation(self, rotation: Rotation) -> None:
        if rotation not in self._ROTATION_MAPS:
            raise RotationError(f"Unsupported rotation: {rotation}")

        mapping = self._ROTATION_MAPS[rotation]
        old_faces = self.faces.copy()
        new_faces = {}
        for new_pos, old_pos in mapping.items():
            new_faces[new_pos] = old_faces[old_pos]
        self.faces = new_faces
        self.move_count += 1
        self.rotation_history.append(rotation)
        logger.debug("Rotation %s applied. Move count = %d.", rotation.name, self.move_count)

    def rotate(self, rotation: Union[Rotation, str]) -> None:
        if isinstance(rotation, str):
            try:
                rotation = Rotation[rotation.upper()]
            except KeyError:
                raise RotationError(f"Unknown rotation name: {rotation}")
        self._apply_rotation(rotation)

    def rotate_right(self) -> None:
        self.rotate(Rotation.RIGHT)

    def rotate_left(self) -> None:
        self.rotate(Rotation.LEFT)

    def rotate_up(self) -> None:
        self.rotate(Rotation.UP)

    def rotate_down(self) -> None:
        self.rotate(Rotation.DOWN)

    def rotate_clockwise(self) -> None:
        self.rotate(Rotation.CLOCKWISE)

    def rotate_counterclockwise(self) -> None:
        self.rotate(Rotation.COUNTERCLOCKWISE)

    def rotate_face_orientation(self) -> None:
        raise NotImplementedError("rotate_face_orientation will be implemented in a later version.")

    def display(self, mode: Literal["compact", "detailed"] = "detailed", verbose: bool = False) -> None:
        print("\n" + "=" * 60)
        if mode == "compact":
            print("Current cube state (compact):")
        else:
            print("Current cube state (unfolded):")
        print("=" * 60)

        face_lines = {}
        for pos in Position:
            face = self.faces.get(pos)
            if face is None:
                lines = [f"Position {pos.name}: MISSING"]
            else:
                if mode == "compact":
                    lines = [f"Face {face.face_number}", f"det = {face.determinant}"]
                else:
                    lines = face.display_lines(verbose=verbose)
            width = max(len(line) for line in lines) if lines else 0
            face_lines[pos] = (lines, width)

        max_width = max(width for _, width in face_lines.values()) if face_lines else 0

        def pad(lines: List[str], w: int) -> List[str]:
            return [line.ljust(w) for line in lines]

        # The middle row of the unfolded net normally prints LEFT, FRONT,
        # RIGHT, and BACK side by side. That row's width is 4 * max_width
        # plus separators. If the terminal is narrower than that, the
        # terminal itself will soft-wrap the line mid-field, which makes
        # the output look corrupted/misaligned (fields appear to repeat
        # under the wrong column). Detect that case and fall back to a
        # simple stacked, single-column layout instead.
        separator = "  "
        four_wide_row_width = max_width * 4 + len(separator) * 3
        terminal_width = shutil.get_terminal_size(fallback=(80, 24)).columns

        if four_wide_row_width > terminal_width:
            self._display_stacked(face_lines)
            print("=" * 60 + "\n")
            return

        top_lines = pad(face_lines[Position.TOP][0], max_width)
        for line in top_lines:
            print(" " * (max_width * 2 + 4) + line)
        print()

        left_lines = pad(face_lines[Position.LEFT][0], max_width)
        front_lines = pad(face_lines[Position.FRONT][0], max_width)
        right_lines = pad(face_lines[Position.RIGHT][0], max_width)
        back_lines = pad(face_lines[Position.BACK][0], max_width)

        max_height = max(len(left_lines), len(front_lines), len(right_lines), len(back_lines))
        left_padded = left_lines + [""] * (max_height - len(left_lines))
        front_padded = front_lines + [""] * (max_height - len(front_lines))
        right_padded = right_lines + [""] * (max_height - len(right_lines))
        back_padded = back_lines + [""] * (max_height - len(back_lines))

        for i in range(max_height):
            left = left_padded[i]
            front = front_padded[i]
            right = right_padded[i]
            back = back_padded[i]
            print(f"{left: <{max_width}}  {front: <{max_width}}  {right: <{max_width}}  {back: <{max_width}}")
        print()

        bottom_lines = pad(face_lines[Position.BOTTOM][0], max_width)
        for line in bottom_lines:
            print(" " * (max_width * 2 + 4) + line)

        print("=" * 60 + "\n")

    @staticmethod
    def _display_stacked(face_lines: Dict[Position, Tuple[List[str], int]]) -> None:
        """Print each face's block one after another, in a fixed reading
        order (TOP, LEFT, FRONT, RIGHT, BACK, BOTTOM). Used automatically
        by display() whenever the terminal is too narrow for the normal
        4-column unfolded-net layout, so long fields (like eigenvalues)
        never get soft-wrapped into the wrong column.
        """
        order = [Position.TOP, Position.LEFT, Position.FRONT,
                 Position.RIGHT, Position.BACK, Position.BOTTOM]
        for pos in order:
            lines, _ = face_lines[pos]
            print(f"--- {pos.name} ---")
            for line in lines:
                print(line)
            print()

    def statistics(self, print_output: bool = True) -> Dict[str, Any]:
        if not self.faces:
            if print_output:
                print("\n" + "=" * 60)
                print("Cube Statistics")
                print("=" * 60)
                print("Cube is empty.")
                print("=" * 60 + "\n")
            return {}

        faces = list(self.faces.values())
        dets = [f.determinant for f in faces]
        traces = [f.trace for f in faces]
        ranks = [f.rank for f in faces]
        conds = [f.condition_number for f in faces if f.condition_number is not None]
        densities = [f.density for f in faces]
        sparsities = [f.sparsity for f in faces]
        all_eigs = np.concatenate([f.eigenvalues for f in faces])
        all_eigs_real = np.real(all_eigs)

        stats: Dict[str, Any] = {
            "faces": len(faces),
            "generation_strategy": self.generation_strategy.name,
            "total_moves": self.move_count,
            "average_determinant": sum(dets) / len(dets),
            "average_trace": sum(traces) / len(traces),
            "average_rank": sum(ranks) / len(ranks),
            "average_condition_number": (sum(conds) / len(conds)) if conds else None,
            "max_determinant": max(dets),
            "min_determinant": min(dets),
            "largest_eigenvalue": float(np.max(all_eigs_real)),
            "smallest_eigenvalue": float(np.min(all_eigs_real)),
            "average_density": sum(densities) / len(densities),
            "average_sparsity": sum(sparsities) / len(sparsities),
            "generation_time": self.generation_stats.generation_time,
            "validation_time": self.last_validation_time,
            "successful_row_ops": self.generation_stats.successful_row_ops,
            "successful_col_ops": self.generation_stats.successful_col_ops,
            "rejected_ops": self.generation_stats.rejected_ops,
            "restarts": self.generation_stats.restarts,
        }

        try:
            self.validate()
            stats["valid"] = True
        except ValidationError:
            stats["valid"] = False

        if print_output:
            print("\n" + "=" * 60)
            print("Cube Statistics")
            print("=" * 60)
            print(f"Faces                     : {stats['faces']}")
            print(f"Generation Strategy       : {stats['generation_strategy']}")
            print(f"Total Moves               : {stats['total_moves']}")
            print(f"Average determinant       : {stats['average_determinant']:.2f}")
            print(f"Average trace             : {stats['average_trace']:.2f}")
            print(f"Average rank              : {stats['average_rank']:.2f}")
            if stats['average_condition_number'] is not None:
                print(f"Average condition number  : {stats['average_condition_number']:.3f}")
            else:
                print("Average condition number  : N/A")
            print(f"Maximum determinant       : {stats['max_determinant']}")
            print(f"Minimum determinant       : {stats['min_determinant']}")
            print(f"Largest eigenvalue        : {stats['largest_eigenvalue']:.3f}")
            print(f"Smallest eigenvalue       : {stats['smallest_eigenvalue']:.3f}")
            print(f"Average matrix density    : {stats['average_density']:.3f}")
            print(f"Average sparsity          : {stats['average_sparsity']:.3f}")
            print(f"Generation time           : {stats['generation_time']:.4f}s")
            print(f"Validation time           : {stats['validation_time']:.6f}s")
            print(f"Successful row operations : {stats['successful_row_ops']}")
            print(f"Successful col operations : {stats['successful_col_ops']}")
            print(f"Rejected operations       : {stats['rejected_ops']}")
            print(f"Generation restarts       : {stats['restarts']}")
            print(f"Cube Valid                : {'Yes' if stats['valid'] else 'No'}")
            print("=" * 60 + "\n")

        return stats

    def summary(self) -> None:
        print("\n" + "=" * 60)
        print(f"MatrixDice v{VERSION}")
        print("=" * 60)
        print(f"Author          : {AUTHOR}")
        print(f"License         : {LICENSE}")
        print(f"Faces           : {len(self.faces)}")
        print(f"Generation Strategy: {self.generation_strategy.name}")
        try:
            self.validate()
            valid = True
        except ValidationError:
            valid = False
        print(f"Cube Valid      : {'Yes' if valid else 'No'}")
        print(f"Moves           : {self.move_count}")
        if self.rotation_history:
            print("Rotation History:")
            for rot in self.rotation_history:
                print(f"  {rot.name}")
        else:
            print("Rotation History: (none)")
        if self.faces:
            dets = [face.determinant for face in self.faces.values()]
            traces = [face.trace for face in self.faces.values()]
            print("Face Statistics:")
            print(f"  Determinants: {dets}")
            print(f"  Traces     : {traces}")
        print("=" * 60 + "\n")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": VERSION,
            "generation_strategy": self.generation_strategy.name,
            "move_count": self.move_count,
            "rotation_history": [r.name for r in self.rotation_history],
            "faces": {
                pos.name: face.to_dict()
                for pos, face in self.faces.items()
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MatrixCube":
        version = data.get("version")
        if version is None:
            raise SerializationError("Missing version in saved data.")
        if version != VERSION and not version.startswith("1."):
            raise SerializationError(f"Unsupported version: {version} (expected {VERSION})")

        strategy_name = data.get("generation_strategy")
        if strategy_name not in GenerationStrategy.__members__:
            raise SerializationError(f"Unknown generation strategy: {strategy_name}")
        strategy = GenerationStrategy[strategy_name]

        cube = cls(strategy=strategy)
        cube.move_count = data.get("move_count", 0)
        rotation_names = data.get("rotation_history", [])
        cube.rotation_history = []
        for name in rotation_names:
            if name in Rotation.__members__:
                cube.rotation_history.append(Rotation[name])
            else:
                logger.warning(f"Ignoring unknown rotation: {name}")

        faces_data = data.get("faces", {})
        for pos_name, face_data in faces_data.items():
            if pos_name not in Position.__members__:
                logger.warning(f"Ignoring unknown position: {pos_name}")
                continue
            pos = Position[pos_name]
            face = Face.from_dict(face_data)
            cube.faces[pos] = face

        try:
            cube.validate()
        except ValidationError as e:
            raise SerializationError(f"Loaded cube validation failed: {e}")

        return cube

    def save_json(self, filepath: Union[str, Path]) -> None:
        try:
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.debug(f"Cube saved to {filepath}.")
        except Exception as e:
            raise SerializationError(f"Failed to save cube: {e}")

    @classmethod
    def load_json(cls, filepath: Union[str, Path]) -> "MatrixCube":
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            cube = cls.from_dict(data)
            logger.debug(f"Cube loaded from {filepath}.")
            return cube
        except Exception as e:
            raise SerializationError(f"Failed to load cube: {e}")

    def self_test(self) -> bool:
        results = []

        try:
            test_cube = MatrixCube(strategy=GenerationStrategy.RANDOM_CONSTRUCTIVE)
            test_cube.generate_random_cube()
            results.append(("Cube generation", True))
        except Exception as e:
            results.append((f"Cube generation (error: {e})", False))

        try:
            test_cube.validate()
            results.append(("Validation", True))
        except Exception as e:
            results.append((f"Validation (error: {e})", False))

        try:
            test_cube.rotate_right()
            test_cube.rotate_up()
            test_cube.rotate_left()
            test_cube.rotate_down()
            face_numbers_after = sorted([face.face_number for face in test_cube.faces.values()])
            results.append(("Rotations (integrity)", face_numbers_after == [1, 2, 3, 4, 5, 6]))
        except Exception as e:
            results.append((f"Rotations (error: {e})", False))

        try:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tf:
                temp_path = tf.name
            test_cube.save_json(temp_path)
            loaded = MatrixCube.load_json(temp_path)
            for pos, face in test_cube.faces.items():
                if not np.array_equal(face.matrix, loaded.faces[pos].matrix):
                    raise SerializationError("Matrix mismatch after load.")
                if face.determinant != loaded.faces[pos].determinant:
                    raise SerializationError("Determinant mismatch after load.")
            Path(temp_path).unlink()
            results.append(("Serialization/Deserialization", True))
        except Exception as e:
            results.append((f"Serialization (error: {e})", False))

        try:
            dets = [face.determinant for face in test_cube.faces.values()]
            results.append(("Determinants preserved", sorted(dets) == [1, 2, 3, 4, 5, 6]))
        except Exception as e:
            results.append((f"Determinants (error: {e})", False))

        for name, passed in results:
            status = "OK " if passed else "FAIL"
            print(f"  [{status}] {name}")

        all_passed = all(passed for _, passed in results)
        if not all_passed:
            raise AssertionError("Self-test failed.")
        return all_passed

    def tower_of_hanoi_validation(self) -> None:
        raise NotImplementedError("tower_of_hanoi_validation will be implemented in v2.0.")

    def scramble(self) -> None:
        raise NotImplementedError("scramble will be implemented in v2.0.")

    def solve_astar(self) -> None:
        raise NotImplementedError("solve_astar will be implemented in v2.0.")

    def solve_bfs(self) -> None:
        raise NotImplementedError("solve_bfs will be implemented in v2.0.")

    def solve_dfs(self) -> None:
        raise NotImplementedError("solve_dfs will be implemented in v2.0.")

    def solve_idastar(self) -> None:
        raise NotImplementedError("solve_idastar will be implemented in v2.0.")

    def solve_mcts(self) -> None:
        raise NotImplementedError("solve_mcts will be implemented in v2.0.")

    def solve_rl(self) -> None:
        raise NotImplementedError("solve_rl will be implemented in v2.0.")

    def visualize_3d(self) -> None:
        raise NotImplementedError("visualize_3d will be implemented in v2.0.")

    def gui(self) -> None:
        raise NotImplementedError("gui will be implemented in v2.0.")

    def multiplayer_mode(self) -> None:
        raise NotImplementedError("multiplayer_mode will be implemented in a later version.")

    def difficulty_score(self) -> None:
        raise NotImplementedError("difficulty_score will be implemented in v2.0.")


class Visualizer:
    FACE_COLORS: ClassVar[Dict[int, str]] = {
        1: "#C9A8E0",
        2: "#F5A65B",
        3: "#88C7E8",
        4: "#F2938C",
        5: "#A8DDB0",
        6: "#F7D774",
    }
    MISSING_COLOR: ClassVar[str] = "#EEEEEE"
    TITLE_COLOR: ClassVar[str] = "#1A1A1A"

    _NET_ORDER: ClassVar[List[Tuple[Position, int, int]]] = [
        (Position.TOP, 0, 1),
        (Position.LEFT, 1, 0),
        (Position.FRONT, 1, 1),
        (Position.RIGHT, 1, 2),
        (Position.BACK, 1, 3),
        (Position.BOTTOM, 2, 1),
    ]

    @classmethod
    def _color_for(cls, face: Optional[Face]) -> str:
        if face is None:
            return cls.MISSING_COLOR
        return cls.FACE_COLORS.get(face.face_number, cls.MISSING_COLOR)

    @classmethod
    def _draw_face_card(
        cls, ax, x: float, y: float, size: float,
        face: Optional[Face], position: Optional[Position],
        show_matrix: bool = True, stats_mode: Literal["full", "det", "none"] = "full",
        title_fontsize: float = 13, body_fontsize: float = 11,
    ) -> None:
        pad = size * 0.04
        color = cls._color_for(face)
        box = FancyBboxPatch(
            (x + pad, y + pad), size - 2 * pad, size - 2 * pad,
            boxstyle=f"round,pad=0,rounding_size={size * 0.06}",
            linewidth=2.2, edgecolor="black", facecolor=color,
        )
        ax.add_patch(box)

        cx = x + size / 2
        if face is None:
            label = position.name if position else "MISSING"
            ax.text(cx, y + size / 2, f"{label}\nMISSING", ha="center", va="center",
                    fontsize=title_fontsize, fontweight="bold", color="black")
            return

        top_y = y + size * 0.90
        ax.text(cx, top_y, f"Face {face.face_number}", ha="center", va="top",
                fontsize=title_fontsize, fontweight="bold", color="black")
        if position is not None:
            ax.text(cx, top_y - size * 0.095, position.name, ha="center", va="top",
                    fontsize=title_fontsize * 0.75, fontweight="bold", color="#2B2B2B")

        if show_matrix:
            matrix_text = "\n".join(
                "[" + " ".join(f"{v:>2}" for v in row) + "]" for row in face.matrix.tolist()
            )
            matrix_y = y + size * (0.56 if stats_mode == "full" else 0.48)
            ax.text(cx, matrix_y, matrix_text, ha="center", va="center",
                    family="monospace", fontsize=body_fontsize, color="black")

        if stats_mode == "full":
            cond_str = f"{face.condition_number:.2f}" if face.condition_number is not None else "N/A"
            stats_text = (
                f"det = {face.determinant}\n"
                f"rank = {face.rank}\n"
                f"trace = {face.trace}\n"
                f"cond = {cond_str}"
            )
            ax.text(cx, y + size * 0.14, stats_text, ha="center", va="bottom",
                    fontsize=body_fontsize * 0.85, color="#222222")
        elif stats_mode == "det":
            ax.text(cx, y + size * 0.10, f"det = {face.determinant}", ha="center", va="bottom",
                    fontsize=body_fontsize * 0.95, fontweight="bold", color="#222222")

    @classmethod
    def _draw_net(
        cls, ax, cube: "MatrixCube", show_matrix: bool, stats_mode: Literal["full", "det", "none"],
        size: float = 1.7, show_position: bool = True,
        title_fontsize: float = 13, body_fontsize: float = 11,
    ) -> None:
        ax.set_xlim(-0.15 * size, 4.15 * size)
        ax.set_ylim(-0.15 * size, 3.15 * size)
        ax.set_aspect("equal")
        ax.axis("off")
        for pos, row, col in cls._NET_ORDER:
            face = cube.faces.get(pos)
            x = col * size
            y = (2 - row) * size
            cls._draw_face_card(
                ax, x, y, size, face, pos if show_position else None,
                show_matrix=show_matrix, stats_mode=stats_mode,
                title_fontsize=title_fontsize, body_fontsize=body_fontsize,
            )

    @classmethod
    def draw_cube_net(cls, cube: "MatrixCube", output_path: Union[str, Path]) -> None:
        fig, ax = plt.subplots(figsize=(14, 11), facecolor="white")
        ax.set_facecolor("white")
        fig.suptitle("MatrixDice \u2014 Cube State", fontsize=26, fontweight="bold", color=cls.TITLE_COLOR)
        cls._draw_net(ax, cube, show_matrix=True, stats_mode="full")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(output_path, dpi=150, facecolor="white")
        plt.close(fig)

    @classmethod
    def draw_cube_compact(cls, cube: "MatrixCube", output_path: Union[str, Path]) -> None:
        fig, ax = plt.subplots(figsize=(13, 10), facecolor="white")
        ax.set_facecolor("white")
        fig.suptitle("MatrixDice \u2014 Cube Net", fontsize=26, fontweight="bold", color=cls.TITLE_COLOR)
        cls._draw_net(ax, cube, show_matrix=True, stats_mode="none")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(output_path, dpi=150, facecolor="white")
        plt.close(fig)

    @staticmethod
    def _projected_pixel_height(ax, points_3d: List[Tuple[float, float, float]]) -> float:
        """Return the vertical screen-space extent (in pixels) that a set
        of 3D points occupies once projected through the current 3D view.

        Used to size text labels per-face: in an isometric view, faces
        are foreshortened by different amounts depending on orientation
        (e.g. with a low elevation angle, the TOP face projects to a much
        shorter diamond than the FRONT/RIGHT faces do), so a single fixed
        font size for every face either overflows the short faces or
        looks tiny on the tall ones.
        """
        xs, ys, zs = zip(*points_3d)
        proj = ax.get_proj()
        x2d, y2d, _ = proj3d.proj_transform(xs, ys, zs, proj)
        display_points = [ax.transData.transform((x, y)) for x, y in zip(x2d, y2d)]
        pixel_heights = [p[1] for p in display_points]
        return max(pixel_heights) - min(pixel_heights)

    @classmethod
    def draw_cube_isometric(cls, cube: "MatrixCube", output_path: Union[str, Path]) -> None:
        fig = plt.figure(figsize=(9, 8), facecolor="white")
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("white")

        # Matplotlib's default 3D projection is a perspective projection,
        # which is non-linear: the mathematical centroid of a face's four
        # corners does not always project to the visual center of that
        # face on screen (it drifts toward whichever edge is nearer the
        # camera). Since face labels are anchored at the 3D centroid, that
        # drift could push multi-line text into the cube's border edges.
        # An orthographic projection is affine and preserves centroids,
        # so labels stay centered in their face regardless of the
        # matrix's content (line count) or the current view angle.
        ax.set_proj_type("ortho")

        vertices = {
            Position.TOP: [(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)],
            Position.FRONT: [(-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)],
            Position.RIGHT: [(1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1)],
        }

        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-2, 2)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("MatrixDice \u2014 Isometric View", fontsize=20, fontweight="bold",
                      color=cls.TITLE_COLOR, pad=18)
        ax.view_init(elev=20, azim=-55)

        # Faces and transData must reflect the final view/limits before we
        # can measure projected pixel heights, so force a draw now.
        fig.canvas.draw()

        # A 5-line label (Face N / 3 matrix rows / det = N) at fontsize F
        # needs roughly F * 1.7 px per line; used only as a starting guess.
        # We then measure the *actual* rendered text extent and rescale to
        # fit, since matplotlib's real line spacing can vary slightly by
        # version/backend and an estimate alone isn't reliable enough to
        # guarantee no overlap with the face edges.
        max_fontsize = 9.5
        min_fontsize = 5.0
        lines_per_label = 5
        px_per_line_per_pt = 1.7
        margin_fraction = 0.75  # use only ~75% of the face's height for text

        for pos, pts in vertices.items():
            face = cube.faces.get(pos)
            color = cls._color_for(face)
            poly = Poly3DCollection([pts], facecolor=color, edgecolor="black", linewidths=2.2, alpha=0.88)
            ax.add_collection3d(poly)
            cx = sum(p[0] for p in pts) / 4
            cy = sum(p[1] for p in pts) / 4
            cz = sum(p[2] for p in pts) / 4

            face_height_px = cls._projected_pixel_height(ax, pts)
            usable_px = face_height_px * margin_fraction
            initial_guess = usable_px / (lines_per_label * px_per_line_per_pt)
            fontsize = max(min_fontsize, min(max_fontsize, initial_guess))

            if face is not None:
                lines = [f"Face {face.face_number}"]
                lines.extend(
                    "[" + " ".join(f"{v}" for v in row) + "]" for row in face.matrix.tolist()
                )
                lines.append(f"det = {face.determinant}")
                label = "\n".join(lines)
            else:
                label = f"{pos.name}\nMISSING"

            text_obj = ax.text(cx, cy, cz, label, ha="center", va="center",
                                family="monospace", fontsize=fontsize, fontweight="bold",
                                color="black", linespacing=1.6, zorder=10)

            # Measure the actual rendered height and shrink further if it
            # still overflows the face's usable space.
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            rendered_height_px = text_obj.get_window_extent(renderer=renderer).height
            if rendered_height_px > usable_px and rendered_height_px > 0:
                scale = usable_px / rendered_height_px
                fontsize = max(min_fontsize, fontsize * scale)
                text_obj.set_fontsize(fontsize)

        fig.tight_layout()
        fig.savefig(output_path, dpi=150, facecolor="white")
        plt.close(fig)

    @classmethod
    def draw_rotation_sequence(cls, cube: "MatrixCube", output_path: Union[str, Path]) -> None:
        rotations: List[Optional[Rotation]] = [None, Rotation.RIGHT, Rotation.UP, Rotation.LEFT, Rotation.DOWN]
        labels = ["Initial", "RIGHT", "UP", "LEFT", "DOWN"]

        working = MatrixCube(strategy=cube.generation_strategy)
        working.faces = dict(cube.faces)

        fig, axes = plt.subplots(1, 5, figsize=(22, 5), facecolor="white")
        fig.suptitle("MatrixDice \u2014 Rotation Sequence", fontsize=24, fontweight="bold", color=cls.TITLE_COLOR)
        for ax, rotation, label in zip(axes, rotations, labels):
            if rotation is not None:
                working.rotate(rotation)
            ax.set_facecolor("white")
            cls._draw_net(
                ax, working, show_matrix=False, stats_mode="none", show_position=False,
                size=1.0, title_fontsize=15, body_fontsize=10,
            )
            ax.set_title(label, fontsize=15, fontweight="bold", color=cls.TITLE_COLOR)

        fig.tight_layout(rect=[0, 0, 1, 0.90])
        fig.savefig(output_path, dpi=150, facecolor="white")
        plt.close(fig)

    @classmethod
    def draw_matrix_gallery(cls, cube: "MatrixCube", output_path: Union[str, Path]) -> None:
        fig, axes = plt.subplots(2, 3, figsize=(14, 9), facecolor="white")
        fig.suptitle("MatrixDice \u2014 Matrix Gallery", fontsize=24, fontweight="bold", color=cls.TITLE_COLOR)
        faces_sorted = sorted(cube.faces.values(), key=lambda f: f.face_number)
        for ax, face in zip(axes.flat, faces_sorted):
            ax.set_facecolor("white")
            ax.axis("off")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            cls._draw_face_card(
                ax, 0, 0, 1, face, position=None,
                show_matrix=True, stats_mode="det",
                title_fontsize=15, body_fontsize=13,
            )

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(output_path, dpi=150, facecolor="white")
        plt.close(fig)

    @classmethod
    def generate_all(cls, cube: "MatrixCube", output_dir: Union[str, Path]) -> List[Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        outputs = []
        jobs = [
            ("cube_net.png", cls.draw_cube_net),
            ("cube_compact.png", cls.draw_cube_compact),
            ("cube_isometric.png", cls.draw_cube_isometric),
            ("rotation_sequence.png", cls.draw_rotation_sequence),
            ("matrix_gallery.png", cls.draw_matrix_gallery),
        ]
        for filename, fn in jobs:
            path = output_dir / filename
            fn(cube, path)
            outputs.append(path)
        return outputs


def _section(title: str) -> None:
    print("=" * 50)
    print(title)
    print("=" * 50)


def _ok(msg: str) -> None:
    print(f"[OK] {msg}")


def main() -> None:
    try:
        print()
        _section(f"MatrixDice v{VERSION}")
        print()

        cube = MatrixCube(strategy=GenerationStrategy.RANDOM_CONSTRUCTIVE)

        print("Generating random matrices...")
        cube.generate_faces_verbose = True
        pos_order = [Position.FRONT, Position.BACK, Position.LEFT, Position.RIGHT, Position.TOP, Position.BOTTOM]
        cube.faces.clear()
        cube.generation_stats = GenerationStats()
        for i, pos in zip(range(1, 7), pos_order):
            matrix, stats = MatrixGenerator.generate(i, cube.generation_strategy)
            cube.faces[pos] = Face(face_number=i, matrix=matrix)
            cube.generation_stats.merge(stats)
            _ok(f"Face {i} generated")

        cube.validate()
        _ok("Cube validated")
        print()

        print("Generating visualizations...")
        output_dir = Path("output")
        generated = Visualizer.generate_all(cube, output_dir)
        for path in generated:
            _ok(str(path))
        print()

        print("Saving JSON...")
        json_path = Path("matrixdice_save.json")
        cube.save_json(json_path)
        _ok(str(json_path))
        print()

        print("Running self-test...")
        cube.self_test()
        print()

        cube.display(mode="detailed", verbose=True)

        cube.rotate_right()
        cube.rotate_up()
        cube.rotate_left()
        cube.rotate_down()
        cube.display(mode="compact")

        cube.statistics()
        cube.summary()

        print("Done.")
        print()

    except ValidationError as e:
        print(f"Validation error(s):\n{e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logger.exception("An unexpected error occurred: %s", e)
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()